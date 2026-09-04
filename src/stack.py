"""F6 — stacking meta-model (FOC-211): leak-free OOF base layer + meta variants.

Two CLI phases (contract: docs/FOC-174-meta-model.md §4–§8):

  phase 1 (--run-stack-base --axis <axis>): for every contributing arm, grouped
      k-fold OOF (customers as groups, deterministic fold assignment from
      sorted customer ids) WITHIN TRAIN -> out-of-fold proba for every TRAIN
      row; plus one fit on full TRAIN -> TEST proba. Cached per (arm, axis) to
      artifacts/stack/oof__<arm>__<axis>.npz with row-identity keys (customer,
      timestamp, row index) + manifest__<axis>.json holding sha256 digests,
      fold assignments and per-fold fit sizes. The registry's arm factories and
      the runner's split code path are imported and reused — nothing is
      re-implemented.

  phase 2 (--run-stack --axis <axis>): meta matrix = TRAIN OOF proba block ⊕
      raw base+client tabular ⊕ 1280-d latent fusion. Variants meta-attn /
      meta-ftt / meta-logit / meta-blend / meta-xgb (plus meta-attn-sens on the
      chronological axis only) evaluated through the same harness protocol:
      ONE stratified 25% meta-val carve cut from TRAIN (used both for early
      stopping / C sweep and the frozen threshold — the PRD's single-carve
      contract), one-shot frozen-threshold test evaluation, PR-AUC / ROC-AUC /
      F1 / recall@precision, percentile bootstrap CIs. Meta rows land in the
      same results JSONL with the same schema (no wall-clock fields).

Leak-free contract (asserted, not assumed):
  * fold i's OOF proba never comes from a model fit on fold i's rows — every
    fold's fit set is asserted disjoint from the fold on (customer, timestamp,
    row index) identity, never positional; per-fold fit sizes are recorded.
  * cached blocks are aligned to the frame by identity, never by position.
  * latent + tabular blocks are stateless, label-free functions of the
    enriched frame (the build_features contract); no reduction is fitted
    outside folds.
  * every fit is re-seeded (seed 42; torch.manual_seed + cudnn.deterministic).
  * npz serialization is byte-deterministic (fixed zip timestamps), so a
    --verify re-run must reproduce the cached files bit-for-bit (AC1).

Deviation note (deliberate): meta rows carry no cv_* keys. The base layer IS
the out-of-fold generalization signal; a row-stratified CV on top of OOF
features would re-use folds twice (report §6.7 open item), so meta rows follow
the runner's cv=False shape instead.

Missing-arm tolerance: an arm whose manifest entry is not "ok" (ArmSkipped /
dropped / never built) contributes a proba column of NaN plus a mask flag of
1.0; models handle it (blend nanmean, logit/ftt/attn fill 0.5 and see the mask,
xgb consumes NaN natively). The pre-registered chronological sensitivity
variant (meta-attn-sens) instead EXCLUDES latent-nn-dist / face-features /
demo-features entirely — a sensitivity row, not a purity certificate (§5).
"""

import argparse
import hashlib
import io
import json
import math
import random
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score

import fraud_pipeline as fp

SRC_DIR = Path(__file__).resolve().parent
DEFAULT_STACK_DIR = SRC_DIR.parent / "artifacts" / "stack"

RANDOM_STATE = 42
DEFAULT_K = 5
DEFAULT_TEST_SIZE = 0.2
MISSING_FILL = 0.5  # neutral proba for arms a model cannot see (see mask flags)

SENS_VARIANT = "meta-attn-sens"
BASE_VARIANTS = ["meta-attn", "meta-ftt", "meta-logit", "meta-blend", "meta-xgb"]
CHRON_AXIS = "chronological"
SENS_DROP_ARMS = ("latent-nn-dist", "face-features", "demo-features")

# Latent fusion block widths (face 512 + text 384 + demo 384, arms_fusion.py).
LATENT_BLOCKS = (512, 384, 384)


class StackCacheError(Exception):
    """Phase 2 preconditions unmet (no manifest / no contributing arms)."""


def _one_line(exc):
    return " ".join(str(exc).split())


def _seed_everything():
    # Project determinism convention (arms_tabnet.py): re-pin before EVERY fit
    # so repeated fits in one process, CV clones and fresh CLI processes agree.
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    try:
        import torch  # lazy: optional dependency

        torch.manual_seed(RANDOM_STATE)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def _sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _input_digests(all_trxns_path, exchange_rates_path, dim_customer_path):
    digests = {}
    for name, path in (
        ("all_trxns", all_trxns_path),
        ("exchange_rates", exchange_rates_path),
        ("dim_customer", dim_customer_path),
    ):
        digests[name] = _sha256_file(path)
    return digests


# ---------------------------------------------------------------------------
# Deterministic npz: numpy's savez stamps zip entries with the current time,
# so an identical re-run would produce a different sha256. Fixed-date zip
# members keep the cache byte-stable (AC1) and sha256-comparable.
def _npz_bytes(arrays):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
        for name, arr in arrays.items():
            member = io.BytesIO()
            np.lib.format.write_array(member, np.asanyarray(arr), allow_pickle=False)
            info = zipfile.ZipInfo(filename="%s.npy" % name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            zf.writestr(info, member.getvalue())
    return buf.getvalue()


def _save_npz_deterministic(path, arrays):
    Path(path).write_bytes(_npz_bytes(arrays))


# ---------------------------------------------------------------------------
# Row identity (F3 r3 lesson): joins and asserts on (customer, timestamp,
# row index) — never positional.
def _identity_arrays(enriched, idx):
    sub = enriched.loc[idx]
    assert sub["timestamp"].notna().all(), "identity: NaT timestamps in split"
    customer = np.array(sub["customer"].astype(str).tolist())  # U-dtype, not object
    return (
        customer,
        sub["timestamp"].astype("int64").to_numpy(),  # ns since epoch
        np.asarray(sub.index, dtype=np.int64),
    )


def _identity_key(identity):
    customer, ts_ns, row_index = identity
    order = np.lexsort((ts_ns, customer, row_index))
    return (customer[order], ts_ns[order], row_index[order])


def _assert_identity_match(identity_a, identity_b, what):
    key_a, key_b = _identity_key(identity_a), _identity_key(identity_b)
    for a, b in zip(key_a, key_b):
        assert np.array_equal(a, b), "%s: row identity mismatch" % what


def _assert_identity_disjoint(identity_a, identity_b, what):
    # AC2 leakage probe: no (customer, timestamp, row index) triple may appear
    # in both a fold and the fit set that produced that fold's OOF proba.
    merged = tuple(np.concatenate([a, b]) for a, b in zip(identity_a, identity_b))
    customer, ts_ns, row_index = _identity_key(merged)
    dup = (
        (row_index[1:] == row_index[:-1])
        & (ts_ns[1:] == ts_ns[:-1])
        & (customer[1:] == customer[:-1])
    )
    assert not dup.any(), "%s: fold rows present in the fold's fit set" % what


# ---------------------------------------------------------------------------
# Deterministic grouped folds (PRD §4): customers as groups, assignment
# derived from sorted customer ids. Greedy balance on (fold frauds, fold rows,
# fold index): a fold whose train-side complement holds zero positives would
# fit degenerate OOF models, so positives are spread alongside rows.
def assign_folds(customer_rows, customer_frauds, k):
    fold_of = {}
    fold_frauds = [0] * k
    fold_rows = [0] * k
    for customer in sorted(customer_rows):
        idx = min(range(k), key=lambda i: (fold_frauds[i], fold_rows[i], i))
        fold_of[customer] = idx
        fold_frauds[idx] += int(customer_frauds.get(customer, 0))
        fold_rows[idx] += int(customer_rows[customer])
    return fold_of


# ---------------------------------------------------------------------------
def _manifest_path(stack_dir, axis):
    return Path(stack_dir) / ("manifest__%s.json" % axis)


def _load_manifest(stack_dir, axis):
    path = _manifest_path(stack_dir, axis)
    if not path.exists():
        return {"axis": axis, "arms": {}}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.setdefault("arms", {})
    return manifest


def _save_manifest(stack_dir, axis, manifest):
    # sort_keys + no wall-clock: byte-stable across identical re-runs (AC1).
    _manifest_path(stack_dir, axis).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Phase 1 — base OOF layer.
def oof_one_arm(arm_name, enriched, y, train_idx, test_idx, k):
    # Grouped k-fold OOF within TRAIN + one full-TRAIN fit -> TEST proba, for
    # one arm. Same build_features path as the runner (label-free, full frame).
    arm = fp.ARMS[arm_name]
    assert not arm["placeholder"], "%s is a placeholder arm" % arm_name
    X_raw = arm["build_features"](enriched)
    X_tr, X_te = X_raw.loc[train_idx], X_raw.loc[test_idx]
    y_tr = y.loc[train_idx]
    y_tr_arr = np.asarray(y_tr)

    cust = enriched.loc[train_idx, "customer"].astype(str)
    customer_rows = cust.value_counts().to_dict()
    customer_frauds = y_tr.groupby(cust).sum().astype(int).to_dict()
    fold_of = assign_folds(customer_rows, customer_frauds, k)
    fold_labels = cust.map(fold_of).to_numpy().astype(np.int8)

    train_ident = _identity_arrays(enriched, train_idx)
    oof = np.full(len(train_idx), np.nan)
    fold_report = []
    for i in range(k):
        fold_pos = np.flatnonzero(fold_labels == i)
        fit_pos = np.flatnonzero(fold_labels != i)
        # AC2 fold integrity: the fold's OOF proba must never come from a model
        # fit on that fold's rows — asserted on row identity, per fold.
        _assert_identity_disjoint(
            tuple(arr[fit_pos] for arr in train_ident),
            tuple(arr[fold_pos] for arr in train_ident),
            "%s fold %d leakage probe" % (arm_name, i),
        )
        assert len(fit_pos) + len(fold_pos) == len(train_idx), "%s fold %d lost rows" % (
            arm_name, i,
        )
        fold_report.append(
            {
                "fold": i,
                "fit_rows": int(len(fit_pos)),
                "fit_positives": int(y_tr_arr[fit_pos].sum()),
                "fold_rows": int(len(fold_pos)),
                "fold_positives": int(y_tr_arr[fold_pos].sum()),
            }
        )
        _seed_everything()
        model = arm["make_model"](y_tr.iloc[fit_pos])
        model.fit(X_tr.iloc[fit_pos], y_tr.iloc[fit_pos])
        oof[fold_pos] = np.asarray(model.predict_proba(X_tr.iloc[fold_pos]))[:, 1]

    _seed_everything()
    full_model = arm["make_model"](y_tr)
    full_model.fit(X_tr, y_tr)
    test_proba = np.asarray(full_model.predict_proba(X_te))[:, 1]

    assert np.isfinite(oof).all(), "%s: non-finite OOF proba" % arm_name
    assert np.isfinite(test_proba).all(), "%s: non-finite test proba" % arm_name
    return {
        "k": k,
        "fold_of": fold_of,
        "fold_labels": fold_labels,
        "fold_report": fold_report,
        "train_ident": train_ident,
        "test_ident": _identity_arrays(enriched, test_idx),
        "oof": oof,
        "test_proba": test_proba,
        "n_features": int(X_raw.shape[1]),
    }


def _npz_arrays(result):
    oof_ident, test_ident = result["train_ident"], result["test_ident"]
    return {
        "customer": oof_ident[0],
        "timestamp_ns": oof_ident[1],
        "row_index": oof_ident[2],
        "fold": result["fold_labels"],
        "oof_proba": result["oof"],
        "test_customer": test_ident[0],
        "test_timestamp_ns": test_ident[1],
        "test_row_index": test_ident[2],
        "test_proba": result["test_proba"],
        "k": np.int64(result["k"]),
    }


def run_stack_base(
    axis,
    arm_names,
    k,
    stack_dir,
    refit=False,
    verify=False,
    all_trxns_path=fp.DEFAULT_ALL_TRXNS_PATH,
    exchange_rates_path=fp.DEFAULT_EXCHANGE_RATES_PATH,
    dim_customer_path=fp.DEFAULT_DIM_CUSTOMER_PATH,
):
    stack_dir = Path(stack_dir)
    enriched, y = fp.load_enriched(
        all_trxns_path=all_trxns_path,
        exchange_rates_path=exchange_rates_path,
        dim_customer_path=dim_customer_path,
    )
    train_idx, test_idx = fp.axis_split(axis, enriched, y)  # runner's split path
    stack_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(stack_dir, axis)
    manifest["axis"] = axis
    manifest["axis_description"] = fp.AXES[axis]["description"]
    manifest["k_default"] = k
    manifest["test_size"] = DEFAULT_TEST_SIZE
    manifest["n_rows"] = int(len(enriched))
    manifest["n_train"] = int(len(train_idx))
    manifest["n_test"] = int(len(test_idx))
    manifest["input_sha256"] = _input_digests(
        all_trxns_path, exchange_rates_path, dim_customer_path
    )

    failures, any_ok = [], False
    for arm_name in arm_names:
        entry = manifest["arms"].get(arm_name, {})
        npz_path = stack_dir / ("oof__%s__%s.npz" % (arm_name, axis))
        if verify:
            failures.append(_verify_one(arm_name, entry, npz_path, enriched, y, train_idx, test_idx, k))
            continue
        if npz_path.exists() and not refit and entry.get("status") == "ok":
            # resume: built by an earlier invocation; manifest keeps its digest
            print("cached   %-24s %s" % (arm_name, npz_path.name))
            any_ok = True
            continue
        try:
            result = oof_one_arm(arm_name, enriched, y, train_idx, test_idx, k)
        except fp.ArmSkipped as exc:
            manifest["arms"][arm_name] = {"status": "skipped", "reason": _one_line(exc)}
            _save_manifest(stack_dir, axis, manifest)
            print("skipped  %-24s %s" % (arm_name, _one_line(exc)))
            continue
        arrays = _npz_arrays(result)  # fold_labels computed inside oof_one_arm
        npz_bytes = _npz_bytes(arrays)
        npz_path.write_bytes(npz_bytes)
        manifest["arms"][arm_name] = {
            "status": "ok",
            "k_used": int(k),
            "npz": npz_path.name,
            "npz_sha256": _sha256_bytes(npz_bytes),
            "n_oof_rows": int(len(train_idx)),
            "n_test_rows": int(len(test_idx)),
            "n_features": int(result["n_features"]),
            "fold_assignments": {c: int(f) for c, f in result["fold_of"].items()},
            "fold_report": result["fold_report"],
        }
        _save_manifest(stack_dir, axis, manifest)
        any_ok = True
        print(
            "built    %-24s k=%d  oof=%d rows, test=%d rows, sha %s..."
            % (arm_name, k, len(train_idx), len(test_idx), manifest["arms"][arm_name]["npz_sha256"][:12])
        )
    _save_manifest(stack_dir, axis, manifest)
    if verify:
        return 0 if not any(failures) else 1
    return 0 if any_ok else 1


def _verify_one(arm_name, entry, npz_path, enriched, y, train_idx, test_idx, k):
    if not npz_path.exists():
        print("verify   %-24s FAIL (no cache)" % arm_name)
        return "missing cache"
    if entry.get("status") != "ok":
        print("verify   %-24s SKIP (manifest status %s)" % (arm_name, entry.get("status")))
        return None
    try:
        result = oof_one_arm(arm_name, enriched, y, train_idx, test_idx, int(entry["k_used"]))
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        print("verify   %-24s FAIL (%s)" % (arm_name, _one_line(exc)))
        return _one_line(exc)
    arrays = _npz_arrays(result)
    with np.load(npz_path) as cached:
        for name, arr in arrays.items():
            if not np.array_equal(np.asarray(cached[name]), np.asanyarray(arr)):
                print("verify   %-24s FAIL (array %s differs)" % (arm_name, name))
                return "array %s differs" % name
    if _sha256_file(npz_path) != _sha256_bytes(arrays):
        print("verify   %-24s FAIL (file digest differs)" % arm_name)
        return "file digest differs"
    print("verify   %-24s PASS (bit-identical re-run)" % arm_name)
    return None


# ---------------------------------------------------------------------------
# Phase 2 — meta layer.
class MetaLayout:
    """Column layout of the meta matrix: [arm probas | masks | tabular | latent]."""

    def __init__(self, arm_names, mask_names, tabular_names, n_latent, latent_blocks):
        self.arm_names = list(arm_names)
        self.mask_names = list(mask_names)
        self.tabular_names = list(tabular_names)
        self.n_latent = int(n_latent)
        self.latent_blocks = tuple(latent_blocks)
        a, m, t = len(self.arm_names), len(self.mask_names), len(self.tabular_names)
        self.arm_slice = slice(0, a)
        self.mask_slice = slice(a, a + m)
        self.tabular_slice = slice(a + m, a + m + t)
        self.latent_slice = slice(a + m + t, a + m + t + self.n_latent)

    @property
    def n_features(self):
        return len(self.arm_names) + len(self.mask_names) + len(self.tabular_names) + self.n_latent


def load_stack_cache(axis, stack_dir, enriched, train_idx, test_idx):
    stack_dir = Path(stack_dir)
    manifest = _load_manifest(stack_dir, axis)
    if not manifest.get("arms"):
        raise StackCacheError("no OOF manifest for axis %s — run --run-stack-base first" % axis)
    cache, contributing = {}, []
    for arm_name in fp.ARMS:
        if fp.ARMS[arm_name]["placeholder"]:
            continue
        entry = manifest["arms"].get(arm_name)
        if entry is None or entry.get("status") != "ok":
            continue  # missing arm -> mask flag downstream
        with np.load(stack_dir / entry["npz"]) as data:
            _assert_identity_match(
                (data["customer"], data["timestamp_ns"], data["row_index"]),
                _identity_arrays(enriched, train_idx),
                "%s/%s OOF identity" % (arm_name, axis),
            )
            _assert_identity_match(
                (data["test_customer"], data["test_timestamp_ns"], data["test_row_index"]),
                _identity_arrays(enriched, test_idx),
                "%s/%s test identity" % (arm_name, axis),
            )
            assert int(data["k"]) == int(entry["k_used"]), "%s: k mismatch" % arm_name
            cache[arm_name] = {
                "oof": np.asarray(data["oof_proba"], dtype=np.float64),
                "test_proba": np.asarray(data["test_proba"], dtype=np.float64),
            }
        assert np.isfinite(cache[arm_name]["oof"]).all(), "%s: non-finite OOF" % arm_name
        assert np.isfinite(cache[arm_name]["test_proba"]).all(), "%s: non-finite test proba" % arm_name
        contributing.append(arm_name)
    if not contributing:
        raise StackCacheError("axis %s: no cached arms — run --run-stack-base first" % axis)
    return cache, contributing, manifest


def build_meta_matrix(enriched, train_idx, test_idx, cache, contributing, drop_arms=()):
    # Missing arms (not ok in the manifest, not deliberately dropped) contribute
    # a NaN proba column + a mask flag; dropped arms (sensitivity pre-reg) are
    # excluded entirely — no column, no mask.
    n = len(enriched)
    arm_names = [a for a in contributing if a not in drop_arms]
    masked = [
        a
        for a in fp.ARMS
        if not fp.ARMS[a]["placeholder"] and a not in contributing and a not in drop_arms
    ]
    proba_cols, mask_cols = {}, {}
    for arm_name in arm_names:
        col = np.full(n, np.nan)
        col[np.asarray(enriched.index.get_indexer(train_idx))] = cache[arm_name]["oof"]
        col[np.asarray(enriched.index.get_indexer(test_idx))] = cache[arm_name]["test_proba"]
        proba_cols["proba__" + arm_name] = col
        mask_cols["mask__" + arm_name] = np.zeros(n)
    for arm_name in masked:
        proba_cols["proba__" + arm_name] = np.full(n, np.nan)
        mask_cols["mask__" + arm_name] = np.ones(n)
    arm_block = pd.DataFrame(proba_cols, index=enriched.index)
    mask_block = pd.DataFrame(mask_cols, index=enriched.index)

    latent = fp._features_latent_pure(enriched)  # stateless 1280-d fusion
    assert latent.index.equals(enriched.index), "latent block index mismatch"
    tabular = fp._features_xgb_client(enriched)  # label-free base+client matrix
    assert tabular.index.equals(enriched.index), "tabular block index mismatch"

    matrix = pd.concat([arm_block, mask_block, tabular, latent], axis=1)
    layout = MetaLayout(
        arm_names=arm_names + masked,
        mask_names=arm_names + masked,
        tabular_names=list(tabular.columns),
        n_latent=latent.shape[1],
        latent_blocks=LATENT_BLOCKS,
    )
    assert matrix.shape[1] == layout.n_features, "matrix width != layout"
    if arm_names:
        covered = matrix[["proba__" + a for a in arm_names]].notna().all().all()
        assert covered, "contributing arm proba column has uncovered rows"
    return matrix, layout


# --- meta variants -----------------------------------------------------------
class _MetaBlend:
    """meta-blend: equal-weight mean of the available arm probas — zero-fit."""

    def __init__(self, layout):
        self.layout = layout

    def fit(self, X_fit, y_fit, X_val=None, y_val=None):
        return self

    def predict_proba(self, X):
        block = np.asarray(X, dtype=np.float64)[:, self.layout.arm_slice]
        with np.errstate(invalid="ignore"):
            score = np.nanmean(block, axis=1)  # absent arms stay NaN until here
        score = np.where(np.isnan(score), MISSING_FILL, score)
        return np.column_stack([1.0 - score, score])


class _MetaLogit:
    """meta-logit: logistic stack on [scores ⊕ masks ⊕ tabular ⊕ latent],
    C swept on the meta-val carve (PRD §6)."""

    C_GRID = (0.01, 0.1, 1.0)

    def __init__(self, layout):
        self.layout = layout

    def fit(self, X_fit, y_fit, X_val, y_val):
        from sklearn.linear_model import LogisticRegression  # lazy sibling
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        Xf = np.nan_to_num(np.asarray(X_fit, dtype=np.float64), nan=MISSING_FILL)
        Xv = np.nan_to_num(np.asarray(X_val, dtype=np.float64), nan=MISSING_FILL)
        y_fit_arr = np.asarray(y_fit).ravel()
        best_model, best_c, best_score = None, None, -np.inf
        for c in self.C_GRID:
            model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("lr", LogisticRegression(C=c, max_iter=5000, random_state=RANDOM_STATE)),
                ]
            )
            model.fit(Xf, y_fit_arr)
            score = average_precision_score(
                np.asarray(y_val).ravel(), model.predict_proba(Xv)[:, 1]
            )
            if score > best_score:
                best_model, best_c, best_score = model, c, score
        self.model_, self.selected_c_, self.val_pr_auc_ = best_model, best_c, float(best_score)
        return self

    def predict_proba(self, X):
        Xn = np.nan_to_num(np.asarray(X, dtype=np.float64), nan=MISSING_FILL)
        return self.model_.predict_proba(Xn)


class _MetaXGB:
    """meta-xgb: small shallow XGB with strong regularization (PRD §6)."""

    def __init__(self, layout):
        self.layout = layout

    def fit(self, X_fit, y_fit, X_val, y_val):
        import xgboost  # lazy

        y_fit_arr = np.asarray(y_fit).ravel()
        params = dict(
            learning_rate=0.05,
            max_depth=3,
            n_estimators=400,
            min_child_weight=5.0,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=5.0,
            reg_alpha=0.5,
            random_state=RANDOM_STATE,
            tree_method="hist",
            eval_metric="aucpr",
            early_stopping_rounds=50,
            scale_pos_weight=float((y_fit_arr == 0).sum() / max(int(y_fit_arr.sum()), 1)),
        )
        self.model_ = xgboost.XGBClassifier(**params)
        self.model_.fit(
            np.asarray(X_fit, dtype=np.float32),
            y_fit_arr,
            eval_set=[(np.asarray(X_val, dtype=np.float32), np.asarray(y_val).ravel())],
            verbose=False,
        )
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(np.asarray(X, dtype=np.float32))


def _make_meta_attn_net(layout):
    # meta-attn (PRD §6, architecture B): shared 1->16 MLP embed + learned
    # per-arm embedding, one 2-head attention layer over arm tokens (mask
    # covers absent arms), masked mean-pool -> 32-d method summary; latent
    # 1280 -> Linear+GELU+LayerNorm -> 64; tabular -> MLP -> 64; head =
    # concat [32, 64, 64] -> residual MLP (2x64) -> 1 logit.
    import torch  # lazy

    class _AttnNet(torch.nn.Module):
        def __init__(self, layout):
            super().__init__()
            self.arm_slice = layout.arm_slice
            self.mask_slice = layout.mask_slice
            self.tabular_slice = layout.tabular_slice
            self.latent_slice = layout.latent_slice
            self.drop = torch.nn.Dropout(0.4)
            self.arm_w = torch.nn.Linear(1, 16)
            self.arm_emb = torch.nn.Embedding(len(layout.arm_names), 16)
            self.mha = torch.nn.MultiheadAttention(embed_dim=16, num_heads=2, batch_first=True)
            self.token_proj = torch.nn.Linear(16, 32)
            self.latent_net = torch.nn.Sequential(
                torch.nn.Linear(layout.n_latent, 64), torch.nn.GELU(), torch.nn.LayerNorm(64)
            )
            self.tab_net = torch.nn.Sequential(
                torch.nn.Linear(len(layout.tabular_names), 64), torch.nn.GELU(), torch.nn.LayerNorm(64)
            )
            self.head_in = torch.nn.Linear(32 + 64 + 64, 64)
            self.head_res = torch.nn.Linear(64, 64)
            self.head_out = torch.nn.Linear(64, 1)

        def forward(self, x, return_attention=False):
            x_arm = x[:, self.arm_slice]
            pad = x[:, self.mask_slice] > 0.5
            tokens = self.arm_w(x_arm.unsqueeze(-1)).squeeze(-1) + self.arm_emb.weight.unsqueeze(0)
            att_out, att_w = self.mha(
                tokens, tokens, tokens, key_padding_mask=pad, need_weights=True,
                average_attn_weights=True,
            )
            pool_w = att_w.mean(dim=1)  # (B, A): mean attention each token receives
            values = self.token_proj(att_out)
            summary = torch.bmm(pool_w.unsqueeze(1), values).squeeze(1)
            z = torch.cat(
                [self.drop(summary), self.drop(self.latent_net(x[:, self.latent_slice])),
                 self.drop(self.tab_net(x[:, self.tabular_slice]))],
                dim=1,
            )
            h1 = torch.nn.functional.gelu(self.head_in(z))
            h2 = torch.nn.functional.gelu(self.head_res(h1))
            logit = self.head_out(h1 + h2).squeeze(-1)
            if return_attention:
                return logit, pool_w
            return logit

    return _AttnNet(layout)


def _make_meta_ftt_net(layout):
    # meta-ftt (PRD §6, architecture A): small FT-Transformer, per-feature
    # tokens — 19 score tokens + mask/tabular tokens + 9 leak-free pooled
    # latent tokens (per modality block: mean/std/max), 2 layers, d_model 64.
    import torch  # lazy

    class _FTTNet(torch.nn.Module):
        def __init__(self, layout):
            super().__init__()
            self.scalar_slice = slice(0, layout.latent_slice.start)  # arm + mask + tabular
            self.latent_slice = layout.latent_slice
            self.latent_blocks = layout.latent_blocks
            self.n_scalar = layout.latent_slice.start
            self.d = 64
            self.weight = torch.nn.Parameter(torch.empty(self.n_scalar, self.d))
            self.bias = torch.nn.Parameter(torch.zeros(self.n_scalar, self.d))
            n_latent_tokens = 3 * len(self.latent_blocks)
            self.latent_w = torch.nn.Parameter(torch.empty(n_latent_tokens, self.d))
            self.latent_b = torch.nn.Parameter(torch.zeros(n_latent_tokens, self.d))
            self.cls = torch.nn.Parameter(torch.empty(1, 1, self.d))
            layer = torch.nn.TransformerEncoderLayer(
                d_model=self.d, nhead=4, dim_feedforward=128, dropout=0.1,
                batch_first=True, norm_first=True,
            )
            self.encoder = torch.nn.TransformerEncoder(layer, num_layers=2)
            self.head = torch.nn.Sequential(torch.nn.LayerNorm(self.d), torch.nn.Linear(self.d, 1))

        def _pooled_latent(self, x_lat):
            # 3 stats per modality block (mean/std/max) = 9 leak-free pooled tokens.
            stats, start = [], 0
            for width in self.latent_blocks:
                blk = x_lat[:, start:start + width]
                stats += [blk.mean(dim=1), blk.std(dim=1, unbiased=False), blk.max(dim=1).values]
                start += width
            return torch.stack(stats, dim=1)

        def forward(self, x):
            scalars = x[:, self.scalar_slice]
            tokens = scalars.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
            lat = self._pooled_latent(x[:, self.latent_slice])
            lat_tokens = lat.unsqueeze(-1) * self.latent_w.unsqueeze(0) + self.latent_b.unsqueeze(0)
            cls = self.cls.expand(x.shape[0], 1, self.d)
            z = self.encoder(torch.cat([cls, tokens, lat_tokens], dim=1))
            return self.head(z[:, 0]).squeeze(-1)

    return _FTTNet(layout)


def _train_torch(model, X_fit, y_fit, X_val, y_val, max_epochs=200, patience=10, batch_size=256):
    # AdamW (lr 1e-3, wd 1e-2), pos_weight for imbalance, early stopping on the
    # meta-val PR-AUC (patience 10, max 200 epochs, batch 256) — PRD §6.
    import torch  # lazy

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _seed_everything()
    model = model.to(device)
    x_fit_t = torch.tensor(
        np.nan_to_num(np.asarray(X_fit, dtype=np.float32), nan=MISSING_FILL), device=device
    )
    y_fit_t = torch.tensor(np.asarray(y_fit).ravel().astype(np.float32), device=device)
    x_val_t = torch.tensor(
        np.nan_to_num(np.asarray(X_val, dtype=np.float32), nan=MISSING_FILL), device=device
    )
    y_val_arr = np.asarray(y_val).ravel().astype(np.int32)

    pos_weight = torch.tensor([(y_fit_t == 0).sum() / (y_fit_t == 1).sum()], device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    gen = torch.Generator().manual_seed(RANDOM_STATE)
    best_state, best_score, bad, epoch = None, -np.inf, 0, 0
    n = x_fit_t.shape[0]
    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n, generator=gen)
        for start in range(0, n, batch_size):
            sel = perm[start:start + batch_size].to(device)
            opt.zero_grad()
            loss = loss_fn(model(x_fit_t[sel]).squeeze(-1), y_fit_t[sel])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_score = torch.sigmoid(model(x_val_t).squeeze(-1)).cpu().numpy()
        score = average_precision_score(y_val_arr, val_score)
        if score > best_score:
            best_score, bad = score, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad > patience:
                break
    model.load_state_dict(best_state)
    return {"epochs": int(epoch + 1), "val_pr_auc": float(best_score)}


class _MetaAttn:
    """meta-attn wrapper: fit/eval + mean attention weight per arm (AC7)."""

    def __init__(self, layout):
        self.layout = layout

    def fit(self, X_fit, y_fit, X_val, y_val):
        import torch  # lazy

        _seed_everything()
        self.net_ = _make_meta_attn_net(self.layout)
        self.train_ = _train_torch(self.net_, X_fit, y_fit, X_val, y_val)
        device = next(self.net_.parameters()).device
        x_val_t = torch.tensor(
            np.nan_to_num(np.asarray(X_val, dtype=np.float32), nan=MISSING_FILL), device=device
        )
        self.net_.eval()
        with torch.no_grad():
            _, pool_w = self.net_(x_val_t, return_attention=True)
        weights = pool_w.mean(dim=0).cpu().numpy()
        total = weights.sum()
        self.attention_ = weights / total if total > 0 else weights  # mean over val rows
        return self

    def predict_proba(self, X):
        import torch  # lazy

        device = next(self.net_.parameters()).device
        x_t = torch.tensor(
            np.nan_to_num(np.asarray(X, dtype=np.float32), nan=MISSING_FILL), device=device
        )
        self.net_.eval()
        with torch.no_grad():
            p = torch.sigmoid(self.net_(x_t)).squeeze(-1).cpu().numpy()
        return np.column_stack([1.0 - p, p])


class _MetaFTT:
    def __init__(self, layout):
        self.layout = layout

    def fit(self, X_fit, y_fit, X_val, y_val):
        _seed_everything()
        self.net_ = _make_meta_ftt_net(self.layout)
        self.train_ = _train_torch(self.net_, X_fit, y_fit, X_val, y_val)
        return self

    def predict_proba(self, X):
        import torch  # lazy

        device = next(self.net_.parameters()).device
        x_t = torch.tensor(
            np.nan_to_num(np.asarray(X, dtype=np.float32), nan=MISSING_FILL), device=device
        )
        self.net_.eval()
        with torch.no_grad():
            p = torch.sigmoid(self.net_(x_t)).squeeze(-1).cpu().numpy()
        return np.column_stack([1.0 - p, p])


META_MODELS = {
    "meta-attn": _MetaAttn,
    "meta-ftt": _MetaFTT,
    "meta-logit": _MetaLogit,
    "meta-blend": _MetaBlend,
    "meta-xgb": _MetaXGB,
    # same attention model; run_stack() swaps in the drop-arms matrix (PRD §5)
    SENS_VARIANT: _MetaAttn,
}


def make_meta_model(variant, layout):
    if variant not in META_MODELS:
        raise ValueError("unknown meta variant: %s" % variant)
    return META_MODELS[variant](layout)


def run_meta_on_split(
    variant,
    matrix,
    layout,
    y,
    train_idx,
    test_idx,
    target_precision=fp.TARGET_PRECISION,
    bootstrap_iters=1000,
):
    # Same harness protocol as run_arm_on_split: ONE stratified 25% meta-val
    # carve cut from TRAIN (seed 42, stratify) — used for early stopping /
    # C sweep AND the frozen threshold — then one-shot frozen-threshold test
    # evaluation with the runner's metric + bootstrap functions verbatim.
    X_tr, X_te = matrix.loc[train_idx], matrix.loc[test_idx]
    y_tr, y_te = y.loc[train_idx], y.loc[test_idx]
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_tr, y_tr, test_size=0.25, random_state=RANDOM_STATE, stratify=y_tr
    )
    model = make_meta_model(variant, layout)
    model.fit(X_fit, y_fit, X_val, y_val)
    threshold = fp.best_f1_threshold(y_val, model.predict_proba(X_val)[:, 1])
    test_proba = np.asarray(model.predict_proba(X_te))[:, 1]
    metrics = fp.rich_test_metrics(y_te, test_proba, threshold, target_precision)
    if bootstrap_iters > 0 and not math.isnan(metrics["pr_auc"]):
        ci = fp.bootstrap_auc_ci(y_te, test_proba, n_iters=bootstrap_iters)
    else:
        ci = {}
    row = {
        "axis": None,  # filled by the caller
        "arm": variant,
        "status": "ok",
        "test_rows": int(len(test_idx)),
        "test_positives": int(np.asarray(y_te).sum()),
        "chance_level": float(np.asarray(y_te).mean()),
        "frozen_threshold": float(threshold),
        "n_features": int(matrix.shape[1]),
        **metrics,
        **ci,
    }
    return row, model


def _save_attn_diagnostics(stack_dir, axis, variant, model, layout, drop_arms):
    payload = {
        "axis": axis,
        "variant": variant,
        "arm_names": list(layout.arm_names),
        "drop_arms": list(drop_arms),
        "mean_attention": {
            arm: float(w) for arm, w in zip(layout.arm_names, model.attention_)
        },
        "val_pr_auc": float(model.train_["val_pr_auc"]),
        "epochs": int(model.train_["epochs"]),
    }
    path = Path(stack_dir) / ("attn__%s__%s.json" % (axis, variant))
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    ranked = sorted(payload["mean_attention"].items(), key=lambda kv: -kv[1])
    print("  attention diagnostics (%s): top=%s | bottom=%s" % (
        variant,
        ["%s=%.3f" % kv for kv in ranked[:3]],
        ["%s=%.3f" % kv for kv in ranked[-3:]],
    ))
    return path


def run_stack(
    axis,
    variants,
    stack_dir,
    results_path,
    all_trxns_path=fp.DEFAULT_ALL_TRXNS_PATH,
    exchange_rates_path=fp.DEFAULT_EXCHANGE_RATES_PATH,
    dim_customer_path=fp.DEFAULT_DIM_CUSTOMER_PATH,
    target_precision=fp.TARGET_PRECISION,
    bootstrap_iters=1000,
    test_size=0.2,
):
    if SENS_VARIANT in variants and axis != CHRON_AXIS:
        raise SystemExit(
            "%s is pre-registered for the %s axis only (PRD §5)" % (SENS_VARIANT, CHRON_AXIS)
        )
    enriched, y = fp.load_enriched(
        all_trxns_path=all_trxns_path,
        exchange_rates_path=exchange_rates_path,
        dim_customer_path=dim_customer_path,
    )
    train_idx, test_idx = fp.axis_split(axis, enriched, y, test_size)
    test_positives = int(y.loc[test_idx].sum())
    print(
        "[%s] %d train / %d test rows | test positives: %d (chance PR-AUC %.4f)"
        % (axis, len(train_idx), len(test_idx), test_positives, test_positives / len(test_idx))
    )

    try:
        cache, contributing, _manifest = load_stack_cache(axis, stack_dir, enriched, train_idx, test_idx)
        matrix, layout = build_meta_matrix(enriched, train_idx, test_idx, cache, contributing)
    except (StackCacheError, fp.ArmSkipped) as exc:
        reason = _one_line(exc)
        rows = [
            {"axis": axis, "arm": variant, "status": "skipped", "reason": reason}
            for variant in variants
        ]
        _persist_rows(results_path, rows)
        for row in rows:
            print("skipped  %-16s %s" % (row["arm"], reason))
        return 1

    sens_matrix = sens_layout = None
    new_rows = []
    for variant in variants:
        drop_arms = SENS_DROP_ARMS if variant == SENS_VARIANT else ()
        if drop_arms:
            if sens_matrix is None:
                sens_matrix, sens_layout = build_meta_matrix(
                    enriched, train_idx, test_idx, cache, contributing, drop_arms
                )
            matrix_v, layout_v = sens_matrix, sens_layout
        else:
            matrix_v, layout_v = matrix, layout
        try:
            _seed_everything()
            row, model = run_meta_on_split(
                variant, matrix_v, layout_v, y, train_idx, test_idx,
                target_precision=target_precision, bootstrap_iters=bootstrap_iters,
            )
            row["axis"] = axis
        except fp.ArmSkipped as exc:
            row = {"axis": axis, "arm": variant, "status": "skipped", "reason": _one_line(exc)}
            model = None
        except Exception as exc:  # one bad variant never crashes the whole run
            row = {
                "axis": axis,
                "arm": variant,
                "status": "error",
                "reason": _one_line("%s: %s" % (type(exc).__name__, exc)),
            }
            model = None
        new_rows.append(row)
        if row["status"] == "ok":
            print(
                "ok       %-16s PR-AUC %.4f (CI %.4f–%.4f) | ROC-AUC %.4f | F1 %.4f | thr %.4f"
                % (variant, row["pr_auc"], row["pr_auc_ci_low"], row["pr_auc_ci_high"],
                   row["roc_auc"], row["f1"], row["frozen_threshold"])
            )
            if variant in ("meta-attn", SENS_VARIANT):
                _save_attn_diagnostics(stack_dir, axis, variant, model, layout_v, drop_arms)
        else:
            print("%-8s %-16s %s" % (row["status"], variant, row.get("reason", "")))

    _persist_rows(results_path, new_rows)
    print("results: %s (%d new meta rows)" % (results_path, len(new_rows)))
    return 0 if any(row["status"] == "ok" for row in new_rows) else 1


def _persist_rows(results_path, new_rows):
    all_rows = fp.merge_results(fp.load_results(results_path), new_rows)
    fp.save_results(results_path, all_rows)


# ---------------------------------------------------------------------------
# CLI
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="F6 stacking meta-model: leak-free OOF base layer + meta variants (FOC-211).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python src/stack.py --run-stack-base --axis random-grouped\n"
            "  python src/stack.py --run-stack-base --axis chronological --arms tabnet --k 3\n"
            "  python src/stack.py --run-stack-base --axis grouped --verify\n"
            "  python src/stack.py --run-stack --axis chronological\n"
        ),
    )
    phase = parser.add_mutually_exclusive_group(required=True)
    phase.add_argument("--run-stack-base", action="store_true", help="phase 1: per-arm grouped OOF cache")
    phase.add_argument("--run-stack", action="store_true", help="phase 2: meta variants")
    parser.add_argument("--axis", required=True, choices=list(fp.AXES), help="evaluation axis")
    parser.add_argument(
        "--arms",
        default=None,
        help="phase 1: comma-separated arm subset (default: every non-placeholder arm)",
    )
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="phase 1: OOF folds (default 5)")
    parser.add_argument("--refit", action="store_true", help="phase 1: rebuild even if cached")
    parser.add_argument(
        "--verify", action="store_true",
        help="phase 1: re-run every listed arm and require bit-identical cached arrays",
    )
    parser.add_argument(
        "--variants",
        default=None,
        help="phase 2: comma-separated variants (default: all base variants; +meta-attn-sens on chronological)",
    )
    parser.add_argument("--stack-dir", default=str(DEFAULT_STACK_DIR))
    parser.add_argument("--results", default=str(fp.DEFAULT_RESULTS_PATH))
    parser.add_argument("--target-precision", type=float, default=fp.TARGET_PRECISION)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--bootstrap-iters", type=int, default=1000)
    parser.add_argument("--all-trxns", default=str(fp.DEFAULT_ALL_TRXNS_PATH))
    parser.add_argument("--exchange-rates", default=str(fp.DEFAULT_EXCHANGE_RATES_PATH))
    parser.add_argument("--dim-customer", default=str(fp.DEFAULT_DIM_CUSTOMER_PATH))
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.k < 2:
        raise SystemExit("--k must be >= 2")
    if args.run_stack_base:
        arm_names = (
            [name.strip() for name in args.arms.split(",")]
            if args.arms
            else [n for n, a in fp.ARMS.items() if not a["placeholder"]]
        )
        unknown = [n for n in arm_names if n not in fp.ARMS]
        if unknown:
            raise SystemExit("unknown arms: %s" % ", ".join(unknown))
        return run_stack_base(
            args.axis,
            arm_names,
            args.k,
            args.stack_dir,
            refit=args.refit,
            verify=args.verify,
            all_trxns_path=args.all_trxns,
            exchange_rates_path=args.exchange_rates,
            dim_customer_path=args.dim_customer,
        )

    if args.variants:
        variants = [v.strip() for v in args.variants.split(",")]
    else:
        variants = list(BASE_VARIANTS) + ([SENS_VARIANT] if args.axis == CHRON_AXIS else [])
    unknown = [v for v in variants if v not in META_MODELS]
    if unknown:
        raise SystemExit("unknown variants: %s" % ", ".join(unknown))
    return run_stack(
        args.axis,
        variants,
        args.stack_dir,
        args.results,
        all_trxns_path=args.all_trxns,
        exchange_rates_path=args.exchange_rates,
        dim_customer_path=args.dim_customer,
        target_precision=args.target_precision,
        bootstrap_iters=args.bootstrap_iters,
        test_size=args.test_size,
    )


if __name__ == "__main__":
    sys.exit(main())