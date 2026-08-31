"""Unified experiment pipeline for the fraud R&D arms (FOC-175).

One registry of arms (feature-set x model pairs sharing an sklearn-style
fit/predict_proba interface) evaluated on three train/test axes under the
nb7/nb8 protocol: fixed XGB-style parameters with rare-class weighting, a
threshold frozen on a stratified validation carve cut from the training side,
and held-out test metrics (PR-AUC, ROC-AUC, F1 at the frozen threshold,
recall@target-precision) reported next to the test positive count and the
chance level (test positive rate). Test PR-AUC/ROC-AUC carry percentile
bootstrap intervals — honest-but-cheap uncertainty at 11-24 test positives.

Wired arms (lifted from nb7/nb8, see the per-arm comments): xgb-baseline,
xgb-client, dictionary, sce. The F3 notebook arms (gbdt-ensemble, tabnet,
timesfm-features, sequential) are reserved as placeholders — running one
reports a SKIPPED row with the reason, never a crash of the whole run. Heavy
dependencies (xgboost, sce, torch, tabnet, timesfm) are imported lazily inside
the arm factories so this module loads without any of them installed.

Results accumulate as JSONL under results/ (one row per axis x arm; a re-run
supersedes its earlier row so the table always shows the latest measurement
per pair). Rows are fully deterministic by construction — no wall-clock
fields — so two identical invocations produce byte-identical files.

CLI:
  python src/fraud_pipeline.py --list-arms
  python src/fraud_pipeline.py                                    # all wired arms x all axes
  python src/fraud_pipeline.py --run-arm xgb-baseline --axis all  # one arm, every axis
  python src/fraud_pipeline.py --axis random-grouped              # every arm, one axis
  python src/fraud_pipeline.py --run-arm sce --axis chronological

Notebook API:
  from fraud_pipeline import ARMS, AXES, load_enriched, run_arm_on_axis
  enriched, y = load_enriched()
  row = run_arm_on_axis("xgb-baseline", "chronological", enriched, y)
"""

import argparse
import itertools
import json
import math
import sys
import time
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from funs import (
    chronological_split,
    createDictionary,
    createMetaDictionary,
    cross_validate_model,
    dataPreparation,
    dictionaryModel,
    grouped_split,
    random_grouped_split,
)

SRC_DIR = Path(__file__).resolve().parent
DATA_DIR = SRC_DIR.parent / "data"
DEFAULT_ALL_TRXNS_PATH = DATA_DIR / "all_trxns.csv"
DEFAULT_EXCHANGE_RATES_PATH = DATA_DIR / "exchange_rates.csv"
DEFAULT_DIM_CUSTOMER_PATH = DATA_DIR / "dim_customer.csv"
DEFAULT_RESULTS_PATH = SRC_DIR.parent / "results" / "fraud_pipeline_results.jsonl"

RANDOM_STATE = 42
TARGET_PRECISION = 0.5

# ---------------------------------------------------------------------------
# Feature sets (nb7/nb8 convention).
BASE_FEATURES = [
    "customer_country",
    "counterparty_country",
    "type",
    "ccy",
    "customer_type",
    "weekday",
    "month",
    "quarter",
    "hour",
    "amount_eur_bucket",
]
CLIENT_NUMERIC = ["age", "account_tenure_years"]
CLIENT_CATEGORICAL = [
    "gender",
    "employment_industry",
    "employment_status",
    "income_band",
    "device",
    "channel",
]
# SCE grouping candidates: the 10 transaction categoricals + the 6 categorical
# client attributes + the 2 binned client numerics. The customer id itself is
# deliberately excluded (nb8): a per-customer fraud rate would re-import the
# identity signal and could not transfer to unseen customers.
GROUPING_COLS = BASE_FEATURES + CLIENT_CATEGORICAL + ["age_band", "tenure_band"]

DICT_VARS = list(BASE_FEATURES)
DICT_SCORE_COLS = [
    "expected_fraud_probability",
    "sd_flags",
    "quantile_flags",
    "quantile_1_flags",
    "quantile_25_flags",
    "quantile_75_flags",
    "quantile_9_flags",
]
DICT_FP_GRID = np.arange(0.05, 0.40, 0.025)
DICT_FLAG_GRID = [0, 1, 2, 3]

FIXED_XGB_PARAMS = dict(
    learning_rate=0.05,
    max_depth=6,
    n_estimators=200,
    reg_lambda=1.0,
    random_state=42,
    eval_metric="logloss",
)


def xgb_params(y_fit):
    # Lifted from src/8. SCE Enrichment.ipynb (nb8): identical hyperparameters
    # everywhere; the rare-class weight is the neg/pos ratio of the fit carve.
    params = dict(FIXED_XGB_PARAMS)
    params["scale_pos_weight"] = (y_fit == 0).sum() / (y_fit == 1).sum()
    return params


def best_f1_threshold(y_true, y_score):
    # Lifted from nb8: same computation as funs.evaluateModel's
    # best_threshold_f1, without the plot/print side effects.
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if len(thresholds) == 0:
        return 0.5
    f1_scores = 2 * precision * recall / (precision + recall + 1e-12)
    return float(thresholds[int(np.nanargmax(f1_scores[:-1]))])


def encode_dummies(frame, cat_cols):
    # Lifted from nb8: one-hot encode; XGBoost forbids '[', ']' and '<' in
    # feature names (the income_band label '1_<20k' and interval buckets would
    # inject them).
    X = pd.get_dummies(frame[cat_cols], columns=cat_cols)
    X.columns = [
        c.replace("[", "").replace("]", "").replace("<", "under") for c in X.columns
    ]
    return X


# ---------------------------------------------------------------------------
# sklearn-style wrappers for the arms whose features are fitted objects.
# Both are lifted from src/8. SCE Enrichment.ipynb (nb8) with heavy imports
# moved inside fit() so this module loads without the optional dependencies.
class SCERateEnricher(BaseEstimator):
    """XGB on base features + SCE fraud-rate context features.

    fit() runs engine.fit_transform on the fitting rows: with the production
    config (use_cross_fitting + strategy="time") every fitting row receives
    fraud-rate statistics built exclusively from strictly earlier rows. The
    engine also keeps full-fitting-frame statistics, so predict_proba()/
    predict() enrich out-of-sample rows through engine.transform(). Refitting
    inside fit() is what makes cross_validate_model honest for this arm: each
    fold's clone rebuilds the context features from its own fold-train rows.
    """

    def __init__(self, config, base_cols, model_params=None):
        self.config = config
        self.base_cols = base_cols
        self.model_params = model_params

    def fit(self, X, y):
        from sce import StatisticalContextEngine  # lazy: optional dependency
        from xgboost import XGBClassifier  # lazy: keeps module import light

        engine = StatisticalContextEngine(self.config)
        with redirect_stdout(StringIO()):  # engine prints a banner per fit
            enriched = engine.fit_transform(X, y)
        self.engine_ = engine
        self.raw_columns_ = list(X.columns)
        dummies = encode_dummies(enriched, self.base_cols)
        self.dummy_columns_ = dummies.columns.tolist()
        self.sce_columns_ = [
            c for c in enriched.columns if c not in self.raw_columns_
        ]
        matrix = pd.concat(
            [dummies, enriched.reindex(columns=self.sce_columns_)], axis=1
        )
        self.model_ = XGBClassifier(**self.model_params)
        self.model_.fit(matrix, np.asarray(y).ravel())
        return self

    def _enriched(self, X):
        with redirect_stdout(StringIO()):
            return self.engine_.transform(X)

    def _matrix(self, X):
        enriched = self._enriched(X)
        dummies = encode_dummies(enriched, self.base_cols).reindex(
            columns=self.dummy_columns_, fill_value=0
        )
        return pd.concat(
            [dummies, enriched.reindex(columns=self.sce_columns_)], axis=1
        )

    def predict_proba(self, X):
        return self.model_.predict_proba(self._matrix(X))

    def predict(self, X):
        return self.model_.predict(self._matrix(X))


class DictionaryRateEnricher(BaseEstimator):
    """nb5-style dictionary features, rebuilt inside fit().

    Mirrors nb5: createDictionary + createMetaDictionary per dictionary
    variable (count_filter=0, nb5's choice) and the three-threshold F1 grid —
    ALL computed on the fitting rows only. dictionaryModel then scores any row
    set by looking up its group values; the scores never read the row's own
    target (dictionaryModel requires the fraud_flag column as input, but the
    label-derived columns are dropped and only the 7 aggregate scores become
    features). Wrapping nb5's construction keeps the CV folds and the
    validation carve out-of-sample.
    """

    def __init__(self, base_cols, dict_vars, model_params=None):
        self.base_cols = base_cols
        self.dict_vars = dict_vars
        self.model_params = model_params

    def fit(self, X, y):
        from xgboost import XGBClassifier  # lazy: keeps module import light

        rows = X.copy()
        y_arr = np.asarray(y).ravel()
        self.dictionaries_ = {
            v: createDictionary(rows, colname=v, count_filter=0)
            for v in self.dict_vars
        }
        self.meta_dictionary_ = pd.concat(
            [
                createMetaDictionary(
                    rows, colname=v, quantile_threshold=0.9, count_filter=0
                )
                for v in self.dict_vars
            ],
            ignore_index=True,
        )
        # nb5's threshold grid on the fitting rows: the permissive call returns
        # the aggregation columns, the grid scans the three-threshold space.
        agg = dictionaryModel(
            rows, self.dictionaries_, self.meta_dictionary_, 0.0, -1, -1
        )
        best_f1, self.thresholds_ = -1.0, (0.2, 0, 0)
        for fp_t, sd_t, qf_t in itertools.product(
            DICT_FP_GRID, DICT_FLAG_GRID, DICT_FLAG_GRID
        ):
            pred = (
                (agg["expected_fraud_probability"] > fp_t)
                & (agg["sd_flags"] > sd_t)
                & (agg["quantile_flags"] > qf_t)
            ).astype(int)
            score = f1_score(y_arr, pred, zero_division=0)
            if score > best_f1:
                best_f1 = score
                self.thresholds_ = (float(fp_t), int(sd_t), int(qf_t))

        dummies = encode_dummies(rows, self.base_cols)
        self.dummy_columns_ = dummies.columns.tolist()
        matrix = pd.concat([dummies, self._dictionary_scores(rows)], axis=1)
        self.model_ = XGBClassifier(**self.model_params)
        self.model_.fit(matrix, y_arr)
        return self

    def _dictionary_scores(self, rows):
        out = dictionaryModel(
            rows, self.dictionaries_, self.meta_dictionary_, *self.thresholds_
        )
        assert len(out) == len(rows), "dictionaryModel changed the row count"
        scores = out[DICT_SCORE_COLS].reset_index(drop=True)
        scores.index = rows.index  # dictionaryModel preserves input row order
        return scores

    def _matrix(self, X):
        dummies = encode_dummies(X, self.base_cols).reindex(
            columns=self.dummy_columns_, fill_value=0
        )
        return pd.concat([dummies, self._dictionary_scores(X)], axis=1)

    def predict_proba(self, X):
        return self.model_.predict_proba(self._matrix(X))

    def predict(self, X):
        return self.model_.predict(self._matrix(X))


# ---------------------------------------------------------------------------
# Arms registry: each arm = (feature-set, model) pair. make_model(y_fit)
# returns a sklearn-style estimator (fit / predict / predict_proba); for the
# enricher arms the model rebuilds its features inside every fit(), which
# keeps cross_validate_model honest. build_features(enriched) returns the
# frame the estimator consumes. Heavy imports happen inside make_model/fit —
# never at registration time.
ARMS = {}


class ArmSkipped(Exception):
    """Raised by an arm when it cannot run (missing dependency, pending
    implementation, missing artifact); the runner turns it into a SKIPPED
    result row instead of crashing the run."""


def register_arm(
    name,
    description,
    make_model,
    build_features,
    supports_cv=True,
    n_features=None,
    placeholder=False,
):
    if name in ARMS:
        raise ValueError("arm already registered: %s" % name)
    ARMS[name] = {
        "name": name,
        "description": description,
        "make_model": make_model,
        "build_features": build_features,
        "supports_cv": supports_cv,
        "n_features": n_features,
        "placeholder": placeholder,
    }


# --- wired arms (lifted from nb7/nb8 — do not re-implement) -----------------

def _make_fixed_xgb(y_fit):
    # nb7 arm model: fixed XGB params, rare-class weight from the fit carve.
    from xgboost import XGBClassifier  # lazy: keeps module import light

    return XGBClassifier(**xgb_params(y_fit))


def _features_xgb_baseline(enriched):
    # nb7 arm a: base transaction features, one-hot encoded.
    return encode_dummies(enriched, BASE_FEATURES)


def _features_xgb_client(enriched):
    # nb7 arm b: base + categorical client features one-hot, numeric client
    # features pass through as numbers.
    return pd.concat(
        [encode_dummies(enriched, BASE_FEATURES + CLIENT_CATEGORICAL), enriched[CLIENT_NUMERIC]],
        axis=1,
    )


def _features_dictionary(enriched):
    # nb8 arm d input: dictionary variables + the id columns dictionaryModel
    # needs. fraud_flag rides along only as a merge key inside dictionaryModel
    # — the label-derived columns are dropped and the 7 aggregate scores
    # (built from fit-row dictionaries only) become the features.
    return enriched[DICT_VARS + ["customer", "counterparty", "timestamp", "fraud_flag"]]


def _make_dictionary(y_fit):
    return DictionaryRateEnricher(
        BASE_FEATURES, DICT_VARS, model_params=xgb_params(y_fit)
    )


def _n_features_dictionary(model):
    return len(model.dummy_columns_) + len(DICT_SCORE_COLS)


def _features_sce(enriched):
    # nb8 arm c input: grouping columns + timestamp for the SCE engine.
    # stat-context 0.4.0 quirk (nb8 / FOC-174 report §3.3): its out-of-fold
    # assignment string-casts converted-dtype grouping columns (int hour ->
    # str hour, Interval bucket -> str bucket) while untouched rows keep the
    # original values, so one column ends up holding mixed int/str values and
    # fragments the engine groups. Normalize the grouping columns to str at
    # the boundary: groups stay exact, dummies stay unique. The timestamp
    # keeps its dtype — the time strategy needs it.
    X_sce_raw = enriched[GROUPING_COLS + ["timestamp"]].copy()
    X_sce_raw[GROUPING_COLS] = X_sce_raw[GROUPING_COLS].astype(str)
    return X_sce_raw


def _make_sce_model(y_fit):
    # nb8 arm c model: the engine config is built here (lazy import — nothing
    # sce-related may run at registration time) and the model shares one
    # scale_pos_weight from the fit carve (nb7 behaviour).
    from sce import AggregationMethod, ContextConfig  # lazy: optional dependency

    config = ContextConfig(
        target_col="fraud",
        categorical_cols=GROUPING_COLS,
        aggregations=[AggregationMethod.MEAN, AggregationMethod.COUNT],
        min_group_size=5,
        use_cross_fitting=True,
        cross_fit_strategy="time",
        time_col="timestamp",
        n_folds=5,
        random_state=42,  # unused by the deterministic time strategy; kept explicit
        include_interactions=True,
        include_global_stats=True,
        include_relative_features=False,  # library WARNING: target leakage
    )
    return SCERateEnricher(config, BASE_FEATURES, model_params=xgb_params(y_fit))


def _n_features_sce(model):
    return len(model.dummy_columns_) + len(model.sce_columns_)


register_arm(
    "xgb-baseline",
    "base features -> dummies -> XGBClassifier, validation-frozen threshold (nb7 arm a)",
    make_model=_make_fixed_xgb,
    build_features=_features_xgb_baseline,
)
register_arm(
    "xgb-client",
    "base + client features -> XGBClassifier, validation-frozen threshold (nb7 arm b)",
    make_model=_make_fixed_xgb,
    build_features=_features_xgb_client,
)
register_arm(
    "dictionary",
    "rule-based dictionary scores (funs.dictionaryModel) + XGB on top (nb8 arm d)",
    make_model=_make_dictionary,
    build_features=_features_dictionary,
    n_features=_n_features_dictionary,
)
register_arm(
    "sce",
    "base + SCE cross-fitted fraud-rate context, engine refit per fit (nb8 arm c); "
    "CV off by default — every fold would refit the engine",
    make_model=_make_sce_model,
    build_features=_features_sce,
    supports_cv=False,
    n_features=_n_features_sce,
)

# ---------------------------------------------------------------------------
# F3 notebook arms (FOC-175) — registration hooks, intentionally without
# implementations yet. The names are reserved so the registry and the results
# file already speak the final vocabulary; running one reports a SKIPPED row
# with the reason (never a crash), and the default selection excludes them.
#
# To land one: follow the wired arms — a build_features(enriched) frame plus a
# make_model(y_fit) factory returning a BaseEstimator with fit/predict_proba
# (wrap non-sklearn models like SCERateEnricher above), import heavy deps
# LAZILY inside make_model/fit so this module keeps importing without
# torch / pytorch_tabnet / timesfm installed, then flip placeholder=False:
#
#   def _make_tabnet(y_fit):
#       from pytorch_tabnet.tab_model import TabNetClassifier  # lazy
#       ...
#
#   register_arm("tabnet", "...", make_model=_make_tabnet,
#                build_features=_features_xgb_client, placeholder=False)

def _features_gbdt_ensemble(enriched):
    # nb9 arm input: the xgb-client matrix with LightGBM-safe column names
    # (arms_gbdt.sanitize_feature_names — the amount_eur_bucket interval labels
    # carry a comma, which LightGBM rejects). Same values and column order as
    # _features_xgb_client; only names differ, so n_features matches.
    import arms_gbdt  # lazy: keeps module import light

    return arms_gbdt.sanitize_feature_names(_features_xgb_client(enriched))


def _make_gbdt_ensemble(y_fit):
    # F3 nb9 arm (src/arms_gbdt.py): XGB + LightGBM + CatBoost soft vote on the
    # base+client feature set. Lazy sibling import — this module must load
    # without lightgbm/catboost installed (other machines), and a missing
    # dependency surfaces as a SKIPPED row (ArmSkipped), never a crash.
    import arms_gbdt  # lazy: imports lightgbm/catboost only inside fit()

    missing = arms_gbdt.check_dependencies()
    if missing is not None:
        raise ArmSkipped("gbdt-ensemble: missing dependency (%s)" % missing)
    return arms_gbdt.make_ensemble(y_fit)


register_arm(
    "gbdt-ensemble",
    "GBDT ensemble (XGB + LightGBM + CatBoost) soft vote on base+client features "
    "(nb9 arm); 3 x 200 trees per fit — cheap enough for full CV, members refit per fold",
    make_model=_make_gbdt_ensemble,
    build_features=_features_gbdt_ensemble,
    supports_cv=True,
)
def _features_tabnet(enriched):
    # nb10 arm input: the xgb-client matrix with LightGBM-safe column names
    # (arms_gbdt.sanitize_feature_names) — the exact frame gbdt-ensemble
    # consumes, so the two F3 arms stay feature-identical (n_features 116;
    # values and column order unchanged, only names mapped).
    import arms_gbdt  # lazy: keeps module import light

    return arms_gbdt.sanitize_feature_names(_features_xgb_client(enriched))


def _make_tabnet(y_fit):
    # F3 nb10 arm (src/arms_tabnet.py): TabNetClassifier (pytorch-tabnet) on the
    # base+client feature set, CUDA when available else CPU. Lazy sibling
    # import — this module must load without torch/pytorch_tabnet installed
    # (other machines), and a missing dependency surfaces as a SKIPPED row
    # (ArmSkipped), never a crash.
    import arms_tabnet  # lazy: imports torch/pytorch_tabnet only inside fit()

    missing = arms_tabnet.check_dependencies()
    if missing is not None:
        raise ArmSkipped("tabnet: missing dependency (%s)" % missing)
    return arms_tabnet.make_tabnet(y_fit)


register_arm(
    "tabnet",
    "TabNet (pytorch-tabnet 4.1.0) on base+client features (nb10 arm); early "
    "stopping on an internal stratified carve — CV off by default ([no-cv]): "
    "per-fold GPU refits are not worth it at this scale",
    make_model=_make_tabnet,
    build_features=_features_tabnet,
    supports_cv=False,
)
register_arm(
    "timesfm-features",
    "TimesFM-encoded amount series appended to base+client features",
    make_model=None,
    build_features=None,
    supports_cv=False,
    placeholder=True,
)
register_arm(
    "sequential",
    "sequence model over per-customer transaction history",
    make_model=None,
    build_features=None,
    supports_cv=False,
    placeholder=True,
)


# ---------------------------------------------------------------------------
# Evaluation axes. All three run on the same underlying frame; every arm on an
# axis is measured on identical row indices.
AXES = {
    "random-grouped": {
        "description": "cohort-random customer-grouped (PRIMARY axis): seeded random "
        "customer assignment, customer-disjoint, no time ordering",
        "split": lambda X, y, enriched, test_size: random_grouped_split(
            X, y, enriched["customer"], test_size=test_size, random_state=RANDOM_STATE
        ),
    },
    "grouped": {
        "description": "cohort-ordered customer-grouped (stress test): test = "
        "latest-seen customers (funs.grouped_split as-is)",
        "split": lambda X, y, enriched, test_size: grouped_split(
            X,
            y,
            enriched["customer"],
            timestamp=enriched["timestamp"],
            test_size=test_size,
            random_state=RANDOM_STATE,
        ),
    },
    "chronological": {
        "description": "chronological: test strictly later than train "
        "(funs.chronological_split as-is)",
        "split": lambda X, y, enriched, test_size: chronological_split(
            X, y, timestamp=enriched["timestamp"], test_size=test_size, random_state=RANDOM_STATE
        ),
    },
}


def load_enriched(
    all_trxns_path=DEFAULT_ALL_TRXNS_PATH,
    exchange_rates_path=DEFAULT_EXCHANGE_RATES_PATH,
    dim_customer_path=DEFAULT_DIM_CUSTOMER_PATH,
):
    # Canonical frame (funs.dataPreparation) + client join + banded client
    # numerics — the nb7/nb8 data section, shared by every arm.
    trxns = dataPreparation(
        all_trxns_path=all_trxns_path, exchange_rates_path=exchange_rates_path
    )
    dim_customer = pd.read_csv(dim_customer_path)
    enriched = trxns.merge(
        dim_customer, on="customer", how="left", validate="many_to_one"
    )
    assert len(enriched) == len(trxns), "join lost rows"
    client_cols = [c for c in dim_customer.columns if c != "customer"]
    assert int(enriched[client_cols].isna().sum().sum()) == 0, "join introduced nulls"

    # SCE groups on categories, so the two numeric client attributes enter the
    # grouping set as binned versions. Edges chosen so no bin is empty on this
    # table (age spans 18-82, tenure 0-25 years) — nb8.
    enriched["age_band"] = pd.cut(
        enriched["age"],
        bins=[17, 25, 40, 55, 120],
        labels=["18-25", "26-40", "41-55", "56+"],
    )
    enriched["tenure_band"] = pd.cut(
        enriched["account_tenure_years"],
        bins=[-1, 2, 7, 15, 100],
        labels=["0-2", "3-7", "8-15", "16+"],
    )
    assert enriched["age_band"].notna().all(), "age binning produced NaN"
    assert enriched["tenure_band"].notna().all(), "tenure binning produced NaN"

    y = enriched["fraud_flag"].map({"N": 0, "Y": 1})
    assert y.notna().all(), "unexpected fraud_flag label outside {N, Y}"
    return enriched, y.astype(int)


def verify_split(axis, train_idx, test_idx, enriched, test_size=0.2):
    # nb8's split guards, re-asserted per axis so a splitter regression cannot
    # silently change what every arm is measured on.
    assert not (set(train_idx) & set(test_idx)), "%s split overlaps" % axis
    assert len(train_idx) + len(test_idx) == len(enriched), "%s split lost rows" % axis
    if axis == "chronological":
        assert len(test_idx) == int(np.ceil(len(enriched) * test_size)), (
            "chronological test rows != ceil(test_size * rows)"
        )
    else:
        train_customers = set(enriched.loc[train_idx, "customer"])
        test_customers = set(enriched.loc[test_idx, "customer"])
        assert train_customers.isdisjoint(test_customers), (
            "%s split shares customers" % axis
        )
        assert len(test_customers) == int(
            np.ceil(enriched["customer"].nunique() * test_size)
        ), "%s test customers != ceil(test_size * customers)" % axis


def axis_split(axis, enriched, y, test_size=0.2):
    # One split per axis; every arm on that axis runs on these exact indices
    # (nb7's identical-splits discipline).
    X_ref = enriched[["customer"]]
    X_train, X_test, _, _ = AXES[axis]["split"](X_ref, y, enriched, test_size)
    train_idx, test_idx = X_train.index, X_test.index
    verify_split(axis, train_idx, test_idx, enriched, test_size)
    return train_idx, test_idx


def rich_test_metrics(y_true, y_score, threshold, target_precision=TARGET_PRECISION):
    # Quiet mirror of funs.evaluateModel's rich path: same metric definitions,
    # no figures/prints (the CLI run is noisy enough without plots).
    y_true_arr = np.asarray(y_true).ravel()
    y_pred = (y_score >= threshold).astype(int)
    metrics = {
        "pr_auc": float("nan"),
        "roc_auc": float("nan"),
        "f1": float(f1_score(y_true_arr, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true_arr, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred, zero_division=0)),
        "recall_at_precision": 0.0,
        "target_precision": float(target_precision),
    }
    try:
        metrics["pr_auc"] = float(average_precision_score(y_true_arr, y_score))
        metrics["roc_auc"] = float(roc_auc_score(y_true_arr, y_score))
    except ValueError:
        pass  # degenerate test set (single class) — cross_validate_model convention
    precision, recall, thresholds = precision_recall_curve(y_true_arr, y_score)
    if len(thresholds) > 0:
        prec_arr, rec_arr = precision[:-1], recall[:-1]
        meets = prec_arr >= target_precision
        metrics["recall_at_precision"] = (
            float(rec_arr[meets].max()) if meets.any() else 0.0
        )
    return metrics


def bootstrap_auc_ci(y_true, y_score, n_iters=1000, ci=95, random_state=RANDOM_STATE):
    # Percentile bootstrap over TEST rows: resample test rows with replacement,
    # recompute both AUCs, take the ci% percentile range. Honest-but-cheap
    # uncertainty — with 11-24 test positives the intervals are wide and must
    # be read as indicative, not inferential.
    y_true_arr = np.asarray(y_true).ravel()
    y_score_arr = np.asarray(y_score).ravel()
    rng = np.random.RandomState(random_state)
    n = len(y_true_arr)
    pr_samples, roc_samples = [], []
    for _ in range(n_iters):
        idx = rng.randint(0, n, n)
        sample_y = y_true_arr[idx]
        if sample_y.min() == sample_y.max():
            continue  # degenerate resample: AUC undefined, skipped
        sample_score = y_score_arr[idx]
        pr_samples.append(average_precision_score(sample_y, sample_score))
        roc_samples.append(roc_auc_score(sample_y, sample_score))
    if not pr_samples:
        return {}
    alpha = (100 - ci) / 2
    return {
        "pr_auc_ci_low": float(np.percentile(pr_samples, alpha)),
        "pr_auc_ci_high": float(np.percentile(pr_samples, 100 - alpha)),
        "roc_auc_ci_low": float(np.percentile(roc_samples, alpha)),
        "roc_auc_ci_high": float(np.percentile(roc_samples, 100 - alpha)),
        "bootstrap_resamples": int(len(pr_samples)),
    }


def run_arm_on_split(
    arm_name,
    enriched,
    y,
    train_idx,
    test_idx,
    cv=True,
    cv_k=5,
    target_precision=TARGET_PRECISION,
    bootstrap_iters=1000,
):
    # nb7/nb8 protocol for one arm on one prepared split: stratified validation
    # carve from the TRAIN side only, threshold frozen on validation, then one
    # frozen-threshold evaluation on the held-out test rows.
    arm = ARMS[arm_name]
    if arm["placeholder"]:
        raise ArmSkipped("%s: not implemented yet — pending the F3 notebook" % arm_name)

    X_raw = arm["build_features"](enriched)
    X_tr, X_te = X_raw.loc[train_idx], X_raw.loc[test_idx]
    y_tr, y_te = y.loc[train_idx], y.loc[test_idx]
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_tr, y_tr, test_size=0.25, random_state=RANDOM_STATE, stratify=y_tr
    )

    model = arm["make_model"](y_fit)
    cv_summary = None
    if cv and arm["supports_cv"]:
        cv_summary = cross_validate_model(
            model, X_tr, y_tr, k=cv_k, stratified=True, random_state=RANDOM_STATE
        )["summary"]
    model.fit(X_fit, y_fit)

    threshold = best_f1_threshold(y_val, model.predict_proba(X_val)[:, 1])
    test_proba = np.asarray(model.predict_proba(X_te))[:, 1]
    metrics = rich_test_metrics(y_te, test_proba, threshold, target_precision)

    if bootstrap_iters > 0 and not math.isnan(metrics["pr_auc"]):
        ci = bootstrap_auc_ci(y_te, test_proba, n_iters=bootstrap_iters)
    else:
        ci = {}

    if arm["n_features"] is not None:
        n_features = arm["n_features"](model)
    else:
        n_features = int(X_raw.shape[1])

    row = {
        "axis": None,  # filled by the caller (run_arm_on_axis)
        "arm": arm_name,
        "status": "ok",
        "test_rows": int(len(test_idx)),
        "test_positives": int(np.asarray(y_te).sum()),
        "chance_level": float(np.asarray(y_te).mean()),
        "frozen_threshold": float(threshold),
        "n_features": int(n_features),
        **metrics,
        **ci,
    }
    if cv_summary is not None:
        row["cv_pr_auc_mean"] = float(cv_summary["pr_auc_mean"])
        row["cv_pr_auc_std"] = float(cv_summary["pr_auc_std"])
    return row


def run_arm_on_axis(
    arm_name,
    axis,
    enriched,
    y,
    test_size=0.2,
    cv=True,
    cv_k=5,
    target_precision=TARGET_PRECISION,
    bootstrap_iters=1000,
):
    # Convenience entry point: one arm, one axis, split handled here.
    train_idx, test_idx = axis_split(axis, enriched, y, test_size)
    row = run_arm_on_split(
        arm_name,
        enriched,
        y,
        train_idx,
        test_idx,
        cv=cv,
        cv_k=cv_k,
        target_precision=target_precision,
        bootstrap_iters=bootstrap_iters,
    )
    row["axis"] = axis
    return row


# ---------------------------------------------------------------------------
# Results persistence: JSONL, one result row per line, accumulating across
# invocations under results/.
def load_results(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError("%s:%d is not valid JSON: %s" % (path, line_no, exc))
    return rows


def _jsonable(value):
    # JSONL hygiene: numpy scalars -> python scalars, NaN -> null.
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def save_results(path, rows):
    # sort_keys keeps the file byte-stable; no wall-clock fields are persisted,
    # so two identical invocations produce byte-identical files.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")


def merge_results(existing, new_rows):
    # Accumulate across invocations: rows from other (axis, arm) pairs are
    # kept; a re-measured pair supersedes its earlier row so the comparison
    # table always shows the latest measurement per pair.
    replaced = {(row["axis"], row["arm"]) for row in new_rows}
    kept = [
        row
        for row in existing
        if (row.get("axis"), row.get("arm")) not in replaced
    ]
    return kept + new_rows


TABLE_COLUMNS = [
    "axis",
    "arm",
    "status",
    "test_positives",
    "test_rows",
    "chance_level",
    "pr_auc",
    "pr_auc_ci_low",
    "pr_auc_ci_high",
    "roc_auc",
    "roc_auc_ci_low",
    "roc_auc_ci_high",
    "f1",
    "recall_at_precision",
    "frozen_threshold",
    "cv_pr_auc_mean",
    "cv_pr_auc_std",
    "n_features",
]


def print_comparison_table(rows, title="comparison table (accumulated, latest per axis x arm)"):
    # Every row carries test positives and chance level next to the metrics —
    # 11-24 test positives make the absolute numbers unreadable without them.
    axis_order = {axis: i for i, axis in enumerate(AXES)}
    arm_order = {name: i for i, name in enumerate(ARMS)}
    rows = sorted(
        rows,
        key=lambda r: (
            axis_order.get(r.get("axis"), 99),
            arm_order.get(r.get("arm"), 99),
        ),
    )
    frame = pd.DataFrame(rows)
    columns = [c for c in TABLE_COLUMNS if c in frame.columns]
    if any(row.get("status") != "ok" for row in rows):
        if "reason" not in frame.columns:
            frame["reason"] = ""
        columns.append("reason")
    print("\n== %s ==" % title)
    print(frame[columns].round(4).to_string(index=False))


def _one_line(text):
    return " ".join(str(text).split())


def _run_line(row):
    if row["status"] != "ok":
        return "[%s | %s] %s: %s" % (
            row["axis"], row["arm"], row["status"].upper(), row["reason"]
        )
    ci = (
        " [CI %.3f, %.3f]" % (row["pr_auc_ci_low"], row["pr_auc_ci_high"])
        if "pr_auc_ci_low" in row
        else ""
    )
    cv_line = (
        " | CV PR-AUC %.4f+-%.4f" % (row["cv_pr_auc_mean"], row["cv_pr_auc_std"])
        if "cv_pr_auc_mean" in row
        else ""
    )
    return (
        "[%s | %s] %.1fs | features=%d | threshold=%.4f | test PR-AUC %.4f%s "
        "| ROC-AUC %.4f | F1 %.4f | R@P=%.2f %.4f%s"
        % (
            row["axis"], row["arm"], row["runtime_s"], row["n_features"],
            row["frozen_threshold"], row["pr_auc"], ci, row["roc_auc"],
            row["f1"], row["target_precision"], row["recall_at_precision"], cv_line,
        )
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Unified arms registry x evaluation axes runner (FOC-175).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python src/fraud_pipeline.py --list-arms\n"
            "  python src/fraud_pipeline.py --run-arm xgb-baseline --axis all\n"
            "  python src/fraud_pipeline.py --axis random-grouped\n"
            "  python src/fraud_pipeline.py --run-arm sce --axis chronological\n"
        ),
    )
    parser.add_argument(
        "--run-arm",
        dest="arms",
        action="append",
        choices=sorted(ARMS),
        metavar="ARM",
        help="arm to run (repeatable; default: every non-placeholder arm)",
    )
    parser.add_argument(
        "--axis",
        choices=list(AXES) + ["all"],
        default="all",
        help="evaluation axis (default: all)",
    )
    parser.add_argument(
        "--list-arms", action="store_true", help="print the registry and exit"
    )
    parser.add_argument(
        "--no-cv",
        action="store_true",
        help="skip cross_validate_model (arms with per-fold refits skip it anyway)",
    )
    parser.add_argument("--cv-k", type=int, default=5)
    parser.add_argument("--target-precision", type=float, default=TARGET_PRECISION)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--bootstrap-iters",
        type=int,
        default=1000,
        help="percentile-bootstrap iterations for test AUC CIs (0 disables)",
    )
    parser.add_argument(
        "--results",
        default=str(DEFAULT_RESULTS_PATH),
        help="JSONL results file (accumulates across runs)",
    )
    parser.add_argument("--all-trxns", default=str(DEFAULT_ALL_TRXNS_PATH))
    parser.add_argument("--exchange-rates", default=str(DEFAULT_EXCHANGE_RATES_PATH))
    parser.add_argument("--dim-customer", default=str(DEFAULT_DIM_CUSTOMER_PATH))
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.list_arms:
        print("registered arms:")
        for name, arm in ARMS.items():
            marker = " (placeholder — pending F3 notebook)" if arm["placeholder"] else ""
            cv = "cv" if arm["supports_cv"] else "no-cv"
            print("  %-16s [%s]%s %s" % (name, cv, marker, arm["description"]))
        print("axes: %s" % ", ".join(AXES))
        return 0

    axis_names = list(AXES) if args.axis == "all" else [args.axis]
    arm_names = []
    for name in args.arms or [n for n, a in ARMS.items() if not a["placeholder"]]:
        if name not in arm_names:
            arm_names.append(name)

    enriched, y = load_enriched(
        all_trxns_path=args.all_trxns,
        exchange_rates_path=args.exchange_rates,
        dim_customer_path=args.dim_customer,
    )

    print(
        "fraud_pipeline: %d txns, %d frauds (%.2f%%), %d customers | arms=%s | axes=%s"
        % (
            len(enriched), int(y.sum()), 100 * y.mean(), enriched["customer"].nunique(),
            ",".join(arm_names), ",".join(axis_names),
        )
    )

    new_rows = []
    for axis in axis_names:
        train_idx, test_idx = axis_split(axis, enriched, y, args.test_size)
        test_positives = int(y.loc[test_idx].sum())
        print(
            "\n[%s] %s\n[%s] %d train / %d test rows | test positives: %d "
            "(chance PR-AUC %.4f)"
            % (
                axis, AXES[axis]["description"], axis, len(train_idx), len(test_idx),
                test_positives, test_positives / len(test_idx),
            )
        )
        for arm_name in arm_names:
            started = time.perf_counter()
            try:
                row = run_arm_on_split(
                    arm_name,
                    enriched,
                    y,
                    train_idx,
                    test_idx,
                    cv=not args.no_cv,
                    cv_k=args.cv_k,
                    target_precision=args.target_precision,
                    bootstrap_iters=args.bootstrap_iters,
                )
                row["axis"] = axis
            except ArmSkipped as exc:
                row = {
                    "axis": axis,
                    "arm": arm_name,
                    "status": "skipped",
                    "reason": _one_line(exc),
                }
            except Exception as exc:  # one bad arm never crashes the whole run
                row = {
                    "axis": axis,
                    "arm": arm_name,
                    "status": "error",
                    "reason": _one_line("%s: %s" % (type(exc).__name__, exc)),
                }
            row["runtime_s"] = round(time.perf_counter() - started, 1)
            new_rows.append(row)
            print(_run_line(row))

    persisted = [
        {key: value for key, value in row.items() if key != "runtime_s"}
        for row in new_rows
    ]
    all_rows = merge_results(load_results(args.results), persisted)
    save_results(args.results, all_rows)

    print_comparison_table(all_rows)
    if args.bootstrap_iters > 0 and any(
        row.get("status") == "ok" and row.get("test_positives", 0) <= 30
        for row in all_rows
    ):
        print(
            "\nnote: <=30 test positives — percentile bootstrap CIs are wide and "
            "indicative only"
        )
    print(
        "results: %s (%d rows total, %d new)" % (args.results, len(all_rows), len(persisted))
    )
    return 0 if any(row["status"] == "ok" for row in new_rows) else 1


if __name__ == "__main__":
    sys.exit(main())
