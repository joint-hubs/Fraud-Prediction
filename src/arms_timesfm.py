"""TimesFM feature-extractor arm (FOC-175 F3): a forecaster, not a classifier.

Implements the `timesfm-features` arm of the unified runner (fraud_pipeline.py).
TimesFM 2.5-200m (pytorch checkpoint) is used STRICTLY as a feature extractor:
for every transaction the pretrained forecaster is asked to predict the
customer's next amount / next arrival gap from that customer's STRICTLY EARLIER
transactions, and the forecast residuals — never any TimesFM "score" — become
extra columns on the base+client matrix. The registered arm's model is the SAME
fixed XGBClassifier factory as xgb-client (fraud_pipeline.xgb_params), so the
arm's hypothesis is "do forecast-residual features add signal", held by keeping
the model identical. TimesFM must never be presented as a classifier.

Series encodings (per customer, transactions sorted by timestamp; ties broken
by frame row order via a stable sort):

- amounts: amount_eur in time order. Forecast context for transaction i =
  that customer's first i amounts (strictly earlier rows only — a row never
  enters its own context, the same past-only bar as nb7/nb8).
- gaps: inter-transaction gaps in HOURS. Chosen over per-period counts because
  a gap localizes the arrival rhythm to the actual transaction, needs no
  arbitrary binning, and gives the forecaster one value per step. Context for
  transaction i = the i-1 gaps strictly before it (the gap at i itself is the
  value being predicted, so gap features back off for i < 2).

Features (6 columns, TIMESFM_FEATURES — all scale-free or bounded):

- tmf_amount_residual_ratio  (amount - point forecast) / point forecast,
  clipped to +/-RESIDUAL_CLIP; 0.0 = "exactly as forecast".
- tmf_amount_quantile_pos    fraction of the forecast's 0.1..0.9 quantile-head
  columns strictly below the realized amount, in [0, 1]; 0.5 = median.
- tmf_amount_spread_rel      forecast (q0.9 - q0.1) width over the point
  forecast (context uncertainty around the row), clipped to [0, RESIDUAL_CLIP].
- tmf_gap_residual_ratio     same residual ratio for the gap series.
- tmf_gap_quantile_pos       same quantile position for the gap series.
- tmf_context_len            number of strictly-earlier transactions used as
  context — lets the model discount short-history rows instead of the code
  silently trusting them.

Neutral backoffs (documented, honest no-information values): a customer's
first transaction has no context at all and its gap has no predecessor, so
residuals/spread fall back to 0.0, quantile positions to 0.5, and
tmf_context_len to 0 — the second transaction gets real amount features but
backed-off gap features.

Past-only discipline over the WHOLE frame: features are extracted once for all
rows with each row's context being strictly earlier transactions of its own
customer (labels never enter a forecast — the forecaster is unsupervised).
On the customer-disjoint axes a test row's context is entirely its own
customer's earlier test rows (the complete genuine history, production-equivalent);
on the chronological axis a test row may additionally read that customer's
earlier TRAIN rows — legitimate past, never the future. This is why one cached
full-frame extraction is split-respecting everywhere, and why CV is honest:
fold membership never changes a feature value and the context carries no
target information, so per-fold clones refit only the XGB (supports_cv=True).

Cost: all ~10.3k per-transaction forecasts (amounts + gaps) are batched into
ONE GPU call (~26 s on the RTX 5070 Ti, checkpoint load ~3-7 s extra) — the
feature extraction runs once per process and is cached on a fingerprint of
(customer, timestamp, amount_eur); every later call, including all three axes
and the 5 CV folds, reuses it.

Determinism: fixed seed, eval-mode inference (no dropout, no RNG), fixed batch
composition (stable sort + the API's own padding) — two extractions produce
bit-identical features (verified in nb11 by a row-identity-aligned
re-extraction check; this extractor returns rows in its own stable
(customer, timestamp) order, so a positional comparison of two runs
misaligns them and reports a bogus delta). torch_compile is left OFF: it
would add ~minutes of warm-up on this stack for no accuracy gain.

Checkpoint discipline: the `google/timesfm-2.5-200m-pytorch` snapshot is fully
cached locally; it is loaded with local_files_only=True end to end
(snapshot_download probe + from_pretrained) so NO network fetch happens at run
time. Exact loading calls:

  snapshot = snapshot_download("google/timesfm-2.5-200m-pytorch", local_files_only=True)
  model = TimesFM_2p5_200M_torch.from_pretrained(snapshot, local_files_only=True, torch_compile=False)
  model.compile(ForecastConfig(max_context=512, max_horizon=128,
                               per_core_batch_size=128,
                               use_continuous_quantile_head=True,
                               normalize_inputs=True))
  points, quantiles = model.forecast(horizon=1, inputs=[np.ndarray, ...])

points is (n_inputs, 1); quantiles is (n_inputs, 1, 10) — head column 0 is the
raw mean head, columns 1..9 are the q0.1..q0.9 quantiles and column 5 (the
median) equals the point forecast. Library quirk: forecast() pads the CALLER'S
input list in place to a multiple of the batch size, so callers pass a fresh
list. max_context=512 (16 patches) covers this table's longest per-customer
prefix (94 transactions) with headroom; max_horizon=128 is one decode step.

check_dependencies() probes timesfm + torch + the checkpoint's presence in the
local HF cache (local_files_only snapshot_download) and returns a short
message instead of raising when something is missing — the runner's factory
turns that into a SKIPPED row (fraud_pipeline.ArmSkipped), never a crash.

No arms_gbdt.sanitize_feature_names needed: the new columns are consumed by
XGBoost (xgb-client naming rules already satisfied; tmf_* names are safe).

Notebook API:
  from arms_timesfm import append_features, extract_timesfm_features, get_forecaster
  enriched_tmf = append_features(enriched)          # cached
  fresh = extract_timesfm_features(enriched)        # cache bypass (determinism check)
"""

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download

RANDOM_SEED = 42

REPO_ID = "google/timesfm-2.5-200m-pytorch"

# Compiled-decode budget: one prefill pass per batch (horizon 1 <= one output
# patch), 16 context patches (512 / 32) cover the longest customer prefix here.
MAX_CONTEXT = 512
MAX_HORIZON = 128
PER_CORE_BATCH_SIZE = 128

# Forecast quantile-head layout (n, horizon, 10): column 0 raw mean head,
# columns 1..9 the q0.1..q0.9 quantiles (column 5 = median = point forecast).
QUANTILE_LOW = 1
QUANTILE_HIGH = 9

# Ratio-type features are clipped to this band: fraud-relevant surprises live
# far inside it, while a near-zero forecast denominator would otherwise inject
# astronomically large values into the tree splits.
RESIDUAL_CLIP = 10.0

TIMESFM_FEATURES = [
    "tmf_amount_residual_ratio",
    "tmf_amount_quantile_pos",
    "tmf_amount_spread_rel",
    "tmf_gap_residual_ratio",
    "tmf_gap_quantile_pos",
    "tmf_context_len",
]

_FORECASTER = None
_FEATURE_CACHE = {}


def check_dependencies():
    # Probe-import the heavy libraries, then probe the checkpoint cache WITHOUT
    # network (local_files_only snapshot_download raises when the snapshot is
    # absent). Returns None when everything is in place, otherwise a short
    # message — the runner's factory turns a non-None result into an ArmSkipped
    # row (both in build_features and make_model; build_features runs first).
    missing = []
    for module in ("timesfm", "torch", "huggingface_hub"):
        try:
            __import__(module)
        except ImportError as exc:
            missing.append("%s (%s)" % (module, exc))
    if missing:
        return "; ".join(missing)
    try:
        snapshot_download(REPO_ID, local_files_only=True)
    except Exception as exc:
        return "checkpoint %s not in the local HF cache (%s: %s)" % (
            REPO_ID,
            type(exc).__name__,
            exc,
        )
    return None


def get_forecaster():
    # Load + compile the cached checkpoint once per process (registry path:
    # fraud_pipeline._make_timesfm -> extraction in build_features). Local
    # files only — the snapshot must already be in the HF cache.
    global _FORECASTER
    if _FORECASTER is None:
        import torch  # lazy: optional dependency
        from timesfm import ForecastConfig, TimesFM_2p5_200M_torch  # lazy

        torch.manual_seed(RANDOM_SEED)
        snapshot = snapshot_download(REPO_ID, local_files_only=True)
        model = TimesFM_2p5_200M_torch.from_pretrained(
            snapshot, local_files_only=True, torch_compile=False
        )
        model.compile(
            ForecastConfig(
                max_context=MAX_CONTEXT,
                max_horizon=MAX_HORIZON,
                per_core_batch_size=PER_CORE_BATCH_SIZE,
                use_continuous_quantile_head=True,
                normalize_inputs=True,
            )
        )
        _FORECASTER = model
    return _FORECASTER


def extract_timesfm_features(frame):
    """Past-only forecast-residual features for every row of `frame`.

    `frame` needs the customer, timestamp and amount_eur columns. Returns a
    float DataFrame indexed like `frame` with TIMESFM_FEATURES columns. Every
    row's features read ONLY that customer's strictly-earlier transactions;
    the ~10.3k per-transaction forecasts run as one batched GPU call.
    """
    assert frame["amount_eur"].notna().all(), "amount_eur has NaN — cannot build series"
    # Stable sort: ties in (customer, timestamp) keep frame row order, so the
    # per-customer series order — and therefore the features — is deterministic.
    sorted_frame = frame.sort_values(["customer", "timestamp"], kind="stable")
    customers = sorted_frame["customer"].to_numpy()
    amounts = sorted_frame["amount_eur"].to_numpy(dtype=float)
    gaps = (
        sorted_frame.groupby("customer", sort=False)["timestamp"]
        .diff()
        .dt.total_seconds()
        / 3600.0
    ).to_numpy()  # NaN at each customer's first row

    n = len(sorted_frame)
    # Start offset of each row's customer (running max over customer starts).
    starts = np.flatnonzero(np.r_[True, customers[1:] != customers[:-1]])
    row_start = np.zeros(n, dtype=int)
    row_start[starts] = starts
    row_start = np.maximum.accumulate(row_start)
    pos_in_customer = np.arange(n) - row_start

    # Rows with a real amount forecast (>=1 earlier txn) / gap forecast
    # (>=2 earlier txns: the gap needs a predecessor to be defined).
    amount_rows = np.flatnonzero(pos_in_customer >= 1)
    gap_rows = np.flatnonzero(pos_in_customer >= 2)
    amount_inputs = [amounts[row_start[gi]:gi] for gi in amount_rows]
    gap_inputs = [gaps[row_start[gi] + 1:gi] for gi in gap_rows]

    # One batched call for everything; fresh list — forecast() pads its input
    # list IN PLACE to a multiple of the global batch size (library quirk).
    points, quantiles = get_forecaster().forecast(
        horizon=1, inputs=list(amount_inputs) + list(gap_inputs)
    )
    assert len(points) == len(amount_inputs) + len(gap_inputs), "forecast row count mismatch"

    # Neutral backoffs first (no-context rows keep them): 0.0 residual/spread
    # = "exactly as forecast", 0.5 quantile position = the median.
    out = {
        "tmf_amount_residual_ratio": np.zeros(n),
        "tmf_amount_quantile_pos": np.full(n, 0.5),
        "tmf_amount_spread_rel": np.zeros(n),
        "tmf_gap_residual_ratio": np.zeros(n),
        "tmf_gap_quantile_pos": np.full(n, 0.5),
        "tmf_context_len": pos_in_customer.astype(float),
    }
    eps = 1e-9
    for k, gi in enumerate(amount_rows):
        point = float(points[k, 0])
        head = quantiles[k, 0]
        actual = amounts[gi]
        out["tmf_amount_residual_ratio"][gi] = np.clip(
            (actual - point) / (abs(point) + eps), -RESIDUAL_CLIP, RESIDUAL_CLIP
        )
        out["tmf_amount_quantile_pos"][gi] = float(
            np.mean(head[QUANTILE_LOW:QUANTILE_HIGH + 1] < actual)
        )
        out["tmf_amount_spread_rel"][gi] = np.clip(
            (head[QUANTILE_HIGH] - head[QUANTILE_LOW]) / (abs(point) + eps),
            0.0,
            RESIDUAL_CLIP,
        )
    for k, gi in enumerate(gap_rows, start=len(amount_rows)):
        point = float(points[k, 0])
        head = quantiles[k, 0]
        actual = gaps[gi]
        out["tmf_gap_residual_ratio"][gi] = np.clip(
            (actual - point) / (abs(point) + eps), -RESIDUAL_CLIP, RESIDUAL_CLIP
        )
        out["tmf_gap_quantile_pos"][gi] = float(
            np.mean(head[QUANTILE_LOW:QUANTILE_HIGH + 1] < actual)
        )
    return pd.DataFrame(out, index=sorted_frame.index, columns=TIMESFM_FEATURES)


def append_features(frame, use_cache=True):
    """frame + the TIMESFM_FEATURES columns, extracted once and cached.

    The cache key is a fingerprint of exactly the columns the features depend
    on (customer, timestamp, amount_eur), so any re-load of the same table
    reuses the GPU extraction; use_cache=False forces a fresh pass (nb11's
    determinism check). Returns a NEW frame — the input is never mutated.
    """
    key = None
    if use_cache:
        hashed = pd.util.hash_pandas_object(
            frame[["customer", "timestamp", "amount_eur"]], index=True
        )
        key = (len(frame), tuple(frame.columns), int(hashed.sum()))
        cached = _FEATURE_CACHE.get(key)
        if cached is not None:
            return pd.concat([frame, cached], axis=1)
    features = extract_timesfm_features(frame)
    if use_cache:
        _FEATURE_CACHE[key] = features
    return pd.concat([frame, features], axis=1)
