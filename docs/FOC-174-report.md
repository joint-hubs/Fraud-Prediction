# FOC-174 — Fraud-Prediction R&D Report

> Phased technical report for the fraud-prediction research repository. This
> file is appended to phase-by-phase; each phase contributes one numbered
> section. Section numbering matches the F0–F5 phases.

## §1 F0 — Method review, eval harness & environment setup

This phase did not train new models for production. It audited the existing
notebooks, stood up a shared evaluation harness, fixed the environment, and
repaired the most damaging leakage and correctness bugs so that F1–F5 build
on a trustworthy baseline.

### 1.1 Environment

- Python 3.11.9 venv at `.venv/`. System Python 3.13 was deliberately avoided
  for better ML-library compatibility (notably `pytorch-tabnet`).
- torch 2.11.0+cu128 installed; **CUDA verified True** on an NVIDIA RTX 5070 Ti
  Laptop GPU (Blackwell sm_120), 12 GB. GPU matmul sanity check passed.
- `requirements.txt` extended from 14 to 23 packages, grouped and commented:
  - **F0 core:** `imbalanced-learn`, `ipykernel`, `nbformat`.
  - **F1:** `lightgbm`, `catboost`.
  - **F1 tabnet:** `pytorch-tabnet` 4.1.0. The brief asked for 3.3.0, but that
    version does not exist on PyPI; 4.1.0 is the actual latest and installs
    cleanly on Python 3.11.
  - **F2:** `timesfm`, `sentence-transformers`, `facenet-pytorch`.
  - **F2 SCE:** `stat-context` from github `joint-hubs/sce` — commented out
    and deferred to F2 (needs auth).
- Jupyter kernel `fraud-f0` registered.
- **Caveat (honest):** `facenet-pytorch` pins `torch>=2.2,<2.3` and during
  install tried to downgrade the CUDA build to a CPU torch (`cuda=False`).
  Reinstalled torch cu128 afterwards to restore CUDA. Recommendation for
  future F2+ installs of ML packages that pin torch: use `--no-deps` and
  manually verify `torch.cuda.is_available()` afterwards.

### 1.2 Shared evaluation harness (`funs.py`)

- `evaluateModel(y_test, y_pred, y_score=None, target_precision=0.5)` was
  refactored. It stays backward-compatible: with no `y_score` it behaves as
  before — accuracy + confusion matrix + `classification_report`, returns
  `None`. With `y_score` (probabilities) it returns a dict
  `{accuracy, precision, recall, f1, roc_auc, pr_auc, best_threshold_f1,
  recall_at_precision, target_precision}` and plots the precision-recall
  curve. **Accuracy is demoted** from the headline metric to one dict key.
- New `chronological_split(X, y, timestamp=None, test_size=0.2,
  random_state=42)` — time-based split: sort by timestamp, test set strictly
  later than train, with a stratified fallback when no timestamp is passed.
- New `cross_validate_model(model, X, y, k=5, stratified=True, shuffle=True,
  random_state=42)` — StratifiedKFold CV, returns per-fold and summary
  metrics.
- The harness is the shared import for all later notebooks (F1–F5).

### 1.3 Methods review — what was wrong

Bulleted findings describing the "before" state, each in one sentence:

- Only accuracy was used everywhere; on a 1.72% positive rate this is
  misleading.
- No AUC / PR-AUC / recall@precision / PR curve were reported.
- No cross-validation (nb5 ran a grid search that was discarded — best params
  computed, then ignored; the final model used a hardcoded `reg_lambda=0.5`
  outside the grid).
- Inconsistent splits: `test_size` 0.2 vs 0.3, and `stratify` applied only
  sometimes.
- Hand-picked inconsistent thresholds: in the dictionary model, train used
  0.15 vs test 0.18, and the train rule had an extra
  `quantile_9_flags>2` clause the test rule omitted — a structural
  divergence, not just tuning.
- XGB hyperparams too aggressive for ~4k train rows
  (`learning_rate=1`, `max_depth=10` → overfit risk).
- Feature-selection leakage: `SelectKBest` was fit on the full data (nb2 and
  nb3).
- `amount_eur` FX conversion was 100% broken: the `rate` merge on
  `[ccy, date]` never matched because `date` was `datetime.date` on one side
  and strings on the other, so all 5302 rates were NaN and a silent
  `rate=1` fallback masked it — every `amount_eur` was just the raw
  `amount`.
- A fragile `drop_duplicates(["timestamp","customer","counterparty","name"])`
  inside `dictionaryModel`.

### 1.4 Leakage & security fixes applied

- `amount_eur` FX: the date dtype mismatch was fixed so real rates now join
  (4712 of 5302 rows get their actual FX rate). EUR is the base currency
  (confirmed `base="EUR"` in the Currencies API notebook), so EUR rows get
  `rate=1.0`, `amount_eur=amount`, flag `False`. A boolean
  `amount_eur_fx_missing` column flags genuinely-missing rates (non-EUR
  ccy/date absent from the rates file) instead of silently assuming 1:1.
  `amount_eur` is NaN for genuinely-missing rows.
- `SelectKBest` now fits on TRAIN only (nb2 and nb3): split before fit, then
  transform both partitions.
- **Security:** a hardcoded apilayer.com API key was purged from the working
  copy of `src/Currencies API.ipynb` (replaced with
  `os.environ.get("APILAYER_KEY", "")` plus a TODO pointing to FOC-181).
  This is a working-copy scrub only — NO git history rewrite and NO key
  rotation (deferred to FOC-181 by decision). Verified: the key literal is
  absent from the entire working tree.

### 1.5 Model fixes (nb3, nb4, nb5)

- **nb3 (Tree Based Models / XGB):** `learning_rate=1` / `max_depth=10`
  replaced with conservative values; class imbalance addressed via
  `scale_pos_weight` / SMOTE-ADASYN / class weights; decision threshold tuned
  on the PR curve through the harness (`best_threshold_f1`,
  `recall_at_precision`); split standardized.
- **nb4 (Dictionary):** hand-tuned train/test thresholds replaced by a
  principled grid search maximizing F1 on the train set, with the SAME rule
  applied to test (no train/test divergence).
- **nb5 (Combined):** the discarded grid search is now used — `scoring`
  switched from `'accuracy'` to `'average_precision'` (PR-AUC); the final
  model uses `grid.best_params_` instead of hardcoded overrides; `reg_lambda`
  stays within the searched grid.
- Metrics reported per the new harness (test set, `evaluateModel` with `y_score`):
  all four notebooks execute end-to-end on the venv (nbconvert exit 0). nb2 is
  analytical (chi-squared / SelectKBest) and does not call `evaluateModel`. The
  three model notebooks, on the chronological/stratified test split:

  | Notebook | PR-AUC | ROC-AUC | F1 (frozen thr) | Threshold source | Recall@prec=0.50 |
  |----------|--------|---------|------------------|------------------|------------------|
  | nb3 (XGB, full feats) | 0.053 | 0.636 | 0.000 | 0.965 (val) | — |
  | nb3 (XGB, selected feats) | 0.033 | 0.548 | 0.000 | 0.980 (val) | — |
  | nb4 (Dictionary) | 0.329 | 0.625 | 0.397 | 0.310 (grid, train) | — |
  | nb5 (Combined dict→XGB) | 0.507 | 0.857 | 0.400 | 0.068 (grid, train) | 0.556 |

  These are baseline numbers after the F0 fixes, not tuned production metrics —
  they establish the reproducible harness + consistent splits so F1+ can compare
  against them. Note the dictionary-model `expected_fraud_probability` is used as
  `y_score` in nb4/nb5 (a continuous risk score), so their PR-AUC is directly
  comparable across the two-layer pipeline. nb5 (dict features → XGB) clearly
  dominates the pure dictionary model, as expected.

  > The nb3 F1 of 0.0 is honest, not broken: the threshold is now tuned on a
  > **validation split of TRAIN** (frozen) and applied once to the held-out test.
  > `scale_pos_weight` inflates fraud probabilities on the val set, so the
  > val-derived best-F1 threshold lands near ~0.97; on the test set that threshold
  > yields zero positive predictions. PR-AUC (threshold-independent) is the correct
  > ranking-quality read on nb3. This is exactly the leak-free behaviour the
  > review required (see §1.5.1).

### 1.5.1 Round-1 review fixes (leakage / correctness)

Three blocking findings from the round-1 review, all fixed on this branch:

- **FIX1 — nb2 broken inline FX (data-leakage via mis-pricing).** nb2 rolled its
  own FX merge on `[ccy, date]` that never matched (`currency_rates.date` was a
  string vs `trxns_data.date` a `datetime.date`), so all 5302 rates were NaN and a
  silent `np.where(isna, 1)` priced every row at 1:1 — 88.9% of rows are non-EUR,
  so `amount_eur` was the raw `amount`. nb2's SelectKBest ranking (which feeds
  `my_feature_names` in nb3) was therefore built on mis-priced features. **Fix:**
  nb2 now calls the shared `funs.dataPreparation` (which normalizes the date dtype,
  joins real rates, sets EUR=1.0, flags FX-missing) instead of the inline merge.
  Verified: a USD row now shows `amount 56691.27 → amount_eur 46514.01`.
- **FIX2 — nb3 threshold tuned on the TEST set.** cells 43/55 derived
  `best_threshold_f1` from `precision_recall_curve(y_test, y_proba)` and re-applied
  it to the same `y_test` — optimistic bias, test no longer held out.
  `evaluateModel` itself was correct. **Fix:** a validation set is carved from
  TRAIN (`train_test_split(X_train, y_train, stratify=y_train)`), the threshold is
  tuned on that val set and **frozen**, then the test set is evaluated exactly once
  with the frozen threshold (no read-back of `best_threshold_f1` from the test
  call). Both XGB passes (full + selected features) fixed.
- **FIX3 — nb5 target left in the feature matrix.** cell 5 did `X = data` (alias)
  then `data = data.drop(target)` — only rebinds the name, so `X`/`X_train`
  physically still carried `fraud_flag_trans` (the 0/1 target). It reached no XGB
  only by accident (a `dictionaryModel` whitelist omission). **Fix:** the target
  column is dropped **before** `X` is assigned (`X = data.drop(columns=target).copy()`),
  so the feature matrix cannot contain the label by construction. Note: the
  original `fraud_flag` (Y/N) is intentionally retained on the frame because
  `funs.dictionaryModel` needs it to build `fraud_flag_transformed`; it is dropped
  from the XGB input frame before `get_dummies`. Verified: no `fraud_flag*` column
  is present in `model_train_data` (94 cols).

### 1.6 Open items / notes

- `drop_duplicates` in `dictionaryModel` left as-is — out of scope for this
  phase, flagged for review.
- `stat-context` (SCE) install deferred to F2.
- F1–F5 libs installed best-effort; only `stat-context` was deferred.

## §2 F1 — Client enrichment (synthetic dim_customer)

F1 asks the phase's real R&D question: does a client dimension table — joined
onto transactions by the anonymous `customer` id — move the XGB baseline?
Phase decision D3 forbids PII, so the client table is fully synthetic: every
attribute is a deterministic draw seeded by the customer id. Chunk A (nb7)
built the table and the join; chunk B answers the question with a controlled
A/B evaluation through the F0 harness. Verdict up front: **client enrichment
moves the measured baseline (test PR-AUC 0.0532 → 0.2374), but the honest
reading is customer-identity signal, not transferable demographic signal** —
see §2.6.

### 2.1 Environment

- nb7 runs on system Python 3.13.7 (kernel `fraud-f0`): pandas 2.3.3,
  scikit-learn 1.7.2, xgboost 2.1.4. The F0 `.venv` (Python 3.11) is unusable
  in this worktree — Windows App Control blocks its pandas DLLs — so all
  metrics are re-established within this single run, and absolute numbers may
  differ slightly from §1.5. The A/B delta is computed within this one run and
  is internally consistent. Encouragingly, the in-run arm-(a) baseline lands
  within rounding of §1.5's nb3 numbers (PR-AUC 0.0532 vs 0.053, ROC-AUC
  0.6361 vs 0.636, F1 0.000, frozen threshold 0.9652 vs 0.965).
- No new dependencies; `funs.py` untouched; evaluation only through the F0
  harness (`chronological_split`, `cross_validate_model`, `evaluateModel`),
  with accuracy demoted per the F0 convention.

### 2.2 Method — synthetic `dim_customer`

- **Determinism via SHA-256 → seed.** Each attribute draw comes from a
  `random.Random` instance seeded with the SHA-256 of the customer id, so the
  seed travels with the id — same id → same profile on every run and machine,
  with no global seed to lose. `generate_dim_customer()` is a pure function of
  the id list; nb7 builds it twice and asserts the two frames identical (an
  in-run determinism guard).
- **Distributions.** Age ~ N(45, 15) clipped to [18, 85] (drawn mean 44.3,
  range 18–82); employment status follows age (students young, retirees old);
  income ~ lognormal, banded into 6 ordinal bands whose `1_`–`6_` label
  prefixes keep sort order == band order; account tenure grows with age
  (drawn age/tenure correlation 0.79); device/channel mobile-first skews.
  Every categorical stays ≤ 8 levels so one-hot encoding stays compact on 100
  customers.
- **Sampling artifact (no fix needed).** The `5_90k_140k` income band has 0
  customers in this 100-customer draw — a natural consequence of the
  lognormal draw at this sample size. The band stays in the label space for
  future draws; nothing downstream breaks on the empty level.
- **D3 (no PII).** Attributes are fully synthetic: no names, emails or
  account numbers; the only key is the anonymous `customer` id.

### 2.3 Join

- `dataPreparation()` output (5302 × 19) is left-merged with `dim_customer`
  (100 × 9) on `customer`, validated as `many_to_one` (many transactions per
  single customer row) → enriched frame **5302 × 27**.
- Sanity checks pass: no rows lost, the customer set is unchanged, and the 8
  new columns introduce zero nulls.

### 2.4 A/B experiment design

- **Arm (a) baseline** — the 10 base transaction features of nb3
  (`customer_country`, `counterparty_country`, `type`, `ccy`, `customer_type`,
  `weekday`, `month`, `quarter`, `hour`, `amount_eur_bucket`), one-hot encoded
  exactly as nb3 (87 encoded features).
- **Arm (b) enriched** — the same 10 features plus the 8 client columns;
  `age` and `account_tenure_years` pass through as numbers, the 6 categorical
  client columns go through the same `get_dummies` flow (116 encoded
  features).
- Everything except the feature matrix is IDENTICAL between arms: same `y`
  (`fraud_flag` → {N:0, Y:1}), same chronological split (same timestamps,
  `test_size=0.2`, `random_state=42` — asserted row-identical between arms),
  same validation carve from TRAIN (`test_size=0.25`, stratified), same
  stratified 5-fold CV on the chronological train split, same XGB
  hyperparameters (lr 0.05, depth 6, 200 trees, `reg_lambda=1.0`,
  `scale_pos_weight` = neg/pos on the fit split, `eval_metric="logloss"`),
  same threshold procedure (tuned on the validation split via
  `evaluateModel`'s `best_threshold_f1`, frozen, applied once to test).
- One encoding accommodation: XGBoost forbids `[`, `]` and `<` in feature
  names. nb3 already stripped `[`/`]` (amount-bucket intervals); nb7
  additionally maps the `<` in the `1_<20k` income-band label.

### 2.5 Results

Test metrics use each arm's frozen validation threshold; CV is stratified
5-fold on the chronological train split. Δ is arm (b) − arm (a).

| Metric | (a) base features | (b) base + client | Δ (b−a) |
|--------|-------------------|-------------------|---------|
| Test PR-AUC | 0.0532 | 0.2374 | +0.1842 |
| Test ROC-AUC | 0.6361 | 0.7692 | +0.1331 |
| Test F1 (frozen threshold) | 0.0000 | 0.2667 | +0.2667 |
| Test recall@precision=0.50 | 0.0417 | 0.2083 | +0.1667 |
| CV PR-AUC mean±std (train) | 0.4633 ± 0.0660 | 0.5253 ± 0.0837 | +0.0620 |
| Frozen threshold (tuned on val) | 0.9652 | 0.6648 | −0.3004 |
| Encoded features | 87 | 116 | +29 |

Two honesty notes on the table. First, arm (a)'s F1 of 0.0000 repeats the
§1.5.1 behaviour, not a new defect: `scale_pos_weight` inflates fraud
probabilities on the val split, the val-tuned threshold lands at 0.9652, and
applied to test it yields zero positive predictions. Second, arm (b)'s F1
gain partly rides on its threshold landing at a usable 0.6648 (test precision
0.667 / recall 0.167) — but the PR-AUC gain (+0.1842) is threshold-independent
ranking quality, so the lift is not a threshold artifact.

### 2.6 Interpretation (honest read)

- **Memorization caveat.** Every client attribute is a deterministic function
  of the `customer` id, so the 8 columns act as a low-cardinality proxy for
  customer identity. The chronological split does not hold customers out —
  customers appearing in both the train and test windows carry identical
  attribute vectors — so the measured lift may reflect customer-identity
  memorization rather than transferable demographic signal.
- **Low cardinality, constant per customer.** 100 unique customers carry 91
  fraud transactions (~1.72% positive rate); each client feature has ≤ 8
  levels and is constant within a customer, so the model can learn at most a
  per-customer prior. The A/B difference is dominated by which customers
  happen to fall into the later test window, not by demographics that would
  generalize to new customers.
- **Where the evidence points.** CV PR-AUC improves too, but far less
  (+0.0620 vs +0.1842 on test); both comparisons mix customers across folds,
  so neither isolates transferable signal. Verdict in plain words: client
  enrichment **moves** the measured baseline, and the signal it adds is best
  described as **who the customer is**, not **what kind of customer they are**.

### 2.7 Open items / notes

- A customer-grouped (disjoint) train/test split would cleanly separate
  memorization from transferable demographic signal — candidate follow-up
  before any claim that client demographics help fraud detection.
- The synthetic attributes are a stand-in for real client data; when a real
  `dim_customer` lands, the same A/B protocol (identical-splits assertion,
  val-frozen threshold) is directly reusable.
- The empty `5_90k_140k` income band is a sampling artifact of the 100-row
  draw (§2.2), kept for future draws.
- nb7 executes end-to-end via nbconvert (exit 0), including the in-run
  determinism assertion and the row-identical A/B split assertion.

## §3 F2 — SCE context enrichment (cross-fitted statistics)

F2 asks the phase's R&D question: does leakage-safe context enrichment —
per-group fraud-rate statistics computed by a cross-fitted engine — beat the
client-identity signal of §2 and the dictionary signal of nb4/nb5? The engine
is `stat-context` 0.4.0 (upstream `joint-hubs/sce`), driven through the F0
harness in a four-arm comparison (`src/8. SCE Enrichment.ipynb`, "nb8" below)
on both the chronological split and a NEW customer-grouped split. Verdict up
front: **no — the SCE arm underperforms even the un-enriched baseline on the
chronological split, and on the customer-grouped split every arm measured in
F0–F2 collapses to at-or-below chance.** The negative result is the finding:
what the program has measured so far is identity- and period-memorization,
not transferable fraud detection (§3.6).

### 3.1 Environment

- `stat-context` 0.4.0, installed from **PyPI** (public wheel; upstream
  project `joint-hubs/sce`, Apache-2.0). Installed from PyPI, not from
  GitHub, and no substitute implementation was used — the §1.1 deferral
  ("needs auth") was resolved by the public wheel.
- nb8 runs in this phase's fresh worktree venv built from the pinned
  requirements: CPython 3.11.9, kernel `python3` (pandas 2.3.3,
  scikit-learn 1.7.2, xgboost 2.1.4, stat-context 0.4.0 — nb8's second cell
  prints the full resolved set at execution time). Execution is via nbclient
  with the Windows Selector event-loop policy (the nbconvert CLI kernel
  start fails on this machine). nb8 is committed WITH its outputs and is
  the executed source of every number in this section.
- Target is binary fraud (0/1); the engine computes per-group fraud-rate
  statistics (mean/count aggregations) over declared groupings.

### 3.2 Method — cross-fitted context features

- **Engine config.** `min_group_size=5`, aggregations MEAN+COUNT,
  `include_interactions=True` (pairs, max depth 2), global stats on,
  `include_relative_features=False` (the library itself warns this causes
  target leakage), `include_fold_variance=False` (only meaningful for the
  random strategy), `random_state=42`.
- **Leakage safety via time cross-fitting.** `use_cross_fitting=True`,
  `cross_fit_strategy="time"`, `time_col=timestamp`, `n_folds=5`: the engine
  sorts by time and runs `TimeSeriesSplit`, and each fold's group statistics
  are computed on that fold's train rows only — every row's context features
  are strictly past-only.
- **Documented artifact, verified in-run.** The earliest ≈ 1/(n_folds+1) ≈
  17% of fitting rows keep NaN context features (they precede every fold's
  statistics source) and back off to the global fraud mean. The
  leakage-verification cell asserts that flipping targets after a
  60th-percentile time cut leaves pre-cut rows' features bit-identical (the
  control confirms post-cut rows do move), and that unseen groups at
  `transform()` back off to the global fraud mean.
- **Groupings — 18, all present in the data (none invented).** 10
  transaction categoricals (the nb3 base set: `customer_country`,
  `counterparty_country`, `type`, `ccy`, `customer_type`, `weekday`,
  `month`, `quarter`, `hour`, `amount_eur_bucket`) + 6 client categoricals
  from §2's `dim_customer` (`gender`, `employment_industry`,
  `employment_status`, `income_band`, `device`, `channel`) + 2 binned client
  numerics (`age_band`, edges 17/25/40/55/120; `tenure_band`, edges
  −1/2/7/15/100).
- **The customer id itself is deliberately excluded** — a per-customer fraud
  rate would re-import exactly the identity signal §2 measured, defeating
  the point of the leakage-safe arm.
- Column count reconciles cleanly: 18 singles + 153 pairwise interactions
  (C(18,2)) + 1 global = **172 grouping level-sets**; with MEAN and COUNT
  aggregations each, **344 statistic columns**.

### 3.3 Engine dtype quirk (found & worked around)

- stat-context 0.4.0's out-of-fold assignment string-casts converted-dtype
  grouping columns (int `hour` → str `hour`, `Interval` `amount_eur_bucket`
  → str) while untouched rows keep their original values — mixed int/str
  within one column. This fragments engine groups (24 `hour` groups became
  48) and makes downstream one-hot encoding emit duplicate column names.
- **Fix:** all grouping columns are normalized to str at the engine boundary
  (`timestamp` keeps its dtype). Worth reporting upstream to
  `joint-hubs/sce` (§3.7); the workaround is documented in the notebook.

### 3.4 Experiment design — four arms × two splits

- Four arms run through the shared F0 harness (`src/funs.py`) under nb7's
  §2 protocol: identical row splits across arms (asserted), a 25% stratified
  validation carve from train, XGBClassifier with §2's fixed
  hyperparameters, `scale_pos_weight` = neg/pos, F1-optimal threshold frozen
  on validation, one-shot test evaluation, stratified 5-fold CV on train.
- **Arms:**
  - **(a) base** — the 10 base transaction features (87 encoded), as §2.
  - **(b) base + client** — the 8 raw `dim_customer` columns added (116
    encoded); identical to §2's arm (b).
  - **(c) base + SCE** — base features plus the 344 cross-fitted context
    statistics; the SCE engine is **refit inside every CV fold** (custom
    sklearn-compatible wrapper; `transform()` for validation/test), so the
    CV scores are leakage-safe too. That `transform()` is purely in-memory:
    the wrapper enriches the frame it is handed through the fold-fitted
    engine and builds the model matrix from the returned DataFrame — nothing
    is persisted between calls, and the enrichment is recomputed on every
    `predict_proba()`/`predict()` invocation (each fold's clone refits its
    own engine). Out-of-sample rows are scored with the engine's
    full-fitting-frame statistics — computed from the fitting rows only —
    and unseen groups back off hierarchically to the global rate rather than
    erroring (asserted empirically in nb8).
  - **(d) base + dictionary** — nb5-style dictionary scores as features;
    dictionaries and thresholds built on train rows only.
- **Two splits for every arm:**
  - **Chronological** — last ceil(20%) = 1061 rows as test (the §1/§2
    convention).
  - **Customer-grouped (new `grouped_split` in `funs.py`)** — customers
    ordered by first-seen transaction (ties broken by id), last ceil(20%) =
    **20 of 100 customers** → test; train/test customers are disjoint.
    Granularity is customer-entry-time, not row granularity (§3.7).

### 3.5 Results

Test metrics use each arm's frozen validation threshold; CV is stratified
5-fold on the train split. Δ is the arm's test PR-AUC minus arm (a)'s.

Chronological split (last 1061 rows as test):

| Arm | Test PR-AUC | Test ROC-AUC | Test F1 (frozen thr) | Recall@prec=0.50 | CV PR-AUC (train) | Encoded features | Δ PR-AUC vs (a) |
|-----|-------------|--------------|----------------------|------------------|-------------------|------------------|-----------------|
| a: base | 0.0532 | 0.6361 | 0.0000 | 0.0417 | 0.4633 ± 0.0660 | 87 | — |
| b: base + client | 0.2374 | 0.7692 | 0.2667 | 0.2083 | 0.5227 ± 0.0809 | 116 | +0.1842 |
| c: base + SCE | 0.0340 | 0.5542 | 0.0000 | 0.0000 | 0.2214 ± 0.1415 | 429 | −0.0192 |
| d: base + dictionary | 0.1034 | 0.7653 | 0.0769 | 0.0417 | 0.3539 ± 0.0814 | 92 | +0.0502 |

Honesty notes on this table. First, arms a and c show F1 = 0.0000 — the
known frozen-threshold artifact (§1.5.1, §2.5), not a new defect:
`scale_pos_weight` inflates fraud probabilities on the validation carve, the
val-tuned threshold lands high, and applied once to test it yields zero
positive predictions; PR-AUC (threshold-independent) is the ranking read,
and by that read arm c is the worst arm, not a tied one. Second, encoded
feature counts for arms c and d differ slightly between splits (c: 429 here
vs 431 grouped; d: 92 vs 94) — the SCE engine and the dictionaries are fit
per split on that split's train rows, so emitted column counts can differ;
the per-column difference was not itemized in the run. Third, arms a/b
reproduce §2.5's test numbers exactly (0.0532 / 0.2374 — the harness
consistency check passes); arm a's CV aggregate matches §2.5 exactly too,
while arm b's lands a third decimal off (0.5227 ± 0.0809 vs 0.5253 ±
0.0837) — nb8 re-runs the pipeline within this phase's single run, and only
the test-side numbers are claimed as reproductions.

Customer-grouped split (20 held-out customers as test):

| Arm | Test PR-AUC | Test ROC-AUC | Test F1 (frozen thr) | Recall@prec=0.50 | CV PR-AUC (train) | Encoded features | Δ PR-AUC vs (a) |
|-----|-------------|--------------|----------------------|------------------|-------------------|------------------|-----------------|
| a: base | 0.0101 | 0.3152 | 0.0000 | 0.0000 | 0.6202 ± 0.0959 | 87 | — |
| b: base + client | 0.0091 | 0.2543 | 0.0000 | 0.0000 | 0.7277 ± 0.0599 | 116 | −0.0010 |
| c: base + SCE | 0.0119 | 0.3670 | 0.0000 | 0.0000 | 0.4532 ± 0.1693 | 431 | +0.0018 |
| d: base + dictionary | 0.0123 | 0.4338 | 0.0000 | 0.0000 | 0.4786 ± 0.1349 | 94 | +0.0022 |

Honesty note on this table: the grouped test holds 11 positives in 831
rows (the executed notebook prints the count), so a random ranking lands
near 11/831 ≈ 0.0132 on PR-AUC and at 0.5 on ROC-AUC — not near the 0.0172
global prevalence — and all four arms sit below both baselines. F1 = 0.0000
across the board is again partly the frozen-threshold artifact, but unlike
the chronological table the PR-AUC/ROC-AUC columns show the ranking itself
fails, so this is not a threshold story. The CV column is the tell:
stratified 5-fold CV mixes customers across folds, so within-customer
signal survives there (b: 0.7277) while the customer-disjoint test
collapses. At this scale the Δ column distinguishes nothing — all arms sit
in a 0.009–0.012 band at or below the 0.0132 chance level.

### 3.6 Interpretation (honest read)

- **Chronological ranking: client (b, 0.2374) > dictionary (d, 0.1034) >
  base (a, 0.0532) > SCE (c, 0.0340). The F2 question — does leakage-safe
  enrichment beat the dictionary signal? — is answered NO.** SCE
  underperforms even the un-enriched baseline (Δ −0.0192 PR-AUC vs base,
  −0.0695 vs the dictionary arm, as measured in the run).
- **Why SCE loses here — hypotheses, stated honestly (not established
  causes):** 344 sparse statistic columns against 67 positives in the
  chronological train split (count from the round-1 reviewer cohort control,
  not from nb8); heavy group fragmentation under
  `min_group_size=5` (172 level-sets on 5.3k rows); the time strategy's NaN
  head (≈17% of fitting rows) backs off to the global mean; fixed XGB
  hyperparameters (nothing was tuned for any arm, by the nb7 discipline);
  frozen-threshold artifacts drive F1 to 0 for arms a and c. CV agrees: arm
  c has the lowest CV PR-AUC (0.2214) with the highest variance (±0.1415).
- **Customer-grouped split: everything collapses.** The grouped test holds
  11 positives in 831 rows (chance PR-AUC ≈ 0.0132, printed by the executed
  notebook). All four arms land at PR-AUC 0.009–0.012 — at or below chance —
  and ROC-AUC 0.25–0.44 is below chance. The §2 winner (client features,
  0.2374) collapses hardest (→ 0.0091); the dictionary collapses too
  (0.1034 → 0.0123); SCE is no better (0.0119). **No signal measured in
  F0–F2 survives customer-disjoint evaluation: what looked like fraud
  detection is memorization, not transferable fraud patterns.**
- **Two distinct mechanisms hide in that collapse — separated (review
  round-1 control run incorporated).** (i) *Unseen customers alone kill the
  signal*: on a random customer-cohort split (same protocol, no time shift)
  arms a/b fall to chance — PR-AUC 0.0171 / 0.0144 against a 0.0133 chance
  level, ROC-AUC 0.5622 / 0.4844. The features carry no information about
  customers they were not fit on. (ii) *The later-period shift is a second,
  separate effect*: only in the time-shifted grouped split do ROC-AUCs drop
  BELOW 0.5 (0.25–0.44) — a period-cohort reversal (feature–target
  relationships flip between the 2021 train population and the held-out
  cohort's period), not a universal anti-signal: the same base features
  rank at chance (a: ROC 0.5622), not below it, when the cohort is drawn at
  random. Below-chance ROC is a property of the time-shifted evaluation,
  and the two mechanisms are reported separately rather than conflated
  under the unseen-customer label.
- **Train-side CV makes it explicit.** On the grouped split, CV PR-AUC stays
  high (b: 0.7277, a: 0.6202) while test collapses — the identity signal is
  real within known customers and transfers to no one.
- **Historical context — do not over-claim.** nb4 (0.329) / nb5 (0.507) used
  nb5's own protocol with a grid-searched XGB. This phase's fixed-parameter
  dictionary arm (d) reaches 0.1034 on the same chronological split — most
  of that gap is protocol (tuning), not signal; 0.507 is not comparable to
  the tables above.

### 3.7 Open items / notes

- The SCE arm was not tuned (fixed XGB parameters everywhere, per the nb7
  discipline); a feature-selection pass over the 344 SCE columns and per-arm
  tuning are natural follow-ups — as is F3's model-experiment phase.
- The engine dtype quirk (§3.3) deserves an upstream issue at
  `joint-hubs/sce`; the workaround is documented in the notebook.
- `grouped_split` guarantees customer-entry-time ordering, not row-level
  separation — documented in `funs.py`.
- **stat-context hash-pin unresolved** — a single `--hash=` line activates pip
  require-hashes mode, which demands a fully-hashed closure including
  transitive dependencies, and the requirements file's deliberately unpinned
  F4 block makes that unresolvable without pip-compile-style tooling (full
  note and the verified wheel sha256: §4.5).
- **Program reframe (the actionable outcome of this phase):** before any F3
  model work, the evaluation axis that matters is the customer-grouped
  split; every F3/F4 candidate must be judged there, not on the
  chronological split.

## §4 F3 — unified experiment pipeline & model experiments (FOC-175)

F3 shifts the question from features to model families: does a
gradient-boosting ensemble, a deep tabular net (TabNet), a pretrained
time-series forecaster used as a feature extractor (TimesFM), or a
per-customer sequence model (LSTM / Transformer) add fraud signal beyond the
F0–F2 arms? All nine arms — the four F0–F2 feature arms plus the four F3
families (five arms) — are measured by one runner under one protocol on three
train/test axes. Verdict up front: **no. On the cohort-random
customer-grouped axis — the PRIMARY axis, per the F2 reviewer recommendation
(§3.7) — all nine arms are statistically indistinguishable from a random
ranking; only the chronological axis, which lets customer identity straddle
train and test, separates arms from chance.** The null result is the finding,
and the phase's deliverable is the pipeline that makes re-running the
comparison cheap when more data arrives (§4.4).

### 4.1 Pipeline architecture (`src/fraud_pipeline.py`)

- **One registry, one protocol.** `fraud_pipeline.py` is a unified experiment
  runner: an arms registry (`ARMS`, populated via `register_arm`) of
  feature-set × model pairs sharing an sklearn-style `fit` / `predict_proba`
  interface, and three evaluation axes (`AXES`). Every arm is measured with
  the nb7/nb8 protocol: one split per axis shared by all arms (identical row
  indices, re-asserted per axis), a stratified 25% validation carve cut from
  TRAIN only, an F1-optimal threshold frozen on that carve, and a one-shot
  frozen-threshold test evaluation reporting PR-AUC, ROC-AUC, F1 and
  recall@precision — always printed next to the test positive count and the
  chance level (the test positive rate), because at 11–24 test positives the
  absolute numbers are unreadable without them. PR-AUC and ROC-AUC carry
  percentile bootstrap CIs (1000 resamples over test rows); the runner prints
  an explicit caveat whenever any test set holds ≤ 30 positives — the
  intervals are wide and indicative only.
- **Three axes.** `random-grouped` — customers assigned to train/test by a
  seeded random draw, customer-disjoint, no time ordering — is the PRIMARY
  axis (customer-disjoint but temporally unbiased); `grouped` (cohort-ordered,
  latest-seen customers held out) is the stress test that additionally removes
  recency overlap; `chronological` (the §1/§2 convention) is retained because
  it is the only axis where the identity signal is in play.
- **Results accumulate** as JSONL in `results/fraud_pipeline_results.jsonl`
  (one row per axis × arm; a re-run supersedes its earlier row, so the
  comparison table always shows the latest measurement per pair). Rows carry
  no wall-clock fields, so two identical invocations produce byte-identical
  files.
- **Determinism.** Seed 42 everywhere; the torch arms re-pin
  `torch.manual_seed` / `np.random.seed` and `cudnn.deterministic=True`
  (benchmark off) at every fit. The runner's metric rows are byte-identical
  across repeated CLI invocations, and the within-kernel checks were
  bit-identical across all three torch notebooks: nb10's three identical
  fits produced max probability delta 0.000e+00 with a full re-execution
  reproducing every printed number; nb12's CLI re-invocations matched to
  the last digit (only wall-clock differed); nb11's TimesFM feature
  extraction re-run in-process is bit-identical once the two extractions
  are compared with row-identity alignment (max |delta feature| =
  0.000e+00 in the committed nb11 output). The extractor returns rows in
  its own stable (customer, timestamp) order, so a positional comparison
  of the two frames misaligns them; an earlier version of the nb11 check
  did exactly that and misreported the row-order artifact as GPU
  nondeterminism — the committed notebook carries the corrected, aligned
  check. Comparison conclusions are unchanged either way: the nb11 arm
  lands at chance on the primary (cohort-random) axis, its CI covering
  chance (§4.2).
- **Notebook execution** goes through `src/run_notebook.py` — nbclient with
  the Windows Selector event-loop policy, because the nbconvert CLI kernel
  start is broken on this machine (§3.1).
- **Plugging in a new arm takes three steps** (all four F3 families followed
  exactly this path):
  1. write `src/arms_<name>.py` exposing an sklearn-style
     `fit` / `predict_proba` estimator plus `check_dependencies()`, which
     probe-imports the heavy libraries (and, for TimesFM, the local HF
     checkpoint) and returns a short message instead of raising when
     something is missing;
  2. add a registry entry in `fraud_pipeline.py` with a lazy sibling import —
     the module must keep loading without torch / pytorch_tabnet / timesfm
     installed on other machines — plus an optional `build_features(enriched)`
     and a `supports_cv` flag; a missing dependency surfaces as a SKIPPED
     result row (`ArmSkipped`), never a crash of the run, and
     `supports_cv=False` arms are exempt from cross-validation (as is
     everything under `--no-cv`);
  3. run `.venv/Scripts/python.exe src/fraud_pipeline.py --run-arm <name>
     --axis <axis>`; the row lands in the results file and every later
     comparison table.

### 4.2 Comparison tables

Rows are in registry order; numbers are the accumulated
`results/fraud_pipeline_results.jsonl` values (the CLI comparison print rounds
to four decimals). CI is the 95% percentile bootstrap interval on test PR-AUC.
Recall@precision targets precision 0.50.

Random-grouped axis (PRIMARY; 977 test rows, 13 positives, chance 0.0133):

| Arm | Test PR-AUC | 95% CI | ROC-AUC | F1@frozen | Recall@prec=0.50 | Test positives | Chance level |
|-----|-------------|--------|---------|-----------|------------------|----------------|--------------|
| xgb-baseline | 0.0171 | 0.009–0.035 | 0.5622 | 0.0000 | 0.00 | 13 | 0.0133 |
| xgb-client | 0.0144 | 0.008–0.030 | 0.4844 | 0.0000 | 0.00 | 13 | 0.0133 |
| dictionary | 0.0231 | 0.012–0.047 | 0.6642 | 0.0000 | 0.00 | 13 | 0.0133 |
| sce | 0.0134 | 0.008–0.024 | 0.5115 | 0.0000 | 0.00 | 13 | 0.0133 |
| gbdt-ensemble | 0.0145 | 0.008–0.034 | 0.4941 | 0.0000 | 0.00 | 13 | 0.0133 |
| tabnet | 0.0138 | 0.008–0.025 | 0.4999 | 0.0000 | 0.00 | 13 | 0.0133 |
| timesfm-features | 0.0131 | 0.008–0.027 | 0.4686 | 0.0000 | 0.00 | 13 | 0.0133 |
| sequential-lstm | 0.0167 | 0.009–0.036 | 0.5725 | 0.0000 | 0.00 | 13 | 0.0133 |
| sequential-transformer | 0.0158 | 0.008–0.040 | 0.4901 | 0.0260 | 0.00 | 13 | 0.0133 |

Grouped axis (stress; 831 test rows, 11 positives, chance 0.0132):

| Arm | Test PR-AUC | 95% CI | ROC-AUC | F1@frozen | Recall@prec=0.50 | Test positives | Chance level |
|-----|-------------|--------|---------|-----------|------------------|----------------|--------------|
| xgb-baseline | 0.0101 | 0.005–0.021 | 0.3152 | 0.0000 | 0.00 | 11 | 0.0132 |
| xgb-client | 0.0091 | 0.005–0.015 | 0.2543 | 0.0000 | 0.00 | 11 | 0.0132 |
| dictionary | 0.0123 | 0.007–0.028 | 0.4338 | 0.0000 | 0.00 | 11 | 0.0132 |
| sce | 0.0119 | 0.006–0.025 | 0.3670 | 0.0000 | 0.00 | 11 | 0.0132 |
| gbdt-ensemble | 0.0094 | 0.005–0.016 | 0.2879 | 0.0000 | 0.00 | 11 | 0.0132 |
| tabnet | 0.0150 | 0.008–0.027 | 0.5516 | 0.0000 | 0.00 | 11 | 0.0132 |
| timesfm-features | 0.0133 | 0.007–0.026 | 0.4792 | 0.0000 | 0.00 | 11 | 0.0132 |
| sequential-lstm | 0.0353 | 0.007–0.193 | 0.4741 | 0.0274 | 0.00 | 11 | 0.0132 |
| sequential-transformer | 0.0158 | 0.007–0.037 | 0.4930 | 0.0000 | 0.00 | 11 | 0.0132 |

Chronological axis (1061 test rows, 24 positives, chance 0.0226):

| Arm | Test PR-AUC | 95% CI | ROC-AUC | F1@frozen | Recall@prec=0.50 | Test positives | Chance level |
|-----|-------------|--------|---------|-----------|------------------|----------------|--------------|
| xgb-baseline | 0.0532 | 0.021–0.161 | 0.6361 | 0.0000 | 0.0417 | 24 | 0.0226 |
| xgb-client | 0.2374 | 0.085–0.426 | 0.7692 | 0.2667 | 0.2083 | 24 | 0.0226 |
| dictionary | 0.1034 | 0.040–0.211 | 0.7653 | 0.0769 | 0.0417 | 24 | 0.0226 |
| sce | 0.0340 | 0.018–0.075 | 0.5542 | 0.0000 | 0.0000 | 24 | 0.0226 |
| gbdt-ensemble | 0.2409 | 0.088–0.423 | 0.7689 | 0.2581 | 0.2083 | 24 | 0.0226 |
| tabnet | 0.0245 | 0.013–0.073 | 0.4325 | 0.0526 | 0.0000 | 24 | 0.0226 |
| timesfm-features | 0.2028 | 0.087–0.379 | 0.7582 | 0.0645 | 0.0417 | 24 | 0.0226 |
| sequential-lstm | 0.0208 | 0.013–0.041 | 0.4285 | 0.0000 | 0.0000 | 24 | 0.0226 |
| sequential-transformer | 0.0237 | 0.014–0.042 | 0.4741 | 0.0000 | 0.0000 | 24 | 0.0226 |

Honesty notes on these tables. First, F1 ≈ 0 on the two customer-disjoint
axes repeats the known frozen-threshold artifact (§1.5.1) — but there it is
not only a threshold story, since the ranking metrics themselves sit at
chance. Second, sequential-lstm's grouped 0.0353 is the only customer-disjoint
point estimate that visibly clears its chance level, and its interval
(0.007–0.193) is far too wide at 11 positives to read as signal. Third, the
chronological ordering repeats §2/§3 with the new families folded in:
xgb-client (0.2374) and gbdt-ensemble (0.2409) lead, timesfm-features
(0.2028) sits just below them, and the customer-disjoint collapse applies to
every arm including the new ones.

### 4.3 Per-family notes

- **GBDT ensemble (`gbdt-ensemble`, nb9, `arms_gbdt.py`).** Equal-weight soft
  vote of XGB + LightGBM + CatBoost on the shared base+client matrix, with
  per-member rare-class weights from the fitting carve, fixed seeds and
  single-threaded CPU fits (LightGBM `deterministic=True`, CatBoost
  `thread_count=1`) — cheap enough (3 × 200 trees) to register with full CV
  support. **Ensembling hurts on the primary axis**: the best lone member
  (lgbm) reached test PR-AUC 0.0213 under the identical protocol vs the soft
  vote's 0.0145 — a delta of −0.0068, well inside the bootstrap band; with
  correlated members trained on ~4k rows the vote mostly averages
  near-identical rankings, so the delta is noise-dominated. Implementation
  note: LightGBM bans JSON special characters in feature names and the
  `amount_eur_bucket` interval labels carry a comma, so the arm maps feature
  names through a sanitizer (values, row order and column order unchanged).
- **TabNet (`tabnet`, nb10, `arms_tabnet.py`).** TabNetClassifier
  (pytorch-tabnet 4.1.0) on the same 116-column matrix, library-default
  architecture, CUDA. Two flags travel with the arm: (1) **stalled
  maintenance** — 4.1.0 is the last release, so any bug or CUDA-compat gap
  found downstream is unlikely to be fixed upstream; (2) **weak native
  imbalance handling** — there is no `scale_pos_weight`-style loss weight;
  the only documented classifier knob is `fit(weights=1)`, an inverse-frequency
  `WeightedRandomSampler` (library-managed minority oversampling), used here
  and disclosed — nothing beyond it. Determinism is bit-identical in
  practice: three identical fits inside one kernel produced max probability
  delta 0.000e+00, and a full re-execution reproduced every printed number.
  Attention-mask aggregation over the fitting rows: base categorical ≈ 73%,
  client categorical ≈ 25%, client numeric ≈ 2% — with the explicit caveat
  that with null test signal these describe train-side fit, not validated
  signal (a statement about which memorizable identity features the model
  leaned on, consistent with the §2/§3 history).
- **TimesFM (`timesfm-features`, nb11, `arms_timesfm.py`).** TimesFM
  2.5-200m (torch-native backend, no JAX) is used **strictly as a feature
  extractor only — a forecaster, not a classifier**. For every transaction
  the pretrained forecaster predicts the customer's next amount and
  inter-transaction gap from that customer's strictly earlier transactions
  (past-only context; labels never enter a forecast), and six
  residual/quantile/context features are appended to the base+client matrix;
  the model on top is the SAME fixed XGBClassifier as xgb-client, so the
  arm's hypothesis is "do the forecast-residual features add signal", held by
  keeping the model identical. Marginal value ≈ zero: on the primary axis
  timesfm-features scores 0.0131 vs xgb-client's 0.0144 — no lift; on the
  chronological axis 0.2028 vs 0.2374 — also no lift. The checkpoint was
  cached locally (HF snapshot loaded `local_files_only=True`, no network at
  run time).
- **Sequential (`sequential-lstm`, `sequential-transformer`, nb12,
  `arms_sequential.py`).** A 1-layer LSTM (hidden 64) and a 2-layer
  Transformer encoder (d_model 64) read per-customer transaction sequences
  (9 embedded factor categoricals + log-scaled amount; no label-derived
  feature anywhere), scoring a transaction against strictly earlier
  same-customer steps — shifted recurrent state for the LSTM, causal
  attention keeping the diagonal for the Transformer — with `pos_weight` in
  `BCEWithLogitsLoss` as the only imbalance handling. Null on all axes.
  Sequence shapes: 100 sequences (one per customer), mean 53 steps, max 94;
  25 customers carry all 91 frauds. The direction, not the score, is the
  finding: per-customer sequence context — "unusual for THIS customer" — is
  the genuinely valuable NN direction on this problem, and it is worth
  re-testing when more data arrives.

### 4.4 Interpretation on the cohort-random axis

- **All nine arms are statistically indistinguishable from chance on the
  primary axis.** Every arm's 95% PR-AUC interval covers the 0.0133 chance
  level, and the ROC-AUC point estimates straddle 0.5 (0.4686–0.6642). The
  dictionary is the only nominal outlier (PR-AUC 0.0231, ROC 0.6642) and even
  its interval still covers chance. Nothing in §4.2's primary table
  distinguishes a trained model from a random ranking.
- **Combined with the chronological lift, this confirms and generalizes the
  F2 conclusion: the learnable signal at this dataset size is
  customer-identity memorization, not generalizable fraud pattern.** The
  chronological axis — the only one where customers straddle train and test —
  is the only one where arms separate (gbdt-ensemble 0.2409, xgb-client
  0.2374, timesfm-features 0.2028); on both customer-disjoint axes every arm,
  old and new, collapses to chance. The CV-vs-test gap reproduces the
  signature with a new model family: gbdt-ensemble's train-side CV PR-AUC is
  0.7132 ± 0.109 against a test 0.0145 on the primary axis — three gradient
  boosting libraries agree on within-customer rankings that transfer to no
  unseen customer.
- **Null results are findings, and the pipeline exists precisely so these
  nine arms can be re-compared cheaply when more data arrives** (the
  re-scope directive from 2026-08-31): one registry entry per new model, one
  CLI invocation per arm × axis, and the accumulated results file updates the
  comparison. Honest expectation-setting for that future comparison: 91
  frauds total, 11–24 test positives per axis — deltas under ~0.05 PR-AUC
  between arms are noise at this size (the noise budget the notebooks print),
  so only a substantially larger labeled set can separate families.

### 4.5 Open items

- **stat-context hash-pin remains unresolved.** A single `--hash=` line on
  the `stat-context` requirement activates pip's require-hashes mode, which
  demands a fully-hashed closure including every transitive dependency; the
  requirements file's deliberately unpinned F4 block ("resolve at F4 install
  time") makes that unresolvable without pip-compile-style tooling. The
  verified wheel sha256 is recorded here so the value is not lost:
  `2749b7b75e3b676bc5c5d03f12dfe950fafff2dea6a66a7ecc7a1f109a10075c`.
  Decision for the owner.
- **`fraud_pipeline.py`'s module docstring is stale (cosmetic).** It still
  describes the F3 arms as reserved placeholders; it was left untouched while
  the notebook arms landed to minimize churn. A one-line docstring update is
  a trivial follow-up.
- **F4 (faces / latent space) is queued AFTER F3** per the 2026-08-31
  decision — out of F3 scope.


## §5 F4 — latent space: multi-modal embeddings + fusion (FOC-178)

F4 asked one question four ways: **does a dense multi-modal representation add anything the
tabular arms cannot already express?** The modalities are face (per-customer synthetic face →
FaceNet, nb13), text (per-transaction synthesized description → MiniLM, nb14 — decision D4),
demographics (per-customer `dim_customer` profile text → the same MiniLM backbone, nb15), and
their fusion (nb15). All four ship as feature-sets in the unified runner
(`face-features`, `text-features-minilm-l6`, `demo-features`, `latent-fusion`), evaluated by
the F3 protocol on all three axes; **`random-grouped` remains the PRIMARY header axis**.

### 5.1 Per-modality notes

**Faces (nb13, decision D5).** No image data exists in this repo, so every face is synthetic:
one StyleGAN2-ADA (FFHQ) image per customer, generated deterministically —
`seed = sha256("face-arm-v1:" + customer_id)[:4]` (32-bit, `np.random.RandomState` range) →
z-vector → 1024² synthesis at `truncation_psi=0.7` (cache: `data/faces/<customer_id>.jpg`,
5.6 MB for all 100; `thispersondoesnotexist.com` was rejected as non-reproducible — no seeded
draw, no hash-verifiable cache). FaceNet (InceptionResnetV1, VGGFace2) embeds each 160 px crop
to a 512-d L2-normalized unit vector broadcast to the customer's rows
(`data/face_embeddings.npz`).
- *Determinism (aligned by identity, never positionally — the F3 r2 rule):* cache pass 2
  regenerated 0 files; a full cache-bypass re-embed reindexed on `customer_id` differed by
  **0.0** (bit-identical); the regenerated JPEG is byte-identical to the committed one.
- *Pre-registered result:* appearance carries **no fraud signal** — on the PRIMARY axis the
  arm sits at/below chance with a CI covering it. On the chronological axis it separates —
  the same identity-memorization artifact F3 measured at 0.24 lift: a per-customer constant
  vector is a customer id in disguise wherever customers straddle train/test. Both
  customer-grouped axes sever that channel by construction, and there the block is noise.
- *Ethical caveat (blocking any real-world use):* appearance-based fraud scoring is
  discriminatory by construction — face embeddings correlate with protected attributes
  (age, gender, ethnicity proxies), so any downstream threshold encodes appearance bias into
  who gets flagged. The block exists here purely as a methods-comparability experiment with a
  pre-registered null expectation; it must not graduate into a deployed scorer. The honest
  use of per-customer embeddings is entity resolution (same face = same customer), not risk.

**Text (nb14, decision D4 — experimental).** The table has **no free-text column** (longest
raw value: 15 chars), so the text is synthesized deterministically from row content —
seed = `sha256(customer|timestamp)`, three template shapes over per-type subjects, amount,
both account ids + countries, fixed amount-band descriptors; a pure function of fields the
tabular arms already see, verified shuffle-pure (5299 distinct texts over 5302 rows).
Candidates were the two locally cached MiniLM checkpoints (no runtime downloads on this
link): L6 scored 0.0122 vs L12 0.0323 on PRIMARY — but L12's CI [0.0084, 0.1604] sits inside
the ~0.05 noise budget at 13 test positives, so the cheaper L6 won. **Synthetic-data caveat:**
these embeddings re-encode known tabular signal — the experiment validates method plumbing,
not real-world text value; a genuine text arm needs a real field (dispute notes, merchant
descriptors) the table does not carry.

**Demographics (nb15).** Each customer's 8 `dim_customer` fields render into one fixed
profile sentence ("gender M, age 39, unemployed in the technology industry, …") embedded by
the same MiniLM-L6 (one backbone across text-shaped modalities, so modality deltas come from
content, not encoder choice). Fields are asserted constant per customer; the corpus digest in
`data/demo_embeddings.npz` invalidates the cache if `dim_customer` changes. This is a dense
re-encoding of columns xgb-client already one-hots — pre-registered expectation: no lift.

### 5.2 Fusion choice: stateless concat, deliberately not alignment

`latent-fusion` concatenates the three L2-normalized blocks (face 512 + text 384 + demo 384 =
1280 dims) on top of the xgb-client matrix. **No fitted projection and no alignment**, for
two reasons stated before any results (nb15): (1) the runner contract evaluates
`build_features` on the full frame *pre-split* — anything fit there (PCA, CCA, Procrustes, a
learned fusion head) leaks test structure; refitting per split inside `make_model` would fix
the leak but make features split-dependent, breaking the cached label-free extraction
contract every arm follows; (2) 100 customers cannot support a fitted shared space. Concat of
unit-norm blocks is the leak-free "one latent space": each modality contributes unit length,
so none dominates by scale, cross-modal geometry stays readable, and the downstream XGB is
per-feature monotone-invariant anyway — the normalization matters for the distance/angle
threshold consumers F5 is planned to need. A `latent-pure` ablation (the 1280 dims WITHOUT
the tabular base, in nb15 only) attributes how much of the fused result is embeddings alone.
Determinism: two cache-bypass fusion builds aligned on row identity — max |delta| = 0.0.

### 5.3 Comparison tables (embedding arms × 3 axes, runner protocol)

random-grouped axis (PRIMARY; 977 test rows, 13 positives, chance 0.0133):

| Arm | Test PR-AUC | 95% CI | ROC-AUC | F1@frozen | Recall@prec=0.50 | CV PR-AUC (mean±std) | Test positives | Chance level |
|---|---|---|---|---|---|---|---|---|
| xgb-client | 0.0144 | 0.0079–0.0300 | 0.4844 | 0.0000 | 0.0000 | 0.6812±0.0768 | 13 | 0.0133 |
| face-features | 0.0110 | 0.0070–0.0198 | 0.3989 | 0.0000 | 0.0000 | 0.7770±0.1381 | 13 | 0.0133 |
| text-features-minilm-l6 | 0.0122 | 0.0069–0.0253 | 0.4220 | 0.0000 | 0.0000 | 0.5331±0.1358 | 13 | 0.0133 |
| demo-features | 0.0114 | 0.0064–0.0239 | 0.3714 | 0.0000 | 0.0000 | 0.7353±0.1124 | 13 | 0.0133 |
| latent-fusion | 0.0174 | 0.0084–0.0503 | 0.5348 | 0.0000 | 0.0000 | 0.6033±0.0924 | 13 | 0.0133 |

grouped axis (stress; 831 test rows, 11 positives, chance 0.0132):

| Arm | Test PR-AUC | 95% CI | ROC-AUC | F1@frozen | Recall@prec=0.50 | CV PR-AUC (mean±std) | Test positives | Chance level |
|---|---|---|---|---|---|---|---|---|
| xgb-client | 0.0091 | 0.0050–0.0154 | 0.2543 | 0.0000 | 0.0000 | 0.7277±0.0599 | 11 | 0.0132 |
| face-features | 0.0165 | 0.0089–0.0350 | 0.5737 | 0.0000 | 0.0000 | 0.7742±0.0813 | 11 | 0.0132 |
| text-features-minilm-l6 | 0.0176 | 0.0080–0.0566 | 0.5551 | 0.0000 | 0.0000 | 0.5025±0.1170 | 11 | 0.0132 |
| demo-features | 0.0118 | 0.0059–0.0268 | 0.3518 | 0.0000 | 0.0000 | 0.7608±0.0884 | 11 | 0.0132 |
| latent-fusion | 0.0194 | 0.0097–0.0480 | 0.6157 | 0.0000 | 0.0000 | 0.5943±0.0859 | 11 | 0.0132 |

chronological axis (identity-straddling; 1061 test rows, 24 positives, chance 0.0226):

| Arm | Test PR-AUC | 95% CI | ROC-AUC | F1@frozen | Recall@prec=0.50 | CV PR-AUC (mean±std) | Test positives | Chance level |
|---|---|---|---|---|---|---|---|---|
| xgb-client | 0.2374 | 0.0850–0.4262 | 0.7692 | 0.2667 | 0.2083 | 0.5227±0.0809 | 24 | 0.0226 |
| face-features | 0.2617 | 0.0991–0.4574 | 0.6928 | 0.0800 | 0.2500 | 0.5880±0.0986 | 24 | 0.0226 |
| text-features-minilm-l6 | 0.0691 | 0.0174–0.1730 | 0.5736 | 0.0800 | 0.0417 | 0.4166±0.0398 | 24 | 0.0226 |
| demo-features | 0.2341 | 0.0898–0.4184 | 0.6851 | 0.0769 | 0.1667 | 0.5585±0.0975 | 24 | 0.0226 |
| latent-fusion | 0.1260 | 0.0383–0.2739 | 0.7191 | 0.1333 | 0.0417 | 0.4738±0.0671 | 24 | 0.0226 |

### 5.4 Interpretation (honest read)

- **PRIMARY axis (cohort-random, customer-grouped): every F4 arm is indistinguishable from
  chance** — PR-AUC at/below 0.02 with 95% CIs covering 0.0133, same as all nine F3 arms. The
  pre-registered null holds for every modality and for the fusion.
- **Grouped (stress) axis:** same collapse — nothing separates.
- **Chronological axis:** face-features (and to a lesser degree the fused space, which
  contains the face block) post the largest lifts — this is the F3 identity-memorization
  artifact again, now via per-customer embedding constants. It is NOT transferable signal:
  the customer-grouped axes are the honest ones, and there every embedding arm is noise.
- The fusion adds nothing over its constituents on any axis — expected, since each block is a
  per-customer or per-row re-encoding of fields the tabular matrix already carries. The
  nb15 ablation (`latent-pure`) confirms the tabular base does the work wherever anything
  separates.
- The F4 deliverable that matters is comparability: four cached, label-free, deterministic
  feature-sets in the runner registry, re-runnable per arm/axis in minutes when more labeled
  data arrives (`python src/fraud_pipeline.py --run-arm <arm> --axis <axis>`).

### 5.5 F4 open items

- F5 (threshold/anomaly layer over this latent space) is intentionally NOT built here —
  the fused space above is its input.
- `data/faces/`, `data/face_embeddings.npz`, `data/demo_embeddings.npz` are committed
  artifacts (5.7 MB total) — regenerating them needs the StyleGAN checkout at `C:/sg2-ada`
  + `C:/faces/ffhq.pkl` and the phase venv; consuming them needs numpy/pandas only.
- StyleGAN2's custom CUDA ops fail to build under torch 2.11 (upfirdn2d/bias_act fall back
  to the reference implementation): slower but deterministic, and irrelevant after the
  one-time generation — noted so a future re-generation does not read the warnings as errors.
- The §4.5 docstring-staleness item is resolved in F4 (module docstring now lists all 13
  wired arms); the stat-context hash-pin decision remains open for the owner.

## §6 F5 — threshold / statistical layer over the latent space (FOC-179)

### 6.1 Environment

Same pinned stack as F4 (no new dependency — the scorers use numpy/sklearn only, both already
pinned); executed on a fresh venv at `C:/venv-f5` (Python 3.11.9, torch 2.11.0+cu128, sklearn
1.7.2, pandas 2.3.3, pyarrow 21.0.0 — `pip check` clean, all `requirements.txt` pins satisfied).
Notebook `src/16. Latent Thresholds.ipynb` runs end-to-end via `src/run_notebook.py` (nbclient +
`WindowsSelectorEventLoopPolicy`) in ~3.5 min, exit 0, committed with outputs.

### 6.2 Method — the dictionary model's threshold logic, ported

nb4's dictionary model scores every transaction by aggregating per-variable fraud probabilities
and **calibrates its thresholds on the training rows only** (a 3-threshold F1 grid over the fit
rows; `DictionaryRateEnricher` mirrors it inside `fit()` in the runner). F5 applies the same
calibration story to the fused 128-d-per-block latent space (face 512 + text 384 + demo 384,
per-modality L2-normalized, stateless concat). Five anomaly scorers + one light classifier
(Mateusz's F5 decision of 2026-09-01, closing the issue's "decide later"), all registered in the
runner as ordinary arms over the **pure latent frame** (the 1280 embedding dims alone — the
geometry-based methods read latent geometry, not the one-hot re-encodings the F4 fusion arm
appends; nb15's `latent-pure` ablation uses the same frame):

| Arm | Score (higher = more anomalous) | Reference fitted on | Dictionary-model analogue |
|---|---|---|---|
| `latent-nn-dist` | distance to the 5th-nearest legitimate fitting row (own row excluded by row identity for calibration) | legit fit carve | per-row "exceeds cutoff" flags |
| `latent-centroid-dist` | Euclidean distance to the legitimate centroid | legit fit carve | distance thresholds |
| `latent-cosine-centroid` | angular score `1 − cos` to the legitimate mean direction | legit fit carve | angle thresholds (blocks are L2-normalized — the angle is the geometry) |
| `latent-cluster-anom` | distance to the nearest k-means center (k=8) | legit fit carve, TRAIN ONLY | anomaly within clusters |
| `latent-gmm-density` | negative log likelihood under PCA(64) + diagonal GMM | legit fit carve, TRAIN ONLY | distributional thresholds |
| `latent-logistic` | supervised logistic-regression probability (C=1, lbfgs) | full fit carve | the "train models too" option, decided 2026-09-01 |

Leakage discipline (the FOC-179 hard requirement): every fitted object — neighbor index, centroid,
k-means, PCA, GMM, logistic weights — is fitted inside the estimator's `fit()` on the rows the
runner hands it (the stratified fit carve of the training side). No clustering or density model
ever sees test rows, and nothing is fitted in `build_features` (which runs pre-split). Each scorer
additionally calibrates its own train-percentile operating point (`q99` of the legitimate fit-row
scores) inside `fit()` — the latent analogue of the dictionary's train-only threshold grid.

**Runner integration (no protocol extension was needed):** the scorers wrap into ordinary
sklearn-style estimators (`src/arms_latent.py`, following the `DictionaryRateEnricher` precedent)
that expose the raw anomaly score as `predict_proba(X)[:, 1]`. PR-AUC/ROC-AUC are rank-based, so
the raw score is directly comparable, and the runner keeps freezing its own best-F1 threshold on
the validation carve — every F5 arm is measured through the *identical* protocol as the 13
F0–F4 arms, on all three axes, with percentile-bootstrap CIs. The one schema-visible difference
is `n_features=1280` (the pure latent frame). Nothing was hand-rolled outside `funs.py`/
`fraud_pipeline.py`.

### 6.3 Comparison tables (F5 arms + anchors × 3 axes, runner protocol)

Anchors: `dictionary` (the conceptual ancestor whose threshold logic F5 ports), `xgb-baseline`,
`latent-fusion` (XGB over the same latent space, from F4). Rows come from the accumulated runner
results (`results/fraud_pipeline_results.jsonl`, 57 rows after F5's 18).

random-grouped axis (PRIMARY; 977 test rows, 13 positives, chance 0.0133):

| Arm | Test PR-AUC | 95% CI | ROC-AUC | F1@frozen | Recall@prec=0.50 | CV PR-AUC (mean±std) | Test positives | Chance level |
|---|---|---|---|---|---|---|---|---|
| dictionary | 0.0231 | 0.0122–0.0469 | 0.6642 | 0.0000 | 0.0000 | 0.5071±0.1027 | 13 | 0.0133 |
| xgb-baseline | 0.0171 | 0.0093–0.0350 | 0.5622 | 0.0000 | 0.0000 | 0.5316±0.1130 | 13 | 0.0133 |
| latent-fusion | 0.0174 | 0.0084–0.0503 | 0.5348 | 0.0000 | 0.0000 | 0.6033±0.0924 | 13 | 0.0133 |
| latent-nn-dist | 0.0142 | 0.0076–0.0312 | 0.4891 | 0.0263 | 0.0000 | 0.0310±0.0089 | 13 | 0.0133 |
| latent-centroid-dist | 0.0194 | 0.0110–0.0366 | 0.6524 | 0.0312 | 0.0000 | 0.0171±0.0047 | 13 | 0.0133 |
| latent-cosine-centroid | 0.0194 | 0.0110–0.0366 | 0.6524 | 0.0312 | 0.0000 | 0.0171±0.0047 | 13 | 0.0133 |
| latent-cluster-anom | 0.0153 | 0.0086–0.0288 | 0.5531 | 0.0263 | 0.0000 | 0.0170±0.0037 | 13 | 0.0133 |
| latent-gmm-density | 0.0229 | 0.0134–0.0439 | 0.7033 | 0.0263 | 0.0000 | 0.0186±0.0034 | 13 | 0.0133 |
| latent-logistic | 0.0224 | 0.0094–0.0742 | 0.5653 | 0.0000 | 0.0000 | 0.3207±0.1118 | 13 | 0.0133 |

grouped axis (stress; 831 test rows, 11 positives, chance 0.0132):

| Arm | Test PR-AUC | 95% CI | ROC-AUC | F1@frozen | Recall@prec=0.50 | CV PR-AUC (mean±std) | Test positives | Chance level |
|---|---|---|---|---|---|---|---|---|
| dictionary | 0.0123 | 0.0065–0.0280 | 0.4338 | 0.0000 | 0.0000 | 0.4786±0.1349 | 11 | 0.0132 |
| xgb-baseline | 0.0101 | 0.0054–0.0214 | 0.3152 | 0.0000 | 0.0000 | 0.6202±0.0959 | 11 | 0.0132 |
| latent-fusion | 0.0194 | 0.0097–0.0480 | 0.6157 | 0.0000 | 0.0000 | 0.5943±0.0859 | 11 | 0.0132 |
| latent-nn-dist | 0.0227 | 0.0099–0.0477 | 0.6252 | 0.0261 | 0.0000 | 0.0839±0.0532 | 11 | 0.0132 |
| latent-centroid-dist | 0.0124 | 0.0064–0.0218 | 0.4563 | 0.0000 | 0.0000 | 0.0290±0.0283 | 11 | 0.0132 |
| latent-cosine-centroid | 0.0124 | 0.0064–0.0218 | 0.4563 | 0.0000 | 0.0000 | 0.0290±0.0283 | 11 | 0.0132 |
| latent-cluster-anom | 0.0171 | 0.0084–0.0342 | 0.5865 | 0.0000 | 0.0000 | 0.0247±0.0148 | 11 | 0.0132 |
| latent-gmm-density | 0.0168 | 0.0081–0.0383 | 0.5469 | 0.0000 | 0.0000 | 0.0638±0.0532 | 11 | 0.0132 |
| latent-logistic | 0.0131 | 0.0061–0.0345 | 0.3940 | 0.0000 | 0.0000 | 0.3089±0.0873 | 11 | 0.0132 |

chronological axis (identity-straddling; 1061 test rows, 24 positives, chance 0.0226):

| Arm | Test PR-AUC | 95% CI | ROC-AUC | F1@frozen | Recall@prec=0.50 | CV PR-AUC (mean±std) | Test positives | Chance level |
|---|---|---|---|---|---|---|---|---|
| dictionary | 0.1034 | 0.0404–0.2113 | 0.7653 | 0.0769 | 0.0417 | 0.3539±0.0814 | 24 | 0.0226 |
| xgb-baseline | 0.0532 | 0.0214–0.1610 | 0.6361 | 0.0000 | 0.0417 | 0.4633±0.0660 | 24 | 0.0226 |
| latent-fusion | 0.1260 | 0.0383–0.2739 | 0.7191 | 0.1333 | 0.0417 | 0.4738±0.0671 | 24 | 0.0226 |
| latent-nn-dist | 0.2369 | 0.0859–0.4183 | 0.7356 | 0.0492 | 0.2083 | 0.0235±0.0096 | 24 | 0.0226 |
| latent-centroid-dist | 0.0281 | 0.0171–0.0478 | 0.5873 | 0.0000 | 0.0000 | 0.0160±0.0070 | 24 | 0.0226 |
| latent-cosine-centroid | 0.0281 | 0.0171–0.0478 | 0.5873 | 0.0000 | 0.0000 | 0.0160±0.0070 | 24 | 0.0226 |
| latent-cluster-anom | 0.0229 | 0.0141–0.0396 | 0.4939 | 0.0503 | 0.0000 | 0.0170±0.0050 | 24 | 0.0226 |
| latent-gmm-density | 0.0426 | 0.0220–0.0904 | 0.6201 | 0.0529 | 0.0000 | 0.0221±0.0157 | 24 | 0.0226 |
| latent-logistic | 0.0651 | 0.0150–0.1688 | 0.4662 | 0.0800 | 0.0417 | 0.3713±0.0628 | 24 | 0.0226 |

### 6.4 Calibration study — the scorers' own train-percentile operating points

The runner's frozen threshold makes the arms comparable; the scorers' own story — the dictionary
model's — is a threshold calibrated on train and applied frozen to test. Refitting each anomaly
scorer on the PRIMARY fit carve (3243 rows, 3185 legit) exactly as the runner does, with the
legitimate-score q95/q99 as the operating point (nb16, 13 test positives, chance 0.0133 —
descriptive, not inferential):

| Arm | Percentile | Threshold | Flagged (of 977) | Precision | Recall |
|---|---|---|---|---|---|
| latent-nn-dist | q95 | 0.7896 | 977 | 0.0133 | 1.0000 |
| latent-centroid-dist | q95 | 1.2928 | 112 | 0.0000 | 0.0000 |
| latent-cosine-centroid | q95 | 0.3320 | 112 | 0.0000 | 0.0000 |
| latent-cluster-anom | q95 | 1.2107 | 401 | 0.0175 | 0.5385 |
| latent-gmm-density | q95 | −36.6767 | 427 | 0.0281 | 0.9231 |
| latent-nn-dist | q99 | 0.8716 | 977 | 0.0133 | 1.0000 |
| latent-centroid-dist | q99 | 1.3436 | 18 | 0.0000 | 0.0000 |
| latent-cosine-centroid | q99 | 0.3628 | 18 | 0.0000 | 0.0000 |
| latent-cluster-anom | q99 | 1.2620 | 228 | 0.0132 | 0.2308 |
| latent-gmm-density | q99 | −29.9726 | 196 | 0.0204 | 0.3077 |

The `latent-nn-dist` rows are degenerate and instructive: **every** test row lands above the q99
calibration point. On the customer-grouped axes the fit rows have their own customer's other
transactions as near-duplicate neighbors, while held-out test customers have no representative in
the reference set at all — so the k-NN distance measures *customer novelty*, not fraud. That is
the customer-disjoint discipline doing its job, and it is why the arm's PR-AUC sits at chance
despite flagging everything: the score carries no fraud information that transfers across
customers. The GMM's q95 point (precision 0.0281, recall 0.9231) is the only operating point that
beats chance precision at meaningful recall — read with §6.6's caveat below.

One calibration subtlety was fixed after review (TEST finding on FOC-179): the self-match
exclusion originally used a distance cutoff (`dists[:, 0] < 1e-12`), but sklearn's self distance
is ~3e-8 of dot-product-expansion noise — not exact 0 — so 2350 of the 3185 legitimate fit rows
kept their own match inside the percentile calibration, deflating the operating points (q99 0.8481
instead of 0.8716). Self is now excluded by neighbour-index identity (`_calibration_scores` in
`arms_latent.py`); the table above shows the corrected values. The fix is calibration-purity only:
out-of-sample scores are bit-identical (max |delta| = 0.0 over the 977 test rows), so PR-AUC,
ROC-AUC, F1@frozen and the runner's own frozen threshold are unchanged — the canonical JSONL rows
for `latent-nn-dist` re-ran byte-identical.

### 6.5 Determinism

- **Identity-aligned feature check (F3 r3 discipline):** two `_features_latent_pure` builds of the
  full frame asserted index-equal BEFORE diffing — max |delta| = 0.0 over 5302 × 1280 (nb16).
  No positional comparison anywhere.
- **Re-run equality:** `latent-cluster-anom` on PRIMARY run twice through the runner — 19 result
  fields byte-identical. All randomized components carry fixed seeds (repo `RANDOM_STATE = 42`:
  k-means `n_init=10`, PCA randomized SVD, GMM init); the JSONL carries no wall-clock fields.
- Notebook vs CLI agreement: the nb16-measured F5 rows match the canonical CLI rows (same
  protocol, same seeds) — e.g. `latent-gmm-density` PRIMARY 0.0229 [0.0134–0.0439] in both.
- **Review-fix verification (nn-dist calibration purity):** the old distance-cutoff heuristic was
  reproduced exactly (self-match missed for 2350/3185 legit fit rows; old q95/q99 = 0.7763/0.8481),
  the corrected identity-based path lands at 0.7896/0.8716, a re-fit reproduces the corrected
  thresholds exactly, and the regenerated canonical JSONL rows are byte-identical to the pre-fix
  ones (out-of-sample scores unchanged).

### 6.6 Interpretation (honest read)

- **The pre-registered null holds on PRIMARY — with one razor-thin exception that should be read
  as noise.** Seven of the nine F5-family rows cover chance. The exception,
  `latent-gmm-density` (0.0229, CI 0.0134–0.0439 vs chance 0.013306), separates by 7×10⁻⁵ on the
  lower CI bound — a boundary at the fourth decimal of a 1000-resample percentile bootstrap with
  13 positives, in a table of 54 axis×arm comparisons. One arm scraping past at that margin is
  exactly what noise does under multiplicity; the honest verdict is "indistinguishable from
  chance, formally marginal." Its ROC-AUC (0.7033, CI 0.6115–0.7890) shows the rare-class pattern:
  ranking mostly-correct among ~964 negatives inflates ROC while average precision stays at the
  chance floor — PR-AUC is the decision metric here (§1.2/§6 DoD).
- **A threshold layer cannot exceed its input representation.** This was the pre-registered F5
  expectation and it is confirmed: the latent space itself carries ~no customer-transferable fraud
  signal at 5.3k transactions (F4), so distances, angles, clusters and densities over that space
  cannot manufacture one. The dictionary model remains the best PRIMARY arm overall (0.0231) —
  and its tabular per-variable lookup logic still beats its own latent-space port.
- **The chronological "lifts" are the known identity/novelty artifacts, in two flavours.**
  `latent-nn-dist` posts 0.2369 — the largest chronological number of the whole study — while its
  CV PR-AUC is 0.0235 (≈ chance): test rows are scored against a reference that contains their own
  customers' rows (the axis straddles identity), so the score partly measures known-customer
  proximity, the same mechanism F4 documented for `face-features` (0.2617). `latent-logistic`
  shows the mirror image: CV 0.3713 (in-sample customers memorizable from the embedding constants)
  vs test 0.0651 with a CI (0.0150–0.1688) that covers chance — the supervised read of the same
  non-transferable signal. On the honest, customer-disjoint axes nothing separates.
- **The calibration study adds the operational caution:** a train-percentile threshold on these
  scores flags 1–45% of test traffic at chance-level precision (`latent-nn-dist` flags 100%).
  Threshold methods calibrated on a customer-disjoint reference need more data — or a
  customer-representative reference — before their operating points mean anything here.
- **What F5 delivers** is the machinery: six deterministic, cached, cheaply re-runnable arms
  (`python src/fraud_pipeline.py --run-arm <arm> --axis <axis>`, ~5–40 s each) implementing the
  dictionary model's calibration story over the latent space inside the one runner protocol —
  ready for the day the dataset outgrows the null.

### 6.7 F5 open items

- The bootstrap-CI multiplicity point above (one marginal "separation" among 54 comparisons) is
  reported, not corrected for — no multiple-comparison machinery was added, consistent with the
  indicative role the CIs were given in §1.
- `cross_validate_model` is row-stratified, not customer-grouped: CV PR-AUC therefore overstates
  transfer on every arm whose features encode customer identity (all embedding arms; §4/§5 note
  the same). A grouped-CV variant is a natural follow-up, deliberately NOT built in F5.
- No new dependency was introduced; nothing to pin. `arms_latent.py` uses numpy + sklearn only.
- Scope boundary respected: SEC (FOC-181), merge to main, push — all out of F5.

## §7 F6 — stacked meta-model over method predictions + the latent space (FOC-211)

### 7.1 Environment

Same worktree venv (`C:/venv-f5`, Python 3.11.9, torch 2.11.0+cu128, sklearn 1.7.2, xgboost 2.1.4 —
all already pinned by F0-F4; **no new dependency**). Two machine-level determinism pins live in
`src/stack.py` and are re-stated by nb17's env cell BEFORE the first numpy import: CPU
BLAS/OpenMP threads forced to 1 (§6.5's 1-ulp KMeans story) and `CUBLAS_WORKSPACE_CONFIG=:4096:8`
forced before CUDA init (§7.5's GPU story). OOF caches live in `artifacts/stack/`
(`oof__<arm>__<axis>.npz` + one `manifest__<axis>.json` with sha256 digests, deterministic fold
assignments and per-fold fit sizes); attention diagnostics in `attn__<axis>__<variant>.json`.

### 7.2 Method — leak-free stacked generalization

F6's question: does a **second-level model over the arms' own predictions** (plus the raw inputs)
beat the arms? The protocol (`src/stack.py`, two CLI phases — one `--run-stack-base` per axis,
one `--run-stack` per axis) is stacked generalization under the F0-F4 runner discipline:

- **Out-of-fold base predictions.** For each of the 19 base arms, the TRAIN side is split into
  k=5 folds grouped by customer — a deterministic greedy assignment over sorted customer ids
  balancing fold fraud count, then row count, then fold index (`assign_folds`). Fold i's OOF
  probabilities come from a model fitted on the other k-1 folds only; a per-fold row-identity
  disjointness probe asserts the fit never saw the rows it predicts. A separate full-TRAIN fit
  produces the TEST predictions. Expensive arms were NOT dropped: all 19 arms run at k=5 on every
  axis (largest single-arm OOF cost ~2 min; no k=3 fallback was needed).
- **Meta matrix.** `[19 arm OOF probas | 19 missing-arm mask flags | 116 raw tabular |
  1280-d fused latent]` = 1434 columns (1428 for the sensitivity variant, which drops three arms
  entirely — no proba column, no mask). Column order keeps the FTT variant's scalar tokens
  contiguous. Missing-arm tolerance: an arm skipped at base time contributes a NaN proba + a mask
  flag (all 19 arms are `ok` on every axis here, so the masks are all-zero on this dataset — the
  tolerance machinery is exercised, its effect is not).
- **Single meta-val carve.** ONE stratified 25% carve of TRAIN (seed 42, `stratify`) serves both
  model selection (torch early stopping on val PR-AUC; logistic C sweep over {0.01, 0.1, 1}) and
  the frozen best-F1 threshold — the same single-carve contract the arms use. Meta rows therefore
  carry no `cv_*` fields (deliberate schema deviation from the base-arm rows, justified in §7.6:
  the OOF layer IS the generalization signal; a row-stratified CV over OOF features would reuse
  the folds it came from). Test evaluation reuses the runner's `best_f1_threshold`,
  `rich_test_metrics` and `bootstrap_auc_ci` verbatim.
- **Variants.** `meta-attn` (PRIMARY: per-arm embeddings + multi-head attention pooling over the
  19 arm tokens, mean attention per arm logged as diagnostics), `meta-ftt` (per-scalar-feature
  tokens + 2-layer FT-Transformer, d=64), `meta-logit` (standardized logistic, C swept on the
  carve), `meta-blend` (zero-fit nanmean of available arm probas — the reference every learned
  variant must beat), `meta-xgb` (shallow histogram tree, early stopping 50), plus
  `meta-attn-sens` on the chronological axis only: `meta-attn` re-fit without
  `latent-nn-dist`/`face-features`/`demo-features` — **labeled sensitivity, not a purity
  certificate**. Torch variants seed python/numpy/torch + cudnn.deterministic + deterministic
  algorithms before every fit and train with grad-norm clipping (1.0); the blend reads raw arm
  outputs (§7.6 notes the scale consequence).

Every join runs on the identity triple `(customer, timestamp, row_index)` (U-dtype strings, int64
ns, int64 row index) with asserted match and disjointness; the deterministic npz writer (fixed
ZipInfo timestamps, ZIP_STORED, `allow_pickle=False`) is what makes the caches byte-stable.

### 7.3 Comparison tables (meta variants + context × 3 axes, runner protocol)

Context rows (best single arm + anchors) come from the accumulated JSONL (73 rows after F6's 16).
Meta rows carry no CV column (single-carve protocol, §7.2); threshold and n_features are shown to
make the protocol explicit.

random-grouped axis (PRIMARY; 977 test rows, 13 positives, chance 0.0133):

| Arm | Test PR-AUC | 95% CI | ROC-AUC | F1@frozen | Recall@prec=0.50 | Frozen threshold | n_features | Chance level |
|---|---|---|---|---|---|---|---|---|
| dictionary (best base arm) | 0.0231 | 0.0122–0.0469 | 0.6642 | 0.0000 | 0.0000 | 0.8355 | 94 | 0.0133 |
| meta-attn | 0.0240 | 0.0117–0.0606 | 0.5789 | 0.0000 | 0.0000 | 0.9841 | 1434 | 0.0133 |
| meta-ftt | 0.0205 | 0.0111–0.0449 | 0.6444 | 0.0000 | 0.0000 | 0.9972 | 1434 | 0.0133 |
| meta-logit | 0.0217 | 0.0095–0.0582 | 0.5496 | 0.0000 | 0.0000 | 0.6307 | 1434 | 0.0133 |
| meta-blend | 0.0211 | 0.0112–0.0491 | 0.6507 | 0.0263 | 0.0000 | −2.5597 | 1434 | 0.0133 |
| meta-xgb | 0.0200 | 0.0098–0.0469 | 0.6025 | 0.0000 | 0.0000 | 0.9106 | 1434 | 0.0133 |

Every meta CI covers chance (lowest CI bound 0.0095 vs chance 0.013306). The pre-registered
PRIMARY null holds.

grouped axis (stress; 831 test rows, 11 positives, chance 0.0132):

| Arm | Test PR-AUC | 95% CI | ROC-AUC | F1@frozen | Recall@prec=0.50 | Frozen threshold | n_features | Chance level |
|---|---|---|---|---|---|---|---|---|
| sequential-lstm (best base arm) | 0.0353 | 0.0070–0.1934 | 0.4741 | 0.0274 | 0.0000 | 0.5935 | 12 | 0.0132 |
| meta-attn | 0.0117 | 0.0058–0.0287 | 0.3701 | 0.0000 | 0.0000 | 0.9794 | 1434 | 0.0132 |
| meta-ftt | 0.0102 | 0.0052–0.0226 | 0.2981 | 0.0000 | 0.0000 | 0.9852 | 1434 | 0.0132 |
| meta-logit | 0.0132 | 0.0065–0.0292 | 0.4548 | 0.0000 | 0.0000 | 0.2716 | 1434 | 0.0132 |
| meta-blend | 0.0154 | 0.0077–0.0334 | 0.5242 | 0.0000 | 0.0000 | −0.8196 | 1434 | 0.0132 |
| meta-xgb | 0.0201 | 0.0087–0.0495 | 0.5754 | 0.0000 | 0.0000 | 0.3990 | 1434 | 0.0132 |

chronological axis (identity-straddling; 1061 test rows, 24 positives, chance 0.0226):

| Arm | Test PR-AUC | 95% CI | ROC-AUC | F1@frozen | Recall@prec=0.50 | Frozen threshold | n_features | Chance level |
|---|---|---|---|---|---|---|---|---|
| face-features (best base arm) | 0.2617 | 0.0991–0.4574 | 0.6928 | 0.0800 | 0.2500 | 0.9274 | 628 | 0.0226 |
| meta-attn | 0.1076 | 0.0479–0.2348 | 0.7657 | 0.0667 | 0.0417 | 0.9862 | 1434 | 0.0226 |
| meta-ftt | 0.0521 | 0.0212–0.1560 | 0.6487 | 0.0741 | 0.0417 | 0.9088 | 1434 | 0.0226 |
| meta-logit | 0.0213 | 0.0131–0.0360 | 0.4737 | 0.0000 | 0.0000 | 0.4241 | 1434 | 0.0226 |
| meta-blend | 0.0952 | 0.0312–0.2050 | 0.6880 | 0.1356 | 0.0417 | −1.7838 | 1434 | 0.0226 |
| meta-xgb | 0.1015 | 0.0268–0.2273 | 0.6510 | 0.0667 | 0.0417 | 0.5360 | 1434 | 0.0226 |
| meta-attn-sens (drops latent-nn-dist, face-features, demo-features) | 0.1064 | 0.0381–0.2186 | 0.7530 | 0.0769 | 0.0417 | 0.9924 | 1428 | 0.0226 |

Four chronological rows separate from chance on the CI bound (meta-attn, meta-blend, meta-xgb,
meta-attn-sens) — §7.6 reads these as the known artifact axis, not as stacking value: the meta
variants sit far below that axis's best single arm (0.2617), and F4/F5 already documented why
chronological lift is identity/novelty, not transferable signal.

### 7.4 Attention diagnostics (meta-attn)

The PRIMARY variant's own account of which inputs it used — mean attention per arm over the
meta-val carve (nb17 cell 7; identical tables were verified bitwise against the CLI-written
`attn__<axis>__<variant>.json`). Top-5 and bottom-2 per fit:

| Fit (val PR-AUC, epochs) | Top arms | Bottom arms |
|---|---|---|
| random-grouped `meta-attn` (0.5327, 30) | latent-gmm-density 0.3255, xgb-baseline 0.0520, latent-nn-dist 0.0490, latent-centroid-dist 0.0478, latent-cluster-anom 0.0465 | sequential-lstm 0.0265, sce 0.0248 |
| grouped `meta-attn` (0.5994, 45) | latent-gmm-density 0.3272, xgb-baseline 0.0553, latent-centroid-dist 0.0477, latent-nn-dist 0.0471, latent-cluster-anom 0.0466 | gbdt-ensemble 0.0275, sce 0.0253 |
| chronological `meta-attn` (0.5883, 38) | latent-gmm-density 0.4125, xgb-baseline 0.0505, dictionary 0.0488, latent-centroid-dist 0.0422, latent-nn-dist 0.0383 | gbdt-ensemble 0.0224, sce 0.0196 |
| chronological `meta-attn-sens` (0.5783, 31) | latent-gmm-density 0.5129, latent-cluster-anom 0.0797, dictionary 0.0610, latent-centroid-dist 0.0398, timesfm-features 0.0380 | gbdt-ensemble 0.0195, xgb-baseline 0.0172 |

Read: the pool concentrates on the anomaly-scorer block — above all `latent-gmm-density` (~0.33–0.41
of all attention, despite that arm sitting at chance-level PR-AUC on the honest axes) — and the
attention mass is fairly flat elsewhere (~0.02–0.05 per arm). When the sensitivity fit removes the
three synthetic-modality arms, the mass re-concentrates further onto the same scorer block (gmm
0.41 → 0.51) with test metrics nearly unchanged (0.1076 → 0.1064) — the sensitivity story: the
pool's shape is robust to the drop, but nothing about it creates separation the inputs did not
have. With 13–24 test positives these weights are descriptive, not inferential.

### 7.5 Determinism

- **OOF caches (two-run bitwise agreement).** The 19-arm × 3-axis caches were built twice in
  independent processes: 45/57 arm-axis npz files were bit-identical (git reported no diff despite
  a full rewrite). The 12 that differed are exactly the threaded-BLAS-sensitive scorers
  (`latent-cluster-anom`, `latent-cosine-centroid`, `latent-gmm-density`, `latent-logistic` × 3
  axes) built before the CPU thread pin landed; rebuilt under the forced single-thread pin and
  re-verified — 12/12 PASS (bit-identical re-runs, `--run-stack-base --verify`).
- **Fold integrity (nb17 cell 4, every arm × every axis).** Per-fold `fit+fold` row and positive
  sums equal the TRAIN totals; the committed fold assignments reproduce the deterministic greedy
  assignment exactly; every cached npz is identity-aligned to the enriched frame (match asserted
  on all three identity fields, TRAIN and TEST); per-fold disjointness probes pass on the cached
  arrays themselves. Fit sizes 3057–3882 (random-grouped), 3234–3976 (grouped), 2526–3776
  (chronological); smallest fold positives 14/14/10.
- **GPU meta training (three stacked root causes found by probe and fixed).** (1) `_FTTNet`
  parameters came from `torch.empty` with no init — CUDA garbage often carries NaN/1e38 bit
  patterns; the probe showed FORWARD-NaN at epoch 0 batch 0 with finite inputs. Fixed with seeded
  `normal_(std=0.02)` init. (2) Even fit-able runs drifted bitwise: flash + memory-efficient SDPA
  backward and cublas split-k accumulate with atomics, which `cudnn.deterministic` does not cover —
  the probe showed whole different trajectories across processes (12 vs 26 epochs, val PR-AUC
  0.018 vs 0.389). Fixed: math-SDPA-only + `torch.use_deterministic_algorithms(True)` in
  `_seed_everything`, `CUBLAS_WORKSPACE_CONFIG=:4096:8` forced before CUDA init. (3) Grad-norm
  clipping (1.0) kept as a spike guard, uniform across torch variants. Probe verdict: attn + ftt
  sha256 digests of (weights, val probas) bitwise identical in-process AND across processes.
- **nb17 ↔ CLI contract.** Identity-aligned meta matrix rebuild max |delta| = 0.0 over 5302 ×
  1434; `meta-attn`/`meta-logit`/`meta-blend` re-runs 19 metric fields byte-identical; **all 16
  meta rows byte-identical to the canonical JSONL (CLI == notebook)**; the four attention
  diagnostics tables match the CLI-written JSONs bitwise. No wall-clock fields anywhere.

### 7.6 Interpretation (honest read)

- **The pre-registered PRIMARY null holds — this was the expected outcome, not a failure.** All
  five meta variants cover chance (lowest CI bound 0.0095 vs chance 0.013306). Stacking adds
  capacity, not information: the OOF arm probabilities it reads are themselves at-chance on the
  customer-disjoint axes (F0-F4), the latent block is the representation F5 showed carries ~no
  transferable signal, and the tabular block is the arms' own feature space. A second-level model
  cannot manufacture separation its inputs do not contain. `meta-attn` posts the nominally best
  PRIMARY number of the whole table (0.0240 vs dictionary 0.0231) — a 9×10⁻⁴ margin inside a
  table of 48 comparisons, both CIs overlapping the other to the last digit that matters; the
  honest verdict is "indistinguishable from its best single input".
- **Stacking never beats its best input on any axis.** random-grouped: dictionary 0.0231 ≈
  meta-attn 0.0240. grouped: sequential-lstm 0.0353 > every meta variant (best meta 0.0201).
  chronological: face-features 0.2617 > every meta variant (best meta 0.1076). With 11–24 test
  positives and percentile bootstrap CIs this is not a photo-finish question — the gaps are the
  size of the CIs themselves. The zero-fit reference (`meta-blend`) landing within ~2× of the
  learned variants on every axis is the quiet confirmation: there is no complementarity between
  arms for a second level to harvest.
- **The blend's scale wart is on the record.** `meta-blend` nanmeans RAW arm outputs whose scales
  differ wildly (gmm's negative log-likelihood −60..−5, distance scores ~0.9–1.5, calibrated
  probas 0..1), so it is dominated by the gmm scale (negative frozen thresholds −0.82/−2.56
  observed). The learned variants absorb scale (logit standardizes, trees split by rank,
  attention/FTT embed); the blend, by design, cannot. Its near-null is partly a scale artifact of
  the zero-fit reference, not evidence about arm informativeness — and the anomaly arms' "proba"
  columns are scores, not calibrated probabilities, which the protocol tolerates (thresholds are
  rank-based) but the blend does not.
- **The chronological "separations" are the known artifact axis, not stacking value.** Four meta
  rows separate on the CI bound there — while sitting at less than half that axis's best single
  arm (0.2617, the identity/novelty number F4 already flagged). The meta model inherits the
  artifact through its inputs (the OOF columns include the same face/tabular information the
  base arms read; the chronological split lets test rows see their own customers' rows in every
  fit carve). read as: the axis lifts, the second level does not fix, amplify or purify anything.
- **Multiplicity, on the record.** 5 variants × 3 axes + 1 sensitivity fit = 16 reported fits
  (48 axis-variant comparisons with CIs) over 13–24 test positives; the nominally best rows are a
  selection effect, not a discovery. `meta-attn-sens` is **labeled sensitivity, not a purity
  certificate**: dropping three synthetic-modality arms changes the attention re-concentration
  and leaves test metrics within noise (0.1076 → 0.1064, CIs overlapping) — a robustness probe of
  the pool's shape, not evidence that any subset of arms is pure or that purity was achieved.
- **What F6 delivers** is the machinery: a leak-free OOF stack inside the one runner protocol —
  new arms drop in as proba columns, skipped arms become mask flags, folds are deterministic and
  asserted (identity + per-fold disjointness), caches are byte-stable and re-verifiable, and the
  whole layer re-runs end-to-end in minutes. Ready for the day the dataset outgrows the null —
  the same hand F5's machinery is waiting for.

### 7.7 F6 open items

- Meta rows omit the `cv_*` fields by design (single-carve protocol, §7.2). If schema uniformity
  across base/meta rows matters downstream, a grouped-CV-over-OOF variant is the natural
  follow-up — deliberately NOT built here (it would reuse the folds the OOF features came from;
  the honest generalization signal for a meta model is the meta-val carve + test, not CV).
- The blend reads raw arm outputs; if a future round gives the blend a rank-transform or z-scores
  the anomaly scorers, the zero-fit reference becomes scale-coherent — a one-line matrix change,
  left out of F6 to keep the pre-registered definition intact.
- GPU determinism for the torch variants relies on math-SDPA-only + deterministic algorithms +
  the pinned cublas workspace; if a future arm or variant needs a kernel without a deterministic
  implementation, the contract (byte-identical re-runs) will fail loudly — that is the intended
  behaviour, but it bounds what can be added without revisiting the pin.
- No new dependency was introduced; nothing to pin. `stack.py` uses numpy/pandas/sklearn/xgboost/
  torch only — all already in the F0 pins. Scope boundary respected: no new base arms, no changes
  to existing arm definitions/splits/recorded rows, SEC (FOC-181), merge to main, push — all out
  of F6.
