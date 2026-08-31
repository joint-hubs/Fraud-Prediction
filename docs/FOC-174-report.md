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
- **Program reframe (the actionable outcome of this phase):** before any F3
  model work, the evaluation axis that matters is the customer-grouped
  split; every F3/F4 candidate must be judged there, not on the
  chronological split.

<!-- F0-F2 complete; F3 appends §4. -->

