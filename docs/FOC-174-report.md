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

<!-- F0 complete; F1 appends §2. -->
