# Notebooks

The repository contains a collection of Jupyter notebooks and Python scripts that were used to train and evaluate a machine learning model for fraud detection in financial transactions.

The main files in the repository are:

- `funs.py`: This Python script contains customized functions used throughout the project.
- `1. Currencies.ipynb`: This notebook collects the currency conversion rates from the static table on the website and saves the data to a CSV file that is used in the solution.
- `2. Analysis.ipynb`: This notebook contains an initial data exploration and analysis to gain an understanding of the data. It includes steps such as identifying the number of fraud cases, performing a chi-squared test to identify variables associated with fraud, and using visualization techniques to identify factors within variables associated with fraud. One-hot encoding and ANOVA F-test techniques are also applied to the data in this notebook.
- `3. Tree Based Models.ipynb`: This notebook builds and explains the decision tree and XGBoost models. The notebook includes steps such as performance evaluation, feature importances, and plotting the decision tree.
- `4. Dictionary Model.ipynb`: This notebook contains an algorithm based on dictionaries and performs a series of steps to find the best thresholds for expected fraud probability, standard deviation flags, and quantile flags. The notebook evaluates the model using the training and test data sets.
- `5. Combined Model.ipynb`: contains a two-layered fraud detection model that combines an algorithm based on dictionaries with an XGBoost model. 
- `6. Evaluation Harness.ipynb`: smoke test for the F0 helpers added to `funs.py` (FX-leakage-safe `dataPreparation`, `chronological_split`, `cross_validate_model`, the refactored `evaluateModel`) on an intentionally trivial feature set.
- `7. Customer Enrichment.ipynb`: builds the synthetic, deterministic (SHA-256-seeded, no-PII) `dim_customer` table and runs the base-vs-base+client A/B through the F0 harness (FOC-174 report §2).
- `8. SCE Enrichment.ipynb`: four-arm comparison (base / base+client / base+SCE / base+dictionary) under the shared protocol on the chronological and customer-grouped splits (FOC-174 report §3).
- `9. GBDT Ensemble.ipynb`: F3 — XGB + LightGBM + CatBoost soft vote (`arms_gbdt.py`) through the unified runner on all three axes, plus a per-member diagnostic on the primary axis (FOC-174 report §4).
- `10. TabNet Experiment.ipynb`: F3 — TabNet (pytorch-tabnet 4.1.0) arm (`arms_tabnet.py`) with attention-mask feature-group aggregation, benchmarked against gbdt-ensemble (FOC-174 report §4).
- `11. TimesFM Features.ipynb`: F3 — TimesFM 2.5-200m as a per-customer past-only forecaster (`arms_timesfm.py`); six forecast-residual features appended to base+client and evaluated for marginal value against xgb-client (FOC-174 report §4).
- `12. Sequential Models.ipynb`: F3 — per-customer LSTM and Transformer sequence arms (`arms_sequential.py`) with past-only scoring, benchmarked against xgb-client (FOC-174 report §4).
- `fraud_pipeline.py`: unified experiment runner (FOC-175) — a registry of arms (feature-set x model pairs: `xgb-baseline`, `xgb-client`, `dictionary`, `sce`, `gbdt-ensemble`, `tabnet`, `timesfm-features`, `sequential-lstm`, `sequential-transformer`) evaluated on three train/test axes (cohort-random customer-grouped, cohort-ordered grouped, chronological) under the nb7/nb8 protocol (validation-frozen threshold, PR-AUC/ROC-AUC with bootstrap CIs, test positives and chance level on every table). Results accumulate to `results/fraud_pipeline_results.jsonl`. CLI: `python src/fraud_pipeline.py --list-arms`, `python src/fraud_pipeline.py --run-arm sce --axis chronological`.
- `run_notebook.py`: executes a repo notebook in place via nbclient with the Windows selector event-loop policy (the nbconvert CLI is broken on this machine), saving it with outputs and exiting nonzero if any cell raised: `python src/run_notebook.py "src/8. SCE Enrichment.ipynb"`.
