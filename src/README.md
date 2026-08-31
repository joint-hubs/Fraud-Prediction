# Notebooks

The repository contains a collection of Jupyter notebooks and Python scripts that were used to train and evaluate a machine learning model for fraud detection in financial transactions.

The main files in the repository are:

- `funs.py`: This Python script contains customized functions used throughout the project.
- `1. Currencies.ipynb`: This notebook collects the currency conversion rates from the static table on the website and saves the data to a CSV file that is used in the solution.
- `2. Analysis.ipynb`: This notebook contains an initial data exploration and analysis to gain an understanding of the data. It includes steps such as identifying the number of fraud cases, performing a chi-squared test to identify variables associated with fraud, and using visualization techniques to identify factors within variables associated with fraud. One-hot encoding and ANOVA F-test techniques are also applied to the data in this notebook.
- `3. Tree Based Models.ipynb`: This notebook builds and explains the decision tree and XGBoost models. The notebook includes steps such as performance evaluation, feature importances, and plotting the decision tree.
- `4. Dictionary Model.ipynb`: This notebook contains an algorithm based on dictionaries and performs a series of steps to find the best thresholds for expected fraud probability, standard deviation flags, and quantile flags. The notebook evaluates the model using the training and test data sets.
- `5. Combined Model.ipynb`: contains a two-layered fraud detection model that combines an algorithm based on dictionaries with an XGBoost model. 
- `fraud_pipeline.py`: unified experiment runner (FOC-175) — a registry of arms (feature-set x model pairs: `xgb-baseline`, `xgb-client`, `dictionary`, `sce`; the F3 arms are reserved placeholders) evaluated on three train/test axes (cohort-random customer-grouped, cohort-ordered grouped, chronological) under the nb7/nb8 protocol (validation-frozen threshold, PR-AUC/ROC-AUC with bootstrap CIs, test positives and chance level on every table). Results accumulate to `results/fraud_pipeline_results.jsonl`. CLI: `python src/fraud_pipeline.py --list-arms`, `python src/fraud_pipeline.py --run-arm sce --axis chronological`.
- `run_notebook.py`: executes a repo notebook in place via nbclient with the Windows selector event-loop policy (the nbconvert CLI is broken on this machine), saving it with outputs and exiting nonzero if any cell raised: `python src/run_notebook.py "src/8. SCE Enrichment.ipynb"`.
