import pandas as pd
import numpy as np
import re
from scipy.stats import chi2_contingency
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split, StratifiedKFold


def chi2_independence(df, factor_col, fraud_col, type="description"):
    # Variable Statistical Test of Independence
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3900058/
    # The null hypothesis is that there is no association between the two variables,
    # and the alternative hypothesis is that there is a significant association between them.

    # Summarize the data by the given factor and fraud flag
    count_group = df.groupby([factor_col, fraud_col]).size().reset_index(name="n")

    # Create a contingency table
    table = pd.pivot_table(count_group, values="n", index=fraud_col, columns=factor_col)

    # Perform the chi-square test of independence
    chi2, pval, dof, expected = chi2_contingency(table)

    if type == "description":
        # Print the results
        print("Chi-square statistic: ", chi2)
        print("p-value: ", pval)
        print("Degrees of freedom: ", dof)
        print("Expected frequencies: ")
        print(pd.DataFrame(expected, index=table.index, columns=table.columns))
    elif type == "table":
        return chi2_contingency(table)[3]


def highlightTable(trxns_data, factor_col, fraud_col):
    # Outlining Factors - chi2_independence based
    # Prepare the data
    count_group = (
        trxns_data.groupby([factor_col, fraud_col]).size().reset_index(name="n")
    )
    table = pd.pivot_table(count_group, values="n", index=fraud_col, columns=factor_col)

    # Calculate expected counts
    expected_counts = chi2_independence(trxns_data, factor_col, fraud_col, type="table")

    # Calculate the standardized residuals for each cell and highlight the cells with residuals greater than 2 or less than -2
    for i in range(len(table.index)):
        for j in range(len(table.columns)):
            expected = expected_counts[i][j]
            observed = table.iloc[i, j]
            residual = (observed - expected) / np.sqrt(expected)
            if np.abs(residual) > 2:
                table.iloc[i, j] = "{:.0f}*".format(observed)
            else:
                table.iloc[i, j] = "{:.0f}".format(observed)

    return table


def plotDictionary(
    trxns_data, colname="weekday", quantile_threshold=0.9, count_filter=5
):
    # Variable Dictionary Analysis
    # Convert the variable
    if not pd.api.types.is_categorical_dtype(trxns_data[colname]):
        trxns_data[colname] = trxns_data[colname].astype("category")

    # Summarize the data by a given factor
    count_group = (
        trxns_data.assign(value=trxns_data[colname])
        .groupby(["value", "fraud_flag"])["value"]
        .count()
        .reset_index(name="n")
        .assign(prob=lambda x: x["n"] / x.groupby("value")["n"].transform("sum"))
        .reset_index()
    )
    count_group = count_group[count_group["n"] > count_filter]

    # Get the paired values only
    count_group_names = count_group.groupby("value")["n"].count().reset_index(name="n_")
    count_group_names = count_group_names[count_group_names["n_"] == 2].drop(
        columns=["n_"]
    )

    # Reshape the data to wide format
    count_group_wide = count_group.pivot(
        index="fraud_flag", columns="value", values="n"
    )
    count_group_wide_prob = count_group.pivot(
        index="fraud_flag", columns="value", values="prob"
    )

    # Bind the data frames
    long_table = pd.concat(
        [
            count_group_wide.assign(group_type="count"),
            count_group_wide_prob.assign(group_type="prob"),
        ],
        axis=0,
        ignore_index=False,
    ).reset_index()
    long_table = pd.melt(
        long_table,
        id_vars=["fraud_flag", "group_type"],
        var_name="name",
        value_name="value",
    )

    # Prepare the statistics
    table_analysis = (
        long_table[long_table["group_type"] == "prob"]
        .groupby("fraud_flag")
        .agg(
            mean=pd.NamedAgg(column="value", aggfunc=lambda x: round(np.mean(x), 4)),
            sd=pd.NamedAgg(column="value", aggfunc=np.std),
            q_t=pd.NamedAgg(
                column="value", aggfunc=lambda x: np.quantile(x, quantile_threshold)
            ),
        )
        .reset_index()
    )

    mean_v = table_analysis.loc[table_analysis["fraud_flag"] == "Y", "mean"].values[0]
    sd_v = table_analysis.loc[table_analysis["fraud_flag"] == "Y", "sd"].values[0]
    q_v = table_analysis.loc[table_analysis["fraud_flag"] == "Y", "q_t"].values[0]

    result_table = long_table[
        (long_table["fraud_flag"] == "Y")
        & (long_table["group_type"] == "prob")
        & (long_table["name"].isin(count_group_names["value"]))
    ].assign(sd_flag=mean_v + sd_v, q_flag=q_v + sd_v)
    plot_result_table = sns.catplot(
        x="value",
        y="name",
        data=result_table,
        kind="bar",
        height=5,
        aspect=1.5,
        palette="Spectral",
    )
    plot_result_table.set(
        xlabel="", ylabel="", title="Fraud probability by '" + colname + "' factor"
    )

    # Add text labels to the bars
    for index in range(len(result_table)):
        plot_result_table.ax.text(
            result_table.iloc[index]["value"],
            index,
            round(result_table.iloc[index]["value"], 4),
            color="black",
            ha="center",
        )

    plot_result_table.ax.axvline(x=mean_v, color="k", linestyle="--")
    plot_result_table.ax.axvline(x=mean_v + 1.5 * sd_v, color="r", linestyle="--")
    plot_result_table.ax.axvline(x=mean_v + 2 * sd_v, color="r", linestyle="--")
    plot_result_table.ax.axvline(x=q_v + sd_v, color="r")
    plt.show()


def dataPreparation(
    all_trxns_path="all_trxns.csv", exchange_rates_path="exchange_rates.csv"
):
    # Prepare the data

    # Data Collection
    # Raw Data
    all_trxns = pd.read_csv(all_trxns_path, dtype={"counterparty": str})

    # Exchange Rates -> more info in currencies.ipynb
    currency_rates = pd.read_csv(
        exchange_rates_path, header=None, names=["ccy", "date", "rate"]
    )
    # Normalize the rate date to a date object so it matches the trxns date type
    # (trxns date comes from .dt.date; the rates CSV date is a string). Without
    # this the merge on [ccy, date] joins on mismatched types and matches nothing.
    currency_rates["date"] = pd.to_datetime(currency_rates["date"]).dt.date

    # Data Cleaning and Preprocessing

    # Transform the variables and provide additional features
    trxns_data = all_trxns.copy()
    # Convert the timestamp to datetime
    trxns_data["timestamp"] = pd.to_datetime(
        trxns_data["timestamp"], infer_datetime_format=True
    )
    # Add the date and the exchange rate
    trxns_data["date"] = trxns_data["timestamp"].dt.date
    trxns_data = trxns_data.merge(currency_rates, on=["ccy", "date"], how="left")
    # FX leakage fix: do NOT silently treat missing rates as 1:1 EUR (that would
    # mis-price non-EUR rows at face value). Flag missing rates explicitly and
    # leave amount_eur as NaN for those rows so downstream code cannot mistake a
    # fabricated price for a real one.
    trxns_data["amount_eur_fx_missing"] = trxns_data["rate"].isna()
    # Clean and convert the amount to EUR
    trxns_data["amount"] = trxns_data["amount"].apply(
        lambda x: float(re.sub("[^0-9.]", "", x))
    )
    trxns_data["amount_eur"] = trxns_data["amount"] / trxns_data["rate"]
    # Extract the customer type from the customer id
    trxns_data["customer_type"] = trxns_data["customer"].str[0]
    # Extract the weekday, month, quarter and hour from the timestamp
    trxns_data["weekday"] = trxns_data["date"].apply(lambda x: x.strftime("%A"))
    trxns_data["month"] = trxns_data["date"].apply(lambda x: x.strftime("%B"))
    trxns_data["quarter"] = trxns_data["date"].apply(
        lambda x: "Q" + str((x.month - 1) // 3 + 1)
    )
    trxns_data["hour"] = trxns_data["timestamp"].dt.hour
    # Replace missing values in the "counterparty_country" column with "unknown"
    trxns_data["counterparty_country"] = np.where(
        trxns_data["counterparty_country"].isna(),
        "unknown",
        trxns_data["counterparty_country"],
    )
    # Clean the names of counterparty countries
    trxns_data["counterparty_country"] = np.where(
        trxns_data["counterparty_country"].isin(["United States", "USA"]),
        "US",
        trxns_data["counterparty_country"],
    )

    # Calculate the thresholds for the equally sized buckets of the amount in EUR.
    # Use nanquantile so rows with a missing FX rate (amount_eur_fx_missing) are
    # excluded from the bin-edge computation rather than poisoning it with NaN.
    amount_eur_quantile = np.nanquantile(
        trxns_data["amount_eur"], q=np.arange(0, 1.2, 0.2)
    )

    # Add amount_eur buckets
    trxns_data["amount_eur_bucket"] = pd.cut(
        trxns_data["amount_eur"], bins=amount_eur_quantile, include_lowest=True
    )

    return trxns_data


def createMetaDictionary(
    trxns_data, colname="weekday", quantile_threshold=0.9, count_filter=5
):
    # Prepare Dictionary Summary row
    # Convert the variable
    if not pd.api.types.is_categorical_dtype(trxns_data[colname]):
        trxns_data[colname] = trxns_data[colname].astype("category")

    # Summarize the data by a given factor
    count_group = (
        trxns_data.assign(value=trxns_data[colname])
        .groupby(["value", "fraud_flag"])["value"]
        .count()
        .reset_index(name="n")
        .assign(prob=lambda x: x["n"] / x.groupby("value")["n"].transform("sum"))
        .reset_index()
    )
    count_group = count_group[count_group["n"] > count_filter]

    # Get the paired values only
    count_group_names = count_group.groupby("value")["n"].count().reset_index(name="n_")
    count_group_names = count_group_names[count_group_names["n_"] == 2].drop(
        columns=["n_"]
    )

    # Reshape the data to wide format
    count_group_wide = count_group.pivot(
        index="fraud_flag", columns="value", values="n"
    )
    count_group_wide_prob = count_group.pivot(
        index="fraud_flag", columns="value", values="prob"
    )

    # Bind the data frames
    long_table = pd.concat(
        [
            count_group_wide.assign(group_type="count"),
            count_group_wide_prob.assign(group_type="prob"),
        ],
        axis=0,
        ignore_index=False,
    ).reset_index()
    long_table = pd.melt(
        long_table,
        id_vars=["fraud_flag", "group_type"],
        var_name="name",
        value_name="value",
    )

    # Prepare the statistics
    table_analysis = (
        long_table[long_table["group_type"] == "prob"]
        .groupby("fraud_flag")
        .agg(
            mean=pd.NamedAgg(column="value", aggfunc=lambda x: round(np.mean(x), 4)),
            sd=pd.NamedAgg(column="value", aggfunc=np.std),
            q_t=pd.NamedAgg(
                column="value", aggfunc=lambda x: np.nanquantile(x, quantile_threshold)
            ),
            q_1=pd.NamedAgg(column="value", aggfunc=lambda x: np.nanquantile(x, 0.1)),
            q_25=pd.NamedAgg(column="value", aggfunc=lambda x: np.nanquantile(x, 0.25)),
            q_75=pd.NamedAgg(column="value", aggfunc=lambda x: np.nanquantile(x, 0.75)),
            q_9=pd.NamedAgg(column="value", aggfunc=lambda x: np.nanquantile(x, 0.9)),
        )
        .reset_index()
    )

    result_table = (
        table_analysis.query('fraud_flag == "Y"')
        .assign(
            sd_flag=lambda x: x["mean"] + x["sd"],
            q_flag=lambda x: x["q_t"],
            q_1_flag=lambda x: x["q_1"],
            q_25_flag=lambda x: x["q_25"],
            q_75_flag=lambda x: x["q_75"],
            q_9_flag=lambda x: x["q_9"],
            variable_name=colname,
        )
        .loc[
            :,
            [
                "variable_name",
                "sd_flag",
                "q_flag",
                "q_1_flag",
                "q_25_flag",
                "q_75_flag",
                "q_9_flag",
            ],
        ]
    )

    return result_table


def createDictionary(trxns_data, colname="weekday", count_filter=5):
    # Convert the variable
    if not pd.api.types.is_categorical_dtype(trxns_data[colname]):
        trxns_data[colname] = trxns_data[colname].astype("category")

    # Summarize the data by a given factor
    count_group = (
        trxns_data.assign(value=trxns_data[colname])
        .groupby(["value", "fraud_flag"])["value"]
        .count()
        .reset_index(name="n")
        .assign(prob=lambda x: x["n"] / x.groupby("value")["n"].transform("sum"))
        .reset_index()
    )
    count_group = count_group[count_group["n"] > count_filter]

    # Get the paired values only
    count_group_names = count_group.groupby("value")["n"].count().reset_index(name="n_")
    count_group_names = count_group_names[count_group_names["n_"] == 2].drop(
        columns=["n_"]
    )

    # Reshape the data to wide format
    count_group_wide = count_group.pivot(
        index="fraud_flag", columns="value", values="n"
    )
    count_group_wide_prob = count_group.pivot(
        index="fraud_flag", columns="value", values="prob"
    )

    # Bind the data frames
    long_table = pd.concat(
        [
            count_group_wide.assign(group_type="count"),
            count_group_wide_prob.assign(group_type="prob"),
        ],
        axis=0,
        ignore_index=False,
    ).reset_index()
    long_table = pd.melt(
        long_table,
        id_vars=["fraud_flag", "group_type"],
        var_name="name",
        value_name="value",
    )

    # Prepare result table
    result_table = long_table.loc[
        (long_table["fraud_flag"] == "Y")
        & (long_table["group_type"] == "prob")
        & (long_table["name"].isin(count_group_names["value"])),
        :,
    ].copy()
    result_table = result_table.drop(["fraud_flag", "group_type"], axis=1)
    result_table.columns = [colname, colname + "_value"]

    return result_table


def evaluateModel(y_test, y_pred, y_score=None, target_precision=0.5):
    # Fraud-evaluation harness.
    #
    # Backward compatible: evaluateModel(y_test, y_pred) keeps the legacy
    # behaviour (accuracy, confusion matrix, classification_report) so existing
    # notebook callers are unaffected. Accuracy is deliberately demoted from
    # headline status — on a ~1.7% positive rate it is misleading.
    #
    # Rich path: pass y_score (positive-class probabilities) to also compute
    # PR-AUC, ROC-AUC, F1/precision/recall, recall@target_precision (the recall
    # achievable at the threshold that first meets target_precision on the
    # precision-recall curve), the best F1 threshold, and a precision-recall
    # curve plot. Returns a result dict; the legacy path returns None.
    y_test_arr = np.asarray(y_test).ravel()
    y_pred_arr = np.asarray(y_pred).ravel()

    # Guard against empty inputs: return a clearly empty result rather than
    # raising, matching the repo's no-exception convention.
    if y_test_arr.size == 0 or y_pred_arr.size == 0:
        if y_score is None:
            return None
        return {
            "accuracy": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "roc_auc": float("nan"),
            "pr_auc": float("nan"),
            "best_threshold_f1": float("nan"),
            "recall_at_precision": float("nan"),
            "target_precision": target_precision,
        }

    accuracy = accuracy_score(y_test_arr, y_pred_arr)

    # Create confusion matrix
    cm = confusion_matrix(y_test_arr, y_pred_arr)

    # Create classification report
    # https://developers.google.com/machine-learning/crash-course/classification/precision-and-recall
    cr = classification_report(y_test_arr, y_pred_arr)

    labels = ["0", "1"]

    # Plot confusion matrix
    plt.figure(figsize=(6, 4))
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    thresh = cm.max() / 2
    for i, j in np.ndindex(cm.shape):
        plt.text(
            j,
            i,
            format(cm[i, j], "d"),
            ha="center",
            va="center",
            color="white" if cm[i, j] > thresh else "black",
        )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.show()

    print("\nAccuracy: %.2f%%" % (accuracy * 100.0), "\n")
    print(cr)

    # Legacy-only path: no probabilities, nothing richer to compute.
    if y_score is None:
        return None

    y_score_arr = np.asarray(y_score).ravel()

    precision, recall, thresholds = precision_recall_curve(y_test_arr, y_score_arr)
    pr_auc = average_precision_score(y_test_arr, y_score_arr)
    roc_auc = roc_auc_score(y_test_arr, y_score_arr)

    # Best threshold by F1: precision_recall_curve omits the threshold for the
    # last (precision=1, recall=0) point, so guard the index.
    f1_scores = 2 * precision * recall / (precision + recall + 1e-12)
    best_idx = int(np.nanargmax(f1_scores[:-1])) if len(thresholds) > 0 else 0
    best_threshold_f1 = float(thresholds[best_idx]) if len(thresholds) > 0 else 0.5

    # Recall at target precision: highest recall whose precision first meets the
    # target, scanning from high threshold (high precision) down.
    prec_arr = precision[:-1] if len(thresholds) > 0 else precision
    rec_arr = recall[:-1] if len(thresholds) > 0 else recall
    meets = prec_arr >= target_precision
    recall_at_precision = float(rec_arr[meets].max()) if meets.any() else 0.0

    # Precision-recall curve plot
    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, color="darkorange", lw=2, label="PR curve")
    plt.scatter(
        [recall[best_idx]],
        [precision[best_idx]],
        color="red",
        zorder=5,
        label="best F1 (t=%.3f)" % best_threshold_f1,
    )
    plt.axhline(target_precision, color="grey", linestyle="--", linewidth=1)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (PR-AUC=%.3f)" % pr_auc)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

    result = {
        "accuracy": float(accuracy),
        "precision": float(precision_score(y_test_arr, y_pred_arr, zero_division=0)),
        "recall": float(recall_score(y_test_arr, y_pred_arr, zero_division=0)),
        "f1": float(f1_score(y_test_arr, y_pred_arr, zero_division=0)),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "best_threshold_f1": best_threshold_f1,
        "recall_at_precision": recall_at_precision,
        "target_precision": target_precision,
    }

    print(
        "PR-AUC=%.4f  ROC-AUC=%.4f  F1=%.4f  Precision=%.4f  Recall=%.4f"
        % (
            result["pr_auc"],
            result["roc_auc"],
            result["f1"],
            result["precision"],
            result["recall"],
        )
    )
    print(
        "Best F1 threshold=%.4f  Recall@precision=%.2f=%.4f"
        % (best_threshold_f1, target_precision, recall_at_precision)
    )

    return result


def chronological_split(X, y, timestamp=None, test_size=0.2, random_state=42):
    # Chronological (time-based) train/test split for fraud evaluation.
    #
    # Why chronological: random shuffling leaks the future into training (a model
    # can learn temporal patterns it should not have seen). With a ~1-year
    # transaction window we sort by timestamp and cut at a date boundary so the
    # test set is strictly LATER than train — closer to how the model is used.
    #
    # Pass a timestamp series (same index as X/y) to get the chronological path.
    # When no timestamp is given, fall back to a stratified random split
    # (train_test_split with stratify=y), which preserves the ~1.7% positive
    # rate in both folds — appropriate for severe class imbalance.
    #
    # Returns X_train, X_test, y_train, y_test.
    if timestamp is None:
        return train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

    ts = pd.Series(timestamp)
    if ts.isna().any():
        raise ValueError("timestamp contains NaN values; cannot sort chronologically")

    sort_index = ts.sort_values().index
    n_test = int(np.ceil(len(sort_index) * test_size))
    if n_test < 1:
        raise ValueError("test_size too small for the provided data length")
    train_idx = sort_index[:-n_test]
    test_idx = sort_index[-n_test:]

    return (
        X.loc[train_idx],
        X.loc[test_idx],
        y.loc[train_idx],
        y.loc[test_idx],
    )


def cross_validate_model(
    model, X, y, k=5, stratified=True, shuffle=True, random_state=42
):
    # k-fold cross-validation returning per-fold metrics + mean/std.
    #
    # Uses StratifiedKFold by default — mandatory for the ~1.7% positive class,
    # where plain KFold could leave a fold with zero frauds. Works for any
    # sklearn-compatible model exposing fit + predict_proba (the positive-class
    # probability is what drives PR-AUC / ROC-AUC).
    #
    # Returns a dict with per-fold arrays and mean/std summaries.
    y_arr = np.asarray(y).ravel()

    if stratified:
        splitter = StratifiedKFold(
            n_splits=k, shuffle=shuffle, random_state=random_state
        )
    else:
        from sklearn.model_selection import KFold

        splitter = KFold(n_splits=k, shuffle=shuffle, random_state=random_state)

    folds = []
    X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X_df, y_arr), start=1):
        X_tr, X_te = X_df.iloc[train_idx], X_df.iloc[test_idx]
        y_tr, y_te = y_arr[train_idx], y_arr[test_idx]

        fitted = (
            model.__class__(**model.get_params())
            if hasattr(model, "get_params")
            else model
        )
        fitted.fit(X_tr, y_tr)

        y_pred = fitted.predict(X_te)
        if hasattr(fitted, "predict_proba"):
            y_score = fitted.predict_proba(X_te)[:, 1]
        elif hasattr(fitted, "decision_function"):
            y_score = fitted.decision_function(X_te)
        else:
            y_score = y_pred.astype(float)

        fold_metrics = {
            "fold": fold,
            "precision": float(precision_score(y_te, y_pred, zero_division=0)),
            "recall": float(recall_score(y_te, y_pred, zero_division=0)),
            "f1": float(f1_score(y_te, y_pred, zero_division=0)),
            "accuracy": float(accuracy_score(y_te, y_pred)),
        }
        # PR-AUC / ROC-AUC need a score, not hard labels.
        try:
            fold_metrics["pr_auc"] = float(
                average_precision_score(y_te, y_score)
            )
            fold_metrics["roc_auc"] = float(roc_auc_score(y_te, y_score))
        except ValueError:
            fold_metrics["pr_auc"] = float("nan")
            fold_metrics["roc_auc"] = float("nan")
        folds.append(fold_metrics)

    folds_df = pd.DataFrame(folds)
    summary = {}
    for col in ["precision", "recall", "f1", "accuracy", "pr_auc", "roc_auc"]:
        if col in folds_df.columns:
            summary[col + "_mean"] = float(folds_df[col].mean())
            summary[col + "_std"] = float(folds_df[col].std())

    return {"folds": folds_df, "summary": summary}


def dictionaryModel(
    input_data,
    dictionaries,
    meta_dictionary,
    fraud_probability_threshold,
    sd_flags_threshold,
    quantile_flags_threshold,
):
    joined_data_test = (
        input_data.merge(
            dictionaries["customer_country"], on="customer_country", how="left"
        )
        .merge(
            dictionaries["counterparty_country"], on="counterparty_country", how="left"
        )
        .merge(dictionaries["type"], on="type", how="left")
        .merge(dictionaries["ccy"], on="ccy", how="left")
        .merge(dictionaries["customer_type"], on="customer_type", how="left")
        .merge(dictionaries["weekday"], on="weekday", how="left")
        .merge(dictionaries["month"], on="month", how="left")
        .merge(dictionaries["quarter"], on="quarter", how="left")
        .merge(dictionaries["hour"], on="hour", how="left")
        .merge(dictionaries["amount_eur_bucket"], on="amount_eur_bucket", how="left")
        .loc[
            :,
            [
                "timestamp",
                "customer",
                "customer_country",
                "customer_type",
                "counterparty",
                "counterparty_country",
                "type",
                "ccy",
                "amount_eur_bucket",
                "weekday",
                "month",
                "quarter",
                "hour",
                "customer_country_value",
                "counterparty_country_value",
                "type_value",
                "ccy_value",
                "customer_type_value",
                "weekday_value",
                "month_value",
                "quarter_value",
                "hour_value",
                "amount_eur_bucket_value",
            ],
        ]
    )

    joined_data_test = (
        joined_data_test.melt(
            id_vars=[
                "timestamp",
                "customer",
                "counterparty",
                "customer_country",
                "counterparty_country",
                "type",
                "ccy",
                "amount_eur_bucket",
                "weekday",
                "month",
                "quarter",
                "hour",
                "customer_type",
            ],
            value_vars=[
                "customer_country_value",
                "counterparty_country_value",
                "type_value",
                "ccy_value",
                "customer_type_value",
                "weekday_value",
                "month_value",
                "quarter_value",
                "hour_value",
                "amount_eur_bucket_value",
            ],
        )
        .assign(name=lambda x: x["variable"].str.replace("_value", ""))
        .merge(meta_dictionary, left_on="name", right_on="variable_name")
        .drop(columns=["variable_name", "variable"])
    )

    joined_data_test_aggregated = (
        joined_data_test.assign(
            sd_flag=lambda x: np.where(x["value"] > x["sd_flag"], 1, 0)
        )
        .assign(quantile_flag=lambda x: np.where(x["value"] > x["q_flag"], 1, 0))
        .assign(quantile_1_flag=lambda x: np.where(x["value"] > x["q_1_flag"], 1, 0))
        .assign(quantile_25_flag=lambda x: np.where(x["value"] > x["q_25_flag"], 1, 0))
        .assign(quantile_75_flag=lambda x: np.where(x["value"] > x["q_75_flag"], 1, 0))
        .assign(quantile_9_flag=lambda x: np.where(x["value"] > x["q_9_flag"], 1, 0))
        .drop_duplicates(["timestamp", "customer", "counterparty", "name"])
        .groupby(
            [
                "timestamp",
                "customer",
                "counterparty",
                "customer_country",
                "counterparty_country",
                "type",
                "ccy",
                "amount_eur_bucket",
                "weekday",
                "month",
                "quarter",
                "hour",
                "customer_type",
            ]
        )
        .agg(
            expected_fraud_probability=("value", "sum"),
            sd_flags=("sd_flag", "sum"),
            quantile_flags=("quantile_flag", "sum"),
            quantile_1_flags=("quantile_1_flag", "sum"),
            quantile_25_flags=("quantile_25_flag", "sum"),
            quantile_75_flags=("quantile_75_flag", "sum"),
            quantile_9_flags=("quantile_9_flag", "sum"),
        )
        .reset_index()
    )

    model_formula = joined_data_test_aggregated.assign(
        predicted_fraud=np.where(
            (
                joined_data_test_aggregated["expected_fraud_probability"]
                > fraud_probability_threshold
            )
            & (joined_data_test_aggregated["sd_flags"] > sd_flags_threshold)
            & (
                joined_data_test_aggregated["quantile_flags"] > quantile_flags_threshold
            ),
            1,
            0,
        )
    )

    test_model = input_data[
        [
            "timestamp",
            "customer",
            "counterparty",
            "customer_country",
            "counterparty_country",
            "type",
            "ccy",
            "amount_eur_bucket",
            "fraud_flag",
            "weekday",
            "month",
            "quarter",
            "hour",
            "customer_type",
        ]
    ].merge(
        model_formula,
        on=[
            "timestamp",
            "customer",
            "counterparty",
            "customer_country",
            "counterparty_country",
            "type",
            "ccy",
            "amount_eur_bucket",
            "weekday",
            "month",
            "quarter",
            "hour",
            "customer_type",
        ],
        how="left",
    )

    test_model = test_model.assign(
        fraud_flag_transformed=np.where((test_model["fraud_flag"] == "Y"), 1, 0)
    ).fillna(0)

    return test_model
