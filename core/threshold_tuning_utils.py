# threshold_tuning_utils.py

import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

def tune_threshold(y_true, y_probs, thresholds=np.arange(0.3, 0.91, 0.05)):
    results = []
    count = 0
    for t in thresholds:
        count +=1 
        y_pred = (y_probs >= t).astype(int)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        # if count < 5:
        #     print(f"Threshold: {t} | f1 score:{f1}")
        results.append((t, precision, recall, f1))

    results_df = pd.DataFrame(results, columns=["threshold", "precision", "recall", "f1"])
    best_row = results_df.loc[results_df["f1"].idxmax()]
    print(best_row.to_dict())
    return best_row.to_dict(), results_df


# ✅ Step 4: Tune threshold with binary conversion of multiclass targets

def tune_threshold_multiclass(y_true, y_prob, thresholds=None):
    """
    Tune threshold for multiclass model (binary signal from class 3 or 4).
    Converts y_true to binary: 1 if class >= 3 else 0.
    """
    from sklearn.metrics import precision_score, recall_score, f1_score
    import numpy as np

    y_true_binary = (y_true >= 3).astype(int)
    if thresholds is None:
        thresholds = np.arange(0.3, 0.9, 0.05)

    best_f1 = -1
    best_row = None
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        precision = precision_score(y_true_binary, y_pred, zero_division=0)
        recall = recall_score(y_true_binary, y_pred, zero_division=0)
        f1 = f1_score(y_true_binary, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_row = {"threshold": t, "precision": precision, "recall": recall, "f1": f1}

    return best_row, thresholds


def tune_threshold_binary(y_true, y_probs, lower=0.3, upper=0.7, step=0.05):
    best_f1 = 0
    best_threshold = lower
    best_precision = 0
    best_recall = 0
    for threshold in np.arange(lower, upper, step):
        y_pred = (y_probs >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_precision = precision_score(y_true, y_pred)
            best_recall = recall_score(y_true, y_pred)
    return {"threshold": best_threshold, "f1": best_f1, "precision": best_precision,
        "recall": best_recall}, best_f1



# def tune_threshold(y_true, y_probs, df=None, return_series=None):
#     best_result = None
#     best_threshold = None

#     thresholds = np.arange(0.1, 0.9, 0.01)
#     results = []

#     for threshold in thresholds:
#         signal = (y_probs >= threshold).astype(int)

#         if df is not None and return_series is not None:
#             df_temp = df.copy()
#             df_temp["signal"] = signal
#             signal_dates = df_temp[df_temp["signal"] == 1]["Date"]
#             pnl_filtered = return_series[df_temp["Date"].isin(signal_dates)]
#             total_return = (1 + pnl_filtered / 100).prod() - 1
#             results.append({"threshold": threshold, "return": total_return})

#     results_df = pd.DataFrame(results)
#     best_row = results_df.sort_values("return", ascending=False).iloc[0]

#     return best_row, results_df


# def walk_forward_threshold_tuning(df, date_col="Date", n_splits=4, thresholds=np.arange(0.3, 0.91, 0.05)):
#     """
#     Walk-forward threshold tuning using F1 Score.
#     Requires 'target', 'pred_prob', and 'Date' columns in df.
#     """
#     df = df.sort_values(date_col).reset_index(drop=True)
#     split_size = len(df) // (n_splits + 1)
#     results = []

#     for split in range(n_splits):
#         start = 0
#         mid = split_size * (split + 1)
#         end = split_size * (split + 2)

#         train_df = df.iloc[start:mid]
#         val_df = df.iloc[mid:end]

#         y_val = val_df["target"]
#         y_probs = val_df["pred_prob"]

#         for t in thresholds:
#             y_pred = (y_probs >= t).astype(int)
#             precision = precision_score(y_val, y_pred, zero_division=0)
#             recall = recall_score(y_val, y_pred, zero_division=0)
#             f1 = f1_score(y_val, y_pred, zero_division=0)
#             results.append({"threshold": t, "precision": precision, "recall": recall, "f1": f1})

#     results_df = pd.DataFrame(results)
#     avg_results = results_df.groupby("threshold").mean().reset_index()
#     best_row = avg_results.loc[avg_results["f1"].idxmax()]
#     return avg_results, best_row["threshold"]

