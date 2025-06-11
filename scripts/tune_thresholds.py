# scripts/tune_thresholds.py

import os
import pandas as pd
import joblib
from core.utils import get_feature_columns
from core.threshold_tuning_utils import tune_threshold_binary
from core.config import *

def main(bucket):
    df_val = pd.read_csv(f"outputs/{bucket}/val_set_classifier.csv")
    model_path = f"models/{bucket}/model_classifier.pkl"
    model, feature_cols = joblib.load(model_path)

    results = []


    for ticker in df_val["ticker"].unique():
        df_ticker = df_val[df_val["ticker"] == ticker].copy()
        if "ticker" in df_ticker.columns and "ticker" in feature_cols:
            df_ticker["ticker"] = df_ticker["ticker"].astype("category")        
        X_val = df_ticker[feature_cols].copy()
        y_true = df_ticker["target"]
        y_proba = model.predict_proba(X_val)[:, 1]

        result_dict, _ = tune_threshold_binary(y_true, y_proba)
        result_dict["Ticker"] = ticker
        results.append(result_dict)


    df_results = pd.DataFrame(results)
    df_results = df_results[["Ticker", "threshold", "f1", "precision", "recall"]]
    df_results.to_csv(f"outputs/{bucket}/best_thresholds.csv", index=False)
    print(f"✅ Saved thresholds to outputs/{bucket}/best_thresholds.csv")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", type=str, required=True)
    args = parser.parse_args()

    main(bucket=args.bucket)
