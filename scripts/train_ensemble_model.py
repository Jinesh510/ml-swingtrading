# train_ensemble_model.py
import argparse
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from core.config import MODEL_DIR
from core.feature_generator import generate_features
from core.labeling import label_atr_based_gain, label_excess_return_regression, label_prob_3pct_gain
from core.utils import load_and_merge_data, save_model
from core.model_trainer import train_lightgbm_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True, help="Bucket name (e.g. IT_Services)")
    args = parser.parse_args()

    bucket = args.bucket
    tickers_df = pd.read_csv(f"tickers/{bucket}.csv")
    tickers = tickers_df["Ticker"].tolist()

    df_raw = load_and_merge_data(
        tickers=tickers,
        ohlcv_path="data/nse_eod",
        start_date="2010-01-01",
        bucket_name=bucket
    )

    # df_feat = generate_features(df_raw)
    merged_df = df_raw.sort_values("Date").reset_index(drop=True)

    # ✅ Compute cross-sectional ranks/zscores across tickers by date
    merged_df["rank_return_5d"] = merged_df.groupby("Date")["return_5d"].rank(pct=True)

    merged_df["zscore_return_5d"] = (
        merged_df["return_5d"] - merged_df.groupby("Date")["return_5d"].transform("mean")
    ) / merged_df.groupby("Date")["return_5d"].transform("std")

    merged_df["macd_hist_x_zscore_5d"] = merged_df["macd_hist"] * merged_df["zscore_return_5d"]

    # df_clf = label_atr_based_gain(merged_df.copy(), atr_multiplier=2, min_pct_gain=0.05)
    df_clf = label_prob_3pct_gain(merged_df.copy())
    # print("Classifier Target dist:",df_clf.groupby("ticker")["target"].mean().sort_values())

    df_reg = label_excess_return_regression(merged_df.copy())



    # Split classification
    df_clf_train, df_clf_temp = train_test_split(df_clf, test_size=0.4, shuffle=False)
    df_clf_val, df_clf_test = train_test_split(df_clf_temp, test_size=0.5, shuffle=False)

    # Split regression
    df_reg_train, df_reg_temp = train_test_split(df_reg, test_size=0.4, shuffle=False)
    df_reg_val, df_reg_test = train_test_split(df_reg_temp, test_size=0.5, shuffle=False)

    # Train both models
    model_clf, feature_cols_clf = train_lightgbm_model(df_clf_train, val_df=df_clf_val, task="classification", ticker=bucket)
    model_reg, feature_cols_reg = train_lightgbm_model(df_reg_train, val_df=df_reg_val, task="regression", ticker=bucket)

    # Save models
    # os.makedirs(f"models/{bucket}", exist_ok=True)
    save_model(model_clf,feature_cols_clf, MODEL_DIR, bucket, "classifier")
    save_model(model_reg,feature_cols_reg, MODEL_DIR, bucket, "regression")
    print(f"✅ Models saved to models/{bucket}/")


    # Save validation and test sets
    os.makedirs(f"outputs/{bucket}", exist_ok=True)
    df_clf_val.to_csv(f"outputs/{bucket}/val_set_classifier.csv", index=False)
    df_reg_val.to_csv(f"outputs/{bucket}/val_set_regression.csv", index=False)
    df_clf_test.to_csv(f"outputs/{bucket}/test_set_classifier.csv", index=False)
    # df_reg_test.to_csv(f"outputs/{bucket}/test_set_regression.csv", index=False)
