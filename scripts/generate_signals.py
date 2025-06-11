# generate_signals.py

import os
import pandas as pd
import joblib
from core.utils import get_feature_columns
from core.regime_filter import tag_market_regime, apply_regime_based_filtering
from core.signal_generator import generate_ensemble_signals
from core.config import *

def main(bucket):
    df_test = pd.read_csv(f"outputs/{bucket}/test_set_classifier.csv")
    model_clf, feature_cols_clf = joblib.load(f"models/{bucket}/model_classifier.pkl")
    model_reg, feature_cols_reg = joblib.load(f"models/{bucket}/model_regression.pkl")

    # if "ticker" in df_test.columns:
    #     df_test["ticker"] = df_test["ticker"].astype("category")

    # Load thresholds
    df_thresh = pd.read_csv(f"outputs/{bucket}/best_thresholds.csv")
    threshold_map = dict(zip(df_thresh["Ticker"], df_thresh["threshold"]))

    # Generate signals
    df_test = generate_ensemble_signals(
        df=df_test,
        regression_model=model_reg,
        classification_model=model_clf,
        threshold_map=threshold_map
    )

    # Apply regime filter
    df_test = tag_market_regime(df_test)
    df_test = apply_regime_based_filtering(df_test)

    # Save final signals
    os.makedirs(f"outputs/{bucket}", exist_ok=True)
    df_test.to_csv(f"outputs/{bucket}/test_predictions_with_signals.csv", index=False)
    print(f"✅ Saved: outputs/{bucket}/test_predictions_with_signals.csv")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", type=str, required=True)
    args = parser.parse_args()
    main(bucket=args.bucket)
