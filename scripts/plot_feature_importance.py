# plot_feature_importance.py
import os
import argparse
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb

def plot_and_save_feature_importance(model_path, model_type, bucket_name, top_n=20):
    model, feature_cols = joblib.load(model_path)
    booster = model.booster_
    importance = booster.feature_importance(importance_type="gain")
    feature_names = booster.feature_name()

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance
    }).sort_values(by="importance", ascending=False)

    # Save full importance as CSV
    os.makedirs(f"outputs/{bucket_name}/feature_importance", exist_ok=True)
    csv_path = f"outputs/{bucket_name}/feature_importance/feature_importance_{model_type}.csv"
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved CSV: {csv_path}")

    # Save chart
    plt.figure(figsize=(8, 10))
    plt.barh(df.head(top_n)["feature"], df.head(top_n)["importance"])
    plt.gca().invert_yaxis()
    plt.title(f"Feature Importance - {bucket_name} {model_type.capitalize()} Model")
    plt.xlabel("Importance")
    plt.tight_layout()

    save_path = f"outputs/{bucket_name}/feature_importance/feature_importance_{model_type}.png"
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Saved chart: {save_path}")

def main(bucket):
    model_dir = f"models/{bucket}"
    assert os.path.exists(model_dir), f"Model directory not found: {model_dir}"

    clf_path = os.path.join(model_dir, "model_classifier.pkl")
    reg_path = os.path.join(model_dir, "model_regression.pkl")

    assert os.path.exists(clf_path), f"Classifier model not found: {clf_path}"
    assert os.path.exists(reg_path), f"Regression model not found: {reg_path}"

    plot_and_save_feature_importance(clf_path, "classifier", bucket)
    plot_and_save_feature_importance(reg_path, "regression", bucket)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", type=str, required=True, help="Model bucket name (e.g., IT_Services)")
    args = parser.parse_args()
    main(args.bucket)
