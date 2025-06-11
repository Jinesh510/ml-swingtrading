# model_diagnostics.py
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve


def save_prediction_distribution(pred_probs, output_path, ticker=None):
    plt.figure(figsize=(8, 4))
    plt.hist(pred_probs, bins=50, color='skyblue', edgecolor='k')
    title = f"Prediction Probability Distribution"
    if ticker:
        title += f" for {ticker}"
    plt.title(title)
    plt.xlabel("Predicted Probability")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_precision_recall(y_true, y_probs, output_path):
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    plt.figure(figsize=(8, 4))
    plt.plot(thresholds, precision[:-1], label="Precision")
    plt.plot(thresholds, recall[:-1], label="Recall")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Precision-Recall vs Threshold")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def run_diagnostics(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    tickers = df["ticker"].unique()

    for ticker in tickers:
        df_ticker = df[df["ticker"] == ticker]
        if df_ticker["target"].nunique() < 2:
            print(f"⚠️ Skipping {ticker} (only one class in target)")
            continue

        save_prediction_distribution(
            df_ticker["pred_prob"],
            os.path.join(output_dir, f"prob_dist_{ticker}.png"),
            ticker=ticker
        )

        save_precision_recall(
            df_ticker["target"],
            df_ticker["pred_prob"],
            os.path.join(output_dir, f"pr_curve_{ticker}.png")
        )
        print(f"✅ Saved diagnostics for {ticker}")


if __name__ == "__main__":
    # Defaults: assume bucket is IT_Services and file is at standard location
    bucket = "IT_Services"
    input_path = f"outputs/{bucket}/test_predictions_with_signals.csv"
    output_dir = f"outputs/{bucket}/diagnostics"

    df = pd.read_csv(input_path)
    run_diagnostics(df, output_dir)
