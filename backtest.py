import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve
from config import DATA_DIR, BUY_THRESHOLD, MODEL_DIR, PEER_NAME, PLOTS_DIR, SIGNAL_MODE, THRESHOLD, TRADES_DIR, USE_FIXED_THRESHOLD
from threshold_tuning_utils import tune_threshold
from utils import (
    compute_peer_features,
    generate_signals,
    get_target_definition,
    load_stock_data,
    compute_features,
    add_advanced_features,
    merge_peer_features,
    simulate_trades,
    train_lightgbm,
    predict_signal,
    load_sector_features,
    load_index_features,
    get_sector_for_stock
)
from trade_analyzer import analyze_trades
def plot_prediction_distribution(df, ticker):
    if "pred_prob" in df.columns:
        plt.figure(figsize=(8, 4))
        plt.hist(df["pred_prob"], bins=50, color='skyblue', edgecolor='black')
        plt.title(f"Predicted Probability Distribution - {ticker}")
        plt.xlabel("Predicted Probability")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        print("⚠️ 'pred_prob' column not found in dataframe.")
def run_backtest_for_stock(filepath):
    ticker = os.path.basename(filepath).replace(".csv", "")
    print(f"🔍 Running backtest for {ticker}")
    df = load_stock_data(filepath)
    df = add_advanced_features(df,ticker)
    if PEER_NAME:
        peer_name = PEER_NAME
        peer_file_path = os.path.join("data", "nse_eod", f"{peer_name.upper()}.csv")
        df_peer = pd.read_csv(peer_file_path,parse_dates=["Date"])
        df_peer_features = compute_peer_features(df_peer, peer_name)
        df = merge_peer_features(df, df_peer_features)
    df = get_target_definition(df)
    if len(df) < 200:
        print(f"⚠️ Skipping {ticker} due to insufficient data ({len(df)} rows)")
        return None, None
    df = df.sort_values("Date").reset_index(drop=True)
    n = len(df)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)
    df_train = df.iloc[:train_end].copy()
    df_val = df.iloc[train_end:val_end].copy()
    df_test = df.iloc[val_end:].copy()
    model = train_lightgbm(df_train, ticker)
    df_val = generate_signals(df_val, model, threshold=THRESHOLD)
    optimal_threshold = THRESHOLD
    if SIGNAL_MODE == "threshold":
        val_valid = df_val.dropna(subset=["target", "pred_prob"]).reset_index(drop=True)
        y_val_true = val_valid["target"]
        y_val_probs = val_valid["pred_prob"]
        best_threshold_result, _ = tune_threshold(y_val_true, y_val_probs)
        precision_vals, recall_vals, thresholds = precision_recall_curve(y_val_true, y_val_probs)
        plt.figure(figsize=(6, 6))
        plt.plot(recall_vals, precision_vals, marker='.', label="PR Curve")
        plt.title(f"Precision-Recall Curve: {ticker}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.grid(True)
        plt.tight_layout()
        plt.legend()
        plot_path = os.path.join(PLOTS_DIR, f"{ticker}_precision_recall.png")
        plt.savefig(plot_path)
        plt.close()
        if USE_FIXED_THRESHOLD:
            optimal_threshold = THRESHOLD
            print(f"⚙️ Using fixed threshold = {optimal_threshold}")
        else:
            optimal_threshold = best_threshold_result["threshold"]
            print(f"🎯 Using tuned threshold = {optimal_threshold:.2f}")
    df_train = generate_signals(df_train, model,threshold=optimal_threshold)
    df_test = generate_signals(df_test, model,threshold=optimal_threshold)
    plot_prediction_distribution(df_test, ticker)
    trades_train = simulate_trades(df_train, ticker)
    trades_test = simulate_trades(df_test, ticker)
    _, metrics_train = analyze_trades(trades_train, ticker)
    _, metrics_test = analyze_trades(trades_test, ticker, output_file=os.path.join(TRADES_DIR, f"{ticker}_test_trades.csv"))
    train_hit_rate = metrics_train["hit_rate"]
    train_sharpe_ratio = metrics_train["sharpe_ratio"]
    train_max_drawdown = metrics_train["max_drawdown"]
    train_cagr = metrics_train["cagr"]
    test_hit_rate = metrics_test["hit_rate"]
    test_sharpe_ratio = metrics_test["sharpe_ratio"]
    test_max_drawdown = metrics_test["max_drawdown"]
    test_cagr = metrics_test["cagr"]
    df_test["return"] = df_test["Close"].pct_change().shift(-1)
    df_test["strategy_return"] = df_test["signal"] * df_test["return"]
    df_test["cumulative_return"] = (1 + df_test["strategy_return"].fillna(0)).cumprod()
    df_test["Date"] = pd.to_datetime(df_test["Date"])
    df_test.set_index("Date", inplace=True)
    y_true_test = df_test["target"]
    y_pred_test = df_test["signal"]
    accuracy = accuracy_score(y_true_test, y_pred_test)
    precision = precision_score(y_true_test, y_pred_test, zero_division=0)
    recall = recall_score(y_true_test, y_pred_test, zero_division=0)
    f1 = f1_score(y_true_test, y_pred_test, zero_division=0)
    final_return = df_test["cumulative_return"].iloc[-1]
    metrics = {
        "Ticker": ticker,
        "Accuracy": round(accuracy,2),
        "Precision": round(precision,2),
        "Recall": round(recall,2),
        "F1 Score": round(f1,2),
        "optimal_threshold": round(optimal_threshold, 2),
        "Final Cumulative Return": round(final_return,2),
        "train_hit_rate":train_hit_rate,
        "train_sharpe_ratio": train_sharpe_ratio,
        "train_max_drawdown":train_max_drawdown,
        "train_cagr":train_cagr,
        "test_hit_rate":test_hit_rate,
        "test_sharpe_ratio": test_sharpe_ratio,
        "test_max_drawdown":test_max_drawdown,
        "test_cagr":test_cagr
    }
    return df_test[["cumulative_return"]], metrics
def main():
    print("📁 Scanning for stock files...")
    stock_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    try:
        with open("data/stock_universe.txt") as f:
            allowed_stocks = set(line.strip().upper() for line in f)
        stock_files = [f for f in stock_files if os.path.basename(f).replace(".csv", "").upper() in allowed_stocks]
        print(f"✅ Filtered to {len(stock_files)} stocks from universe.")
    except FileNotFoundError:
        print("⚠️ stock_universe.txt not found. Using all stock files.")
    all_results = []
    all_metrics = []
    for filepath in stock_files:
        result, metrics = run_backtest_for_stock(filepath)
        if result is not None:
            df_cumret = result
            ticker = metrics["Ticker"]
            df_cumret.rename(columns={"cumulative_return": ticker}, inplace=True)
            all_results.append(df_cumret)
            all_metrics.append(metrics)
    if all_results:
        combined = pd.concat(all_results, axis=1)
        combined.fillna(method="ffill", inplace=True)
        combined.plot(title="Backtested Cumulative Returns")
        plt.grid()
        plt.tight_layout()
        plt.show()
    else:
        print("⚠️ No backtest results to plot.")
    if all_metrics:
        perf_df = pd.DataFrame([r for r in all_metrics if r is not None])
        perf_df.to_csv(os.path.join(MODEL_DIR, "model_performance_log.csv"), index=False)
        print("📊 Model performance log saved to models/model_performance_log.csv")
        perf_df[["Ticker", "optimal_threshold", "F1 Score"]].to_csv(os.path.join(MODEL_DIR, "optimal_thresholds.csv"), index=False)
        print("✅ Optimal thresholds saved to models/optimal_thresholds.csv")
    else:
        print("⚠️ No models passed evaluation.")
if __name__ == "__main__":
    main()