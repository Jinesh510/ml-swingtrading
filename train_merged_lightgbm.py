import os
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, auc, f1_score, precision_recall_curve, precision_score, recall_score
from sklearn.model_selection import train_test_split
from config import PLOTS_DIR, PROB_DIST_DIR, RESULTS_DIR, SIGNAL_MODE, THRESHOLD, TOP_K, TRADES_DIR, USE_FIXED_THRESHOLD
from threshold_tuning_utils import tune_threshold
# from threshold_tuning_utils import walk_forward_threshold_tuning
from utils import apply_training_filters, generate_signals, label_profitable_trades, load_processed_data, simulate_trades
from utils import train_lightgbm
from backtest import predict_signal
from trade_analyzer import analyze_trades

def load_merged_stock_data(tickers):
    all_dfs = []
    for ticker in tickers:
        df = load_processed_data(ticker)
        # df["ticker"] = ticker
        all_dfs.append(df)
    return pd.concat(all_dfs, ignore_index=True)

def plot_precision_recall(df, ticker):
    y_true = df["target"]
    y_probs = df["pred_prob"]
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    pr_auc = auc(recall, precision)
    plt.figure()
    plt.plot(recall, precision, marker='.')
    plt.title(f"{ticker} Precision-Recall Curve (AUC={pr_auc:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True)
    plt.savefig(os.path.join(PLOTS_DIR,f"pr_curve_{ticker}.png"))
    plt.close()
    return pr_auc

def plot_probability_histogram(df, ticker, set_name="test"):
    plt.figure()
    plt.hist(df[df["target"] == 0]["pred_prob"], bins=50, alpha=0.6, label="Class 0")
    plt.hist(df[df["target"] == 1]["pred_prob"], bins=50, alpha=0.6, label="Class 1")
    plt.axvline(x=0.5, color="red", linestyle="--", label="0.5 Threshold")
    plt.title(f"{ticker} {set_name} Prediction Probability Distribution")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(PROB_DIST_DIR,f"prob_dist_{ticker}_{set_name}.png"))
    plt.close()


if __name__ == "__main__":
    tickers = ["MPHASIS", "PERSISTENT", "COFORGE", "BSOFT", "ECLERX","CYIENT","SONATASOFTW","ZENSARTECH","TATAELXSI"]
    # tickers =["COFORGE"]
    # tickers = ["MPHASIS", "COFORGE", "PERSISTENT", "BIRLASOFT", "LTI", 
    #             "LTTS", "TATAELXSI", "SONATSOFTW", "ZENSARTECH", "CYIENT", 
    #             "KPITTECH", "ECLERX", "DATAPATTNS", "LATENTVIEW", "NEWGEN"]
    merged_df = load_merged_stock_data(tickers)
    merged_df["ticker"] = merged_df["ticker"].astype("category")

    # Sort by date just in case
    merged_df = merged_df.sort_values("Date").reset_index(drop=True)
    # Train joint model
    df_train, df_temp = train_test_split(merged_df, test_size=0.4, shuffle=False)
    df_val, df_test = train_test_split(df_temp, test_size=0.5, shuffle=False)

    # df_train = apply_training_filters(df_train)
    # df_val = apply_training_filters(df_val)
    # df_test = apply_training_filters(df_test)

    model = train_lightgbm(df_train, ticker="MIDCAP_IT")

    # df_train_signals = generate_signals(df_train.copy(), model_temp,use_dynamic_threshold=False)
    # train_trades = simulate_trades(df_train_signals, "MIDCAP_IT")

    # # Label based on profitable trades
    # df_train_labeled = label_profitable_trades(df_train_signals, train_trades)

    # Final training
    # model = train_lightgbm(df_train_labeled, ticker="MIDCAP_IT")


    # Step 3: Label validation and test data using simulated trades
    # df_val_signals = generate_signals(df_val.copy(), model,use_dynamic_threshold=False)

    # df_val_pred = predict_signal(df_val.copy(), model_temp)
    # df_val_pred["signal"] = 1  # temp signal to simulate all rows
    # val_trades = simulate_trades(df_val_pred, "MIDCAP_IT")
    # df_val_labeled = label_profitable_trades(df_val_pred, val_trades)


    # val_trades = simulate_trades(df_val_signals, "MIDCAP_IT")
    # df_val_labeled = label_profitable_trades(df_val_signals, val_trades)

    # df_test_signals = generate_signals(df_test.copy(), model,use_dynamic_threshold=True)
    # test_trades = simulate_trades(df_test_signals, "MIDCAP_IT")
    # df_test_labeled = label_profitable_trades(df_test_signals, test_trades)

    optimal_threshold = THRESHOLD
    # df_val_pred = generate_signals(df_val.copy(), model, threshold=optimal_threshold, topk=TOP_K, signal_mode=SIGNAL_MODE)
    df_val_pred = predict_signal(df_val.copy(), model)

    # Compute predicted probabilities
    # df_val_pred["pred_prob"] = model.predict_proba(df_val_pred[model.feature_name_])[:, 1]


    if SIGNAL_MODE == "threshold":
        print("🔍 Tuning threshold on validation set...")

        # Per-stock threshold tuning
        per_ticker_thresholds = {}

        for ticker in tickers:
            val_subset = df_val_pred[df_val_pred["ticker"] == ticker]
            # val_subset = df_val_labeled[df_val_labeled["ticker"] == ticker]

            # print(f"\n🎯 Ticker: {ticker} — {len(val_subset)} rows")
            # print("Target dist:", val_subset["target"].value_counts().to_dict())
            # print("Pred prob stats:\n", val_subset["pred_prob"].describe())
            # if val_subset.empty:
            #     print(f"⚠️ Skipping {ticker} — no validation data")
            #     continue
            # val_subset["Date"] = pd.to_datetime(val_subset["Date"])
            # avg_perf, best_thresh = walk_forward_threshold_tuning(val_subset)
            # val_trades_subset = simulate_trades(val_subset.copy(), ticker)

            # # Merge back P&L %
            # val_subset = val_subset.merge(
            #     val_trades_subset[["Entry Date", "P&L %"]],
            #     left_on="Date",
            #     right_on="Entry Date",
            #     how="left"
            # )

            # # Now extract P&L for only the trades that would be entered
            # pnl_series = val_subset[val_subset["signal"] == 1]["P&L %"].dropna()
            # # pnl_series = val_subset[val_subset["signal"] == 1]["P&L %"]
            # best_row, _ = tune_threshold(
            #     val_subset["target"],
            #     val_subset["pred_prob"],
            #     df=val_subset,
            #     return_series=pnl_series
            # )

            # val_probs = model.predict_proba(val_subset[model.feature_name_])[:, 1]

            best_row, _ = tune_threshold(val_subset["target"], val_subset["pred_prob"])
            if USE_FIXED_THRESHOLD:
                per_ticker_thresholds[ticker] = THRESHOLD
                print(f"⚙️ Using fixed threshold = {THRESHOLD}")
            else:
                # per_ticker_thresholds[ticker] = best_thresh
                # print(f"✅ {ticker}: Walk-forward optimal threshold = {best_thresh:.2f}")

                per_ticker_thresholds[ticker] = best_row["threshold"]
                print(f"✅ Tuned threshold for {ticker}: {best_row['threshold']:.2f}")
                # print(f"✅ Tuned threshold for {ticker} (Return-Optimized): {best_row['threshold']:.2f}")


                # print(f"🎯 Using tuned threshold = {optimal_threshold:.2f}")

            # per_ticker_thresholds[ticker] = best_row["threshold"]
            # print(f"✅ Tuned threshold for {ticker}: {best_row['threshold']:.2f}")


        # val_valid = df_val_pred.dropna(subset=["target", "pred_prob"]).reset_index(drop=True)
        # y_val_true = val_valid["target"]
        # y_val_probs = val_valid["pred_prob"]

        # best_threshold_result,_ = tune_threshold(y_val_true,y_val_probs)
        # # tuned_threshold = best_threshold_result["optimal_threshold"]
        # # print(f"🎯 Best Threshold (F1): {tuned_threshold:.2f}")
        # if USE_FIXED_THRESHOLD:
        #     optimal_threshold = THRESHOLD
        #     print(f"⚙️ Using fixed threshold = {optimal_threshold}")
        # else:
        #     optimal_threshold = best_threshold_result["threshold"]
        #     print(f"🎯 Using tuned threshold = {optimal_threshold:.2f}")

    results=[]
    print("\n📈 Evaluating on test set per ticker:")
    for test_ticker in tickers:
        df_ticker_test = df_test[df_test["ticker"] == test_ticker].copy()

        # df_ticker_test = df_test_labeled[df_test_labeled["ticker"] == test_ticker].copy()
        tuned_threshold = per_ticker_thresholds[test_ticker]

        df_ticker_test = generate_signals(df_ticker_test, model, threshold=tuned_threshold, signal_mode=SIGNAL_MODE, use_dynamic_threshold=False)
        
        pr_auc = plot_precision_recall(df_ticker_test, test_ticker)
        plot_probability_histogram(df_ticker_test, test_ticker, set_name="test")

        trades_df = simulate_trades(df_ticker_test,test_ticker)
        # trades_df["P&L %"].hist(bins=50)
        # plt.title("Distribution of Trade P&L")
        # plt.xlabel("P&L %")
        # plt.ylabel("Trade Count")
        # plt.grid(True)
        # plt.show()

        trades_df.to_csv(os.path.join(TRADES_DIR, f"trades_{test_ticker}.csv"), index=False)
        print(f"📝 Saved trades_{test_ticker}.csv and PR curve image.")

        _, metrics = analyze_trades(trades_df, test_ticker)

        df_ticker_test.set_index("Date", inplace=True)
        # df_eval = df_ticker_test[(df_ticker_test["signal"] == 1) | (df_ticker_test["target"] == 1)].copy()

        # y_true_test = df_eval["target"]
        # y_pred_test = df_eval["signal"]

        y_true_test = df_ticker_test["target"]
        y_pred_test = df_ticker_test["signal"]
        accuracy = accuracy_score(y_true_test, y_pred_test)
        precision = precision_score(y_true_test, y_pred_test, zero_division=0)
        recall = recall_score(y_true_test, y_pred_test, zero_division=0)
        f1 = f1_score(y_true_test, y_pred_test, zero_division=0)
        trades_df["Cumulative Equity"] = (1 + trades_df["P&L %"] / 100).cumprod()
        final_return = trades_df["Cumulative Equity"].iloc[-1]

        metrics_to_print = {
            "Ticker": test_ticker,
            "Accuracy": round(accuracy,2),
            "Precision": round(precision,2),
            "Recall": round(recall,2),
            "F1 Score": round(f1,2),
            "optimal_threshold": round(tuned_threshold, 2),
            "Final Cumulative Return": round(final_return,2),
            'PR AUC':pr_auc,
            # "train_hit_rate":train_hit_rate,
            # "train_sharpe_ratio": train_sharpe_ratio,
            # "train_max_drawdown":train_max_drawdown,
            # "train_cagr":train_cagr,
            'test_num_trades':metrics['num_trades'],
            "test_hit_rate":metrics['hit_rate'],
            "test_sharpe_ratio": metrics['sharpe_ratio'],
            "test_max_drawdown":metrics['max_drawdown'],
            "test_cagr":metrics['cagr']
            
        }
        
        # metrics["PR AUC"] = pr_auc
        results.append(metrics_to_print)


    summary_df = pd.DataFrame(results)
    summary_df.to_csv(os.path.join(RESULTS_DIR,"merged_model_test_summary.csv"), index=False)
    print("✅ Summary saved to merged_model_test_summary.csv")

