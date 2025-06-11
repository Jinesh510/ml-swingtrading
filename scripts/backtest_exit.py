# backtest_exit.py

import os
import argparse
import pandas as pd
from core.trade_simulator import simulate_trades
from core.trade_analyzer import analyze_trades
from sklearn.metrics import classification_report

def run_exit_strategies(df_signals, tickers, exit_conditions_list, trades_dir, results_dir):
    os.makedirs(trades_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    for i, exit_params in enumerate(exit_conditions_list, start=1):
        print(f"\n🚀 Running Exit Strategy {i} - Params: {exit_params}")
        results = []

        for ticker in tickers:
            df_ticker = df_signals[df_signals["ticker"] == ticker].copy()

            trades_df = simulate_trades(
                df_ticker,
                ticker,
                trailing_stop=exit_params["TRAILING_STOPLOSS"],
                profit_target=exit_params["PROFIT_TARGET"],
                max_hold_days=exit_params["MAX_HOLD_DAYS"]
            )

            trades_df.to_csv(os.path.join(trades_dir, f"trades_{ticker}_exit{i}.csv"), index=False)

            if trades_df.empty:
                print(f"⚠️ No trades for {ticker} under Exit Strategy {i}")
                continue
            _, metrics = analyze_trades(trades_df, ticker)

            df_ticker.set_index("Date", inplace=True)
            y_true = (df_ticker["target"] > 0).astype(int)
            y_pred = df_ticker["signal"]
            print(classification_report(y_true, y_pred))

            trades_df["Cumulative Equity"] = (1 + trades_df["P&L %"] / 100).cumprod()
            final_return = trades_df["Cumulative Equity"].iloc[-1]

            results.append({
                "Exit_Setup": f"Exit_{i}",
                "Ticker": ticker,
                "Final Cumulative Return": round(final_return, 2),
                'test_num_trades': metrics['num_trades'],
                "test_hit_rate": metrics['hit_rate'],
                "test_sharpe_ratio": metrics['sharpe_ratio'],
                "test_max_drawdown": metrics['max_drawdown'],
                "test_cagr": metrics['cagr']
            })

        summary_df = pd.DataFrame(results)
        outpath = os.path.join(results_dir, f"merged_model_test_summary_exit{i}.csv")
        summary_df.to_csv(outpath, index=False)
        print(f"✅ Saved summary for Exit Strategy {i} to {outpath}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", type=str, required=True, help="Model bucket name (e.g. IT_Services)")
    args = parser.parse_args()

    bucket = args.bucket
    signals_path = f"outputs/{bucket}/test_predictions_with_signals.csv"
    tickers_path = f"tickers/{bucket}.csv"
    trades_dir = f"outputs/{bucket}/trades"
    results_dir = f"outputs/{bucket}/results"

    df_signals = pd.read_csv(signals_path)
    # tickers = pd.read_csv(tickers_path)["Ticker"].tolist()

    tickers = df_signals['ticker'].unique()
    print(tickers)

    exit_conditions_list = [

        {"TRAILING_STOPLOSS": 0.10, "PROFIT_TARGET": 0.08, "MAX_HOLD_DAYS": 30},  # Baseline
        {"TRAILING_STOPLOSS": 0.15, "PROFIT_TARGET": 0.12, "MAX_HOLD_DAYS": 45},  # Moderate swing
        {"TRAILING_STOPLOSS": 0.20, "PROFIT_TARGET": 0.15, "MAX_HOLD_DAYS": 50},  # Wider range
        {"TRAILING_STOPLOSS": 0.25, "PROFIT_TARGET": 0.20, "MAX_HOLD_DAYS": 60},  # Strong trend capture
        {"TRAILING_STOPLOSS": 0.30, "PROFIT_TARGET": 0.25, "MAX_HOLD_DAYS": 75},  # High volatility plays

        {"TRAILING_STOPLOSS": 0.10, "PROFIT_TARGET": 0.20, "MAX_HOLD_DAYS": 50},  # Aggressive target, tight SL
        {"TRAILING_STOPLOSS": 0.20, "PROFIT_TARGET": 0.10, "MAX_HOLD_DAYS": 50},  # Conservative PT, loose SL

        {"TRAILING_STOPLOSS": 0.12, "PROFIT_TARGET": 0.15, "MAX_HOLD_DAYS": 40},  # Balanced swing
        {"TRAILING_STOPLOSS": 0.18, "PROFIT_TARGET": 0.22, "MAX_HOLD_DAYS": 60},  # Mid/long swing
        {"TRAILING_STOPLOSS": 0.22, "PROFIT_TARGET": 0.30, "MAX_HOLD_DAYS": 90},  # Long hold, breakout trend
        {"TRAILING_STOPLOSS": 0.20, "PROFIT_TARGET": 0.30, "MAX_HOLD_DAYS": 200},  # Long hold, breakout trend

        {"TRAILING_STOPLOSS": 0.15, "PROFIT_TARGET": 0.20, "MAX_HOLD_DAYS": 200},  # Long hold, breakout trend
        {"TRAILING_STOPLOSS": 0.15, "PROFIT_TARGET": 0.10, "MAX_HOLD_DAYS": 200},  # Long hold, breakout trend
        {"TRAILING_STOPLOSS": 0.25, "PROFIT_TARGET": 0.20, "MAX_HOLD_DAYS": 200},  # Long hold, breakout trend


        {"TRAILING_STOPLOSS": 0.10, "PROFIT_TARGET": 0.10, "MAX_HOLD_DAYS": 20},  # Fast mean reversion
        {"TRAILING_STOPLOSS": 0.05, "PROFIT_TARGET": 0.15, "MAX_HOLD_DAYS": 15},  # Ultra tight scalping
    ]

    run_exit_strategies(df_signals, tickers, exit_conditions_list, trades_dir, results_dir)
