import pandas as pd
import numpy as np
import os
def analyze_trades(df, ticker, output_file=None):
    if df.empty:
        print(f"No trades found for {ticker}")
        return df, {}
    num_trades = len(df)
    hit_rate = (df["P&L %"] > 0).sum() / num_trades
    daily_returns = df["P&L %"] / df["holding_days"]
    sharpe_ratio = (
        daily_returns.mean() / daily_returns.std() * np.sqrt(252)
        if daily_returns.std() != 0 else 0
    )
    df["Cumulative Equity"] = (1 + df["P&L %"] / 100).cumprod()
    peak = df["Cumulative Equity"].cummax()
    drawdown = 1 - df["Cumulative Equity"] / peak
    max_drawdown = drawdown.max()
    # start_date = pd.to_datetime(df["Entry Date"]).iloc[0]
    # end_date = pd.to_datetime(df["Exit Date"]).iloc[-1]
    start_date = pd.to_datetime(df["Entry Date"]).min()
    end_date = pd.to_datetime(df["Exit Date"]).max()
    print(f"{ticker} start : {start_date}, end : {end_date}")
    num_years = (end_date - start_date).days / 365.25
    print(f"{ticker} num years: {num_years}")
    final_return = df["Cumulative Equity"].iloc[-1]
    cagr = (final_return) ** (1 / num_years) - 1 if num_years > 0 else 0
    if output_file:
        df.to_csv(output_file, index=False)
    print(f"\n📊 Exit Reason Distribution:\n{df['exit_reason'].value_counts()}")
    print(f"📅 Avg Holding Days: {df['holding_days'].mean():.2f}")
    print(f"📈 Avg for winners: {df[df['P&L %'] > 0]['holding_days'].mean():.2f}")
    print(f"📉 Avg for losers: {df[df['P&L %'] < 0]['holding_days'].mean():.2f}")
    return df, {
        "ticker": ticker,
        "num_trades": num_trades,
        "hit_rate": round(hit_rate, 2),
        "sharpe_ratio": round(sharpe_ratio, 2),
        "max_drawdown": round(max_drawdown, 2),
        "cagr": round(cagr, 2),
        "trades_df": df
    }
