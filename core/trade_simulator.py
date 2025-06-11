# trade_simulator.py
import pandas as pd

def simulate_trades(df, ticker, trailing_stop=0.1, profit_target=0.08, max_hold_days=30):
    df = df.copy()
    df = df[df['signal'] == 1].sort_values("Date").reset_index(drop=True)

    trades = []
    i = 0
    while i < len(df):
        row = df.iloc[i]
        entry_date = pd.to_datetime(row["Date"])
        entry_price = row["Close"]

        exit_price = None
        exit_date = None
        exit_reason = None

        window = df[(pd.to_datetime(df["Date"]) > entry_date) &
                    (pd.to_datetime(df["Date"]) <= entry_date + pd.Timedelta(days=max_hold_days))]

        for _, future_row in window.iterrows():
            high = future_row["High"]
            low = future_row["Low"]
            current_date = pd.to_datetime(future_row["Date"])

            if high >= entry_price * (1 + profit_target):
                exit_price = entry_price * (1 + profit_target)
                exit_date = current_date
                exit_reason = "TARGET"
                break
            elif low <= entry_price * (1 - trailing_stop):
                exit_price = entry_price * (1 - trailing_stop)
                exit_date = current_date
                exit_reason = "STOPLOSS"
                break

        if exit_price is None:
            exit_row = window.iloc[-1] if not window.empty else row
            exit_price = exit_row["Close"]
            exit_date = pd.to_datetime(exit_row["Date"])
            exit_reason = "MAX_HOLD"

        pnl_pct = 100 * (exit_price - entry_price) / entry_price
        holding_days = (exit_date - entry_date).days

        trades.append({
            "Ticker": ticker,
            "Entry Date": entry_date,
            "Exit Date": exit_date,
            "Entry Price": entry_price,
            "Exit Price": exit_price,
            "P&L %": pnl_pct,
            "Exit Reason": exit_reason,
            "holding_days": holding_days
        })

        # Move pointer past the exit date to avoid overlapping trades
        i += 1
        while i < len(df) and pd.to_datetime(df.iloc[i]["Date"]) <= exit_date:
            i += 1

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        # Ensure required columns exist for downstream code
        trades_df = pd.DataFrame(columns=[
            "Ticker", "Entry Date", "Exit Date", "Entry Price",
            "Exit Price", "P&L %", "Exit Reason", "holding_days"
        ])

    return trades_df
