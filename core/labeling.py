# labeling.py
import pandas as pd
import pandas_ta as ta

def label_prob_3pct_gain(df):
    df = df.sort_values("Date").reset_index(drop=True)
    df["target"] = 0
    # for i in range(len(df)):
    #     window = df["Close"].iloc[i+1:i+11]  # next 10 days
    #     entry_price = df["Close"].iloc[i]
    #     if (window >= entry_price * 1.07).any():
    #         df.at[i, "target"] = 1
    
    df["target"] = (df["return_10d"] > 0.03).astype(int)

    # Diagnostic print
    if "ticker" in df.columns:
        print("\n📊 Target class balance per ticker:")
        print(df.groupby("ticker")["target"].mean().sort_values())
    return df

def label_excess_return_regression(df):
    df = df.copy()
    df = df.dropna(subset=["return_10d", "sector_ret_10d"])
    df["target"] = df["return_10d"] - df["sector_ret_10d"]
    return df

import pandas_ta as ta
import pandas as pd

def label_atr_based_gain(df: pd.DataFrame, atr_multiplier: float = 2.0, min_pct_gain: float = 0.07) -> pd.DataFrame:
    """
    Labels a row as target=1 if price in next 10 days exceeds:
    max(Close + ATR * multiplier, Close * (1 + min_pct_gain))

    Parameters:
        atr_multiplier (float): Multiplier for 10-day ATR threshold
        min_pct_gain (float): Absolute minimum % gain floor (e.g., 0.07 for 7%)
    """
    df = df.sort_values("Date").reset_index(drop=True)
    df["target"] = 0

    # Compute true 10-day ATR
    df["atr_10d"] = ta.atr(high=df["High"], low=df["Low"], close=df["Close"], length=10)

    for i in range(len(df)):
        entry_price = df.loc[i, "Close"]
        atr = df.loc[i, "atr_10d"]

        if pd.isna(atr):
            continue

        threshold = max(entry_price + atr * atr_multiplier, entry_price * (1 + min_pct_gain))
        future_window = df["Close"].iloc[i+1:i+11]

        if (future_window >= threshold).any():
            df.at[i, "target"] = 1

    # Diagnostic print
    if "ticker" in df.columns:
        print("\n📊 Target class balance per ticker:")
        print(df.groupby("ticker")["target"].mean().sort_values())

    return df


