# regime_filter.py

import pandas as pd

def tag_market_regime(df):
    df = df.copy()
    df["market_regime"] = "neutral"
    df.loc[(df["nifty_ret_10d"] > 0.01) & (df["vix_lag1"] < 15), "market_regime"] = "bull"
    df.loc[(df["nifty_ret_10d"] < -0.01) | (df["vix_lag1"] > 20), "market_regime"] = "bear"
    return df

def apply_regime_based_filtering(df):
    df = df.copy()
    df["signal"] = 0

    bull_mask = df["market_regime"] == "bull"
    bear_mask = df["market_regime"] == "bear"
    neutral_mask = df["market_regime"] == "neutral"

    df.loc[bull_mask & (df["final_rank"] >= 0.85), "signal"] = 1
    df.loc[neutral_mask & (df["final_rank"] >= 0.90), "signal"] = 1
    df.loc[bear_mask & (df["final_rank"] >= 0.98), "signal"] = 1

    return df
