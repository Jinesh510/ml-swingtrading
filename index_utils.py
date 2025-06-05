
# index_utils.py

import pandas as pd
import pandas_ta as ta

from config import TARGET_TYPE

# def compute_index_features(df):
#     df["Date"] = pd.to_datetime(df["Date"])
#     df["rsi_14"] = ta.rsi(df["Close"], length=14)
#     df["ret_3d"] = df["Close"].pct_change(3)
#     df["ret_5d"] = df["Close"].pct_change(5)
#     df["volatility_5d"] = df["Close"].rolling(5).std()
#     # df["momentum"] = (df["Close"] - df["Close"].rolling(20).max()) / df["Close"].rolling(20).max()
#     # print(len(df))
#     # print("Unique dates in df:", df["Date"].nunique())
#     df = df.dropna().reset_index(drop=True)
#     # print(len(df))
#     return df


# def compute_index_features(df):
#     df = df.copy()
#     df["Date"] = pd.to_datetime(df["Date"])
#     df = df.sort_values("Date").reset_index(drop=True)

#     # Use pandas_ta for RSI
#     df["rsi_14"] = ta.rsi(df["Close"], length=14)

#     # Return and volatility features
#     df["ret_10d"] = df["Close"].pct_change(10)
#     df["ret_15d"] = df["Close"].pct_change(15)
#     df["volatility_10d"] = df["Close"].rolling(10).std()

#     df = df.dropna().reset_index(drop=True)
#     return df

# for midcaps
def compute_index_features(df):
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    
    if "5d" in TARGET_TYPE:
        df["ret_5d"] = df["Close"].pct_change(5)
    elif "3d" in TARGET_TYPE:
        df["ret_3d"] = df["Close"].pct_change(3)
    elif "10d" in TARGET_TYPE:
        df["ret_10d"] = df["Close"].pct_change(10)

    df["volatility_10d"] = df["Close"].pct_change(1).rolling(10).std()
    df["rsi_14"] = ta.rsi(df["Close"], length=14)
    
    return df.dropna().reset_index(drop=True)


#for small caps
# def compute_index_features(df):
#     df = df.copy()
#     df["Date"] = pd.to_datetime(df["Date"])
#     df = df.sort_values("Date").reset_index(drop=True)

#     # Short-term return
#     df["ret_3d"] = df["Close"].pct_change(3)

#     # Market volatility (rolling std of 1d returns)
#     df["volatility_10d"] = df["Close"].pct_change(1).rolling(10).std()

#     # Optional: Market sentiment
#     df["rsi_14"] = ta.rsi(df["Close"], length=14)

#     df = df.dropna().reset_index(drop=True)
#     return df



