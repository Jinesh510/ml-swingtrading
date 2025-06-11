
# index_utils.py

import os
import pandas as pd
import pandas_ta as ta

from core.config import TARGET_TYPE


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

def compute_sector_features(df):
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    
    if "5d" in TARGET_TYPE:
        if "sector" in TARGET_TYPE:
            df["ret_5d"] = df["Close"].pct_change(5).shift(-5)
        else:
            df["ret_5d"] = df["Close"].pct_change(5)

    elif "3d" in TARGET_TYPE:
        if "sector" in TARGET_TYPE:
            df["ret_3d"] = df["Close"].pct_change(3).shift(-3)
        else:
            df["ret_3d"] = df["Close"].pct_change(3)
    elif "10d" in TARGET_TYPE:
        if "sector" in TARGET_TYPE:
            df["ret_10d"] = df["Close"].pct_change(10).shift(-10)
        else:
            df["ret_10d"] = df["Close"].pct_change(10)

    df["volatility_10d"] = df["Close"].pct_change(1).rolling(10).std()
    df["rsi_14"] = ta.rsi(df["Close"], length=14)
    
    return df.dropna().reset_index(drop=True)


def load_vix_index():
    vix_df = pd.read_csv("india_vix.csv", parse_dates=["Date"])
    vix_df["Date"] = pd.to_datetime(vix_df["Date"], dayfirst=True)
    vix_df = vix_df.sort_values("Date").drop_duplicates(subset="Date")
    vix_df = vix_df.reset_index(drop=True)
    df = vix_df[["Date", "vix"]]
    return df


def get_sector_for_bucket(bucket_name, sector_map_file="data/bucket_sector_index_mapping.csv"):
    try:
        df_map = pd.read_csv(sector_map_file)
        row = df_map[df_map["Bucket"].str.upper() == bucket_name.upper()]
        return row["Sector"].values[0] if not row.empty else None
    except:
        return None



