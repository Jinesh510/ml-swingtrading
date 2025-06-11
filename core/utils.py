from core.feature_generator import generate_features
from core.index_utils import compute_index_features, compute_sector_features, get_sector_for_bucket, load_vix_index
import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import joblib
import pandas_ta as ta
from core.config import CAP_TYPE, FEATURE_COLS, LIGHTGBM_PARAMS, MAX_HOLD_DAYS, MODEL_DIR, PEER_NAME, PROFIT_TARGET, SIGNAL_MODE, TARGET_TYPE, THRESHOLD, TOP_K, TRAILING_STOPLOSS
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def get_feature_columns(target_type = TARGET_TYPE, include_interactions=True, include_index_features=True, include_index_and_sector=True):
    base_features = [
        "return_1d", "return_3d", "return_5d",
        "sma_ratio", "ema_ratio",
        "volatility_5d", "volatility_rank_5d",
        "vix_lag1", "breakout_5d", "open_close_ratio","gap_up_1pct",
        "momentum_score", "vol_adjusted_return","price_trend",
        "breakout_10d", "vol_adj_bar", 
        "ema_diff_lag1", "prev_day_reversal", "bullish_bar","3d_consistent_up","new_high_5d",
        "strong_reversal_candle",
        "rank_return_5d", "zscore_return_5d", 
        "rsi_14", "macd_hist","high_close_ratio","low_close_ratio","true_range"
    ]


    interaction_features = [
        "rsi_14_x_return_1d",
        "volume_ratio_5d_x_volatility_5d",
        "macd_hist_x_zscore_5d",
        "price_range_ratio",
        "body_size_ratio"
    ]

    index_features = ["nifty_ret_10d"]


    # if include_index_and_sector:
#     #     base_features += [
#     #         # "nifty_ret_3d", 
#     #         "nifty_volatility_10d", 
#     #         "nifty_rsi_14",
#     #         # "sector_ret_3d", 
#     #         "sector_volatility_10d", 
#     #         "sector_rsi_14"
#     #     ]

#     #     if "5d" in target_type:
#     #         base_features += ["nifty_ret_5d","sector_ret_5d"]
#     #     elif "3d" in target_type:
#     #         base_features += ["nifty_ret_3d","sector_ret_3d"]
#     #     elif "10d" in target_type:
#     #         base_features += ["nifty_ret_10d","sector_ret_10d"]


    final_features = base_features
    if include_interactions:
        final_features += interaction_features
    if include_index_features:
        final_features += index_features

    final_features = ['vix_lag1',
    'ema_ratio',
    'nifty_ret_10d',
    'macd_hist',
    'volatility_5d',
    'price_range_ratio',
    'rsi_14',
    'breakout_10d',
    'macd_hist_x_zscore_5d',
    'gap_up_1pct',
    'volume_ratio_5d_x_volatility_5d',
    'open_close_ratio']

    return final_features


def drop_feature_columns(target_type):
    cols_to_ignore = ["Date", "signal", "target", "rank", 
        "pred_prob", "pred_value", "final_rank",
        "market_regime"]
    
    more_cols_to_ignore = ["true_range","price_range_ratio",
                           "volatility_5d","volatility_rank_5d",
                           "vol_adjusted_return","volume_ratio_5d_x_volatility_5d"]
    # cols_to_ignore += more_cols_to_ignore

    leakage_cols = []
    if target_type.startswith("3d"):
        leakage_cols.append("return_3d")
    elif "5d" in target_type:
        leakage_cols.append("return_5d")
        if "sector" in target_type:
            leakage_cols.append("sector_ret_5d")

    elif "10d" in target_type:
        leakage_cols.append("return_10d")
        if "sector" in target_type:
            leakage_cols.append("sector_ret_10d")  # if target is relative to this
    
    cols_to_ignore += leakage_cols

    return cols_to_ignore


def load_index_data(index_name):
    filepath = os.path.join("data", "index_eod", f"{index_name}.csv")
    if os.path.exists(filepath):
        df = pd.read_csv(filepath, parse_dates=["Date"])
        return df
    return pd.DataFrame()

def load_sector_data(sector_name):
    filepath = os.path.join("data", "sector_eod", f"{sector_name}.csv")
    if os.path.exists(filepath):
        df = pd.read_csv(filepath, parse_dates=["Date"])
        return df
    return pd.DataFrame()

def load_index_features(index_name):
    filepath = os.path.join("data", "index_eod", f"{index_name}.csv")
    if os.path.exists(filepath):
        df = pd.read_csv(filepath, parse_dates=["Date"])
        return compute_index_features(df)
    return pd.DataFrame()
def load_sector_features(sector_name):
    filepath = os.path.join("data", "sector_eod", f"{sector_name}.csv")
    if os.path.exists(filepath):
        df = pd.read_csv(filepath, parse_dates=["Date"])
        return compute_sector_features(df)
    return pd.DataFrame()


def tag_market_regime(df):
    df = df.copy()
    df["market_regime"] = "neutral"

    df.loc[(df["nifty_ret_10d"] > 0.01) & (df["vix_lag1"] < 15), "market_regime"] = "bull"
    df.loc[(df["nifty_ret_10d"] < -0.01) | (df["vix_lag1"] > 20), "market_regime"] = "bear"

    return df

def merge_index_features(index_name,df):

    nifty = load_index_features(index_name)
    if not nifty.empty:
        df = df.merge(nifty.add_prefix("nifty_"), left_on="Date", right_on="nifty_Date", how="left").drop(columns=["nifty_Date"])
        df = df.drop(columns=["nifty_Open","nifty_High","nifty_Low","nifty_Close"])
        return df
    return df


def merge_sector_features(bucket_name,df):
    sector = get_sector_for_bucket(bucket_name)

    if sector:
        sector_df = load_sector_features(sector)
        if not sector_df.empty:
            sector_df["Date"] = pd.to_datetime(sector_df["Date"])
            df = df.merge(sector_df.add_prefix("sector_"), left_on="Date", right_on="sector_Date", how="left").drop(columns=["sector_Date"])
            df = df.drop(columns=["sector_Open","sector_High","sector_Low","sector_Close"])
            df["rel_return_10d"] = df["return_10d"] - df["sector_ret_10d"]
            return df
        return df
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


def save_model(model,feature_cols, model_dir, bucket_name, model_type):
    os.makedirs(os.path.join(model_dir, bucket_name), exist_ok=True)
    outpath = os.path.join(model_dir, bucket_name, f"model_{model_type}.pkl")
    joblib.dump((model,feature_cols), outpath)



def load_and_merge_data(tickers, ohlcv_path, start_date, bucket_name):
    dfs = []
    valid_tickers = []
    exclude_tickers = ["OFSS"]


    for ticker in tickers:

        print("Ticker:",ticker)

        # if ticker in exclude_tickers:
        #     print(f"⚠️ Skipping {ticker} due to extreme class imbalance")
        #     continue

        file_path = os.path.join(ohlcv_path, f"{ticker}.csv")
        if not os.path.exists(file_path):
            continue
        df = pd.read_csv(file_path, parse_dates=["Date"])

        if df["Date"].min() > pd.Timestamp(start_date):
            print(f"❌ Skipping {ticker}: data starts after {start_date}")
            continue  # Skip tickers with insufficient history

        

        df["ticker"] = ticker
        df = generate_features(df)

        if "ticker" not in df.columns:
            print(f"⚠️ Skipping {ticker}: 'ticker' column missing after feature generation")
            continue

        dfs.append(df)
        print(f"✅ Added {ticker} to dfs")
        valid_tickers.append(ticker)
    print(f"✅ Added {len(valid_tickers)} tickers to dfs")
    if not dfs:
        raise ValueError("No valid tickers with sufficient history found!")

    df_all = pd.concat(dfs).sort_values(["ticker", "Date"])
    df_all = df_all[df_all["Date"] >= start_date]

    # Add sector index
    df_all = merge_sector_features(bucket_name,df_all)

    # Add NIFTY index
    df_all = merge_index_features("NIFTY",df_all)

    print(df_all.head())

    return df_all

