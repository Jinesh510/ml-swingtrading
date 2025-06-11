# feature_generator.py
import pandas as pd
import pandas_ta as ta

from core.config import TARGET_TYPE
from core.index_utils import load_vix_index

def generate_features(df):
    df = df.sort_values("Date").copy()

    # Price returns
    df["return_1d"] = df["Close"].pct_change(1)
    df["return_3d"] = df["Close"].pct_change(3)
    df["return_5d"] = df["Close"].pct_change(5)
    if "10d" in TARGET_TYPE:
        df["return_10d"] = df["Close"].pct_change(10).shift(-10)

    # Volatility
    df["volatility_5d"] = df["Close"].rolling(5).std()
    df["volatility_rank_5d"] = df["volatility_5d"].rank(pct=True)

    #adding vix
    vix_df = load_vix_index()
    # print("Vix head:",vix_df.head())
    vix_df["vix_lag1"] = vix_df["vix"].shift(1)
    df = df.merge(vix_df[["Date", "vix_lag1"]], on="Date", how="left")
    df["vix_lag1"] = df["vix_lag1"].fillna(method="ffill")

    # Technical indicators
    df["rsi_14"] = ta.rsi(df["Close"], length=14)
    df["macd_hist"] = ta.macd(df["Close"])["MACDh_12_26_9"]
    df["sma_5"] = df["Close"].rolling(5).mean()
    df["sma_20"] = df["Close"].rolling(20).mean()
    df["sma_ratio"] = df["sma_5"] / df["sma_20"]

    df["ema_5"] = df["Close"].ewm(span=5, adjust=False).mean()
    df["ema_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["ema_ratio"] = df["ema_5"] / df["ema_20"]

    
    df["ema_diff_lag1"] = df["ema_20"] - df["sma_20"]

    # Momentum/price structure
    df["breakout_5d"] = df["Close"] / df["High"].rolling(5).max()
    df["breakout_10d"] = df["Close"] / df["High"].rolling(10).max()
    df["price_trend"] = df["Close"].rolling(5).mean() / df["Close"]

    # Price action signals
    df["open_close_ratio"] = (df["Close"] - df["Open"]) / df["Open"]
    df["gap_pct"] = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)
    df["gap_up_1pct"] = (df["gap_pct"] > 0.01).astype(int)
    df["bullish_bar"] = ((df["Close"] > df["Open"]) & ((df["Close"] - df["Open"]) > 0.01 * df["Open"])).astype(int)
    df["3d_consistent_up"] = ((df["Close"] > df["Close"].shift(1)) & \
                                (df["Close"].shift(1) > df["Close"].shift(2)) & \
                                (df["Close"].shift(2) > df["Close"].shift(3))).astype(int)
    df["new_high_5d"] = (df["Close"] > df["Close"].shift(1).rolling(5).max()).astype(int)

    # Reversal patterns
    df["prev_day_reversal"] = ((df["Close"].shift(1) < df["Open"].shift(1)) & \
                                (df["Close"] > df["Open"]) & \
                                (df["Close"] > df["Close"].shift(1))).astype(int)
    df["strong_reversal_candle"] = ((df["Close"] > df["Open"]) & \
                                     ((df["High"] - df["Close"]) < 0.005 * df["Close"]))

    # Relative strength
    # df["rel_5d_strength"] = df["return_5d"] - df["sector_ret_10d"]
    df["rank_return_5d"] = df["return_5d"].rank(pct=True)
    df["zscore_return_5d"] = (df["return_5d"] - df["return_5d"].rolling(20).mean()) / (df["return_5d"].rolling(20).std() + 1e-6)

    # Composite features
    df["momentum_score"] = df["return_5d"] * df["ema_ratio"]
    df["vol_adjusted_return"] = df["return_5d"] / (df["volatility_5d"] + 1e-6)

    # Volume features
    if "Volume" in df.columns:
        df["volume_ratio_5d"] = df["Volume"] / df["Volume"].rolling(5).mean()
        df["volume_ratio_5d_x_volatility_5d"] = df["volume_ratio_5d"] * df["volatility_5d"]
        df["vol_adj_bar"] = (df["Close"] > df["Open"]) * df["Volume"].rolling(3).mean()


    # Price ratios
    df["high_close_ratio"] = df["High"] / df["Close"]
    df["low_close_ratio"] = df["Low"] / df["Close"]
    df["true_range"] = df["High"] - df["Low"]
    df["price_range_ratio"] = df["true_range"] / df["Close"]
    df["body_size_ratio"] = (df["Close"] - df["Open"]).abs() / df["true_range"]

    # Interaction terms
    df["rsi_14_x_return_1d"] = df["rsi_14"] * df["return_1d"]

    # Identify and drop constant feature columns only (exclude 'Date', 'ticker', etc.)
    meta_cols = ["Date", "ticker", "Open", "High", "Low", "Close", "Volume"]
    drop_cols = [col for col in df.columns if col not in meta_cols and df[col].nunique() <= 1]
    df = df.drop(columns=drop_cols)

    return df
