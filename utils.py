from index_utils import compute_index_features, compute_sector_features, load_vix_index
import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import joblib
import pandas_ta as ta
from config import CAP_TYPE, FEATURE_COLS, LIGHTGBM_PARAMS, MAX_HOLD_DAYS, MODEL_DIR, PEER_NAME, PROFIT_TARGET, SIGNAL_MODE, TARGET_TYPE, THRESHOLD, TOP_K, TRAILING_STOPLOSS
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from config import (
#     TRAILING_STOPLOSS,
#     MAX_HOLD_DAYS,
#     EXIT_SMA_PERIOD,
#     EXIT_RSI_PERIOD,
#     EXIT_RSI_THRESHOLD
# )

def load_stock_data(filepath):
    df = pd.read_csv(filepath, parse_dates=["Date"])
    df = df.sort_values("Date").drop_duplicates("Date")
    return df

def load_processed_data(ticker):
    filepath = os.path.join("data", "nse_eod", f"{ticker}.csv")
    df = pd.read_csv(filepath, parse_dates=["Date"])
    df = df.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
    
    #only considering data greater than 2010 for model consistency
    df = df[df["Date"] >= "2010-01-01"].copy()

    # Add advanced features
    df["ticker"] = ticker

    # df = add_advanced_features(df, ticker=ticker)
    # df = add_peer_relative_features(df)
    # df = tag_market_regime(df)

    
    # lagged = generate_lagged_features(df)
    # df = pd.concat([df, lagged], axis=1)

    # df = add_enriched_features(df,ticker=ticker)
    # print(df.head())
    # # Merge NIFTY index features
    # nifty = load_index_features("NIFTY")
    # df = df.merge(nifty.add_prefix("nifty_"), left_on="Date", right_on="nifty_Date", how="left").drop(columns=["nifty_Date"])
    # df = df.drop(columns=["nifty_Open","nifty_High","nifty_Low","nifty_Close"])

    # # Merge sector index features
    # sector_df = load_sector_features(ticker)
    # print(sector_df.head())
    # df = df.merge(sector_df.add_prefix("sector_"), left_on="Date", right_on="sector_Date", how="left").drop(columns=["sector_Date"])
    # df = df.drop(columns=["sector_Open","sector_High","sector_Low","sector_Close"])

    #new code for 10d target
    df = generate_clean_features(df)
    df = merge_index_features("NIFTY",df)
    df = merge_sector_features(ticker,df)

    
    # Add target column based on current TARGET_TYPE
    # df = get_target_definition(df, target_type=TARGET_TYPE)

    #multi-class target variable changes
    # df = label_multiclass_excess_return(df)

    #regression model
    df = label_excess_return_regression(df)

    # binary classifier model
    # df = label_prob_3pct_gain(df)

    df = df.dropna().reset_index(drop=True)
    return df


def compute_features(df):
    df["rsi_14"] = ta.rsi(df["Close"], length=14)
    df["sma_20"] = df["Close"].rolling(20).mean()
    df["sma_50"] = df["Close"].rolling(50).mean()
    df = df.dropna().reset_index(drop=True)
    return df
def get_target_definition(df, target_type="3d_1pct"):
    if target_type == "3d_1pct":
        df["target"] = (df["return_3d"] > 0.01).astype(int)
    elif target_type == "3d_2pct":
        df["target"] = (df["return_3d"] > 0.02).astype(int)
    elif target_type == "3d_3pct":
        df["target"] = (df["return_3d"] > 0.03).astype(int)
    elif target_type == "3d_5pct":
        df["target"] = (df["return_3d"] > 0.05).astype(int)
    elif target_type == "5d_vs_sector":
        df["target"] = (df["return_5d"] > df["sector_ret_5d"] + 0.02).astype(int)    
    elif target_type == "10d_vs_sector":
        df["target"] = (df["return_10d"] > df["sector_ret_10d"] + 0.01).astype(int)
    elif target_type == "quantile_top25":
        q75 = df["return_10d"].quantile(0.75)
        df["target"] = (df["return_10d"] > q75).astype(int)
    elif target_type == "quantile_top10":
        q90 = df["return_10d"].quantile(0.90)
        df["target"] = (df["return_10d"] > q90).astype(int)
    else:
        raise ValueError(f"Unsupported target type: {target_type}")
    return df
def get_feature_columns(target_type=TARGET_TYPE, cap_type=CAP_TYPE, peer_name=None,include_index_and_sector=True):
    base_features = [
        "return_1d", 
        "return_3d", 
        "atr_pct", 
        "bb_width",
        "volume_zscore", 
        # "rsi_14", 
        # "macd_hist",
        "price_vs_sma_10",
        "breakout_flag_5d"
    ]

    base_features = [
        
        "return_1d", "return_3d", "return_5d",
        "sma_ratio", "ema_ratio", "price_trend",
        "volatility_5d", "volatility_rank_5d",
        "vix_lag1",
        # New breakout/momentum features
        "breakout_5d",         # Close relative to recent high
        "open_close_ratio",    # Intra-day strength
        "gap_up_1pct",         # Gap-up flag
        "bullish_bar",          # Bullish candle

         # 🟩 New pattern signals
        "3d_consistent_up",
        "new_high_5d",

        # 🟪 Interaction features
        "momentum_score",           # return_5d * ema_ratio
        "vol_adjusted_return",      # return_5d / volatility_5d
        
        #newer
        "breakout_10d",
        "rel_5d_strength",        
        "vol_adj_bar",
        "ema_diff_lag1",

        #index features
        "nifty_ret_10d",

        # Reversal signals (optional)
        "prev_day_reversal", 
        "strong_reversal_candle",

        #cross-sectional features
        "rank_return_5d", 
        "zscore_return_5d",

        #basic technical indicators
        "rsi_14",
        "macd_hist"



    ]

    interaction_features = [
        "rsi_14_x_return_1d",
        "volume_ratio_5d_x_volatility_5d",
        "macd_hist_x_zscore_5d",
        "price_range_ratio",
        "body_size_ratio"
    ]
    
    # base_features += interaction_features

    advanced_technical_features = [
                    "price_to_vwap",
                    "price_to_sma_20",
                    "return_5d",
                    "return_20d",
                    "rsi_30",
                    # "macd_histogram",
                    "bb_position",
                    "volume_ratio",
                    "gap"
                ]

    enriched_features = [
        # 🔸 Fundamental Proxies
        "price_momentum_60d",
        "price_momentum_120d",
        "volatility_rank",
        "volume_trend_30d",

        # 🔸 Market Regime Features
        "market_volatility",
        "volatility_regime",
        "market_momentum_20d",
        "market_trend",
        "market_rsi",

        # 🔸 Relative to NIFTY
        "relative_return_1d",
        "relative_return_5d",
        "relative_return_20d",
        "beta_60d",

        # 🔸 Relative to Sector
        "relative_sector_return_1d",
        "relative_sector_return_5d",
        "relative_sector_return_20d",
    ]

    lagged_features = [
        "Close_lag1", "Close_lag2", "Close_lag3",
        "rsi_14_lag1", "rsi_14_lag2", "rsi_14_lag3",
        "macd_hist_lag1", "macd_hist_lag2", "macd_hist_lag3"
    ]
    
    # base_features = base_features + advanced_technical_features + enriched_features
    # base_features += lagged_features 

    # if include_index_and_sector and cap_type in ["midcap", "largecap"]:
    #     base_features += [
    #         # "nifty_ret_3d", 
    #         "nifty_volatility_10d", 
    #         "nifty_rsi_14",
    #         # "sector_ret_3d", 
    #         "sector_volatility_10d", 
    #         "sector_rsi_14"
    #     ]

    #     if "5d" in TARGET_TYPE:
    #         base_features += ["nifty_ret_5d","sector_ret_5d"]
    #     elif "3d" in TARGET_TYPE:
    #         base_features += ["nifty_ret_3d","sector_ret_3d"]
    #     elif "10d" in TARGET_TYPE:
    #         base_features += ["nifty_ret_10d","sector_ret_10d"]


    


    if peer_name is None:
        peer_name = []
    else:
        base_features += [
            f"{peer_name}_return_3d",
            f"{peer_name}_volume_zscore",
            f"{peer_name}_price_vs_sma_10",
            f"{peer_name}_breakout_flag_5d"
        ]

    ## experiment overriding everything 
    # base_features =[
    #                 'market_volatility',
    #                 'nifty_volatility_10d',
    #                 'market_momentum_20d',
    #                 'sector_rsi_14',
    #                 'sector_ret_3d',
    #                 'sector_volatility_10d',
    #                 'market_rsi',
    #                 'nifty_ret_3d',
    #                 'nifty_rsi_14',
    #                 'atr_pct',
    #                 'volume_zscore',
    #                 'bb_position',
    #                 'price_vs_sma_10',
    #                 'rel_ret_3d',
    #                 'rsi_14_rank_within_peers',
    #                 'vix'

    # ]

    return base_features
def drop_target_related_columns(target_type):
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
    return leakage_cols

def train_lightgbm(df, ticker):
    df.fillna(method="ffill", inplace=True)
    df = df.sort_values("Date")


    feature_cols = get_feature_columns(target_type=TARGET_TYPE, cap_type=CAP_TYPE, peer_name=PEER_NAME)
    leakage_cols = drop_target_related_columns(target_type=TARGET_TYPE)
    # extra_leakage_cols = ["target", "sample_weight"]

    extra_ignore = ["Date", "pred_prob", "signal", "rank"]
    some_more_to_ignore = ["vix_lag1"]
    addl_cols_to_ignore = ["bullish_bar", "gap_up_1pct", "price_trend", "new_high_5d","3d_consistent_up", "rel_5d_strength"]
    cols_to_ignore = leakage_cols + extra_ignore 
    # cols_to_ignore += some_more_to_ignore
    cols_to_ignore += addl_cols_to_ignore
    # Categorical ticker (applies to both single and merged case)
    if "ticker" in df.columns:
        df["ticker"] = df["ticker"].astype("category")
        feature_cols.append("ticker")

    feature_cols= [x for x in feature_cols if x not in cols_to_ignore]
    
    X_train = df[feature_cols]
    y_train = df["target"]
    # sample_weights = df.get("sample_weight", pd.Series([1.0] * len(df)))

    print(f"📊 Features being used for {ticker}: {feature_cols}")
    # model = lgb.LGBMClassifier(**LIGHTGBM_PARAMS)

    # regression model
    model = lgb.LGBMRegressor(**LIGHTGBM_PARAMS)


    if "ticker" in feature_cols:
        model.fit(X_train, y_train ,categorical_feature=["ticker"])
    else:
        model.fit(X_train, y_train)
    # model.fit(X_train, y_train)
    # joblib.dump(model, os.path.join(MODEL_DIR, f"{ticker}.pkl"))
    # joblib.dump(model, os.path.join(MODEL_DIR, f"model_classifier.pkl"))
    joblib.dump(model, os.path.join(MODEL_DIR, f"model_regression.pkl"))

    print(f"✅ Trained with features: {X_train.columns.tolist()}")
    return model



def load_model(ticker):
    model_path = os.path.join(MODEL_DIR, f"{ticker}.pkl")
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

# def predict_signal(df, model):
#     # Predicts probability of class 1 using the model.
#     # Assumes model has a predict_proba method.
#     X = df[model.feature_name_]
#     df["pred_prob"] = model.predict_proba(X)[:, 1]

#     return df

## multi-class target variable changes
# def predict_signal(df, model):
#     """
#     Predicts class probabilities using a multiclass model.
#     Uses probability of class 4 (Strong Buy) as pred_prob.
#     """
#     X = df[model.feature_name_]
#     pred_probs = model.predict_proba(X)  # shape (n_samples, 5)
#     df["pred_class"] = np.argmax(pred_probs, axis=1)
#     df["pred_prob"] = pred_probs[:, 4]  # Use Strong Buy prob for ranking/confidence
#     return df


# regression model
def predict_signal(df, model):
    X = df[model.feature_name_]
    df["pred_value"] = model.predict(X)
    # df["signal"] = (df["pred_value"] > 0.02).astype(int)  # Predicting excess return > 2%

    return df

# binary classifier model

# def predict_signal(df, model):
#     X = df[model.feature_name_]
#     df["pred_prob"] = model.predict_proba(X)[:, 1]  # For binary classification
#     return df



def add_advanced_features(df,ticker):
    
    df["return_1d"] = df["Close"].pct_change(1)

    if "3d" in TARGET_TYPE:
        df["return_3d"] = df["Close"].pct_change(3).shift(-3)
    else:
        df["return_3d"] = df["Close"].pct_change(3)

    if "5d" in TARGET_TYPE:
        df["return_5d"] = df["Close"].pct_change(5).shift(-5)
    else:
        df["return_5d"] = df["Close"].pct_change(5)
 

    df["atr"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)
    df["atr_pct"] = df["atr"] / df["Close"]
    bb = ta.bbands(df["Close"], length=20)
    if bb is not None and not bb.empty:
        df["bb_width"] = bb["BBU_20_2.0"] - bb["BBL_20_2.0"]
    df["volume_avg_20"] = df["Volume"].rolling(20).mean()
    df["volume_zscore"] = (df["Volume"] - df["volume_avg_20"]) / df["volume_avg_20"]
    df["volume_surge_flag"] = (df["Volume"] > 1.2 * df["volume_avg_20"]).astype(int)
    df["rsi_14"] = ta.rsi(df["Close"], length=14)
    df["macd_hist"] = ta.macd(df["Close"])["MACDh_12_26_9"]
    df["sma_10"] = df["Close"].rolling(10).mean()
    df["price_vs_sma_10"] = df["Close"] / df["sma_10"]
    df["recent_high_5d"] = df["Close"].rolling(5).max()
    df["breakout_flag_5d"] = (df["Close"] > df["recent_high_5d"].shift(1)).astype(int)

    # add more features from claude
    # Returns
    # df["return_5d"] = df["Close"].pct_change(5)
    df["return_20d"] = df["Close"].pct_change(20)

    # Price to SMA
    df["sma_20"] = df["Close"].rolling(20).mean()
    df["price_to_sma_20"] = df["Close"] / df["sma_20"]

    # RSI 30
    df["rsi_30"] = ta.rsi(df["Close"], length=30)

    # MACD histogram
    macd_result = ta.macd(df["Close"])
    df["macd_histogram"] = macd_result["MACDh_12_26_9"]

    # Bollinger Band position
    bb_result = ta.bbands(df["Close"], length=20)
    df["bb_upper"] = bb_result["BBU_20_2.0"]
    df["bb_lower"] = bb_result["BBL_20_2.0"]
    df["bb_position"] = (df["Close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    # Volume ratio
    df["volume_sma_20"] = df["Volume"].rolling(20).mean()
    df["volume_ratio"] = df["Volume"] / df["volume_sma_20"]

    # VWAP and price to VWAP
    # Ensure proper datetime index for pandas_ta VWAP
    df = df.sort_values("Date")
    df.set_index("Date", inplace=True)
    df.index = pd.to_datetime(df.index)  # Explicitly ensure datetime

    # Now apply VWAP safely
    df["vwap"] = ta.vwap(df["High"], df["Low"], df["Close"], df["Volume"])

    # Optional: reset index back if you want
    df = df.reset_index()
    print(df.head())
    # df["vwap"] = ta.vwap(df["High"], df["Low"], df["Close"], df["Volume"])
    df["price_to_vwap"] = df["Close"] / df["vwap"]

    # Gap (previous close to current open)
    df["gap"] = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)

    #adding vix
    vix_df = load_vix_index()
    # print("Vix head:",vix_df.head())
    df = df.merge(vix_df, on="Date", how="left")
    df["vix"] = df["vix"].fillna(method="ffill")


    # add nifty and sector features
    nifty = load_index_features("NIFTY")
    if not nifty.empty:
        df = df.merge(nifty.add_prefix("nifty_"), left_on="Date", right_on="nifty_Date", how="left").drop(columns=["nifty_Date"])
        df = df.drop(columns=["nifty_Open","nifty_High","nifty_Low","nifty_Close"])
    sector = get_sector_for_stock(ticker)
    if sector:
        sector_df = load_sector_features(sector)
        if not sector_df.empty:
            sector_df["Date"] = pd.to_datetime(sector_df["Date"])
            df = df.merge(sector_df.add_prefix("sector_"), left_on="Date", right_on="sector_Date", how="left").drop(columns=["sector_Date"])
            df = df.drop(columns=["sector_Open","sector_High","sector_Low","sector_Close"])

    
    df = df.dropna().reset_index(drop=True)
    return df


def calculate_fundamental_proxies(self, df):
    """
    Calculate fundamental proxies from price data
    (In real implementation, you'd use actual fundamental data)
    """
    features = pd.DataFrame(index=df.index)
    
    # Price momentum as growth proxy
    features['price_momentum_60d'] = df['Close'].pct_change(60)
    features['price_momentum_120d'] = df['Close'].pct_change(120)
    
    # Volatility as risk proxy
    features['volatility_rank'] = df['Close'].pct_change().rolling(252).std().rank(pct=True)
    
    # Volume trend as institutional interest proxy
    features['volume_trend_30d'] = df['Volume'].rolling(30).mean() / df['Volume'].rolling(60).mean()
    
    return features


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


def add_enriched_features(df, ticker):
    """
    Add fundamental proxies, market regime, and relative features to stock DataFrame.
    Expects df, nifty_df, sector_df to have a 'Date' column and be sorted.
    """
    df = df.copy()
    df = df.sort_values("Date")
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)

    # --- Fundamental Proxies ---
    df["price_momentum_60d"] = df["Close"].pct_change(60)
    df["price_momentum_120d"] = df["Close"].pct_change(120)
    df["volatility_rank"] = df["Close"].pct_change().rolling(252).std().rank(pct=True)
    df["volume_trend_30d"] = df["Volume"].rolling(30).mean() / df["Volume"].rolling(60).mean()

    # --- Market Regime Features from NIFTY ---
    nifty_df = load_index_data("NIFTY")
    if not nifty_df.empty:
        nifty = nifty_df.copy()
        nifty = nifty.sort_values("Date")
        nifty["Date"] = pd.to_datetime(nifty["Date"])
        nifty.set_index("Date", inplace=True)

        nifty_returns = nifty["Close"].pct_change()
        market_volatility = nifty_returns.rolling(20).std() * np.sqrt(252)
        df["market_volatility"] = market_volatility
        df["volatility_regime"] = (market_volatility > market_volatility.rolling(252).median()).astype(int)
        df["market_momentum_20d"] = nifty["Close"].pct_change(20)
        df["market_trend"] = (nifty["Close"] > nifty["Close"].rolling(50).mean()).astype(int)
        df["market_rsi"] = ta.rsi(nifty["Close"], length=14)

        # Relative Features vs NIFTY
        stock_returns = df["Close"].pct_change()
        nifty_returns = nifty["Close"].pct_change()
        common_dates = stock_returns.index.intersection(nifty_returns.index)

        if not common_dates.empty:
            rel_ret = stock_returns[common_dates] - nifty_returns[common_dates]
            df.loc[common_dates, "relative_return_1d"] = rel_ret
            df["relative_return_5d"] = rel_ret.rolling(5).sum()
            df["relative_return_20d"] = rel_ret.rolling(20).sum()
            df["beta_60d"] = stock_returns.rolling(60).corr(nifty_returns)

    # --- Relative Features vs Sector ---
    sector_name = get_sector_for_stock(ticker)
    sector_df = load_sector_data(sector_name)
    if not sector_df.empty:
        sector_df = sector_df.copy()
        sector_df = sector_df.sort_values("Date")
        sector_df["Date"] = pd.to_datetime(sector_df["Date"])
        sector_df.set_index("Date", inplace=True)

        sector_returns = sector_df["Close"].pct_change()
        common_dates_sec = df.index.intersection(sector_returns.index)

        if not common_dates_sec.empty:
            rel_sec_ret = df["Close"].pct_change()[common_dates_sec] - sector_returns[common_dates_sec]
            df.loc[common_dates_sec, "relative_sector_return_1d"] = rel_sec_ret
            df["relative_sector_return_5d"] = rel_sec_ret.rolling(5).sum()
            df["relative_sector_return_20d"] = rel_sec_ret.rolling(20).sum()

    # df = df.reset_index()
    print(len(df))
    df = df.reset_index()
    df = df.dropna().reset_index(drop=True)
    print(len(df))

    return df



def get_sector_for_stock(ticker, sector_map_file="sector_mapper.csv"):
    try:
        df_map = pd.read_csv(sector_map_file)
        row = df_map[df_map["Stock"].str.upper() == ticker.upper()]
        return row["Sector"].values[0] if not row.empty else None
    except:
        return None
    

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

def apply_post_prediction_filters(df, prob_col="pred_prob", threshold=THRESHOLD):
    df = df.copy()
    # print(f"df len before post pred filter:{len(df)}")
    if "signal" not in df.columns:
        df["signal"] = (df[prob_col] >= threshold).astype(int)
    df["signal"] = df.apply(
        lambda row: 1 if (
            row["signal"] == 1 and
            row["atr_pct"] < 0.04 and
            row["volume_zscore"] > 0.5 and
            row["breakout_flag_5d"] == 1
        ) else 0, axis=1
    )
    # print(len(df))
    return df

def generate_signals(df, model, threshold=THRESHOLD, topk=TOP_K, signal_mode=SIGNAL_MODE, use_dynamic_threshold=False):
    df = predict_signal(df.copy(), model)
    if signal_mode == "topk":
        df["rank"] = df["pred_prob"].rank(method="first", ascending=False)
        df["signal"] = (df["rank"] <= topk).astype(int)
        # df["confidence_weight"] = 1.0  # uniform weight

    else:  # threshold mode
        # df["signal"] = (df["pred_prob"] >= threshold).astype(int)

        # Use dynamic threshold based on regime
        if use_dynamic_threshold:
            def get_dynamic_threshold(row):
                if row["market_regime"] == "bull":
                    return 0.45
                elif row["market_regime"] == "bear":
                    return 0.65
                else:
                    return 0.55

            df["threshold"] = df.apply(get_dynamic_threshold, axis=1)
            # df["signal"] = (df["pred_prob"] >= df["dynamic_threshold"]).astype(int)
        else:
            df["threshold"] = threshold
        
        #binary classifier
        # df["signal"] = (df["pred_prob"] >= df["threshold"]).astype(int)

        #multi-class target variable changes
        # Generate signal if predicted class is Buy (3) or Strong Buy (4) AND confidence > threshold
        # df["signal"] = ((df["pred_class"] >= 3) & (df["pred_prob"] >= df["threshold"])).astype(int)


        # Regression model logic
        df["signal"] = (df["pred_value"] >= threshold).astype(int)


        # df = apply_post_prediction_filters(df)

        # Confidence weight for signal days only
        # df["confidence_weight"] = 0.0
        # df.loc[df["signal"] == 1, "confidence_weight"] = (
        #     df["pred_prob"] - df["threshold"]
        # ) / df["atr_pct"]

    return df


def compute_peer_features(df_peer, peer_name):
    df_peer = df_peer.copy()
    df_peer["Date"] = pd.to_datetime(df_peer["Date"])
    df_peer = df_peer.sort_values("Date").reset_index(drop=True)
    df_peer[f"{peer_name}_return_3d"] = df_peer["Close"].pct_change(3)
    df_peer["volume_avg_20"] = df_peer["Volume"].rolling(20).mean()
    df_peer[f"{peer_name}_volume_zscore"] = (df_peer["Volume"] - df_peer["volume_avg_20"]) / df_peer["volume_avg_20"]
    df_peer[f"{peer_name}_sma_10"] = df_peer["Close"].rolling(10).mean()
    df_peer[f"{peer_name}_price_vs_sma_10"] = df_peer["Close"] / df_peer[f"{peer_name}_sma_10"]
    df_peer[f"{peer_name}_recent_high_5d"] = df_peer["Close"].rolling(5).max()
    df_peer[f"{peer_name}_breakout_flag_5d"] = (df_peer["Close"] > df_peer[f"{peer_name}_recent_high_5d"].shift(1)).astype(int)
    peer_features = df_peer[[
        "Date",
        f"{peer_name}_return_3d",
        f"{peer_name}_volume_zscore",
        f"{peer_name}_price_vs_sma_10",
        f"{peer_name}_breakout_flag_5d"
    ]].dropna().reset_index(drop=True)
    return peer_features

def merge_peer_features(df_main, df_peer_features):
    df_main = df_main.copy()
    df_main["Date"] = pd.to_datetime(df_main["Date"])
    df_merged = df_main.merge(df_peer_features, on="Date", how="left")
    return df_merged

def simulate_trades(df, ticker, trailing_stop=TRAILING_STOPLOSS, profit_target=PROFIT_TARGET, max_hold_days=MAX_HOLD_DAYS):
    # Simulates trades based on entry signals with trailing stoploss, profit target, and max hold.
    # Assumes signal column is binary.
    # import pandas as pd
    trades = []
    last_exit_index = -1
    for i in range(len(df)):
        if df.iloc[i]["signal"] == 1 and i > last_exit_index:
            entry_price = df.iloc[i]["Close"]
            entry_date = df.iloc[i]["Date"]
            max_price = entry_price
            exit_reason = "max_hold"
            exit_price = entry_price
            exit_date = entry_date
            for j in range(i + 1, min(i + max_hold_days, len(df))):
                current_price = df.iloc[j]["Close"]
                max_price = max(max_price, current_price)
                drawdown = (current_price - max_price) / max_price
                gain = (current_price - entry_price) / entry_price
                if gain >= profit_target:
                    exit_reason = "profit_target"
                    exit_price = current_price
                    exit_date = df.iloc[j]["Date"]
                    break
                elif drawdown <= -trailing_stop:
                    exit_reason = "trailing_stoploss"
                    exit_price = current_price
                    exit_date = df.iloc[j]["Date"]
                    break
            else:
                exit_price = df.iloc[j]["Close"]
                exit_date = df.iloc[j]["Date"]
            pnl_pct = ((exit_price - entry_price) / entry_price) * 100
            
            #confidence based allocation
            # weight = df.iloc[i].get("confidence_weight", 1.0)  # fallback = 1.0
            # weighted_pnl = pnl_pct * weight

            holding_days = (pd.to_datetime(exit_date) - pd.to_datetime(entry_date)).days
            trades.append({
                "Entry Date": entry_date,
                "Exit Date": exit_date,
                "Buy Price": round(entry_price, 2),
                "Sell Price": round(exit_price, 2),
                "P&L %": round(pnl_pct, 2),
                "holding_days": holding_days,
                "exit_reason": exit_reason
                # "confidence_weight": round(weight, 3),
                # "weighted_pnl": round(weighted_pnl, 3)
            })
            last_exit_index = j

    return pd.DataFrame(trades)


# def simulate_trades(df, ticker):
#     df = df.copy()
#     df["SMA_X"] = df["Close"].rolling(EXIT_SMA_PERIOD).mean()
#     df["RSI_X"] = ta.rsi(df["Close"], length=EXIT_RSI_PERIOD)

#     trades = []
#     last_exit_index = -1

#     for i in range(len(df)):
#         if df.iloc[i]["signal"] == 1 and i > last_exit_index:
#             entry_price = df.iloc[i]["Close"]
#             entry_date = df.iloc[i]["Date"]
#             max_price = entry_price
#             exit_price = entry_price
#             exit_date = entry_date
#             exit_reason = "max_hold"

#             for j in range(i + 1, min(i + MAX_HOLD_DAYS, len(df))):
#                 current_price = df.iloc[j]["Close"]
#                 sma = df.iloc[j]["SMA_X"]
#                 rsi = df.iloc[j]["RSI_X"]
#                 max_price = max(max_price, current_price)

#                 # 1️⃣ Trailing SL
#                 if (current_price - max_price) / max_price <= -TRAILING_STOPLOSS:
#                     exit_price = current_price
#                     exit_date = df.iloc[j]["Date"]
#                     exit_reason = "trailing_stoploss"
#                     break

#                 # 2️⃣ SMA breach
#                 elif current_price < sma:
#                     exit_price = current_price
#                     exit_date = df.iloc[j]["Date"]
#                     exit_reason = "sma_break"
#                     break

#                 # 3️⃣ RSI drop
#                 elif rsi < EXIT_RSI_THRESHOLD:
#                     exit_price = current_price
#                     exit_date = df.iloc[j]["Date"]
#                     exit_reason = "rsi_drop"
#                     break

#             else:
#                 exit_price = df.iloc[j]["Close"]
#                 exit_date = df.iloc[j]["Date"]

#             pnl_pct = ((exit_price - entry_price) / entry_price) * 100
#             holding_days = (pd.to_datetime(exit_date) - pd.to_datetime(entry_date)).days

#             trades.append({
#                 "Entry Date": entry_date,
#                 "Exit Date": exit_date,
#                 "Buy Price": round(entry_price, 2),
#                 "Sell Price": round(exit_price, 2),
#                 "P&L %": round(pnl_pct, 2),
#                 "holding_days": holding_days,
#                 "exit_reason": exit_reason
#             })

#             last_exit_index = j

#     return pd.DataFrame(trades)




def generate_lagged_features(df):
    """
    Generate lagged features for Close, RSI, MACD hist for last 3 days
    """
    lag_features = pd.DataFrame(index=df.index)
    base_cols = ["Close", "rsi_14", "macd_hist"]

    for col in base_cols:
        for lag in range(1, 4):
            col_name = f"{col}_lag{lag}"
            if col in df.columns:
                lag_features[col_name] = df[col].shift(lag)

    return lag_features

# def label_profitable_trades(df, trades_df):
#     """
#     Assigns target = 1 to entry dates of trades with positive P&L.
#     All other rows get target = 0.
#     """
#     df = df.copy()
#     df["Date"] = pd.to_datetime(df["Date"])
#     df["target"] = 0  # Default: not a good trade

#     # Label entries that resulted in profit
#     profitable_entries = trades_df[trades_df["P&L %"] > 0]["Entry Date"].unique()

#     df.loc[df["Date"].isin(profitable_entries), "target"] = 1

#     return df


def label_profitable_trades(df, trades_df):
    """
    Assigns target = 1 to rows where a trade was entered AND ended with meaningful profit (e.g. >= 2.0%) AND did not exit due to max_hold.
    All other rows get target = 0.
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["target"] = 0

    # qualifying_trades = trades_df[(trades_df["P&L %"] >= 0) & (trades_df["exit_reason"] != "max_hold")]
    # qualifying_dates = pd.to_datetime(qualifying_trades["Entry Date"].unique())

    profitable_entries = trades_df[trades_df["P&L %"] > 2.0]["Entry Date"].unique()
    # profitable_entries = pd.to_datetime(profitable_entries).normalize()


    df.loc[df["Date"].isin(profitable_entries), "target"] = 1
    return df



# def label_profitable_trades(df, trades_df):
#     """
#     Assigns target=1 to rows where a trade was entered and ended with positive P&L.
#     All other rows get target=0. Also assigns sample weights based on P&L magnitude.
#     """
#     df = df.copy()
#     df["Date"] = pd.to_datetime(df["Date"])
#     df["target"] = 0
#     df["sample_weight"] = 1.0

#     trades_df = trades_df.copy()
#     trades_df["Entry Date"] = pd.to_datetime(trades_df["Entry Date"])

#     # Join P&L back to signal dataframe
#     df = df.merge(
#         trades_df[["Entry Date", "P&L %"]],
#         left_on="Date",
#         right_on="Entry Date",
#         how="left"
#     )

#     # Labeling
#     df.loc[df["P&L %"] > 0, "target"] = 1

#     # Sample weights (for all rows with non-null P&L)
#     df.loc[df["P&L %"].notnull(), "sample_weight"] = 1 + (df["P&L %"] / 100).clip(lower=0)

#     df.drop(columns=["Entry Date"], inplace=True, errors="ignore")
#     return df


def apply_training_filters(df):
    return df[
        (df["atr_pct"] < 0.04) &
        (df["volume_zscore"] > 0.5) &
        (df["breakout_flag_5d"] == 1)
    ].copy()

def add_peer_relative_features(df):
    df = df.sort_values(["Date", "ticker"]).copy()

    # rel_ret_3d: 3-day return relative to peer median
    df["rel_ret_3d"] = df.groupby("Date")["return_3d"].transform(
        lambda x: x - x.median()
    )

    # RSI rank within peers
    df["rsi_14_rank_within_peers"] = df.groupby("Date")["rsi_14"].rank(pct=True)

    return df

def tag_market_regime(df):
    df = df.copy()
    df["market_regime"] = "neutral"

    df.loc[(df["nifty_ret_3d"] > 0.02) & (df["vix"] < 15), "market_regime"] = "bull"
    df.loc[(df["nifty_ret_3d"] < -0.02) | (df["vix"] > 18), "market_regime"] = "bear"

    return df

#### New code for 10d sector outperformance exp

def label_sector_outperformers_10d(df, threshold=0.03):
    """
    Labels target = 1 for stocks that outperform their sector by a margin in 10 days.
    Assumes 'return_10d' and 'sector_ret_10d' columns already exist.
    """
    df["target"] = (df["return_10d"] > df["sector_ret_10d"] + threshold).astype(int)
    return df

def generate_clean_features(df):
    """
    Clean, minimal feature set for merged stock training.
    Assumes 'close', 'sector', and 'return_1d' columns are available.
    """
    # Momentum features
    df["return_1d"] = df["Close"].pct_change(1)
    df["return_3d"] = df["Close"].pct_change(3)
    df["return_5d"] = df["Close"].pct_change(5)
    if "10d" in TARGET_TYPE:
        df["return_10d"] = df["Close"].pct_change(10).shift(-10)

    # Moving average ratios
    df["sma_5"] = df["Close"].rolling(5).mean()
    df["sma_20"] = df["Close"].rolling(20).mean()
    df["sma_ratio"] = df["sma_5"] / df["sma_20"]

    df["ema_5"] = df["Close"].ewm(span=5, adjust=False).mean()
    df["ema_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["ema_ratio"] = df["ema_5"] / df["ema_20"]

    df["price_trend"] = (df["ema_5"] > df["ema_20"]).astype(int)

    # Volatility features
    df["volatility_5d"] = df["Close"].rolling(5).std()
    df["volatility_rank_5d"] = df["volatility_5d"].rank(pct=True)

    #adding vix
    vix_df = load_vix_index()
    # print("Vix head:",vix_df.head())
    vix_df["vix_lag1"] = vix_df["vix"].shift(1)
    df = df.merge(vix_df[["Date", "vix_lag1"]], on="Date", how="left")
    df["vix_lag1"] = df["vix_lag1"].fillna(method="ffill")

    # Relative to recent high — breakout strength
    df["breakout_5d"] = df["Close"] / df["Close"].rolling(5).max()

    # Intra-day price strength
    df["open_close_ratio"] = df["Close"] / df["Open"]

    # Gap-up signal (1%+ gap up from previous close)
    df["gap_up_1pct"] = (df["Open"] > df["Close"].shift(1) * 1.01).astype(int)

    # Bullish bar: close > open and body > 1% of close
    df["bullish_bar"] = ((df["Close"] > df["Open"]) & 
                        ((df["Close"] - df["Open"]) > df["Close"] * 0.01)).astype(int)


    # Has the stock been closing higher for 3 days?
    df["3d_consistent_up"] = (
        (df["Close"] > df["Close"].shift(1)) &
        (df["Close"].shift(1) > df["Close"].shift(2)) &
        (df["Close"].shift(2) > df["Close"].shift(3))
    ).astype(int)

    # Is today's close highest of past 5 days?
    df["new_high_5d"] = (df["Close"] == df["Close"].rolling(5).max()).astype(int)

    df["momentum_score"] = df["return_5d"] * df["ema_ratio"]
    df["vol_adjusted_return"] = df["return_5d"] / (df["volatility_5d"] + 1e-6)


    # Price breakout strength
    df["breakout_10d"] = df["Close"] / df["Close"].rolling(10).max()

    # Relative strength compared to group
    df["rel_5d_strength"] = df["return_5d"] - df.groupby("Date")["return_5d"].transform("median")

    # Volume-adjusted signal (if available)
    df["vol_adj_bar"] = (df["Close"] > df["Open"]) * df["Volume"].rolling(3).mean()

    # Lagged signal as a proxy for trend continuation
    df["ema_diff_lag1"] = (df["ema_5"] - df["ema_20"]).shift(1)


    df["prev_day_reversal"] = ((df["Close"].shift(1) < df["Open"].shift(1)) &
                           (df["Close"] > df["Open"])).astype(int)

    df["strong_reversal_candle"] = ((df["Close"] > df["Open"]) & 
                                    ((df["Close"] - df["Open"]) > 2 * (df["Open"] - df["Low"]))).astype(int)


    # Technical indicators
    df["rsi_14"] = ta.rsi(df["Close"], length=14)
    macd = ta.macd(df["Close"])
    df["macd_hist"] = macd["MACDh_12_26_9"]  # MACD histogram (diff between macd and signal)


    # 🧠 Feature Interactions
    df["rsi_14_x_return_1d"] = df["rsi_14"] * df["return_1d"]
    
    df["volume_ratio_5d"] = df["Volume"] / df["Volume"].rolling(5).mean()

    df["volume_ratio_5d_x_volatility_5d"] = df["volume_ratio_5d"] * df["volatility_5d"]
    # df["macd_hist_x_zscore_5d"] = df["macd_hist"] * df["zscore_return_5d"]
    df["price_range_ratio"] = (df["High"] - df["Low"]) / df["Open"]
    df["body_size_ratio"] = abs(df["Close"] - df["Open"]) / (df["High"] - df["Low"] + 1e-6)







    # Drop rows with NaNs due to rolling calculations

    df.dropna(inplace=True)

    return df

def merge_index_features(index_name,df):

    nifty = load_index_features(index_name)
    if not nifty.empty:
        df = df.merge(nifty.add_prefix("nifty_"), left_on="Date", right_on="nifty_Date", how="left").drop(columns=["nifty_Date"])
        df = df.drop(columns=["nifty_Open","nifty_High","nifty_Low","nifty_Close"])
        return df
    return df

def merge_sector_features(ticker,df):
    sector = get_sector_for_stock(ticker)
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

def label_multiclass_excess_return(df):
    """
    Label targets based on excess 10-day return over sector.
    Classes:
    0 = Strong Sell (<= -3%)
    1 = Sell (-3% to -1%)
    2 = Hold (-1% to +1%)
    3 = Buy (+1% to +3%)
    4 = Strong Buy (> +3%)
    """
    df = df.dropna(subset=["return_10d", "sector_ret_10d"])
    excess_ret = df["return_10d"] - df["sector_ret_10d"]
    bins = [-float("inf"), -0.03, -0.01, 0.01, 0.03, float("inf")]
    labels = [0, 1, 2, 3, 4]  # Class labels
    df["target"] = pd.cut(excess_ret, bins=bins, labels=labels)
    df["target"] = df["target"].astype(float).fillna(-1).astype(int)

    return df

def label_excess_return_regression(df):
    df = df.dropna(subset=["return_10d", "sector_ret_10d"])
    df["target"] = df["return_10d"] - df["sector_ret_10d"]
    return df


def extract_signals_from_multiclass_preds(pred_probs):
    """
    Convert predicted class probabilities to discrete signal.
    Args:
        pred_probs: (n_samples, 5) array of class probabilities

    Returns:
        signal: 1 if predicted class is Buy or Strong Buy
        confidence: probability of Strong Buy
    """
    pred_class = np.argmax(pred_probs, axis=1)
    signal = (pred_class >= 3).astype(int)
    confidence = pred_probs[:, 4]  # probability of class 4 (Strong Buy)
    return signal, confidence

def label_prob_3pct_gain(df):
    df = df.sort_values("Date").reset_index(drop=True)
    df["target"] = 0

    for i in range(len(df)):
        window = df["Close"].iloc[i+1:i+11]  # next 10 days
        entry_price = df["Close"].iloc[i]
        if (window >= entry_price * 1.03).any():
            df.at[i, "target"] = 1
    return df

# ✅ Add rule-based momentum score for ensemble ranker

def compute_momentum_score(df):
    """
    Rule-based momentum score combining RSI, EMA crossover, and MACD histogram.
    The higher the score, the stronger the bullish momentum.
    """
    df = df.copy()
    score = 0

    # RSI > 60
    df["rsi_flag"] = (df["rsi_14"] > 60).astype(int)

    # EMA 5 > EMA 20
    df["ema_ratio"] = df["ema_5"] / df["ema_20"]
    df["ema_flag"] = (df["ema_ratio"] > 1.01).astype(int)

    # MACD histogram > 0
    df["macd_flag"] = (df["macd_hist"] > 0).astype(int)

    # Total momentum score = sum of the above
    df["momentum_score"] = df["rsi_flag"] + df["ema_flag"] + df["macd_flag"]

    return df

# ✅ Generate ensemble rank and signal

def generate_ensemble_signals(df, regression_model, classification_model, topk=30):
    df = df.copy()

    # Predict regression model
    X_reg = df[regression_model.feature_name_]
    df["pred_value"] = regression_model.predict(X_reg)
    df["rank_regression"] = df["pred_value"].rank(pct=True)

    # Predict classification model
    X_clf = df[classification_model.feature_name_]
    df["pred_prob"] = classification_model.predict_proba(X_clf)[:, 1]
    df["rank_classifier"] = df["pred_prob"].rank(pct=True)

    # Compute momentum score and rank
    df = compute_momentum_score(df)
    df["rank_momentum"] = df["momentum_score"].rank(pct=True)

    # Final rank: average of all ranks
    df["final_rank"] = (df["rank_regression"] + df["rank_classifier"] + df["rank_momentum"]) / 3

    # Generate signal based on top-k
    df["rank"] = df["final_rank"].rank(method="first", ascending=False)
    df["signal"] = (df["rank"] <= topk).astype(int)

    return df
