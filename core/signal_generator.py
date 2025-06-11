# signal_generator.py

import pandas as pd

def compute_momentum_score(df):
    df = df.copy()
    # df["momentum_score"] = (
    #     0.3 * df["ret_5d"] +
    #     0.2 * df["rsi_14"] +
    #     0.2 * df["MACD_12_26_9"] +
    #     0.3 * df["EMA_20"] / df["Close"]
    # )
    
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

def generate_ensemble_signals(df, regression_model, classification_model, topk=None, threshold_map=None):
    df = df.copy()

    # Predict regression
    if "ticker" in regression_model.feature_name_ and "ticker" in df.columns:
        df["ticker"] = df["ticker"].astype("category")

    X_reg = df[regression_model.feature_name_]
    df["pred_value"] = regression_model.predict(X_reg)
    df["rank_regression"] = df.groupby("Date")["pred_value"].rank(pct=True)

    # Predict classification
    X_clf = df[classification_model.feature_name_]
    df["pred_prob"] = classification_model.predict_proba(X_clf)[:, 1]
    df["rank_classifier"] = df.groupby("Date")["pred_prob"].rank(pct=True)

    # Momentum score and rank
    df = compute_momentum_score(df)
    df["rank_momentum"] = df.groupby("Date")["momentum_score"].rank(pct=True)

    # Final rank
    df["final_rank"] = (
        df["rank_regression"] + df["rank_classifier"] + df["rank_momentum"]
    ) / 3

    # Signal logic
    if threshold_map:
        df["signal"] = 0
        for ticker in df["ticker"].unique():
            if ticker in threshold_map:
                mask = df["ticker"] == ticker
                df.loc[mask, "signal"] = (df.loc[mask, "final_rank"] >= threshold_map[ticker]).astype(int)
    elif topk:
        df["rank"] = df.groupby("Date")["final_rank"].rank(method="first", ascending=False)
        df["signal"] = (df["rank"] <= topk).astype(int)
    else:
        df["signal"] = 0  # fallback

    return df
