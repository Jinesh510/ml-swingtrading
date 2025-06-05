
# live_trade.py

import yfinance as yf
import pandas as pd
from config import MODEL_DIR, BUY_THRESHOLD, MAX_POSITIONS, RISK_PER_TRADE, CAPITAL
from utils import compute_features, load_model, predict_signal

# Define live stock universe

try:
    with open("data/stock_universe.txt") as f:
        live_stocks = [line.strip().upper() + ".NS" for line in f if line.strip()]
except FileNotFoundError:
    live_stocks = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]


def fetch_latest_data(ticker):
    df = yf.download(ticker, period="30d", interval="1d")
    df = df.reset_index()
    df = compute_features(df)
    return df

def calculate_position(entry_price, capital, risk_per_trade):
    stop_loss_price = entry_price * 0.98  # 2% SL
    risk_amount = capital * risk_per_trade
    qty = int(risk_amount / (entry_price - stop_loss_price))
    return qty, stop_loss_price

def main():
    trade_recos = []

    for ticker in live_stocks:
        base_name = ticker.replace(".NS", "")
        model = load_model(base_name)
        if model is None:
            print(f"Model not found for {ticker}, skipping.")
            continue

        df = fetch_latest_data(ticker)
        latest = df.iloc[[-1]]
        df = predict_signal(df, model, threshold=BUY_THRESHOLD)
        signal = df["signal"].iloc[-1]
        prob = df["pred_prob"].iloc[-1]
        entry_price = df["Close"].iloc[-1]

        if signal == 1:
            qty, sl_price = calculate_position(entry_price, CAPITAL, RISK_PER_TRADE)
            trade_recos.append({
                "ticker": ticker,
                "prob": prob,
                "entry_price": entry_price,
                "stop_loss": sl_price,
                "qty": qty
            })

    trade_recos = sorted(trade_recos, key=lambda x: x["prob"], reverse=True)[:MAX_POSITIONS]
    df_trades = pd.DataFrame(trade_recos)
    print("🔔 Today's Trade Recommendations:")
    print(df_trades)

    # Optional: Zerodha Kite API scaffold (requires secure token flow)
    # from kiteconnect import KiteConnect
    # kite = KiteConnect(api_key="your_api_key")
    # data = kite.generate_session("request_token", api_secret="your_secret")
    # kite.set_access_token(data["access_token"])
    # for trade in trade_recos:
    #     kite.place_order(
    #         tradingsymbol=trade["ticker"].replace(".NS", ""),
    #         exchange="NSE",
    #         transaction_type="BUY",
    #         quantity=trade["qty"],
    #         order_type="MARKET",
    #         product="CNC"
    #     )

if __name__ == "__main__":
    main()
