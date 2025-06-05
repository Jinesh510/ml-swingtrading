
# update_eod_data.py

from nsepython import nse_eq, nse_index
import pandas as pd
import os
from datetime import datetime
from config import DATA_DIR
import time

def update_stock_eod(ticker, folder):
    try:
        df = pd.DataFrame(nse_eq(ticker)["data"])
        df["Date"] = pd.to_datetime(df["CH_TIMESTAMP"])
        df = df[["Date", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]]
        df.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
        df = df.sort_values("Date")

        filepath = os.path.join(folder, f"{ticker}.csv")
        if os.path.exists(filepath):
            existing = pd.read_csv(filepath, parse_dates=["Date"])
            df = pd.concat([existing, df])
            df = df.drop_duplicates(subset="Date").sort_values("Date")
        df.to_csv(filepath, index=False)
        print(f"✅ Updated {ticker}")
    except Exception as e:
        print(f"❌ Failed to update {ticker}: {e}")

def update_index_eod(index_code, index_name, folder):
    try:
        df = pd.DataFrame(nse_index(index_code)["data"])
        df["Date"] = pd.to_datetime(df["CH_TIMESTAMP"])
        df = df[["Date", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]]
        df.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
        df = df.sort_values("Date")

        filepath = os.path.join(folder, f"{index_name}.csv")
        if os.path.exists(filepath):
            existing = pd.read_csv(filepath, parse_dates=["Date"])
            df = pd.concat([existing, df])
            df = df.drop_duplicates(subset="Date").sort_values("Date")
        df.to_csv(filepath, index=False)
        print(f"✅ Updated index: {index_name}")
    except Exception as e:
        print(f"❌ Failed to update index {index_name}: {e}")

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    index_dir = os.path.join(DATA_DIR, "index_eod")
    os.makedirs(index_dir, exist_ok=True)

    # Sample stocks from sector_mapper.csv
    stock_list = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "LT", "HINDUNILVR"]
    for stock in stock_list:
        update_stock_eod(stock, DATA_DIR)
        time.sleep(1)  # avoid rate limits

    # Sample indices
    index_map = {
        "NIFTY 50": "NIFTY",
        "NIFTY BANK": "BANKNIFTY",
        "NIFTY IT": "NIFTYIT",
        "NIFTY FIN SERVICE": "NIFTYFIN"
    }
    for name, code in index_map.items():
        update_index_eod(code, name.replace(" ", "_"), index_dir)
        time.sleep(1)

if __name__ == "__main__":
    main()
