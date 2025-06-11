
# clean_eod_data.py

import os
import pandas as pd
from datetime import datetime

DIRS = {
    "stock": ("data/nse_raw", "data/nse_eod"),
    "index": ("data/index_raw", "data/index_eod"),
    "sector": ("data/sector_raw", "data/sector_eod")
}

def clean_and_save_file(src_folder, dst_folder, filename):
    filepath = os.path.join(src_folder, filename)
    try:
        df = pd.read_csv(filepath)

        # Clean and standardize
        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
        df = df.dropna(subset=["Date", "Open", "High", "Low", "Close", "Volume"])
        df = df.sort_values("Date").drop_duplicates(subset="Date")

        os.makedirs(dst_folder, exist_ok=True)
        out_path = os.path.join(dst_folder, filename)
        df.to_csv(out_path, index=False)
        print(f"✅ Cleaned and saved: {filename} → {dst_folder}")
    except Exception as e:
        print(f"❌ Failed to clean {filename} in {src_folder}: {e}")

def main():
    for label, (src, dst) in DIRS.items():
        if not os.path.exists(src):
            print(f"⚠️ Source directory not found: {src}")
            continue
        for file in os.listdir(src):
            if file.endswith(".csv"):
                clean_and_save_file(src, dst, file)

if __name__ == "__main__":
    main()
