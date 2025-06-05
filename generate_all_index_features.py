
# generate_all_index_features.py

import os
import pandas as pd
from index_utils import compute_index_features

INPUT_DIRS = {
    "index": "data/index_eod",
    "sector": "data/sector_eod"
}

def process_folder(folder_path):
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            path = os.path.join(folder_path, file)
            try:
                df = pd.read_csv(path, parse_dates=["Date"])
                df_feat = compute_index_features(df)
                df_feat.to_csv(path, index=False)
                print(f"✅ Features updated for: {file}")
            except Exception as e:
                print(f"❌ Failed for {file}: {e}")

def main():
    for label, folder in INPUT_DIRS.items():
        print(f"🔄 Processing {label} files in {folder}...")
        process_folder(folder)

if __name__ == "__main__":
    main()
