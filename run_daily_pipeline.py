
# run_daily_pipeline.py

import subprocess
import os
import pandas as pd
from config import MODEL_DIR, PROMOTION_CRITERIA

def run_script(script_name):
    result = subprocess.run(["python", script_name], capture_output=True, text=True)
    print(f"--- Output of {script_name} ---")
    print(result.stdout)
    if result.stderr:
        print(f"--- Errors in {script_name} ---")
        print(result.stderr)

def promote_models(performance_file):
    if not os.path.exists(performance_file):
        print("⚠️ No performance file found. Skipping promotion.")
        return

    df = pd.read_csv(performance_file)
    promoted = df[
        (df["F1 Score"] >= PROMOTION_CRITERIA["f1_score_min"]) &
        (df["Final Cumulative Return"] >= PROMOTION_CRITERIA["final_return_min"])
    ]

    # Remove models that don't meet criteria
    all_models = set(f for f in os.listdir(MODEL_DIR) if f.endswith(".pkl"))
    keep_models = set(f"{ticker}.pkl" for ticker in promoted["Ticker"])
    remove_models = all_models - keep_models

    for model in remove_models:
        try:
            os.remove(os.path.join(MODEL_DIR, model))
            print(f"🗑️ Removed unqualified model: {model}")
        except Exception as e:
            print(f"❌ Error removing {model}: {e}")

    print(f"✅ Promoted {len(promoted)} models.")

def main():
    run_script("update_eod_data.py")
    run_script("backtest.py")
    performance_log = os.path.join(MODEL_DIR, "model_performance_log.csv")
    promote_models(performance_log)

if __name__ == "__main__":
    main()
