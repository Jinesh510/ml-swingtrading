
# config.py
import os
# Global trading parameters
CAPITAL = 1_000_000  # INR
MAX_POSITIONS = 5
RISK_PER_TRADE = 0.02  # 2%

# Signal threshold for model confidence
BUY_THRESHOLD = 0.7

# Feature columns
FEATURE_COLS = ["rsi_14", "sma_20", "sma_50"]

# Directory paths
DATA_DIR = "data/nse_eod"
MODEL_DIR = "models"
PLOTS_DIR = os.path.join(MODEL_DIR, "plots")
TRADES_DIR = os.path.join(MODEL_DIR, "trades")
PROB_DIST_DIR = os.path.join(MODEL_DIR, "prob_dist")
RESULTS_DIR = os.path.join(MODEL_DIR, "results")



PROMOTION_CRITERIA = {"f1_score_min": 0.6, "final_return_min": 1.2}

USE_FIXED_THRESHOLD = False
SIGNAL_MODE = "threshold"       # "topk" or "threshold"
TOP_K = 50                 # used only if mode is topk
THRESHOLD = 0.4            # used only if mode is threshold

TARGET_TYPE = "10d_vs_sector"  # Options: "3d_1pct","5d_vs_sector", "10d_vs_sector", "quantile_top25"
CAP_TYPE = "midcap"      # Options: "largecap", "midcap", "smallcap"
PEER_NAME = None

# original exit conditions aligned with target
TRAILING_STOPLOSS = 0.1   # 4%
PROFIT_TARGET = 0.08       # 3%
MAX_HOLD_DAYS = 30        # Optional limit

# Exit Logic Configuration (for simulate_trades)
# EXIT_SMA_PERIOD = 5             # for SMA_X breach
# EXIT_RSI_PERIOD = 14            # for RSI
# EXIT_RSI_THRESHOLD = 50         # below this = exit
# TRAILING_STOPLOSS = 0.04        # 4% trailing SL
# MAX_HOLD_DAYS = 20              # max number of days to hold a position



# LIGHTGBM_PARAMS = {
#     "objective": "binary",
#     "random_state": 42,
#     "learning_rate": 0.03,
#     "num_leaves": 31,
#     "min_child_samples": 20,
#     "n_estimators": 300,
#     "subsample": 0.8,
#     "colsample_bytree": 0.8,
#     "class_weight": "balanced"
# }

# LIGHTGBM_PARAMS = {
#     "objective": "binary",
#     "metric": "binary_logloss",
#     "verbosity": -1
# }

# LIGHTGBM_PARAMS={
#     'num_leaves': 64,
# 'max_depth': 6,
# 'min_data_in_leaf': 10,
# 'learning_rate': 0.03,
# 'n_estimators': 300,
# 'reg_alpha': 1.0,
# 'reg_lambda': 1.0,
# 'class_weight': 'balanced'

# }

# LIGHTGBM_PARAMS = {
#     "num_leaves": 64,
#     "max_depth": 7,
#     "learning_rate": 0.03,
#     "n_estimators": 500,
#     "subsample": 0.9,
#     "colsample_bytree": 0.9,
#     "min_child_samples": 30,
#     "random_state": 42
# }

#multi-class target
# LIGHTGBM_PARAMS = {
#     "objective": "multiclass",
#     "num_class": 5,
#     "metric": "multi_logloss",
#     "learning_rate": 0.03,
#     "num_leaves": 64,
#     "max_depth": 6,
#     "subsample": 0.9,
#     "colsample_bytree": 0.9,
#     "random_state": 42
# }

#regression model

LIGHTGBM_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.03,
    "num_leaves": 64,
    "max_depth": 6,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "random_state": 42
}


SIGNAL_THRESHOLD = 0.02

