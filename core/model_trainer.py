# model_trainer.py
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from core.config import TARGET_TYPE
from core.threshold_tuning_utils import tune_threshold
from core.utils import drop_feature_columns, get_feature_columns
from lightgbm import early_stopping, log_evaluation



def train_lightgbm_model(train_df, val_df=None, task="classification", ticker=None):
    train_df = train_df.dropna(subset=["target"]).copy()
    train_df = train_df.sort_values("Date")

    feature_cols = get_feature_columns(target_type=TARGET_TYPE)
    cols_to_ignore = drop_feature_columns(target_type=TARGET_TYPE)

    feature_cols.append("ticker")

    # Remove 'ticker' for classification models
    if task == "classification" and "ticker" in feature_cols:
        feature_cols.remove("ticker")

    # Cast 'ticker' to category only if it's still in features
    if "ticker" in feature_cols and "ticker" in train_df.columns:
        train_df["ticker"] = train_df["ticker"].astype("category")
        if val_df is not None and "ticker" in val_df.columns:
            val_df["ticker"] = val_df["ticker"].astype("category")

    # if "ticker" in train_df.columns:
    #     train_df["ticker"] = train_df["ticker"].astype("category")
    #     val_df["ticker"] = val_df["ticker"].astype("category")
    #     feature_cols.append("ticker")

    feature_cols= [x for x in feature_cols if x not in cols_to_ignore]


    X_train = train_df[feature_cols]
    y_train = train_df["target"]

    print("Feature cols:", feature_cols)
    print("Feature summary:\n", X_train.describe())
    print("NaN summary:\n", X_train.isna().sum())
    print("Feature variance:\n", X_train.std())

    if val_df is not None:
        val_df = val_df.dropna(subset=["target"]).copy()
        val_df = val_df.sort_values("Date")

        X_val = val_df[feature_cols]
        y_val = val_df["target"]
    else:
        X_val = None
        y_val = None

    classifier_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "seed": 42,
        "class_weight":'balanced'
    }

    regression_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "seed": 42
    }


    if task == "classification":
        model = lgb.LGBMClassifier(**classifier_params)
    elif task == "regression":
        model = lgb.LGBMRegressor(**regression_params)
    else:
        raise ValueError("task must be 'classification' or 'regression'")

    model.fit(X_train, y_train,
              eval_set=[(X_train, y_train), (X_val, y_val)],
              callbacks=[early_stopping(stopping_rounds=50), log_evaluation(period=100)])

    
    print(f"✅ Trained model for {ticker}")

    return model, feature_cols
