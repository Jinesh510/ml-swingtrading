import pandas as pd
import lightgbm as lgb
from itertools import product
from sklearn.metrics import f1_score, precision_score, recall_score
from utils import add_advanced_features, train_lightgbm, get_target_definition, drop_target_related_columns, get_feature_columns
from config import TARGET_TYPE, CAP_TYPE

# Load your training dataframe
ticker="MPHASIS"
filepath = "./data/nse_eod/"+ ticker + ".csv"
df = pd.read_csv(filepath)
df["Date"] = pd.to_datetime(df["Date"])

df = add_advanced_features(df,ticker)
df = get_target_definition(df, TARGET_TYPE)
feature_cols = get_feature_columns(target_type=TARGET_TYPE, cap_type=CAP_TYPE)
leakage_cols = drop_target_related_columns(target_type=TARGET_TYPE)

feature_cols= [x for x in feature_cols if x not in leakage_cols]

df = df.dropna().reset_index(drop=True)
df = df.sort_values("Date").reset_index(drop=True)
n = len(df)
train_end = int(n * 0.6)
val_end = int(n * 0.8)


df = df.iloc[:val_end].copy()

# df_train = df.iloc[:train_end].copy()
# df_val = df.iloc[train_end:val_end].copy()
# df_test = df.iloc[val_end:].copy()


# df = drop_target_related_columns(df, TARGET_TYPE)

# Basic preprocessing
# feature_cols = get_feature_columns(target_type=TARGET_TYPE, cap_type=CAP_TYPE)
X = df[feature_cols]
y = df["target"]

# Train/Val split (time-based)
# split_idx = int(len(df) * 0.8)
X_train, X_val = X.iloc[:train_end], X.iloc[train_end:]
y_train, y_val = y.iloc[:train_end], y.iloc[train_end:]

# Grid search space
search_space = {
    "learning_rate": [0.03, 0.05, 0.1],
    "num_leaves": [15, 25, 31],
    "min_child_samples": [10, 20, 40]
}

results = []

for lr, leaves, min_child in product(
    search_space["learning_rate"],
    search_space["num_leaves"],
    search_space["min_child_samples"]
):
    params = {
        "objective": "binary",
        "random_state": 42,
        "learning_rate": lr,
        "num_leaves": leaves,
        "min_child_samples": min_child,
        "n_estimators": 300,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "class_weight": "balanced"
    }

    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    f1 = f1_score(y_val, y_pred, zero_division=0)
    precision = precision_score(y_val, y_pred, zero_division=0)
    recall = recall_score(y_val, y_pred, zero_division=0)

    results.append({
        "learning_rate": lr,
        "num_leaves": leaves,
        "min_child_samples": min_child,
        "f1": f1,
        "precision": precision,
        "recall": recall
    })

# Save to CSV
results_df = pd.DataFrame(results)
results_df.to_csv(f"./models/optimization/lightgbm_grid_search_results_{ticker}.csv", index=False)
print("✅ Grid search completed and saved.")
