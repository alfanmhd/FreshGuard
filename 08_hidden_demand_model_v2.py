
import pandas as pd
import numpy as np
import os
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

print("Loading synthetic data...")

base_path = "data/raw/synthetic"

sales = pd.read_csv(f"{base_path}/sales_daily.csv")
truth = pd.read_csv(f"{base_path}/ground_truth_demand.csv")

df = sales.merge(
    truth,
    on=["date", "store_id", "sku_id"],
    validate="one_to_one"
)

df["date"] = pd.to_datetime(df["date"])

# ==========================================
# 1. CREATE HISTORICAL FEATURES
# ==========================================

df = df.sort_values(
    ["store_id", "sku_id", "date"]
).reset_index(drop=True)

groups = df.groupby(["store_id", "sku_id"])

# Previous observed sales
for lag in [1, 7, 14, 28]:
    df[f"lag_{lag}"] = groups["units_sold"].shift(lag)

# Previous stockout duration
df["previous_stockout_hours"] = (
    groups["stockout_hours"].shift(1)
)

# Rolling averages using previous observations only
for window in [7, 28]:
    df[f"rolling_mean_{window}"] = (
        groups["units_sold"]
        .transform(
            lambda s: s.shift(1).rolling(
                window=window,
                min_periods=1
            ).mean()
        )
    )

# ==========================================
# 2. CALENDAR FEATURES
# ==========================================

df["month"] = df["date"].dt.month
df["day_of_month"] = df["date"].dt.day
df["day_of_year"] = df["date"].dt.day_of_year

# ==========================================
# 3. TEMPORAL SPLIT
# ==========================================

cutoff = pd.Timestamp("2025-07-01")

train_df = df[
    (df["date"] < cutoff) &
    (df["stockout_flag"] == 0)
].copy()

test_df = df[
    (df["date"] >= cutoff) &
    (df["stockout_flag"] == 1)
].copy()

print("\n========== DATA SPLIT ==========")
print("Training records:", len(train_df))
print("Test records:", len(test_df))

# ==========================================
# 4. FEATURES
# ==========================================

categorical_features = [
    "store_id",
    "sku_id"
]

numerical_features = [
    "price",
    "discount_pct",
    "promo_flag",
    "temp_c",
    "rain_mm",
    "dow",
    "is_weekend",
    "is_holiday",
    "is_pre_holiday",
    "month",
    "day_of_month",
    "day_of_year",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28",
    "previous_stockout_hours"
]

features = categorical_features + numerical_features

X_train = train_df[features]
y_train = train_df["true_demand"]

X_test = test_df[features]
y_test = test_df["true_demand"]

# ==========================================
# 5. PREPROCESSING
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            SimpleImputer(strategy="median"),
            numerical_features
        )
    ]
)

# ==========================================
# 6. RANDOM FOREST V2
# ==========================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=150,
                max_depth=20,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

print("\nTraining Random Forest V2...")

model.fit(X_train, y_train)

print("Training completed!")

# ==========================================
# 7. PREDICTIONS
# ==========================================

predictions = np.maximum(
    model.predict(X_test),
    0
)

# ==========================================
# 8. EVALUATION
# ==========================================

def evaluate(actual, predicted):
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(
            mean_squared_error(actual, predicted)
        ),
        "Bias": np.mean(predicted - actual)
    }

baseline = evaluate(
    y_test,
    test_df["units_sold"]
)

rf_v1 = evaluate(
    y_test,
    joblib.load(
        "models/hidden_demand_random_forest.joblib"
    ).predict(
        test_df[
            [
                "store_id", "sku_id", "price",
                "discount_pct", "promo_flag", "temp_c",
                "rain_mm", "dow", "is_weekend",
                "is_holiday", "is_pre_holiday",
                "month", "day_of_month", "day_of_year"
            ]
        ]
    )
)

rf_v2 = evaluate(y_test, predictions)

results = pd.DataFrame([
    {"Model": "Recorded Sales Baseline", **baseline},
    {"Model": "Random Forest V1", **rf_v1},
    {"Model": "Random Forest V2", **rf_v2}
])

print("\n========== MODEL COMPARISON ==========")
print(results.round(4).to_string(index=False))

# ==========================================
# 9. SAVE RESULTS AND MODEL
# ==========================================

os.makedirs("reports", exist_ok=True)
os.makedirs("models", exist_ok=True)

results.to_csv(
    "reports/hidden_demand_comparison.csv",
    index=False
)

joblib.dump(
    model,
    "models/hidden_demand_random_forest_v2.joblib"
)

print("\nRandom Forest V2 saved successfully.")
