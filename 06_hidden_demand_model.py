
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error

import joblib
import os

print("Loading synthetic data...")

# ==========================================
# 1. LOAD AND MERGE DATA
# ==========================================

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
# 2. FEATURE ENGINEERING
# ==========================================

df["month"] = df["date"].dt.month
df["day_of_month"] = df["date"].dt.day
df["day_of_year"] = df["date"].dt.dayofyear

# ==========================================
# 3. TEMPORAL TRAIN/TEST SPLIT
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
print("Training observations:", len(train_df))
print("Stockout test observations:", len(test_df))

# ==========================================
# 4. SELECT FEATURES
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
    "day_of_year"
]

features = categorical_features + numerical_features

X_train = train_df[features]
y_train = train_df["true_demand"]

X_test = test_df[features]
y_test = test_df["true_demand"]

# ==========================================
# 5. BUILD RANDOM FOREST PIPELINE
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
            "passthrough",
            numerical_features
        )
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=100,
                max_depth=18,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

# ==========================================
# 6. TRAIN MODEL
# ==========================================

print("\nTraining Random Forest...")

model.fit(X_train, y_train)

print("Training completed!")

# ==========================================
# 7. PREDICT HIDDEN DEMAND
# ==========================================

predictions = model.predict(X_test)

# Demand cannot be negative
predictions = np.maximum(predictions, 0)

# ==========================================
# 8. EVALUATE MODEL
# ==========================================

mae = mean_absolute_error(y_test, predictions)

rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)

bias = np.mean(predictions - y_test)

# Baseline: recorded sales directly
baseline_predictions = test_df["units_sold"].to_numpy()

baseline_mae = mean_absolute_error(
    y_test,
    baseline_predictions
)

baseline_rmse = np.sqrt(
    mean_squared_error(y_test, baseline_predictions)
)

baseline_bias = np.mean(
    baseline_predictions - y_test
)

print("\n========== MODEL RESULTS ==========")

print("Random Forest MAE:", round(mae, 4))
print("Random Forest RMSE:", round(rmse, 4))
print("Random Forest Bias:", round(bias, 4))

print("\n========== BASELINE RESULTS ==========")

print("Baseline MAE:", round(baseline_mae, 4))
print("Baseline RMSE:", round(baseline_rmse, 4))
print("Baseline Bias:", round(baseline_bias, 4))

# ==========================================
# 9. SAVE RESULTS
# ==========================================

results = pd.DataFrame([
    {
        "model": "Recorded Sales Baseline",
        "MAE": baseline_mae,
        "RMSE": baseline_rmse,
        "Bias": baseline_bias
    },
    {
        "model": "Random Forest",
        "MAE": mae,
        "RMSE": rmse,
        "Bias": bias
    }
])

os.makedirs("reports", exist_ok=True)
os.makedirs("models", exist_ok=True)

results.to_csv(
    "reports/hidden_demand_results.csv",
    index=False
)

# Save model
joblib.dump(
    model,
    "models/hidden_demand_random_forest.joblib"
)

print("\nModel saved successfully.")
print("Results saved in reports/hidden_demand_results.csv")
