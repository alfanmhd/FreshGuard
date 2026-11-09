
import os
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.metrics import mean_absolute_error, mean_squared_error


# --------------------------------------------------
# 1. PATHS AND CONFIGURATION
# --------------------------------------------------

DATA_DIR = "data/raw/synthetic"
MODEL_DIR = "models"
REPORT_DIR = "reports"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

CUTOFF_DATE = "2025-07-01"


# --------------------------------------------------
# 2. LOAD AND MERGE DATA
# --------------------------------------------------

print("\nLoading FreshGuard synthetic dataset...")

sales = pd.read_csv(f"{DATA_DIR}/sales_daily.csv")
truth = pd.read_csv(f"{DATA_DIR}/ground_truth_demand.csv")

df = sales.merge(
    truth,
    on=["date", "store_id", "sku_id"],
    how="inner",
    validate="one_to_one"
)

df["date"] = pd.to_datetime(df["date"])

print(f"Total records: {len(df):,}")


# --------------------------------------------------
# 3. FEATURE ENGINEERING
# --------------------------------------------------

df["month"] = df["date"].dt.month
df["day_of_year"] = df["date"].dt.day_of_year

features = [
    "store_id",
    "sku_id",
    "units_sold",
    "opening_stock",
    "stockout_hours",
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
    "day_of_year"
]

target = "lost_sales"

# Use categorical features for store and product IDs.
for col in ["store_id", "sku_id"]:
    df[col] = df[col].astype("category")


# --------------------------------------------------
# 4. TIME-BASED TRAIN / TEST SPLIT
# --------------------------------------------------

train_df = df[df["date"] < CUTOFF_DATE].copy()
test_df = df[df["date"] >= CUTOFF_DATE].copy()

# Focus specifically on days where stockouts occurred.
train_df = train_df[train_df["stockout_flag"] == 1].copy()
test_df = test_df[test_df["stockout_flag"] == 1].copy()

X_train = train_df[features]
y_train = train_df[target]

X_test = test_df[features]
y_test = test_df[target]

print(f"\nTraining stockout records: {len(train_df):,}")
print(f"Testing stockout records:  {len(test_df):,}")


# --------------------------------------------------
# 5. TRAIN FINAL LIGHTGBM MODEL
# --------------------------------------------------

print("\nTraining final LightGBM model...")

model = lgb.LGBMRegressor(
    objective="regression",
    n_estimators=800,
    learning_rate=0.035,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=30,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbosity=-1
)

model.fit(X_train, y_train)

print("Model training completed.")


# --------------------------------------------------
# 6. PREDICT LOST SALES AND RECONSTRUCT DEMAND
# --------------------------------------------------

predicted_lost_sales = model.predict(X_test)

# Lost sales cannot be negative.
predicted_lost_sales = np.maximum(predicted_lost_sales, 0)

# Reconstruct total demand.
estimated_demand = (
    X_test["units_sold"].to_numpy() + predicted_lost_sales
)

actual_demand = test_df["true_demand"].to_numpy()

# Baseline: assume observed sales equal total demand.
baseline_demand = X_test["units_sold"].to_numpy()


# --------------------------------------------------
# 7. MODEL EVALUATION
# --------------------------------------------------

def evaluate(name, actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    bias = np.mean(predicted - actual)

    return {
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "Bias": bias
    }


results = pd.DataFrame([
    evaluate("Observed Sales Baseline", actual_demand, baseline_demand),
    evaluate("Final LightGBM", actual_demand, estimated_demand)
])

print("\nFINAL MODEL RESULTS")
print(results.to_string(index=False))

baseline_mae = results.iloc[0]["MAE"]
model_mae = results.iloc[1]["MAE"]

improvement = (
    (baseline_mae - model_mae) / baseline_mae
) * 100

print(f"\nMAE improvement over baseline: {improvement:.2f}%")

if model_mae < baseline_mae:
    print("LightGBM outperformed the baseline.")
else:
    print("LightGBM did not outperform the baseline.")


# --------------------------------------------------
# 8. SAVE MODEL AND RESULTS
# --------------------------------------------------

joblib.dump(model, f"{MODEL_DIR}/freshguard_final_model.joblib")

results.to_csv(
    f"{REPORT_DIR}/final_model_results.csv",
    index=False
)

# Save feature importance.
importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

importance.to_csv(
    f"{REPORT_DIR}/final_feature_importance.csv",
    index=False
)

# Save a sample of predictions for analysis.
predictions = test_df[
    ["date", "store_id", "sku_id", "units_sold", "true_demand"]
].copy()

predictions["actual_lost_sales"] = y_test.to_numpy()
predictions["predicted_lost_sales"] = predicted_lost_sales
predictions["estimated_demand"] = estimated_demand

predictions.to_csv(
    f"{REPORT_DIR}/final_demand_predictions.csv",
    index=False
)

print("\nSaved:")
print("- models/freshguard_final_model.joblib")
print("- reports/final_model_results.csv")
print("- reports/final_feature_importance.csv")
print("- reports/final_demand_predictions.csv")

print("\nFreshGuard final model pipeline completed.")
