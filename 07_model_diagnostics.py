
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import mean_absolute_error

print("Loading data and trained model...")

base_path = "data/raw/synthetic"

sales = pd.read_csv(f"{base_path}/sales_daily.csv")
truth = pd.read_csv(f"{base_path}/ground_truth_demand.csv")

df = sales.merge(
    truth,
    on=["date", "store_id", "sku_id"],
    validate="one_to_one"
)

df["date"] = pd.to_datetime(df["date"])

df["month"] = df["date"].dt.month
df["day_of_month"] = df["date"].dt.day
df["day_of_year"] = df["date"].dt.day_of_year

cutoff = pd.Timestamp("2025-07-01")

train_df = df[
    (df["date"] < cutoff) &
    (df["stockout_flag"] == 0)
].copy()

test_df = df[
    (df["date"] >= cutoff) &
    (df["stockout_flag"] == 1)
].copy()

categorical_features = ["store_id", "sku_id"]

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

model = joblib.load(
    "models/hidden_demand_random_forest.joblib"
)

predictions = np.maximum(
    model.predict(test_df[features]),
    0
)

# ======================================
# 1. TARGET DISTRIBUTION
# ======================================

print("\n========== TARGET DISTRIBUTION ==========")

print("\nTraining target:")
print(train_df["true_demand"].describe())

print("\nTest target:")
print(test_df["true_demand"].describe())

# ======================================
# 2. PREDICTION COMPARISON
# ======================================

comparison = pd.DataFrame({
    "Actual Demand": test_df["true_demand"].to_numpy(),
    "Observed Sales": test_df["units_sold"].to_numpy(),
    "RF Prediction": predictions
})

print("\n========== PREDICTION SUMMARY ==========")
print(comparison.describe().to_string())

# ======================================
# 3. ERROR ANALYSIS
# ======================================

comparison["RF Error"] = (
    comparison["RF Prediction"] -
    comparison["Actual Demand"]
)

comparison["Baseline Error"] = (
    comparison["Observed Sales"] -
    comparison["Actual Demand"]
)

print("\n========== ERROR ANALYSIS ==========")

print(
    "RF underprediction percentage:",
    round((comparison["RF Error"] < 0).mean() * 100, 2),
    "%"
)

print(
    "Baseline underprediction percentage:",
    round((comparison["Baseline Error"] < 0).mean() * 100, 2),
    "%"
)

# ======================================
# 4. PRODUCT-LEVEL ERROR
# ======================================

test_df = test_df.copy()

test_df["rf_prediction"] = predictions

product_results = (
    test_df.groupby("sku_id")
    .apply(
        lambda group: pd.Series({
            "observations": len(group),
            "actual_demand": group["true_demand"].mean(),
            "observed_sales": group["units_sold"].mean(),
            "predicted_demand": group["rf_prediction"].mean(),
            "RF_MAE": mean_absolute_error(
                group["true_demand"],
                group["rf_prediction"]
            )
        }),
        include_groups=False
    )
)

print("\n========== PRODUCT-LEVEL RESULTS ==========")
print(
    product_results.sort_values(
        "RF_MAE", ascending=False
    ).head(10).to_string()
)

# ======================================
# 5. SAVE PREDICTIONS
# ======================================

comparison.to_csv(
    "reports/hidden_demand_predictions.csv",
    index=False
)

product_results.to_csv(
    "reports/product_model_diagnostics.csv"
)

print("\nDiagnostics completed.")
