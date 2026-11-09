
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error

sales = pd.read_csv("data/raw/synthetic/sales_daily.csv")
truth = pd.read_csv("data/raw/synthetic/ground_truth_demand.csv")

df = sales.merge(
    truth,
    on=["date", "store_id", "sku_id"],
    validate="one_to_one"
)

df["date"] = pd.to_datetime(df["date"])
df["month"] = df["date"].dt.month
df["day_of_month"] = df["date"].dt.day
df["day_of_year"] = df["date"].dt.day_of_year

test = df[
    (df["date"] >= "2025-07-01") &
    (df["stockout_flag"] == 1)
].copy()

features = [
    "store_id", "sku_id",
    "price", "discount_pct", "promo_flag",
    "temp_c", "rain_mm", "dow",
    "is_weekend", "is_holiday", "is_pre_holiday",
    "month", "day_of_month", "day_of_year"
]

model = joblib.load(
    "models/hidden_demand_random_forest_v3.joblib"
)

raw_predictions = np.maximum(
    model.predict(test[features]),
    0
)

observed_sales = test["units_sold"].to_numpy()
actual_demand = test["true_demand"].to_numpy()

# Total demand cannot be below observed sales
anchored_predictions = np.maximum(
    observed_sales,
    raw_predictions
)

def evaluate(name, predicted):
    return {
        "Model": name,
        "MAE": mean_absolute_error(actual_demand, predicted),
        "RMSE": np.sqrt(
            mean_squared_error(actual_demand, predicted)
        ),
        "Bias": np.mean(predicted - actual_demand)
    }

results = pd.DataFrame([
    evaluate("Recorded Sales", observed_sales),
    evaluate("Random Forest V3", raw_predictions),
    evaluate("Sales-Anchored V3", anchored_predictions)
])

print("\n========== SALES-ANCHORED RESULTS ==========")
print(results.round(4).to_string(index=False))

print(
    "\nPercentage of predictions corrected:",
    round(np.mean(raw_predictions < observed_sales) * 100, 2),
    "%"
)

results.to_csv(
    "reports/sales_anchored_results.csv",
    index=False
)
