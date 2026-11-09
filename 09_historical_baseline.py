
import pandas as pd
import numpy as np
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

cutoff = pd.Timestamp("2025-07-01")

# Historical observations only
history = df[
    (df["date"] < cutoff) &
    (df["stockout_flag"] == 0)
].copy()

# Future stockout observations
test = df[
    (df["date"] >= cutoff) &
    (df["stockout_flag"] == 1)
].copy()

# ==========================================
# 1. SKU-STORE HISTORICAL DEMAND
# ==========================================

sku_store_average = (
    history.groupby(["store_id", "sku_id"])["units_sold"]
    .mean()
)

global_average = history["units_sold"].mean()

test["historical_prediction"] = [
    sku_store_average.get(
        (store, sku),
        global_average
    )
    for store, sku in zip(
        test["store_id"],
        test["sku_id"]
    )
]

# ==========================================
# 2. EVALUATION
# ==========================================

def evaluate(actual, predicted):
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(
            mean_squared_error(actual, predicted)
        ),
        "Bias": np.mean(predicted - actual)
    }

actual = test["true_demand"]

results = []

predictions = {
    "Recorded Sales": test["units_sold"],
    "Historical SKU-Store Average": test["historical_prediction"]
}

for name, prediction in predictions.items():
    metrics = evaluate(actual, prediction)

    results.append({
        "Model": name,
        **metrics
    })

results = pd.DataFrame(results)

print("\n========== HISTORICAL BASELINE COMPARISON ==========")
print(results.round(4).to_string(index=False))

# ==========================================
# 3. SAVE RESULTS
# ==========================================

results.to_csv(
    "reports/historical_baseline_results.csv",
    index=False
)

print("\nHistorical baseline evaluation completed.")
