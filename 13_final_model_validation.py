
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

pred = pd.read_csv("reports/final_demand_predictions.csv")

# Overall metrics
actual = pred["true_demand"]
estimated = pred["estimated_demand"]
baseline = pred["units_sold"]

print("\nOVERALL VALIDATION")
print("Test records:", len(pred))
print("Baseline MAE:", mean_absolute_error(actual, baseline))
print("Model MAE:", mean_absolute_error(actual, estimated))
print("Model RMSE:", np.sqrt(mean_squared_error(actual, estimated)))
print("Model Bias:", (estimated - actual).mean())

# Product-level performance
product_results = (
    pred.groupby("sku_id")
    .apply(
        lambda x: pd.Series({
            "records": len(x),
            "MAE": mean_absolute_error(
                x["true_demand"], x["estimated_demand"]
            ),
            "baseline_MAE": mean_absolute_error(
                x["true_demand"], x["units_sold"]
            )
        }),
        include_groups=False
    )
    .reset_index()
)

product_results["improvement_pct"] = (
    (product_results["baseline_MAE"] - product_results["MAE"])
    / product_results["baseline_MAE"].replace(0, np.nan)
) * 100

product_results.to_csv(
    "reports/product_validation.csv", index=False
)

# Store-level performance
store_results = (
    pred.groupby("store_id")
    .apply(
        lambda x: pd.Series({
            "records": len(x),
            "MAE": mean_absolute_error(
                x["true_demand"], x["estimated_demand"]
            ),
            "baseline_MAE": mean_absolute_error(
                x["true_demand"], x["units_sold"]
            )
        }),
        include_groups=False
    )
    .reset_index()
)

store_results["improvement_pct"] = (
    (store_results["baseline_MAE"] - store_results["MAE"])
    / store_results["baseline_MAE"].replace(0, np.nan)
) * 100

store_results.to_csv(
    "reports/store_validation.csv", index=False
)

print("\nPRODUCT VALIDATION")
print(product_results.round(3).to_string(index=False))

print("\nSTORE VALIDATION")
print(store_results.round(3).to_string(index=False))

print("\nSaved product and store validation reports.")
