
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. LOAD SYNTHETIC DATA
# ==========================================

print("Loading FreshGuard synthetic datasets...")

base_path = "data/raw/synthetic"

sales = pd.read_csv(f"{base_path}/sales_daily.csv")
ground_truth = pd.read_csv(
    f"{base_path}/ground_truth_demand.csv"
)
products = pd.read_csv(f"{base_path}/products.csv")
stores = pd.read_csv(f"{base_path}/stores.csv")
calendar = pd.read_csv(f"{base_path}/calendar.csv")

os.makedirs("reports/figures", exist_ok=True)

# ==========================================
# 2. BASIC DATASET INFORMATION
# ==========================================

print("\n========== DATASET OVERVIEW ==========")

datasets = {
    "Sales": sales,
    "Ground Truth": ground_truth,
    "Products": products,
    "Stores": stores,
    "Calendar": calendar
}

for name, data in datasets.items():
    print(f"\n{name}")
    print("Shape:", data.shape)
    print("Columns:", list(data.columns))
    print("Missing values:", data.isnull().sum().sum())

# ==========================================
# 3. MERGE SALES AND GROUND TRUTH
# ==========================================

keys = ["date", "store_id", "sku_id"]

df = sales.merge(
    ground_truth,
    on=keys,
    how="inner",
    validate="one_to_one"
)

print("\n========== MERGE VALIDATION ==========")
print("Merged records:", len(df))
print("Sales records:", len(sales))
print("Ground truth records:", len(ground_truth))

assert len(df) == len(sales) == len(ground_truth)
assert df[keys].duplicated().sum() == 0

# ==========================================
# 4. VALIDATE GROUND TRUTH RELATIONSHIPS
# ==========================================

print("\n========== DEMAND VALIDATION ==========")

print(
    "Observed sales greater than true demand:",
    (df["units_sold"] > df["true_demand"]).sum()
)

print(
    "Negative lost sales:",
    (df["lost_sales"] < 0).sum()
)

print(
    "Maximum difference between true demand "
    "and observed sales plus lost sales:",
    np.abs(
        df["true_demand"]
        - df["units_sold"]
        - df["lost_sales"]
    ).max()
)

# ==========================================
# 5. DEMAND UNDERESTIMATION
# ==========================================

total_observed = df["units_sold"].sum()
total_true = df["true_demand"].sum()
total_lost = df["lost_sales"].sum()

underestimation = (
    total_lost / total_true * 100
    if total_true > 0 else 0
)

print("\n========== DEMAND UNDERSTIMATION ==========")

print("Total observed sales:", total_observed)
print("Total actual demand:", total_true)
print("Total lost sales:", total_lost)

print(
    "Demand underestimation:",
    round(underestimation, 2),
    "%"
)

# ==========================================
# 6. STOCKOUT VS NON-STOCKOUT COMPARISON
# ==========================================

print("\n========== STOCKOUT COMPARISON ==========")

stockout_comparison = (
    df.groupby("stockout_flag")
    .agg(
        records=("units_sold", "size"),
        observed_sales=("units_sold", "mean"),
        actual_demand=("true_demand", "mean"),
        lost_sales=("lost_sales", "mean"),
        average_stockout_hours=("stockout_hours", "mean")
    )
)

print(stockout_comparison.to_string())

# ==========================================
# 7. VISUALISE OBSERVED VS TRUE DEMAND
# ==========================================

comparison = pd.DataFrame({
    "Demand Type": ["Observed Sales", "True Demand"],
    "Total Units": [total_observed, total_true]
})

plt.figure(figsize=(8, 5))

plt.bar(
    comparison["Demand Type"],
    comparison["Total Units"]
)

plt.title("Observed Sales vs Actual Demand")
plt.ylabel("Total Units")
plt.tight_layout()

plt.savefig(
    "reports/figures/11_observed_vs_true_demand.png"
)
plt.show()

# ==========================================
# 8. LOST SALES DISTRIBUTION
# ==========================================

plt.figure(figsize=(10, 5))

plt.hist(
    df["lost_sales"],
    bins=40,
    edgecolor="black"
)

plt.title("Distribution of Lost Sales")
plt.xlabel("Lost Units")
plt.ylabel("Frequency")
plt.tight_layout()

plt.savefig(
    "reports/figures/12_lost_sales_distribution.png"
)
plt.show()

# ==========================================
# 9. STOCKOUT HOURS VS LOST SALES
# ==========================================

plt.figure(figsize=(9, 5))

plt.scatter(
    df["stockout_hours"],
    df["lost_sales"],
    alpha=0.15
)

plt.title("Stockout Hours vs Lost Sales")
plt.xlabel("Stockout Hours")
plt.ylabel("Lost Units")
plt.tight_layout()

plt.savefig(
    "reports/figures/13_stockout_hours_vs_lost_sales.png"
)
plt.show()

# ==========================================
# 10. PRODUCT CATEGORY ANALYSIS
# ==========================================

product_data = df.merge(
    products[["sku_id", "category"]],
    on="sku_id",
    how="left",
    validate="many_to_one"
)

category_analysis = (
    product_data.groupby("category")
    .agg(
        observed_sales=("units_sold", "sum"),
        true_demand=("true_demand", "sum"),
        lost_sales=("lost_sales", "sum")
    )
)

category_analysis["lost_demand_percentage"] = (
    category_analysis["lost_sales"]
    / category_analysis["true_demand"]
    * 100
)

print("\n========== CATEGORY ANALYSIS ==========")
print(category_analysis.to_string())

category_analysis.to_csv(
    "reports/synthetic_category_analysis.csv"
)

# ==========================================
# 11. BASELINE ESTIMATION
# ==========================================

# Baseline: predict demand using observed sales directly.
# This is deliberately simple and will be our benchmark.

df["baseline_prediction"] = df["units_sold"]

mae = np.mean(
    np.abs(df["true_demand"] - df["baseline_prediction"])
)

rmse = np.sqrt(
    np.mean(
        (df["true_demand"] - df["baseline_prediction"]) ** 2
    )
)

bias = np.mean(
    df["baseline_prediction"] - df["true_demand"]
)

print("\n========== BASELINE EVALUATION ==========")
print("Baseline MAE:", round(mae, 4))
print("Baseline RMSE:", round(rmse, 4))
print("Baseline Bias:", round(bias, 4))

# ==========================================
# 12. SAVE BASELINE RESULTS
# ==========================================

baseline_results = pd.DataFrame([{
    "model": "Observed Sales Baseline",
    "MAE": mae,
    "RMSE": rmse,
    "Bias": bias
}])

baseline_results.to_csv(
    "reports/baseline_results.csv",
    index=False
)

print("\nSYNTHETIC DATA ANALYSIS COMPLETED!")
print("Results saved in reports/")
