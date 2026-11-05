
from datasets import load_from_disk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

print("Loading FreshRetailNet...")

dataset = load_from_disk("data/raw/freshretailnet")
train = dataset["train"]

sample_size = min(200000, len(train))
df = train.shuffle(seed=42).select(range(sample_size)).to_pandas()

os.makedirs("reports/figures", exist_ok=True)

# Use documented 06:00-22:00 stockout count
df["daily_stockout_hours"] = df["stock_hour6_22_cnt"]

df["daily_stockout_flag"] = (
    df["daily_stockout_hours"] > 0
).astype(int)

df["stockout_percentage"] = (
    df["daily_stockout_hours"] / 16
) * 100

# ======================================
# 1. STOCKOUT SUMMARY
# ======================================

print("\n========== CORRECTED STOCKOUT SUMMARY ==========")

print("Total sampled records:", len(df))

print(
    "Records with stockouts:",
    df["daily_stockout_flag"].sum()
)

print(
    "Stockout record percentage:",
    round(df["daily_stockout_flag"].mean() * 100, 2),
    "%"
)

print(
    "Average stockout hours:",
    round(df["daily_stockout_hours"].mean(), 2)
)

print(
    "Average stockout hours among affected records:",
    round(
        df.loc[
            df["daily_stockout_flag"] == 1,
            "daily_stockout_hours"
        ].mean(),
        2
    )
)

# ======================================
# 2. SALES COMPARISON
# ======================================

print("\n========== SALES COMPARISON ==========")

print(
    df.groupby("daily_stockout_flag")["sale_amount"]
    .agg(["count", "mean", "median", "std"])
)

# ======================================
# 3. PRODUCT STOCKOUT ANALYSIS
# ======================================

product_analysis = (
    df.groupby("product_id")
    .agg(
        observations=("daily_stockout_flag", "size"),
        stockout_rate=("daily_stockout_flag", "mean"),
        average_sales=("sale_amount", "mean")
    )
)

# Exclude products with very few observations
reliable_products = product_analysis[
    product_analysis["observations"] >= 50
].sort_values("stockout_rate", ascending=False)

print("\n========== PRODUCTS WITH HIGHEST STOCKOUT RATES ==========")
print(reliable_products.head(10).to_string())

# ======================================
# 4. STOCKOUT DISTRIBUTION
# ======================================

plt.figure(figsize=(10, 5))

plt.hist(
    df["daily_stockout_hours"],
    bins=17,
    edgecolor="black"
)

plt.title("Daily Stockout Hours (06:00-22:00)")
plt.xlabel("Stockout Hours")
plt.ylabel("Number of Records")
plt.tight_layout()

plt.savefig(
    "reports/figures/08_stockout_hours_distribution.png"
)
plt.show()

# ======================================
# 5. SALES COMPARISON GRAPH
# ======================================

sales_comparison = (
    df.groupby("daily_stockout_flag")["sale_amount"].mean()
)

plt.figure(figsize=(8, 5))

plt.bar(
    ["No Stockout", "Stockout"],
    sales_comparison.values
)

plt.title("Average Sales by Stockout Status")
plt.ylabel("Average Normalised Sales")
plt.tight_layout()

plt.savefig(
    "reports/figures/09_stockout_sales_comparison.png"
)
plt.show()

# ======================================
# 6. PRODUCT STOCKOUT RATES
# ======================================

plt.figure(figsize=(12, 5))

reliable_products.head(10)["stockout_rate"].plot(kind="bar")

plt.title("Top Products by Stockout Rate")
plt.xlabel("Product ID")
plt.ylabel("Stockout Rate")
plt.tight_layout()

plt.savefig(
    "reports/figures/10_product_stockout_rates.png"
)
plt.show()

print("\nCorrected stockout analysis completed.")
