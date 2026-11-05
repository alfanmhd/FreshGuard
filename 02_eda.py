
from datasets import load_from_disk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. LOAD DATASET
# ==========================================

print("Loading FreshRetailNet-50K...")

dataset = load_from_disk("data/raw/freshretailnet")
train = dataset["train"]

# Random sample for visual analysis
np.random.seed(42)
sample_size = min(100000, len(train))
random_indices = np.random.choice(
    len(train), size=sample_size, replace=False
)

df = train.select(random_indices.tolist()).to_pandas()

print("Sample loaded:", df.shape)

# ==========================================
# 2. PREPARE OUTPUT FOLDER
# ==========================================

os.makedirs("reports/figures", exist_ok=True)

df["dt"] = pd.to_datetime(df["dt"])

# ==========================================
# 3. BASIC DATA INFORMATION
# ==========================================

print("\n========== DATA OVERVIEW ==========")
print(df.info())

print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe().to_string())

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum().to_string())

# ==========================================
# 4. SALES DISTRIBUTION
# ==========================================

plt.figure(figsize=(10, 5))

upper_limit = df["sale_amount"].quantile(0.99)

plt.hist(
    df["sale_amount"].clip(upper=upper_limit),
    bins=50
)

plt.title("Distribution of Recorded Sales")
plt.xlabel("Sale Amount")
plt.ylabel("Frequency")
plt.tight_layout()

plt.savefig("reports/figures/01_sales_distribution.png")
plt.show()

# ==========================================
# 5. SALES OVER TIME
# ==========================================

daily_sales = (
    df.groupby("dt")["sale_amount"]
    .mean()
    .sort_index()
)

plt.figure(figsize=(12, 5))

plt.plot(daily_sales.index, daily_sales.values)

plt.title("Average Recorded Sales Over Time")
plt.xlabel("Date")
plt.ylabel("Average Sale Amount")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("reports/figures/02_sales_over_time.png")
plt.show()

# ==========================================
# 6. STOCK AVAILABILITY COUNT
# ==========================================

plt.figure(figsize=(10, 5))

plt.hist(df["stock_hour6_22_cnt"], bins=30)

plt.title("Distribution of Stock Hour Count")
plt.xlabel("Stock Hour Count")
plt.ylabel("Frequency")
plt.tight_layout()

plt.savefig("reports/figures/03_stock_hour_count.png")
plt.show()

# ==========================================
# 7. HOURLY SALES PATTERN
# ==========================================

hourly_sales = np.array(df["hours_sale"].tolist())

average_hourly_sales = np.mean(hourly_sales, axis=0)

plt.figure(figsize=(12, 5))

plt.plot(
    range(len(average_hourly_sales)),
    average_hourly_sales,
    marker="o"
)

plt.title("Average Hourly Sales Pattern")
plt.xlabel("Hourly Interval Index")
plt.ylabel("Average Sales")
plt.xticks(range(24))
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig("reports/figures/04_hourly_sales.png")
plt.show()

# ==========================================
# 8. STOCK STATUS VALUE DISTRIBUTION
# ==========================================

stock_values = np.concatenate(
    df["hours_stock_status"].to_numpy()
)

values, counts = np.unique(
    stock_values, return_counts=True
)

plt.figure(figsize=(8, 5))

plt.bar(values.astype(str), counts)

plt.title("Hourly Stock Status Value Distribution")
plt.xlabel("Recorded Status Value")
plt.ylabel("Frequency")
plt.tight_layout()

plt.savefig("reports/figures/05_stock_status_values.png")
plt.show()

# ==========================================
# 9. WEATHER AND SALES RELATIONSHIP
# ==========================================

weather_sample = df.sample(
    n=min(10000, len(df)),
    random_state=42
)

plt.figure(figsize=(9, 5))

plt.scatter(
    weather_sample["avg_temperature"],
    weather_sample["sale_amount"],
    alpha=0.2
)

plt.title("Temperature vs Recorded Sales")
plt.xlabel("Average Temperature")
plt.ylabel("Recorded Sales")
plt.tight_layout()

plt.savefig("reports/figures/06_temperature_sales.png")
plt.show()

# ==========================================
# 10. DISCOUNT AND SALES RELATIONSHIP
# ==========================================

plt.figure(figsize=(9, 5))

plt.scatter(
    weather_sample["discount"],
    weather_sample["sale_amount"],
    alpha=0.2
)

plt.title("Discount vs Recorded Sales")
plt.xlabel("Discount")
plt.ylabel("Recorded Sales")
plt.tight_layout()

plt.savefig("reports/figures/07_discount_sales.png")
plt.show()

# ==========================================
# 11. SAVE SAMPLE SUMMARY
# ==========================================

summary = df.describe(include="all").transpose()

summary.to_csv("reports/eda_summary.csv")

print("\nEDA COMPLETED SUCCESSFULLY!")
print("Graphs saved in reports/figures/")
print("Summary saved in reports/eda_summary.csv")
