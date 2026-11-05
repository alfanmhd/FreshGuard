
from datasets import load_from_disk
import numpy as np
import pandas as pd

print("Loading FreshRetailNet...")

dataset = load_from_disk("data/raw/freshretailnet")
train = dataset["train"]

# Select a random sample
sample = train.shuffle(seed=42).select(range(10000))
df = sample.to_pandas()

# Convert hourly status lists into arrays
status_arrays = np.array(df["hours_stock_status"].tolist())

print("\n========== STOCK STATUS ANALYSIS ==========")

print("Hourly array shape:", status_arrays.shape)

unique_values, counts = np.unique(
    status_arrays, return_counts=True
)

for value, count in zip(unique_values, counts):
    print(f"Status {value}: {count:,} occurrences")

# Test which value represents out-of-stock
# The documented count covers the 06:00-22:00 period.
documented_count = df["stock_hour6_22_cnt"].to_numpy()

print("\n========== COUNT VALIDATION ==========")

for value in unique_values:

    # Test whether this value matches the daily out-of-stock count
    calculated_count = np.sum(
        status_arrays[:, 6:22] == value,
        axis=1
    )

    matches = np.mean(calculated_count == documented_count)

    print(
        f"Status {value}: "
        f"count agreement = {matches:.2%}"
    )

# Validate hourly sales against daily sales
hourly_sales = np.array(df["hours_sale"].tolist())

hourly_totals = hourly_sales.sum(axis=1)
daily_sales = df["sale_amount"].to_numpy()

differences = np.abs(hourly_totals - daily_sales)

print("\n========== SALES VALIDATION ==========")

print("Maximum difference:", differences.max())
print("Mean difference:", differences.mean())

print("\nValidation completed.")
