
import os
from collections import defaultdict, deque

import numpy as np
import pandas as pd


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

DATA_DIR = "data/raw/synthetic"
REPORT_DIR = "reports"

os.makedirs(REPORT_DIR, exist_ok=True)

CUTOFF_DATE = "2025-07-01"

TARGET_STOCK_DAYS = 2
HISTORY_DAYS = 7


# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

print("\nLoading inventory simulation data...")

sales = pd.read_csv(f"{DATA_DIR}/sales_daily.csv")
truth = pd.read_csv(f"{DATA_DIR}/ground_truth_demand.csv")
products = pd.read_csv(f"{DATA_DIR}/products.csv")

predictions = pd.read_csv(
    f"{REPORT_DIR}/final_demand_predictions.csv"
)

sales["date"] = pd.to_datetime(sales["date"])
truth["date"] = pd.to_datetime(truth["date"])
predictions["date"] = pd.to_datetime(predictions["date"])

df = sales.merge(
    truth,
    on=["date", "store_id", "sku_id"],
    validate="one_to_one"
)

# Predictions exist only for stockout days.
df = df.merge(
    predictions[
        ["date", "store_id", "sku_id", "predicted_lost_sales"]
    ],
    on=["date", "store_id", "sku_id"],
    how="left",
    validate="one_to_one"
)

df["predicted_lost_sales"] = (
    df["predicted_lost_sales"].fillna(0)
)

df = df.merge(
    products,
    on="sku_id",
    how="left",
    validate="many_to_one"
)

df = df.sort_values(
    ["store_id", "sku_id", "date"]
).reset_index(drop=True)

df["estimated_demand"] = (
    df["units_sold"] + df["predicted_lost_sales"]
)

test = df[df["date"] >= CUTOFF_DATE].copy()
history_data = df[df["date"] < CUTOFF_DATE].copy()

print("Total test records:", len(test))


# --------------------------------------------------
# 3. PREPARE HISTORICAL DEMAND ESTIMATES
# --------------------------------------------------

# Before the test period, use observed sales only.
# During the test period, use reconstructed demand
# from the previous day's observations.

histories = {}

for key, group in history_data.groupby(
    ["store_id", "sku_id"]
):
    histories[key] = deque(
        group.sort_values("date")["units_sold"].tail(HISTORY_DAYS).tolist(),
        maxlen=HISTORY_DAYS
    )


# --------------------------------------------------
# 4. INVENTORY SIMULATION
# --------------------------------------------------

simulation_results = []

for key, group in test.groupby(["store_id", "sku_id"]):

    group = group.sort_values("date")

    store_id, sku_id = key

    shelf_life = int(group["shelf_life_days"].iloc[0])
    unit_cost = float(group["unit_cost"].iloc[0])
    salvage_value = float(group["salvage_value"].iloc[0])

    history = histories.get(
        key,
        deque(maxlen=HISTORY_DAYS)
    )

    # Inventory lots: each lot stores its age and quantity.
    inventory = []

    first_opening_stock = float(
        group["opening_stock"].iloc[0]
    )

    if first_opening_stock > 0:
        inventory.append({
            "age": 0,
            "quantity": first_opening_stock
        })

    for _, row in group.iterrows():

        # Expire stock before today's trading.
        expired_today = 0

        remaining_lots = []

        for lot in inventory:
            if lot["age"] >= shelf_life:
                expired_today += lot["quantity"]
            else:
                remaining_lots.append(lot)

        inventory = remaining_lots

        # Forecast using only previous days.
        if len(history) > 0:
            forecast = float(np.mean(history))
        else:
            forecast = float(row["units_sold"])

        # Order enough to reach the target stock level.
        stock_on_hand = sum(
            lot["quantity"] for lot in inventory
        )

        # Shelf-life-aware inventory target
        target_coverage_days = min(TARGET_STOCK_DAYS,max(0.5, shelf_life * 0.5))

        target_stock = forecast * target_coverage_days

        order_quantity = max(
            0,
            target_stock - stock_on_hand
        )

        # Assume same-day delivery (zero lead time).
        if order_quantity > 0:
            inventory.append({
                "age": 0,
                "quantity": order_quantity
            })

        # Actual demand is used only to evaluate the policy.
        actual_demand = float(row["true_demand"])

        remaining_demand = actual_demand
        fulfilled = 0

        # FIFO selling: oldest inventory is sold first.
        for lot in inventory:

            sold = min(
                lot["quantity"],
                remaining_demand
            )

            lot["quantity"] -= sold
            remaining_demand -= sold
            fulfilled += sold

            if remaining_demand <= 0:
                break

        # Remove empty lots.
        inventory = [
            lot for lot in inventory
            if lot["quantity"] > 0
        ]

        lost_sales = remaining_demand

        # Age remaining stock by one day.
        for lot in inventory:
            lot["age"] += 1

        closing_stock = sum(
            lot["quantity"] for lot in inventory
        )

        simulation_results.append({
            "date": row["date"],
            "store_id": store_id,
            "sku_id": sku_id,
            "actual_demand": actual_demand,
            "forecast_demand": forecast,
            "order_quantity": order_quantity,
            "fulfilled_sales": fulfilled,
            "lost_sales": lost_sales,
            "expired_units": expired_today,
            "closing_stock": closing_stock,
            "purchase_cost": order_quantity * unit_cost,
            "waste_cost": expired_today * (
                unit_cost - salvage_value
            )
        })

        # Today's reconstructed demand becomes available
        # for tomorrow's forecast, never today's order.
        history.append(float(row["estimated_demand"]))


simulation = pd.DataFrame(simulation_results)


# --------------------------------------------------
# 5. BASELINE AND FRESHGUARD METRICS
# --------------------------------------------------

baseline = {
    "Strategy": "Existing Synthetic Strategy",
    "Demand": test["true_demand"].sum(),
    "Sales Fulfilled": test["units_sold"].sum(),
    "Lost Sales": test["lost_sales"].sum(),
    "Waste Units": test["units_wasted"].sum(),
    "Units Received": test["units_received"].sum(),
    "Purchase Cost": (
        test["units_received"] * test["unit_cost"]
    ).sum(),
    "Waste Cost": (
        test["units_wasted"]
        * (test["unit_cost"] - test["salvage_value"])
    ).sum()
}

freshguard = {
    "Strategy": "FreshGuard Inventory Strategy",
    "Demand": simulation["actual_demand"].sum(),
    "Sales Fulfilled": simulation["fulfilled_sales"].sum(),
    "Lost Sales": simulation["lost_sales"].sum(),
    "Waste Units": simulation["expired_units"].sum(),
    "Units Received": simulation["order_quantity"].sum(),
    "Purchase Cost": simulation["purchase_cost"].sum(),
    "Waste Cost": simulation["waste_cost"].sum()
}

results = pd.DataFrame([baseline, freshguard])

results["Lost Sales Rate (%)"] = (
    results["Lost Sales"] / results["Demand"] * 100
)

results["Waste Rate (%)"] = (
    results["Waste Units"] / results["Units Received"].replace(0, np.nan) * 100
)

results["Total Cost"] = (
    results["Purchase Cost"] + results["Waste Cost"]
)

print("\nINVENTORY SIMULATION RESULTS")
print(results.round(2).to_string(index=False))


# --------------------------------------------------
# 6. CALCULATE IMPROVEMENTS
# --------------------------------------------------

def improvement(old, new):
    if old == 0:
        return np.nan
    return (old - new) / old * 100


print("\nFRESHGUARD PERFORMANCE")

print(
    "Lost sales reduction:",
    round(
        improvement(
            baseline["Lost Sales"],
            freshguard["Lost Sales"]
        ),
        2
    ),
    "%"
)

print(
    "Waste reduction:",
    round(
        improvement(
            baseline["Waste Units"],
            freshguard["Waste Units"]
        ),
        2
    ),
    "%"
)

print(
    "Total cost reduction:",
    round(
        improvement(
            baseline["Purchase Cost"] + baseline["Waste Cost"],
            freshguard["Purchase Cost"] + freshguard["Waste Cost"]
        ),
        2
    ),
    "%"
)


# --------------------------------------------------
# 7. SAVE RESULTS
# --------------------------------------------------

results.to_csv(
    f"{REPORT_DIR}/inventory_simulation_results.csv",
    index=False
)

simulation.to_csv(
    f"{REPORT_DIR}/inventory_simulation_daily.csv",
    index=False
)

print("\nSaved:")
print("- reports/inventory_simulation_results.csv")
print("- reports/inventory_simulation_daily.csv")

print("\nFreshGuard inventory simulation completed.")
