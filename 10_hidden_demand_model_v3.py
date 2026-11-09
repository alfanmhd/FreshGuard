
import pandas as pd
import numpy as np
import os
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
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
df = df.sort_values(["store_id", "sku_id", "date"])

# ==========================================
# 1. CALENDAR FEATURES
# ==========================================

df["month"] = df["date"].dt.month
df["day_of_month"] = df["date"].dt.day
df["day_of_year"] = df["date"].dt.day_of_year

# ==========================================
# 2. TEMPORAL SPLIT
# ==========================================

cutoff = pd.Timestamp("2025-07-01")

# All historical demand labels are available
# in this synthetic benchmark.
train_df = df[df["date"] < cutoff].copy()

# Evaluate only on future stockout observations
test_df = df[
    (df["date"] >= cutoff) &
    (df["stockout_flag"] == 1)
].copy()

print("\n========== DATA SPLIT ==========")
print("Training records:", len(train_df))
print("Training stockout records:", train_df["stockout_flag"].sum())
print("Test stockout records:", len(test_df))

# ==========================================
# 3. FEATURES
# ==========================================

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

X_train = train_df[features]
y_train = train_df["true_demand"]

X_test = test_df[features]
y_test = test_df["true_demand"]

# ==========================================
# 4. PREPROCESSING
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            SimpleImputer(strategy="median"),
            numerical_features
        )
    ]
)

# ==========================================
# 5. RANDOM FOREST V3
# ==========================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=150,
                max_depth=20,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

print("\nTraining Random Forest V3...")

model.fit(X_train, y_train)

print("Training completed!")

# ==========================================
# 6. PREDICTIONS
# ==========================================

predictions = np.maximum(
    model.predict(X_test),
    0
)

# ==========================================
# 7. EVALUATION
# ==========================================

def evaluate(actual, predicted):
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(
            mean_squared_error(actual, predicted)
        ),
        "Bias": np.mean(predicted - actual)
    }

results = [
    {
        "Model": "Recorded Sales Baseline",
        **evaluate(y_test, test_df["units_sold"])
    },
    {
        "Model": "Random Forest V3",
        **evaluate(y_test, predictions)
    }
]

results = pd.DataFrame(results)

print("\n========== V3 MODEL RESULTS ==========")
print(results.round(4).to_string(index=False))

# ==========================================
# 8. SAVE RESULTS
# ==========================================

os.makedirs("reports", exist_ok=True)
os.makedirs("models", exist_ok=True)

results.to_csv(
    "reports/hidden_demand_v3_results.csv",
    index=False
)

joblib.dump(
    model,
    "models/hidden_demand_random_forest_v3.joblib"
)

print("\nV3 model and results saved successfully.")
