# FreshGuard datasets guide

## Recommended strategy: two datasets

| Role | Dataset | Why |
|---|---|---|
| **Primary real data** | FreshRetailNet-50K (Hugging Face: `Dingdong-Inc/FreshRetailNet-50K`) | Real perishable sales with verified hourly stockout flags, about 20% organic stockouts, 863 SKUs, 898 stores, CC BY 4.0 |
| **Evaluation / simulation data** | Synthetic `sales_daily.csv` (this folder) | Real data never reveals true demand. Synthetic data does, so you can measure how well your hidden-demand model recovers it |
| Optional extras | Favorita (Kaggle, has a `perishable` flag and promo), M5 (Kaggle, FOODS category, prices, SNAP) | Good for extra baseline benchmarks. Neither has stockout labels |

## Loading FreshRetailNet-50K
```python
from datasets import load_dataset
ds = load_dataset("Dingdong-Inc/FreshRetailNet-50K")   # train 4.5M rows, eval split
df = ds["train"].to_pandas()
```
Fields include hourly sales, hourly stock status, `activity_flag` (promotion), precipitation, temperature, humidity and wind. Aggregate hours to days to get daily sales and stockout hours.
Baseline code: https://github.com/Dingdong-Inc/frn-50k-baseline

## Synthetic files
- `sales_daily.csv`: what a retailer observes. Use this to train. 12 stores x 40 SKUs x 731 days = 350,880 rows.
- `ground_truth_demand.csv`: `true_demand`, `lost_sales`. **Evaluation only. Never use as a feature.**
- `products.csv`: category, shelf life, price, unit cost, salvage value (for cost simulation).
- `stores.csv`, `calendar.csv`.

### Columns in sales_daily.csv
`units_sold` (censored target), `opening_stock`, `stockout_flag`, `stockout_hours`, `price`, `discount_pct`, `promo_flag`, `units_received`, `units_wasted`, `temp_c`, `rain_mm`, `dow`, `is_weekend`, `is_holiday`, `is_pre_holiday`.

### Realism checks (seed 42)
- Stockout rate 19.2% of series-days (FreshRetailNet-50K reports about 20%)
- Observed sales are 6.7% below true demand overall (the paper reports 7.37% underestimation)
- Observed sales never exceed true demand; they equal it whenever there is no stockout
- Waste is low (about 1%) because FIFO selling plus frequent restocking clears stock. Raise `replen_safety` in the script for a higher-waste scenario.

## Suggested project use
1. Stage 1 (hidden demand): train on non-stockout days, predict demand on stockout days. Score against `true_demand` (synthetic) and against held-out tests (real).
2. Stage 2 (forecast): train forecasters on recovered demand vs raw sales. Compare MAE / RMSE / bias.
3. Inventory simulation: use `unit_cost`, `base_price`, `salvage_value`, `shelf_life_days` to compute stockout rate, waste and profit under different order-up-to policies.

Regenerate with different settings: `python generate_freshguard_data.py --seed 7 --stores 20 --skus 60`
