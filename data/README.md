# Data directory

**Datasets only** — no Python code here. Preprocessing scripts live in [`scripts/`](../scripts/).

## Layout

```
data/
├── raw/           # Transaction-level exports (not in git)
├── processed/     # Aggregated training tables (not in git)
└── synthetic/     # Optional generated data for testing (not in git)
```

## Setup

1. Place your raw file here:

   `data/raw/ecommerce_dynamic_pricing_dataset.csv`

2. Run preprocessing (from project root):

   ```bash
   python -m scripts.preprocess_ecommerce
   ```

3. Output appears at:

   `data/processed/processed_sales.csv`

## File reference

| Path | Description |
|------|-------------|
| `raw/ecommerce_dynamic_pricing_dataset.csv` | Source transactions |
| `processed/processed_sales.csv` | Daily demand by category (for LightGBM / RL) |
| `synthetic/synthetic_sales.csv` | Optional fake data from `scripts.generate_data` |

## Processed schema

`date`, `category`, `demand` (daily conversions), `avg_price`, `log_avg_price`, `transaction_count`, `avg_discount`, `avg_discount_rate`, `day_of_week`, `month`, `is_weekend`, `is_holiday_season`
