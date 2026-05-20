# Data directory

**Datasets only** — no Python code here. Preprocessing scripts live in [`scripts/`](../scripts/).

## Layout

```
data/
├── raw/              # Transaction-level exports (not in git)
├── processed/        # Aggregated tables (not in git)
│   ├── processed_sales.csv    # day × category (RL + LightGBM)
│   ├── product_daily.csv      # day × product_id × category
│   └── product_monthly.csv    # month × product_id × category
└── synthetic/        # Optional generated data (not in git)
```

## Setup

See the main **[README.md](../README.md)** for the full clone → train roadmap.

1. Place your raw file here:

   `data/raw/ecommerce_dynamic_pricing_dataset.csv`

2. Run preprocessing (from project root):

   ```bash
   python -m scripts.preprocess_ecommerce --with-products
   ```

3. Outputs:

   - `data/processed/processed_sales.csv` — category-level (train RL)
   - `data/processed/product_daily.csv` — product-level daily
   - `data/processed/product_monthly.csv` — product-level monthly (dashboard)

## File reference

| Path | Description |
|------|-------------|
| `raw/ecommerce_dynamic_pricing_dataset.csv` | Source transactions |
| `processed/processed_sales.csv` | Daily demand by category (for LightGBM / RL) |
| `synthetic/synthetic_sales.csv` | Optional fake data from `scripts.generate_data` |

## Processed schema

`date`, `category`, `demand` (daily conversions), `avg_price`, `log_avg_price`, `transaction_count`, `avg_discount`, `avg_discount_rate`, `day_of_week`, `month`, `is_weekend`, `is_holiday_season`
