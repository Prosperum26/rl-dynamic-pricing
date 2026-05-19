# rl-dynamic-pricing

AI-powered **dynamic pricing** for e-commerce: forecast demand with **LightGBM**, learn a pricing policy with **PPO** (Stable-Baselines3), and explore results in a **Streamlit** dashboard.

## Features

- **Demand forecasting** — global LightGBM model with product category as a feature
- **RL simulation** — Gymnasium `PricingEnv` driven by the trained demand model
- **PPO agent** — discrete price actions over a calibrated price grid
- **Interactive dashboard** — live episodes, what-if demand curves, strategy comparison, training charts

## Project structure

```
rl-dynamic-pricing/
├── config/                 # Hyperparameters & paths (constants.py)
├── data/                   # Datasets only (CSVs not in git)
│   ├── raw/                # Transaction-level exports
│   ├── processed/          # Aggregated training tables
│   ├── synthetic/          # Optional generated data
│   └── README.md
├── scripts/                # Preprocessing & CLI utilities
│   ├── preprocess_ecommerce.py
│   ├── generate_data.py
│   └── quickstart.py
├── environment/            # Gymnasium PricingEnv
├── training/               # Model training
├── dashboard/              # Streamlit web UI
├── notebooks/
├── models/                 # Saved models (local)
├── logs/                   # PPO training logs (local)
└── requirements.txt
```

## Quick start

### 1. Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### 2. Data

Place raw CSV here:

`data/raw/ecommerce_dynamic_pricing_dataset.csv`

Preprocess:

```bash
python -m scripts.preprocess_ecommerce
```

Output: `data/processed/processed_sales.csv`

### 3. Train demand model

```bash
python -m training.train_demand_predictor
```

### 4. Train PPO agent

```bash
python -m training.train_ppo --timesteps 50000 --category Books
```

### 5. Dashboard

```bash
streamlit run dashboard/streamlit_app.py
```

### Smoke test

```bash
python scripts/quickstart.py
```

## Pipeline

```
data/raw/*.csv
      |
      v
scripts/preprocess_ecommerce.py  -->  data/processed/processed_sales.csv
      |
      v
training/train_demand_predictor.py  -->  models/demand_*.joblib
      |
      v
training/train_ppo.py  -->  models/best_model/best_model.zip
      |
      v
dashboard/streamlit_app.py
```

## Configuration

Edit [`config/constants.py`](config/constants.py) for price grid, inventory, categories, and training hyperparameters. Paths:

| Constant | Default path |
|----------|----------------|
| `RAW_SALES_PATH` | `data/raw/ecommerce_dynamic_pricing_dataset.csv` |
| `PROCESSED_SALES_PATH` | `data/processed/processed_sales.csv` |
| `SYNTHETIC_SALES_PATH` | `data/synthetic/synthetic_sales.csv` |

## CLI reference

| Command | Purpose |
|---------|---------|
| `python -m scripts.preprocess_ecommerce` | Raw CSV → processed |
| `python -m scripts.generate_data` | Synthetic data → `data/synthetic/` |
| `python -m training.train_demand_predictor` | Train LightGBM |
| `python -m training.train_ppo --timesteps N` | Train PPO |
| `python -m training.train_ppo --evaluate models/best_model/best_model.zip` | Evaluate |
| `streamlit run dashboard/streamlit_app.py` | Web UI |

## Git vs local files

| In repository | Local only |
|---------------|------------|
| Source code, `data/README.md`, `.gitkeep` | `data/**/*.csv`, `models/*`, `logs/*`, `.venv/` |

## Requirements

- Python 3.10+
- ~8 GB RAM (CPU is fine for this MVP)

## Documentation

- **[README_LEARNING.md](README_LEARNING.md)** — Giải thích chi tiết toàn project (tiếng Việt): data, training, Streamlit, giới hạn dataset & model.

## License

MIT — see [LICENSE](LICENSE).
