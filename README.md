# rl-dynamic-pricing

AI-powered **dynamic pricing** for e-commerce: forecast demand with **LightGBM**, learn a pricing policy with **PPO** (Stable-Baselines3), evaluate with **standardized benchmarks**, and explore results in a **Streamlit** dashboard.

## Link Data Set
- **E-Commerce Pricing Optimization Dataset** - https://www.kaggle.com/datasets/zoya77/e-commerce-pricing-optimization-dataset/data

## Features

- **Demand forecasting** — global LightGBM (category as feature, time-based hold-out)
- **RL simulation** — Gymnasium `PricingEnv` driven by the trained demand model
- **PPO agent** — discrete price actions; **one checkpoint per category**
- **Product analytics** — top SKUs by month (product-level aggregates)
- **Benchmark reports** — CSV + Markdown under `reports/benchmark/`
- **Interactive dashboard** — live episodes, what-if curves, strategy comparison

## Requirements

- **Python 3.10+**
- **~8 GB RAM** (CPU is sufficient for this MVP; GPU optional for long PPO runs)
- **Git**

---

## Onboarding roadmap (clone → data → train → demo)

Follow these steps in order. Each step produces artifacts used by the next.

| Step | What you do | Output |
|------|-------------|--------|
| 0 | Clone repo, create venv, install deps | `.venv/` |
| 1 | Download raw CSV → `data/raw/` | `ecommerce_dynamic_pricing_dataset.csv` |
| 2 | Preprocess | `data/processed/*.csv` |
| 3 | Train demand (LightGBM) | `models/demand_*.joblib` |
| 4 | Train PPO (per category) | `models/best_model/best_model_*.zip` |
| 5 | Run benchmarks (optional) | `reports/benchmark/` |
| 6 | Launch dashboard | Browser UI |

Estimated time on a typical laptop: **preprocess & demand ~2–5 min**, **PPO all categories ~30–90 min** (100k timesteps each).

---

## Step 0 — Clone and environment

```bash
git clone <your-repo-url> rl-dynamic-pricing
cd rl-dynamic-pricing

python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Verify install:

```bash
python scripts/quickstart.py
```

---

## Step 1 — Obtain and place the dataset

Raw data is **not committed to git** (see `.gitignore`). You must add it locally.

### Where to save

Put the file **exactly** here (filename must match):

```
data/raw/ecommerce_dynamic_pricing_dataset.csv
```

Full path example (Windows):

```
C:\Dev\Project\rl-dynamic-pricing\data\raw\ecommerce_dynamic_pricing_dataset.csv
```

### Required columns

| Column | Description |
|--------|-------------|
| `Product_ID` | SKU identifier |
| `Product_Category` | One of: Books, Clothing, Electronics, Home & Kitchen |
| `Price` | Listed price |
| `Discount` | Discount amount |
| `Purchase_Timestamp` | Event datetime |
| `Purchase Probability` | In this dataset: **0 or 1** (purchase / no purchase) |

Optional: `Transaction_ID` (used for counting transactions if present).

### How to get the file

- Use the dataset provided with your course / team / release package, **or**
- Search Kaggle / data hubs for an e-commerce dynamic pricing export with the columns above and rename it to `ecommerce_dynamic_pricing_dataset.csv`, **or**
- Generate synthetic data for pipeline testing only:

  ```bash
  python -m scripts.generate_data
  # then copy or point preprocess to data/synthetic/synthetic_sales.csv if adapted
  ```

Expected size: **~2,000+ transaction rows** (one row = one interaction, not one day).

More detail: [`data/README.md`](data/README.md).

---

## Step 2 — Preprocess

From project root (venv activated):

```bash
# Category-level (RL + LightGBM) + product-level (dashboard Product insights)
python -m scripts.preprocess_ecommerce --with-products
```

**Outputs:**

| File | Grain | Used for |
|------|-------|----------|
| `data/processed/processed_sales.csv` | day × category | LightGBM, PPO, Live simulation |
| `data/processed/product_daily.csv` | day × product × category | Product trends |
| `data/processed/product_monthly.csv` | month × product × category | Top SKU dashboard |

Category-only preprocess:

```bash
python -m scripts.preprocess_ecommerce
```

Product-only (if category files already exist):

```bash
python -m scripts.preprocess_products
```

---

## Step 3 — Train demand model (LightGBM)

```bash
python -m training.train_demand_predictor
```

**Outputs:**

- `models/demand_predictor.joblib`
- `models/demand_encoder.joblib`
- `models/demand_predictor_metadata.json` (test metrics: R², WMAPE, MAPE, …)

MVP gate (documented in training log): test **R² ≥ 0.30** on time-based hold-out.

---

## Step 4 — Train PPO pricing policies

Train **after** demand model exists. PPO learns inside a simulator that calls LightGBM for demand.

### Recommended — one policy per category (Streamlit auto-load)

```bash
python -m training.train_ppo --train-all-categories --timesteps 100000
```

Saves:

- `models/best_model/best_model_books.zip`
- `models/best_model/best_model_clothing.zip`
- `models/best_model/best_model_electronics.zip`
- `models/best_model/best_model_home_and_kitchen.zip`

### Single category

```bash
python -m training.train_ppo --timesteps 100000 --category Books
```

### One shared policy (random category each episode)

```bash
python -m training.train_ppo --timesteps 100000 --category all
```

→ `models/best_model/best_model_all.zip`

### Evaluate a checkpoint

```bash
python -m training.train_ppo --evaluate models/best_model/best_model_books.zip --category Books
```

---

## Step 5 — Benchmark report (optional, recommended for reports)

```bash
python -m scripts.run_benchmark
python -m scripts.run_benchmark --pricing-episodes 20   # include PPO vs baselines
```

**Outputs:** `reports/benchmark/<timestamp>/` — see [`reports/benchmark/README.md`](reports/benchmark/README.md).

Latest copy: `reports/benchmark/BENCHMARK_REPORT.md`, `benchmark_results.csv`.

---

## Step 6 — Dashboard

```bash
python -m streamlit run dashboard/streamlit_app.py
```

In the sidebar:

- **Product category** — switches demand context and **Auto PPO per category** checkpoint
- **Use LightGBM demand model** / **Use trained PPO** — should be enabled after training

---

## Project structure

```
rl-dynamic-pricing/
├── benchmarks/             # Benchmark thresholds & citations (standards.py)
├── config/                 # Hyperparameters & paths (constants.py)
├── data/                   # Datasets (CSVs not in git)
│   ├── raw/                # ecommerce_dynamic_pricing_dataset.csv
│   └── processed/          # Aggregated tables
├── scripts/                # preprocess_*, run_benchmark, quickstart
├── environment/            # PricingEnv, category wrapper
├── training/               # demand_*, train_ppo, benchmark_eval
├── dashboard/              # Streamlit app
├── reports/benchmark/      # Generated evaluation reports
├── models/                 # Trained artifacts (local)
├── logs/                   # PPO training logs (local)
├── README.md               # This file (onboarding EN)
└── README_LEARNING.md      # Deep dive + report writing (VI)
```

---

## End-to-end pipeline

```
data/raw/ecommerce_dynamic_pricing_dataset.csv
      |
      v
scripts/preprocess_ecommerce [--with-products]
      |
      +--> data/processed/processed_sales.csv
      +--> data/processed/product_daily.csv
      +--> data/processed/product_monthly.csv
      |
      v
training/train_demand_predictor  -->  models/demand_*.joblib
      |
      v
training/train_ppo  -->  models/best_model/best_model_<category>.zip
      |
      +--> scripts/run_benchmark  -->  reports/benchmark/
      |
      v
dashboard/streamlit_app.py
```

---

## Configuration

Edit [`config/constants.py`](config/constants.py) for price grid ($25–$975, step $25), inventory, categories, PPO/LightGBM hyperparameters.

| Constant | Default path |
|----------|----------------|
| `RAW_SALES_PATH` | `data/raw/ecommerce_dynamic_pricing_dataset.csv` |
| `PROCESSED_SALES_PATH` | `data/processed/processed_sales.csv` |
| `PROCESSED_PRODUCT_MONTHLY_PATH` | `data/processed/product_monthly.csv` |

---

## CLI reference

| Command | Purpose |
|---------|---------|
| `python -m scripts.preprocess_ecommerce --with-products` | Raw → category + product tables |
| `python -m training.train_demand_predictor` | Train LightGBM |
| `python -m training.train_ppo --train-all-categories --timesteps N` | Train 4 category policies |
| `python -m training.train_ppo --category Books --timesteps N` | Train one category |
| `python -m training.train_ppo --category all` | Single multi-category policy |
| `python -m training.train_ppo --evaluate <path.zip>` | Evaluate PPO vs baselines |
| `python -m scripts.run_benchmark` | Export benchmark CSV + Markdown |
| `python -m streamlit run dashboard/streamlit_app.py` | Web UI |
| `python scripts/quickstart.py` | Environment smoke test |

---

## Git vs local files

| In repository | Local only (regenerate) |
|---------------|---------------------------|
| Source code, docs, `data/README.md` | `data/**/*.csv` |
| `reports/benchmark/README.md` | `reports/benchmark/**/outputs` |
| `.gitkeep` under `models/`, `logs/` | `models/*`, `logs/*`, `.venv/` |

---

## Documentation

| Document | Audience | Content |
|----------|----------|---------|
| **[README_LEARNING.md](README_LEARNING.md)** | Report / thesis (VI) | Concepts, design decisions, evaluation, limits, future work |
| **[reports/benchmark/README.md](reports/benchmark/README.md)** | Evaluation | How to run standardized benchmarks |
| **[data/README.md](data/README.md)** | Data setup | Schema and paths |

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `Input file not found` | Place CSV in `data/raw/` with exact filename |
| `Demand model not found` | Run `train_demand_predictor` before PPO or Streamlit |
| PPO wrong on Streamlit category change | Enable **Auto PPO per category**; train with `--train-all-categories` |
| `streamlit` not found | Activate `.venv`, use `python -m streamlit run ...` |
| PPO train crashes at end (`SameFileError`) | Pull latest code (archive fix) and re-run |

---

## License

MIT — see [LICENSE](LICENSE).
