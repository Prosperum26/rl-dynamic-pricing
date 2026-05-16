"""
Generate synthetic sales data for testing without a raw e-commerce CSV.

Usage:
    python -m scripts.generate_data
    python -m scripts.generate_data --samples 10000
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.constants import (
    BASE_DEMAND,
    DEMAND_NOISE_STD,
    MAX_PRICE,
    MIN_PRICE,
    PRICE_ELASTICITY,
    RANDOM_SEED,
    SEASONALITY_AMPLITUDE,
    SEASONALITY_PERIOD,
    SYNTHETIC_SALES_PATH,
)


def generate_synthetic_sales_data(
    n_samples: int = 5000,
    start_date: str = "2022-01-01",
    end_date: str = "2023-12-31",
    output_path: str | Path | None = None,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    np.random.seed(seed)
    date_range = pd.date_range(start=start_date, end=end_date, periods=n_samples)

    price_segments = np.random.choice([0.3, 0.6, 0.9], size=n_samples, p=[0.3, 0.5, 0.2])
    prices = MIN_PRICE + price_segments * (MAX_PRICE - MIN_PRICE)
    prices += np.random.normal(0, 3, n_samples)
    prices = np.clip(prices, MIN_PRICE, MAX_PRICE).round(2)

    day_of_week = date_range.dayofweek
    month = date_range.month
    is_weekend = (day_of_week >= 5).astype(int)
    is_holiday_season = ((month == 11) | (month == 12)).astype(int)
    is_summer = ((month >= 6) & (month <= 8)).astype(int)

    price_effect = PRICE_ELASTICITY * (prices - (MIN_PRICE + MAX_PRICE) / 2)
    weekly_pattern = SEASONALITY_AMPLITUDE * 0.5 * np.sin(
        2 * np.pi * day_of_week / SEASONALITY_PERIOD
    )
    monthly_pattern = SEASONALITY_AMPLITUDE * np.sin(2 * np.pi * month / 12)
    noise = np.random.normal(0, DEMAND_NOISE_STD, n_samples)

    demand = (
        BASE_DEMAND
        + price_effect
        + weekly_pattern
        + monthly_pattern
        + is_holiday_season * 15
        + is_summer * 8
        + 0.01 * np.arange(n_samples)
        + noise
    )
    demand = np.maximum(demand, 0).astype(int)

    df = pd.DataFrame({
        "date": date_range,
        "price": prices,
        "demand": demand,
        "day_of_week": day_of_week,
        "month": month,
        "is_weekend": is_weekend,
        "is_holiday_season": is_holiday_season,
    })

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Data saved to: {output_path}")

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic sales data")
    parser.add_argument("--output", type=str, default=str(SYNTHETIC_SALES_PATH))
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--start-date", type=str, default="2022-01-01")
    parser.add_argument("--end-date", type=str, default="2023-12-31")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    print("=" * 60)
    print("Synthetic Data Generation")
    print("=" * 60)

    df = generate_synthetic_sales_data(
        n_samples=args.samples,
        start_date=args.start_date,
        end_date=args.end_date,
        output_path=args.output,
        seed=args.seed,
    )
    print(f"Shape: {df.shape}")
    print(df.head())


if __name__ == "__main__":
    main()
