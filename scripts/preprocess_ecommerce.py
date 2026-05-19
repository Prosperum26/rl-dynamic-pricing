"""
Preprocess transaction-level e-commerce data into aggregated demand time series.

Usage (from project root):
    python -m scripts.preprocess_ecommerce

    python -m scripts.preprocess_ecommerce \\
        --input data/raw/ecommerce_dynamic_pricing_dataset.csv \\
        --output data/processed/processed_sales.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.constants import PROCESSED_SALES_PATH, RAW_SALES_PATH
from scripts.preprocess_common import (
    add_effective_price,
    add_time_features,
    clean_transactions,
    load_raw_data,
    validate_raw_data,
)

PROCESSED_COLUMNS = [
    "date",
    "category",
    "demand",              # daily conversion count (sum of 0/1 purchase_probability)
    "avg_price",
    "log_avg_price",
    "transaction_count",
    "avg_discount",
    "avg_discount_rate",
    "day_of_week",
    "month",
    "is_weekend",
    "is_holiday_season",
]

def aggregate_by_date_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate to daily category level.

    demand = number of predicted conversions (purchase_probability is 0/1 in source data).
    """
    grouped = df.groupby(["date", "Product_Category"], as_index=False).agg(
        demand=("purchase_probability", "sum"),
        avg_price=("effective_price", "mean"),
        transaction_count=("Transaction_ID", "count"),
        avg_discount=("Discount", "mean"),
        day_of_week=("day_of_week", "first"),
        month=("month", "first"),
        is_weekend=("is_weekend", "first"),
    )
    grouped = grouped.rename(columns={"Product_Category": "category"})
    grouped["log_avg_price"] = np.log1p(grouped["avg_price"])
    grouped["avg_discount_rate"] = (
        grouped["avg_discount"] / grouped["avg_price"].replace(0, np.nan)
    ).fillna(0).clip(0, 1)
    grouped["is_holiday_season"] = grouped["month"].isin([11, 12]).astype(int)
    return grouped.sort_values(["date", "category"]).reset_index(drop=True)


def validate_processed_data(df: pd.DataFrame) -> None:
    missing_cols = [c for c in PROCESSED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Processed data missing columns: {missing_cols}")
    if df.duplicated(subset=["date", "category"]).any():
        raise ValueError("Duplicate (date, category) rows found after aggregation.")
    if (df["demand"] < 0).any():
        raise ValueError("Negative demand values found.")
    if (df["avg_price"] <= 0).any():
        raise ValueError("Non-positive avg_price values found.")


def print_summary(df: pd.DataFrame, label: str) -> None:
    print(f"\n{'=' * 60}")
    print(label)
    print("=" * 60)
    print(f"  Rows:        {len(df):,}")
    print(f"  Date range:  {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  Categories:  {sorted(df['category'].unique())}")
    print(f"  demand  mean: {df['demand'].mean():.2f}")
    print(f"  avg_price mean: ${df['avg_price'].mean():.2f}")


def suggest_price_constants(df: pd.DataFrame) -> None:
    prices = df["avg_price"]
    p5, p50, p95 = prices.quantile([0.05, 0.5, 0.95])
    min_price = max(1.0, np.floor(p5 / 5) * 5)
    max_price = np.ceil(p95 / 10) * 10
    step = 25.0 if (max_price - min_price) > 200 else 10.0
    print(f"\nSuggested config/constants.py: MIN_PRICE={min_price:.0f}, MAX_PRICE={max_price:.0f}, PRICE_STEP={step:.0f}")


def preprocess(
    input_path: Path,
    output_path: Path,
    apply_discount: bool = True,
) -> pd.DataFrame:
    print("=" * 60)
    print("E-commerce demand preprocessing")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")
    print("=" * 60)

    df = load_raw_data(input_path)
    validate_raw_data(df)
    df = clean_transactions(df)
    df = add_time_features(df)
    df = add_effective_price(df, apply_discount=apply_discount)
    processed = aggregate_by_date_category(df)
    validate_processed_data(processed)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed[PROCESSED_COLUMNS].to_csv(output_path, index=False)
    print(f"Saved {len(processed):,} rows")

    print_summary(processed, "Processed dataset summary")
    suggest_price_constants(processed)
    return processed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate e-commerce transactions into daily category-level demand."
    )
    parser.add_argument("--input", type=Path, default=RAW_SALES_PATH)
    parser.add_argument("--output", type=Path, default=PROCESSED_SALES_PATH)
    parser.add_argument("--no-apply-discount", action="store_true")
    parser.add_argument(
        "--with-products",
        action="store_true",
        help="Also run product-level preprocessing (product_daily + product_monthly)",
    )
    args = parser.parse_args()

    try:
        preprocess(
            input_path=args.input,
            output_path=args.output,
            apply_discount=not args.no_apply_discount,
        )
        if args.with_products:
            from scripts.preprocess_products import preprocess_products
            from config.constants import (
                PROCESSED_PRODUCT_DAILY_PATH,
                PROCESSED_PRODUCT_MONTHLY_PATH,
            )
            preprocess_products(
                input_path=args.input,
                daily_output=PROCESSED_PRODUCT_DAILY_PATH,
                monthly_output=PROCESSED_PRODUCT_MONTHLY_PATH,
                apply_discount=not args.no_apply_discount,
            )
    except (FileNotFoundError, ValueError) as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
