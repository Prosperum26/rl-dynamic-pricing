"""
Build product-level aggregates from raw transactions (Direction B).

Outputs:
  - data/processed/product_daily.csv   (date x product_id x category)
  - data/processed/product_monthly.csv (year x month x product_id x category)

Usage:
    python -m scripts.preprocess_products
    python -m scripts.preprocess_products --input data/raw/ecommerce_dynamic_pricing_dataset.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.constants import (
    PROCESSED_PRODUCT_DAILY_PATH,
    PROCESSED_PRODUCT_MONTHLY_PATH,
    RAW_SALES_PATH,
)
from scripts.preprocess_common import (
    add_effective_price,
    add_time_features,
    clean_transactions,
    load_raw_data,
    validate_raw_data,
)

PRODUCT_DAILY_COLUMNS = [
    "date",
    "product_id",
    "category",
    "demand",
    "transaction_count",
    "avg_price",
    "avg_discount_rate",
    "revenue_proxy",
    "day_of_week",
    "month",
    "is_weekend",
]

PRODUCT_MONTHLY_COLUMNS = [
    "year",
    "month",
    "product_id",
    "category",
    "demand",
    "transaction_count",
    "avg_price",
    "revenue_proxy",
]


def aggregate_product_daily(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["date", "Product_ID", "Product_Category"], as_index=False).agg(
        demand=("purchase_probability", "sum"),
        transaction_count=("Transaction_ID", "count"),
        avg_price=("effective_price", "mean"),
        avg_discount=("Discount", "mean"),
        revenue_proxy=("revenue_proxy", "sum"),
        day_of_week=("day_of_week", "first"),
        month=("month", "first"),
        is_weekend=("is_weekend", "first"),
    )
    grouped = grouped.rename(columns={
        "Product_ID": "product_id",
        "Product_Category": "category",
    })
    grouped["avg_discount_rate"] = (
        grouped["avg_discount"] / grouped["avg_price"].replace(0, pd.NA)
    ).fillna(0).clip(0, 1)
    grouped = grouped.drop(columns=["avg_discount"])
    return grouped.sort_values(["date", "category", "product_id"]).reset_index(drop=True)


def aggregate_product_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    daily = daily.copy()
    daily["year"] = pd.to_datetime(daily["date"]).dt.year
    grouped = daily.groupby(["year", "month", "product_id", "category"], as_index=False).agg(
        demand=("demand", "sum"),
        transaction_count=("transaction_count", "sum"),
        avg_price=("avg_price", "mean"),
        revenue_proxy=("revenue_proxy", "sum"),
    )
    return grouped.sort_values(
        ["year", "month", "category", "demand"],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)


def preprocess_products(
    input_path: Path,
    daily_output: Path,
    monthly_output: Path,
    apply_discount: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("=" * 60)
    print("Product-level preprocessing")
    print(f"  Input:          {input_path}")
    print(f"  Daily output:   {daily_output}")
    print(f"  Monthly output: {monthly_output}")
    print("=" * 60)

    df = load_raw_data(input_path)
    validate_raw_data(df)
    df = clean_transactions(df)
    df = add_time_features(df)
    df = add_effective_price(df, apply_discount=apply_discount)

    daily = aggregate_product_daily(df)
    monthly = aggregate_product_monthly(daily)

    daily_output.parent.mkdir(parents=True, exist_ok=True)
    daily[PRODUCT_DAILY_COLUMNS].to_csv(daily_output, index=False)
    monthly[PRODUCT_MONTHLY_COLUMNS].to_csv(monthly_output, index=False)

    print(f"\nSaved product_daily:   {len(daily):,} rows")
    print(f"Saved product_monthly: {len(monthly):,} rows")
    print(f"  Unique products: {daily['product_id'].nunique()}")
    print(f"  Date range: {daily['date'].min().date()} to {daily['date'].max().date()}")

    return daily, monthly


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate raw data to product-level tables")
    parser.add_argument("--input", type=Path, default=RAW_SALES_PATH)
    parser.add_argument("--daily-output", type=Path, default=PROCESSED_PRODUCT_DAILY_PATH)
    parser.add_argument("--monthly-output", type=Path, default=PROCESSED_PRODUCT_MONTHLY_PATH)
    parser.add_argument("--no-apply-discount", action="store_true")
    args = parser.parse_args()

    try:
        preprocess_products(
            input_path=args.input,
            daily_output=args.daily_output,
            monthly_output=args.monthly_output,
            apply_discount=not args.no_apply_discount,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
