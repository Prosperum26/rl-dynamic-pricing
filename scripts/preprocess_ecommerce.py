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

REQUIRED_RAW_COLUMNS = [
    "Product_Category",
    "Price",
    "Discount",
    "Purchase_Timestamp",
]

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

PURCHASE_PROB_COL = "Purchase Probability"


def load_raw_data(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    if PURCHASE_PROB_COL in df.columns:
        df = df.rename(columns={PURCHASE_PROB_COL: "purchase_probability"})
    elif "purchase_probability" not in df.columns:
        raise ValueError(
            f"Missing purchase probability column. Expected '{PURCHASE_PROB_COL}' "
            "or 'purchase_probability'."
        )

    return df


def validate_raw_data(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_RAW_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Raw dataset is missing required columns: {missing}")
    if "purchase_probability" not in df.columns:
        raise ValueError("Raw dataset must include purchase probability.")
    if len(df) == 0:
        raise ValueError("Raw dataset is empty.")


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    n_start = len(df)
    df = df.dropna(
        subset=["Purchase_Timestamp", "Price", "purchase_probability", "Product_Category"]
    ).copy()
    df["purchase_probability"] = df["purchase_probability"].clip(0.0, 1.0)
    df = df[df["Price"] > 0]
    df["Discount"] = df["Discount"].fillna(0).clip(lower=0)
    df.loc[df["Discount"] > df["Price"], "Discount"] = df.loc[
        df["Discount"] > df["Price"], "Price"
    ]
    n_dropped = n_start - len(df)
    if n_dropped > 0:
        print(f"  Dropped or fixed rows: {n_dropped} removed, {len(df)} remaining")
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Purchase_Timestamp"] = pd.to_datetime(df["Purchase_Timestamp"], errors="coerce")
    invalid_dates = df["Purchase_Timestamp"].isna().sum()
    if invalid_dates > 0:
        print(f"  Warning: {invalid_dates} rows have unparseable timestamps (dropped)")
        df = df.dropna(subset=["Purchase_Timestamp"])

    df["date"] = df["Purchase_Timestamp"].dt.normalize()
    df["day_of_week"] = df["Purchase_Timestamp"].dt.dayofweek
    df["month"] = df["Purchase_Timestamp"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df


def compute_effective_price(df: pd.DataFrame, apply_discount: bool = True) -> pd.DataFrame:
    df = df.copy()
    if apply_discount:
        discount_rate = (df["Discount"] / df["Price"]).clip(0.0, 1.0)
        df["effective_price"] = df["Price"] * (1.0 - discount_rate)
    else:
        df["effective_price"] = df["Price"]
    return df


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
    df = handle_missing_values(df)
    df = add_time_features(df)
    df = compute_effective_price(df, apply_discount=apply_discount)
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
    args = parser.parse_args()

    try:
        preprocess(
            input_path=args.input,
            output_path=args.output,
            apply_discount=not args.no_apply_discount,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
