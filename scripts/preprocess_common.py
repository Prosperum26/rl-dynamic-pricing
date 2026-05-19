"""
Shared helpers for e-commerce preprocessing scripts.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PURCHASE_PROB_COL = "Purchase Probability"

REQUIRED_RAW_COLUMNS = [
    "Product_ID",
    "Product_Category",
    "Price",
    "Discount",
    "Purchase_Timestamp",
]


def load_raw_data(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    if PURCHASE_PROB_COL in df.columns:
        df = df.rename(columns={PURCHASE_PROB_COL: "purchase_probability"})
    elif "purchase_probability" not in df.columns:
        raise ValueError(
            f"Missing purchase probability column. Expected '{PURCHASE_PROB_COL}'."
        )

    return df


def validate_raw_data(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_RAW_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Raw dataset is missing required columns: {missing}")
    if len(df) == 0:
        raise ValueError("Raw dataset is empty.")


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    n_start = len(df)
    df = df.dropna(
        subset=[
            "Purchase_Timestamp",
            "Price",
            "purchase_probability",
            "Product_Category",
            "Product_ID",
        ]
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
    invalid = df["Purchase_Timestamp"].isna().sum()
    if invalid > 0:
        print(f"  Warning: {invalid} rows with unparseable timestamps (dropped)")
        df = df.dropna(subset=["Purchase_Timestamp"])

    df["date"] = df["Purchase_Timestamp"].dt.normalize()
    df["year"] = df["Purchase_Timestamp"].dt.year
    df["day_of_week"] = df["Purchase_Timestamp"].dt.dayofweek
    df["month"] = df["Purchase_Timestamp"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df


def add_effective_price(df: pd.DataFrame, apply_discount: bool = True) -> pd.DataFrame:
    df = df.copy()
    if apply_discount:
        rate = (df["Discount"] / df["Price"]).clip(0.0, 1.0)
        df["effective_price"] = df["Price"] * (1.0 - rate)
    else:
        df["effective_price"] = df["Price"]
    df["revenue_proxy"] = df["effective_price"] * df["purchase_probability"]
    return df
