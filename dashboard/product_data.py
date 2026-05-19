"""
Load product-level processed tables for Streamlit analytics.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from config.constants import (
    PROCESSED_PRODUCT_DAILY_PATH,
    PROCESSED_PRODUCT_MONTHLY_PATH,
)


@st.cache_data(show_spinner="Loading product monthly data...")
def load_product_monthly(path: str | Path = PROCESSED_PRODUCT_MONTHLY_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner="Loading product daily data...")
def load_product_daily(path: str | Path = PROCESSED_PRODUCT_DAILY_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def top_products_in_period(
    monthly: pd.DataFrame,
    category: str,
    year: int,
    month: int | None = None,
    top_n: int = 10,
    sort_by: str = "demand",
) -> pd.DataFrame:
    """Rank products within a category for a year (optional single month)."""
    if monthly.empty:
        return pd.DataFrame()

    mask = (monthly["category"] == category) & (monthly["year"] == year)
    if month is not None:
        mask &= monthly["month"] == month

    subset = monthly.loc[mask].copy()
    if subset.empty:
        return subset

    ranked = (
        subset.groupby(["product_id", "category"], as_index=False)
        .agg(
            demand=(sort_by, "sum"),
            transaction_count=("transaction_count", "sum"),
            avg_price=("avg_price", "mean"),
            revenue_proxy=("revenue_proxy", "sum"),
        )
        .sort_values(sort_by, ascending=False)
        .head(top_n)
    )
    total = ranked[sort_by].sum()
    ranked["share_pct"] = (ranked[sort_by] / total * 100) if total > 0 else 0.0
    return ranked.reset_index(drop=True)
