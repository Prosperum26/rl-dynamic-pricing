"""
Feature engineering for global demand forecasting (all product categories).

Loads processed_sales.csv, maps columns for LightGBM, and one-hot encodes category.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from config.constants import (
    MODELS_DIR,
    PROCESSED_SALES_PATH,
    PRODUCT_CATEGORIES,
    DEFAULT_PRODUCT_CATEGORY,
    RANDOM_SEED,
)

# Columns required after loading processed CSV
PROCESSED_REQUIRED_COLUMNS = [
    "date",
    "category",
    "demand",
    "day_of_week",
    "month",
    "is_weekend",
]

BASE_FEATURE_COLS = [
    "price",
    "day_of_week",
    "month",
    "is_weekend",
    "is_holiday_season",
]

DEFAULT_PROCESSED_PATH = PROCESSED_SALES_PATH
ENCODER_PATH = MODELS_DIR / "demand_encoder.joblib"
MODEL_PATH = MODELS_DIR / "demand_predictor.joblib"
METADATA_PATH = MODELS_DIR / "demand_predictor_metadata.json"


def load_processed_sales(path: Path) -> pd.DataFrame:
    """
    Load aggregated sales CSV and normalize column names for training.

    Maps avg_price -> price and derives is_holiday_season when missing.
    """
    if not path.exists():
        raise FileNotFoundError(f"Processed data not found: {path}")

    df = pd.read_csv(path, parse_dates=["date"])

    if "avg_price" in df.columns and "price" not in df.columns:
        df["price"] = df["avg_price"]
    elif "price" not in df.columns:
        raise ValueError("Dataset must include 'price' or 'avg_price'.")

    missing = [c for c in PROCESSED_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Processed dataset missing columns: {missing}")

    if "is_holiday_season" not in df.columns:
        df["is_holiday_season"] = df["month"].isin([11, 12]).astype(int)

    df["category"] = df["category"].astype(str)
    df["price"] = df["price"].astype(float)
    df["demand"] = df["demand"].astype(float)

    return df.sort_values("date").reset_index(drop=True)


def time_based_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split by time so the test set is the most recent rows (no future leakage)."""
    n_test = max(1, int(len(df) * test_size))
    train_df = df.iloc[:-n_test].copy()
    test_df = df.iloc[-n_test:].copy()
    return train_df, test_df


class DemandFeatureEncoder:
    """
    One-hot encodes product category and builds the feature matrix for LightGBM.

    Unknown categories at inference time are mapped to all-zero category columns.
    """

    def __init__(self, categories: Optional[List[str]] = None):
        self.categories: List[str] = categories or list(PRODUCT_CATEGORIES)
        self.feature_names_: List[str] = []

    def fit(self, df: pd.DataFrame) -> "DemandFeatureEncoder":
        """Learn category levels from training data (sorted for stable column order)."""
        seen = sorted(df["category"].unique().tolist())
        # Keep known catalog categories first, then any extras in the data
        extras = [c for c in seen if c not in self.categories]
        self.categories = [c for c in self.categories if c in seen] + extras
        if not self.categories:
            self.categories = seen

        self.feature_names_ = BASE_FEATURE_COLS + [
            f"category_{c.replace(' ', '_').replace('&', 'and')}" for c in self.categories
        ]
        return self

    def _category_column_name(self, category: str) -> str:
        safe = category.replace(" ", "_").replace("&", "and")
        return f"category_{safe}"

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Build feature matrix X from a DataFrame with base + category columns."""
        if not self.feature_names_:
            raise RuntimeError("Encoder is not fitted. Call fit() first.")

        n = len(df)
        X = np.zeros((n, len(self.feature_names_)), dtype=np.float64)

        col_index = {name: i for i, name in enumerate(self.feature_names_)}

        for col in BASE_FEATURE_COLS:
            X[:, col_index[col]] = df[col].values

        for i, cat in enumerate(df["category"].values):
            col_name = self._category_column_name(cat)
            if col_name in col_index:
                X[i, col_index[col_name]] = 1.0

        return X

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    def build_row(
        self,
        price: float,
        day_of_week: int,
        month: int,
        category: str,
        is_weekend: Optional[int] = None,
        is_holiday_season: Optional[int] = None,
    ) -> np.ndarray:
        """Build a single feature row for inference."""
        if is_weekend is None:
            is_weekend = int(day_of_week >= 5)
        if is_holiday_season is None:
            is_holiday_season = int(month in [11, 12])

        row_df = pd.DataFrame([{
            "price": price,
            "day_of_week": day_of_week,
            "month": month,
            "is_weekend": is_weekend,
            "is_holiday_season": is_holiday_season,
            "category": category,
        }])
        return self.transform(row_df)

    def save(self, path: Path = ENCODER_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path = ENCODER_PATH) -> "DemandFeatureEncoder":
        return joblib.load(path)


class DemandPredictorWrapper:
    """Loads trained LightGBM + encoder for use in PricingEnv or scripts."""

    def __init__(self, model, encoder: DemandFeatureEncoder):
        self.model = model
        self.encoder = encoder

    @classmethod
    def load(
        cls,
        model_path: Path = MODEL_PATH,
        encoder_path: Path = ENCODER_PATH,
    ) -> "DemandPredictorWrapper":
        if not model_path.exists():
            raise FileNotFoundError(
                f"Demand model not found at {model_path}. "
                "Run: python -m training.train_demand_predictor"
            )
        model = joblib.load(model_path)
        encoder = DemandFeatureEncoder.load(encoder_path)
        return cls(model, encoder)

    def predict_demand(
        self,
        price: float,
        day_of_week: int,
        month: int,
        category: str = DEFAULT_PRODUCT_CATEGORY,
    ) -> float:
        features = self.encoder.build_row(price, day_of_week, month, category)
        row = pd.DataFrame(features.reshape(1, -1), columns=self.encoder.feature_names_)
        return float(self.model.predict(row)[0])

    # Alias for compatibility with sklearn-style .predict()
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


def save_metadata(
    metadata: dict,
    path: Path = METADATA_PATH,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def load_metadata(path: Path = METADATA_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
