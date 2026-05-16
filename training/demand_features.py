"""
Feature engineering for global demand forecasting (all product categories).

Training uses full aggregated fields (discount rate, transaction volume).
At inference (RL env), missing values are filled from per-category training averages.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from config.constants import (
    DEFAULT_PRODUCT_CATEGORY,
    MODELS_DIR,
    PROCESSED_SALES_PATH,
    PRODUCT_CATEGORIES,
    RANDOM_SEED,
)

PROCESSED_REQUIRED_COLUMNS = [
    "date",
    "category",
    "demand",
    "day_of_week",
    "month",
    "is_weekend",
]

# Features always available at pricing time (plus category one-hot)
INFERENCE_FEATURE_COLS = [
    "price",
    "log_price",
    "price_vs_category_median",
    "discount_rate",
    "transaction_count",
    "day_of_week",
    "month",
    "is_weekend",
    "is_holiday_season",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
]

DEFAULT_PROCESSED_PATH = PROCESSED_SALES_PATH
ENCODER_PATH = MODELS_DIR / "demand_encoder.joblib"
MODEL_PATH = MODELS_DIR / "demand_predictor.joblib"
METADATA_PATH = MODELS_DIR / "demand_predictor_metadata.json"


def load_processed_sales(path: Path) -> pd.DataFrame:
    """Load processed CSV and derive modeling columns."""
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

    df["category"] = df["category"].astype(str)
    df["price"] = df["price"].astype(float)
    df["demand"] = df["demand"].astype(float)

    if "is_holiday_season" not in df.columns:
        df["is_holiday_season"] = df["month"].isin([11, 12]).astype(int)

    if "avg_discount_rate" not in df.columns:
        if "avg_discount" in df.columns:
            df["avg_discount_rate"] = (
                df["avg_discount"] / df["price"].replace(0, np.nan)
            ).fillna(0).clip(0, 1)
        else:
            df["avg_discount_rate"] = 0.0

    if "transaction_count" not in df.columns:
        df["transaction_count"] = 1.0

    return df.sort_values("date").reset_index(drop=True)


def time_based_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n_test = max(1, int(len(df) * test_size))
    return df.iloc[:-n_test].copy(), df.iloc[-n_test:].copy()


def _cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["dow_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 7)
    out["month_sin"] = np.sin(2 * np.pi * (out["month"] - 1) / 12)
    out["month_cos"] = np.cos(2 * np.pi * (out["month"] - 1) / 12)
    out["log_price"] = np.log1p(out["price"])
    return out


class DemandFeatureEncoder:
    """
    Builds feature matrix with category one-hot encoding and train-set statistics.

    category_stats stores per-category medians/means for inference imputation.
    """

    def __init__(self, categories: Optional[List[str]] = None):
        self.categories: List[str] = categories or list(PRODUCT_CATEGORIES)
        self.feature_names_: List[str] = []
        self.category_stats_: Dict[str, Dict[str, float]] = {}
        self.global_stats_: Dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> "DemandFeatureEncoder":
        seen = sorted(df["category"].unique().tolist())
        extras = [c for c in seen if c not in self.categories]
        self.categories = [c for c in self.categories if c in seen] + extras
        if not self.categories:
            self.categories = seen

        self.category_stats_ = {}
        for cat in self.categories:
            sub = df[df["category"] == cat]
            self.category_stats_[cat] = {
                "median_price": float(sub["price"].median()),
                "mean_discount_rate": float(sub["avg_discount_rate"].mean()),
                "mean_transaction_count": float(sub["transaction_count"].mean()),
            }

        self.global_stats_ = {
            "median_price": float(df["price"].median()),
            "mean_discount_rate": float(df["avg_discount_rate"].mean()),
            "mean_transaction_count": float(df["transaction_count"].mean()),
        }

        self.feature_names_ = INFERENCE_FEATURE_COLS + [
            f"category_{c.replace(' ', '_').replace('&', 'and')}" for c in self.categories
        ]
        return self

    def _category_column_name(self, category: str) -> str:
        return f"category_{category.replace(' ', '_').replace('&', 'and')}"

    def _enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        out = _cyclical_features(df.copy())

        medians = []
        disc = []
        txn = []
        for _, row in out.iterrows():
            cat = row["category"]
            stats = self.category_stats_.get(cat, {})
            med = stats.get("median_price", self.global_stats_["median_price"])
            medians.append(med if med > 0 else 1.0)
            disc.append(row.get("avg_discount_rate", stats.get("mean_discount_rate", 0)))
            txn.append(row.get("transaction_count", stats.get("mean_transaction_count", 1)))

        out["price_vs_category_median"] = out["price"] / np.array(medians)
        out["discount_rate"] = np.array(disc, dtype=float)
        out["transaction_count"] = np.array(txn, dtype=float)
        return out

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.feature_names_:
            raise RuntimeError("Encoder is not fitted. Call fit() first.")

        enriched = self._enrich(df)
        n = len(enriched)
        X = np.zeros((n, len(self.feature_names_)), dtype=np.float64)
        col_index = {name: i for i, name in enumerate(self.feature_names_)}

        for col in INFERENCE_FEATURE_COLS:
            X[:, col_index[col]] = enriched[col].values

        for i, cat in enumerate(enriched["category"].values):
            name = self._category_column_name(cat)
            if name in col_index:
                X[i, col_index[name]] = 1.0

        return X

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    def build_row(
        self,
        price: float,
        day_of_week: int,
        month: int,
        category: str = DEFAULT_PRODUCT_CATEGORY,
        discount_rate: Optional[float] = None,
        transaction_count: Optional[float] = None,
        is_weekend: Optional[int] = None,
        is_holiday_season: Optional[int] = None,
    ) -> np.ndarray:
        stats = self.category_stats_.get(category, self.global_stats_)
        if discount_rate is None:
            discount_rate = stats.get("mean_discount_rate", self.global_stats_["mean_discount_rate"])
        if transaction_count is None:
            transaction_count = stats.get(
                "mean_transaction_count", self.global_stats_["mean_transaction_count"]
            )
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
            "avg_discount_rate": discount_rate,
            "transaction_count": transaction_count,
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
        discount_rate: Optional[float] = None,
        transaction_count: Optional[float] = None,
    ) -> float:
        features = self.encoder.build_row(
            price=price,
            day_of_week=day_of_week,
            month=month,
            category=category,
            discount_rate=discount_rate,
            transaction_count=transaction_count,
        )
        row = pd.DataFrame(features.reshape(1, -1), columns=self.encoder.feature_names_)
        return float(max(0.0, self.model.predict(row)[0]))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


def save_metadata(metadata: dict, path: Path = METADATA_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def load_metadata(path: Path = METADATA_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
