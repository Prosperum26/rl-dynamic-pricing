"""
Demand Predictor Training Script.

Trains a global LightGBM model on processed_sales.csv with category as a feature.

Usage:
    python -m training.train_demand_predictor

    python -m training.train_demand_predictor --data data/processed_sales.csv

    python -m training.train_demand_predictor --random-split
"""

import argparse
import os
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).parent.parent))

from config.constants import (
    LGB_PARAMS,
    MODELS_DIR,
    PROCESSED_SALES_PATH,
    RANDOM_SEED,
    SYNTHETIC_SALES_PATH,
    TRAIN_TEST_SPLIT,
)
from training.demand_features import (
    DEFAULT_PROCESSED_PATH,
    DemandFeatureEncoder,
    ENCODER_PATH,
    METADATA_PATH,
    MODEL_PATH,
    load_processed_sales,
    save_metadata,
    time_based_split,
)


def generate_synthetic_data(n_samples: int = 5000, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate synthetic sales data (includes category for global model)."""
    from config.constants import PRODUCT_CATEGORIES

    np.random.seed(seed)
    dates = pd.date_range(start="2022-01-01", end="2023-12-31", periods=n_samples)
    prices = np.random.uniform(25, 975, n_samples)
    day_of_week = dates.dayofweek
    month = dates.month
    categories = np.random.choice(PRODUCT_CATEGORIES, n_samples)

    base_demand = 2.0
    price_effect = -0.002 * prices
    weekend_boost = np.where(day_of_week >= 5, 0.5, 0)
    holiday_boost = np.where((month == 11) | (month == 12), 0.8, 0)
    noise = np.random.normal(0, 0.5, n_samples)
    demand = np.maximum(base_demand + price_effect + weekend_boost + holiday_boost + noise, 0)

    return pd.DataFrame({
        "date": dates,
        "category": categories,
        "price": np.round(prices, 2),
        "day_of_week": day_of_week,
        "month": month,
        "is_weekend": (day_of_week >= 5).astype(int),
        "is_holiday_season": ((month == 11) | (month == 12)).astype(int),
        "demand": demand,
    })


def train_model(
    data_path: str | None = None,
    save_model: bool = True,
    generate_data: bool = True,
    time_split: bool = True,
):
    """
    Train global LightGBM demand model with category one-hot features.

    Args:
        data_path: Path to processed or compatible CSV
        save_model: Persist model, encoder, and metadata
        generate_data: Generate synthetic data if no file is available
        time_split: Use chronological train/test split (recommended for real data)
    """
    print("=" * 60)
    print("Training Global Demand Prediction Model")
    print("=" * 60)

    # Load data
    if data_path and os.path.exists(data_path):
        print(f"Loading data from: {data_path}")
        df = load_processed_sales(Path(data_path))
        split_type = "time" if time_split else "random"
    elif DEFAULT_PROCESSED_PATH.exists():
        print(f"Loading default processed data: {DEFAULT_PROCESSED_PATH}")
        df = load_processed_sales(DEFAULT_PROCESSED_PATH)
        split_type = "time" if time_split else "random"
    elif generate_data:
        print("Generating synthetic training data...")
        df = generate_synthetic_data(n_samples=5000)
        df.to_csv(SYNTHETIC_SALES_PATH, index=False)
        print(f"Synthetic data saved to: {SYNTHETIC_SALES_PATH}")
        split_type = "random"
    else:
        raise ValueError(
            "No data found. Run: python -m scripts.preprocess_ecommerce"
        )

    print(f"Dataset shape: {df.shape}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Categories: {sorted(df['category'].unique())}")
    print(f"Demand - mean: {df['demand'].mean():.2f}, max: {df['demand'].max():.2f}")
    print(f"Price - mean: ${df['price'].mean():.2f}, max: ${df['price'].max():.2f}")

    # Encode features (price + calendar + category one-hot)
    encoder = DemandFeatureEncoder()
    encoder.fit(df)

    if time_split:
        train_df, test_df = time_based_split(df, test_size=TRAIN_TEST_SPLIT)
        X_train = encoder.transform(train_df)
        y_train = train_df["demand"].values
        X_test = encoder.transform(test_df)
        y_test = test_df["demand"].values
        print(f"\nTime-based split - train: {len(train_df)}, test: {len(test_df)}")
        print(f"  Train dates: {train_df['date'].min().date()} to {train_df['date'].max().date()}")
        print(f"  Test dates:  {test_df['date'].min().date()} to {test_df['date'].max().date()}")
    else:
        X = encoder.transform(df)
        y = df["demand"].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TRAIN_TEST_SPLIT, random_state=RANDOM_SEED
        )
        split_type = "random"
        print(f"\nRandom split — train: {len(X_train)}, test: {len(X_test)}")

    feature_names = encoder.feature_names_
    print(f"Features ({len(feature_names)}): {feature_names}")

    # Train LightGBM
    print("\nTraining LightGBM model...")
    model = lgb.LGBMRegressor(**LGB_PARAMS, random_state=RANDOM_SEED)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="rmse",
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=0),
        ],
    )

    y_pred = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    print(f"\nTest Set Performance ({split_type} split):")
    print(f"  RMSE: {rmse:.3f}")
    print(f"  MAE:  {mae:.3f}")
    print(f"  R²:   {r2:.4f}")

    print("\nFeature Importance:")
    for name, imp in sorted(
        zip(feature_names, model.feature_importances_),
        key=lambda x: -x[1],
    ):
        print(f"  {name}: {imp:.0f}")

    if save_model:
        MODELS_DIR.mkdir(exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        encoder.save(ENCODER_PATH)
        print(f"\nModel saved to: {MODEL_PATH}")
        print(f"Encoder saved to: {ENCODER_PATH}")

        metadata = {
            "model_type": "LightGBM",
            "global_model": True,
            "split_type": split_type,
            "features": feature_names,
            "categories": encoder.categories,
            "metrics": {"rmse": rmse, "mae": mae, "r2": r2},
            "hyperparameters": LGB_PARAMS,
            "n_train": len(y_train),
            "n_test": len(y_test),
        }
        save_metadata(metadata)
        print(f"Metadata saved to: {METADATA_PATH}")

    print("=" * 60)
    print("Training complete!")
    return model, encoder, {"rmse": rmse, "mae": mae, "r2": r2}


def predict_demand(
    price: float,
    day_of_week: int,
    month: int,
    category: str,
    model_path: Path | None = None,
    encoder_path: Path | None = None,
) -> float:
    """Predict demand for a single (price, time, category) tuple."""
    from training.demand_features import DemandPredictorWrapper

    wrapper = DemandPredictorWrapper.load(
        model_path or MODEL_PATH,
        encoder_path or ENCODER_PATH,
    )
    return wrapper.predict_demand(price, day_of_week, month, category)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train global demand prediction model")
    parser.add_argument(
        "--data",
        type=str,
        default=str(PROCESSED_SALES_PATH),
        help="Path to processed_sales.csv",
    )
    parser.add_argument("--no-save", action="store_true", help="Do not save model")
    parser.add_argument(
        "--no-generate",
        action="store_true",
        help="Fail instead of generating synthetic data",
    )
    parser.add_argument(
        "--random-split",
        action="store_true",
        help="Use random train/test split instead of time-based",
    )

    args = parser.parse_args()

    data_path = args.data if args.data and os.path.exists(args.data) else None
    if data_path is None and PROCESSED_SALES_PATH.exists():
        data_path = str(PROCESSED_SALES_PATH)

    train_model(
        data_path=data_path,
        save_model=not args.no_save,
        generate_data=not args.no_generate,
        time_split=not args.random_split,
    )
