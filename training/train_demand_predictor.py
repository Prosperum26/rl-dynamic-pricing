"""
Demand Predictor Training Script.

Trains a global LightGBM model on processed_sales.csv with enriched features.

Usage:
    python -m scripts.preprocess_ecommerce
    python -m training.train_demand_predictor
"""

import argparse
import os
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from config.constants import (
    DEMAND_TARGET_COL,
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
from training.demand_metrics import compute_demand_metrics, format_metrics


def generate_synthetic_data(n_samples: int = 5000, seed: int = RANDOM_SEED):
    """Generate synthetic data compatible with the enriched feature pipeline."""
    import pandas as pd
    from config.constants import PRODUCT_CATEGORIES

    np.random.seed(seed)
    dates = pd.date_range(start="2022-01-01", end="2023-12-31", periods=n_samples)
    prices = np.random.uniform(25, 975, n_samples)
    day_of_week = dates.dayofweek
    month = dates.month
    categories = np.random.choice(PRODUCT_CATEGORIES, n_samples)

    demand = np.maximum(
        2.0 - 0.002 * prices
        + np.where(day_of_week >= 5, 0.5, 0)
        + np.where((month == 11) | (month == 12), 0.8, 0)
        + np.random.normal(0, 0.5, n_samples),
        0,
    )

    return pd.DataFrame({
        "date": dates,
        "category": categories,
        "price": np.round(prices, 2),
        "avg_price": np.round(prices, 2),
        "day_of_week": day_of_week,
        "month": month,
        "is_weekend": (day_of_week >= 5).astype(int),
        "is_holiday_season": ((month == 11) | (month == 12)).astype(int),
        "avg_discount_rate": np.random.uniform(0.02, 0.15, n_samples),
        "transaction_count": np.random.poisson(3, n_samples) + 1,
        "demand": demand,
    })


def train_model(
    data_path: str | None = None,
    save_model: bool = True,
    generate_data: bool = True,
    time_split: bool = True,
):
    print("=" * 60)
    print("Training Global Demand Prediction Model (v2 features)")
    print("=" * 60)

    if data_path and os.path.exists(data_path):
        print(f"Loading data from: {data_path}")
        df = load_processed_sales(Path(data_path))
        split_type = "time" if time_split else "random"
    elif DEFAULT_PROCESSED_PATH.exists():
        print(f"Loading: {DEFAULT_PROCESSED_PATH}")
        df = load_processed_sales(DEFAULT_PROCESSED_PATH)
        split_type = "time" if time_split else "random"
    elif generate_data:
        print("Generating synthetic training data...")
        df = generate_synthetic_data(n_samples=5000)
        SYNTHETIC_SALES_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(SYNTHETIC_SALES_PATH, index=False)
        split_type = "random"
    else:
        raise ValueError("No data found. Run: python -m scripts.preprocess_ecommerce")

    target_col = DEMAND_TARGET_COL
    print(f"Dataset: {df.shape[0]} rows, target={target_col}")
    print(f"  demand mean={df[target_col].mean():.2f}, zeros={(df[target_col]==0).mean():.1%}")
    print(f"  price mean=${df['price'].mean():.2f}")

    if time_split:
        train_df, test_df = time_based_split(df, test_size=TRAIN_TEST_SPLIT)
        print(f"\nTime split: train={len(train_df)}, test={len(test_df)}")
    else:
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(
            df, test_size=TRAIN_TEST_SPLIT, random_state=RANDOM_SEED
        )
        split_type = "random"
        print(f"\nRandom split: train={len(train_df)}, test={len(test_df)}")

    # Fit encoder on train only (no leakage for category stats)
    encoder = DemandFeatureEncoder()
    encoder.fit(train_df)
    X_train = encoder.transform(train_df)
    y_train = train_df[target_col].values
    X_test = encoder.transform(test_df)
    y_test = test_df[target_col].values

    print(f"Features ({len(encoder.feature_names_)}): {encoder.feature_names_}")

    print("\nTraining LightGBM...")
    model = lgb.LGBMRegressor(**LGB_PARAMS, random_state=RANDOM_SEED)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="rmse",
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=50),
        ],
    )

    y_pred = np.clip(model.predict(X_test), 0, None)
    metrics = compute_demand_metrics(y_test, y_pred)

    print(f"\nTest metrics ({split_type} split):")
    print(format_metrics(metrics))

    print("\nFeature importance (top 10):")
    for name, imp in sorted(
        zip(encoder.feature_names_, model.feature_importances_),
        key=lambda x: -x[1],
    )[:10]:
        print(f"  {name}: {imp:.0f}")

    if save_model:
        MODELS_DIR.mkdir(exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        encoder.save(ENCODER_PATH)

        metadata = {
            "model_type": "LightGBM",
            "model_version": 2,
            "target": target_col,
            "target_description": "daily conversion count per category",
            "global_model": True,
            "split_type": split_type,
            "features": encoder.feature_names_,
            "categories": encoder.categories,
            "category_stats": encoder.category_stats_,
            "metrics": metrics,
            "hyperparameters": LGB_PARAMS,
            "n_train": len(y_train),
            "n_test": len(y_test),
        }
        save_metadata(metadata)
        print(f"\nSaved model  -> {MODEL_PATH}")
        print(f"Saved encoder -> {ENCODER_PATH}")
        print(f"Saved metadata -> {METADATA_PATH}")

    print("=" * 60)
    goal_met = metrics["r2"] >= 0.3
    print(f"R2 goal (>= 0.3): {'PASS' if goal_met else 'BELOW TARGET - consider more data or features'}")
    print("=" * 60)
    return model, encoder, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train demand prediction model")
    parser.add_argument("--data", type=str, default=str(PROCESSED_SALES_PATH))
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--no-generate", action="store_true")
    parser.add_argument("--random-split", action="store_true")
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
