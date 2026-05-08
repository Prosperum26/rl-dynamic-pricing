"""
Demand Predictor Training Script.

Trains a LightGBM model to predict demand based on price and time features.
This can be used as a component in the RL environment or standalone.

Usage:
    python -m training.train_demand_predictor
    
    # With custom data
    python -m training.train_demand_predictor --data data/my_sales.csv
"""

import os
import sys
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import joblib

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config.constants import (
    DATA_DIR, MODELS_DIR,
    LGB_PARAMS, TRAIN_TEST_SPLIT, RANDOM_SEED
)


def generate_synthetic_data(n_samples: int = 5000, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Generate synthetic sales data for training.
    
    Creates realistic data with:
    - Price-demand relationship (elasticity)
    - Weekly seasonality
    - Holiday effects
    - Random noise
    
    Args:
        n_samples: Number of records to generate
        seed: Random seed
        
    Returns:
        DataFrame with synthetic sales data
    """
    np.random.seed(seed)
    
    # Generate random dates over 2 years
    dates = pd.date_range(start='2022-01-01', end='2023-12-31', periods=n_samples)
    
    # Random prices between $10 and $100
    prices = np.random.uniform(10, 100, n_samples)
    
    # Base demand
    base_demand = 50
    
    # Price effect (negative elasticity)
    price_effect = -0.5 * prices
    
    # Day of week effect (higher on weekends)
    day_of_week = dates.dayofweek
    weekend_boost = np.where(day_of_week >= 5, 15, 0)
    
    # Month effect (holiday seasons)
    month = dates.month
    holiday_boost = np.where(
        ((month == 11) | (month == 12)),  # Nov-Dec holidays
        20,
        np.where(
            (month == 7) | (month == 8),    # Summer
            10,
            0
        )
    )
    
    # Random noise
    noise = np.random.normal(0, 10, n_samples)
    
    # Calculate demand
    demand = base_demand + price_effect + weekend_boost + holiday_boost + noise
    demand = np.maximum(demand, 0)  # Non-negative
    demand = demand.astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'price': np.round(prices, 2),
        'day_of_week': day_of_week,
        'month': month,
        'is_weekend': (day_of_week >= 5).astype(int),
        'is_holiday_season': ((month == 11) | (month == 12)).astype(int),
        'demand': demand
    })
    
    return df


def prepare_features(df: pd.DataFrame) -> tuple:
    """
    Prepare feature matrix and target from DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        (X, y, feature_names) tuple
    """
    feature_cols = [
        'price',
        'day_of_week',
        'month',
        'is_weekend',
        'is_holiday_season'
    ]
    
    X = df[feature_cols].values
    y = df['demand'].values
    
    return X, y, feature_cols


def train_model(
    data_path: str = None,
    save_model: bool = True,
    generate_data: bool = True
):
    """
    Train LightGBM demand prediction model.
    
    Args:
        data_path: Path to CSV data file (optional)
        save_model: Whether to save the trained model
        generate_data: Whether to generate synthetic data if no data_path
    """
    print("=" * 60)
    print("Training Demand Prediction Model")
    print("=" * 60)
    
    # Load or generate data
    if data_path and os.path.exists(data_path):
        print(f"Loading data from: {data_path}")
        df = pd.read_csv(data_path, parse_dates=['date'])
    elif generate_data:
        print("Generating synthetic training data...")
        df = generate_synthetic_data(n_samples=5000)
        
        # Save generated data
        output_path = DATA_DIR / "synthetic_sales.csv"
        df.to_csv(output_path, index=False)
        print(f"Synthetic data saved to: {output_path}")
    else:
        raise ValueError("No data provided and generate_data=False")
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nData sample:")
    print(df.head(10))
    
    # Prepare features
    X, y, feature_names = prepare_features(df)
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TRAIN_TEST_SPLIT, random_state=RANDOM_SEED
    )
    
    print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Train model
    print("\nTraining LightGBM model...")
    model = lgb.LGBMRegressor(**LGB_PARAMS, random_state=RANDOM_SEED)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='rmse',
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=0)]
    )
    
    # Evaluate
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nTest Set Performance:")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE:  {mae:.2f}")
    print(f"  R²:   {r2:.4f}")
    
    # Feature importance
    importance = model.feature_importances_
    print(f"\nFeature Importance:")
    for name, imp in zip(feature_names, importance):
        print(f"  {name}: {imp:.0f}")
    
    # Save model
    if save_model:
        MODELS_DIR.mkdir(exist_ok=True)
        
        model_path = MODELS_DIR / "demand_predictor.joblib"
        joblib.dump(model, model_path)
        print(f"\nModel saved to: {model_path}")
        
        # Save metadata
        metadata = {
            'model_type': 'LightGBM',
            'features': feature_names,
            'metrics': {
                'rmse': float(rmse),
                'mae': float(mae),
                'r2': float(r2)
            },
            'hyperparameters': LGB_PARAMS
        }
        
        metadata_path = MODELS_DIR / "demand_predictor_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to: {metadata_path}")
    
    print("=" * 60)
    print("Training complete!")
    
    return model, {
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }


def predict_demand(price: float, day_of_week: int, month: int, 
                   model_path: str = None) -> float:
    """
    Convenience function to predict demand for given inputs.
    
    Args:
        price: Product price
        day_of_week: Day of week (0=Monday, 6=Sunday)
        month: Month (1-12)
        model_path: Path to saved model (uses default if None)
        
    Returns:
        Predicted demand
    """
    if model_path is None:
        model_path = MODELS_DIR / "demand_predictor.joblib"
    
    model = joblib.load(model_path)
    
    features = np.array([[
        price,
        day_of_week,
        month,
        int(day_of_week >= 5),  # is_weekend
        int(month in [11, 12])   # is_holiday_season
    ]])
    
    return model.predict(features)[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train demand prediction model")
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to CSV file with sales data (must have columns: date, price, demand)"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save the trained model"
    )
    parser.add_argument(
        "--no-generate",
        action="store_true",
        help="Don't generate synthetic data if no data file provided"
    )
    
    args = parser.parse_args()
    
    train_model(
        data_path=args.data,
        save_model=not args.no_save,
        generate_data=not args.no_generate
    )
