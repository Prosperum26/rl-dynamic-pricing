"""
Synthetic Data Generation for Testing.

Generates realistic sales data with configurable patterns for:
- Price-demand relationships
- Seasonality (weekly, monthly, yearly)
- Trends and promotional effects
- Random noise

Usage:
    python -m data.generate_data --output data/sales.csv --samples 10000
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import argparse
from datetime import datetime, timedelta

from config.constants import (
    MIN_PRICE, MAX_PRICE,
    BASE_DEMAND, PRICE_ELASTICITY,
    SEASONALITY_AMPLITUDE, SEASONALITY_PERIOD,
    DEMAND_NOISE_STD, RANDOM_SEED,
    DATA_DIR
)


def generate_synthetic_sales_data(
    n_samples: int = 5000,
    start_date: str = "2022-01-01",
    end_date: str = "2023-12-31",
    output_path: str = None,
    seed: int = RANDOM_SEED
) -> pd.DataFrame:
    """
    Generate synthetic sales data with realistic patterns.
    
    Args:
        n_samples: Number of daily records to generate
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        output_path: Path to save CSV (optional)
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with synthetic sales data
    """
    np.random.seed(seed)
    
    # Generate date range
    date_range = pd.date_range(start=start_date, end=end_date, periods=n_samples)
    
    # Generate prices with some variation (could be decisions made)
    # Mix of low, medium, high prices
    price_segments = np.random.choice(
        [0.3, 0.6, 0.9],  # Low, medium, high price ratios
        size=n_samples,
        p=[0.3, 0.5, 0.2]  # More medium prices
    )
    prices = MIN_PRICE + price_segments * (MAX_PRICE - MIN_PRICE)
    prices += np.random.normal(0, 3, n_samples)  # Add noise
    prices = np.clip(prices, MIN_PRICE, MAX_PRICE)
    prices = np.round(prices, 2)
    
    # Calculate time features
    day_of_week = date_range.dayofweek  # 0=Monday
    month = date_range.month
    day_of_year = date_range.dayofyear
    quarter = date_range.quarter
    is_weekend = (day_of_week >= 5).astype(int)
    
    # Holiday flags (simplified)
    is_holiday_season = ((month == 11) | (month == 12)).astype(int)
    is_summer = ((month >= 6) & (month <= 8)).astype(int)
    
    # Calculate demand components
    base = BASE_DEMAND
    
    # Price effect (elasticity)
    price_effect = PRICE_ELASTICITY * (prices - (MIN_PRICE + MAX_PRICE) / 2)
    
    # Weekly seasonality (higher on weekends)
    weekly_pattern = SEASONALITY_AMPLITUDE * 0.5 * np.sin(
        2 * np.pi * day_of_week / SEASONALITY_PERIOD
    )
    
    # Monthly seasonality
    monthly_pattern = SEASONALITY_AMPLITUDE * np.sin(
        2 * np.pi * month / 12
    )
    
    # Holiday boost
    holiday_boost = is_holiday_season * 15
    
    # Summer boost
    summer_boost = is_summer * 8
    
    # Trend (slight growth over time)
    trend = 0.01 * np.arange(n_samples)
    
    # Promotions (random spikes)
    is_promotion = (np.random.random(n_samples) < 0.05).astype(int)
    promotion_boost = is_promotion * 20
    
    # Random noise
    noise = np.random.normal(0, DEMAND_NOISE_STD, n_samples)
    
    # Calculate final demand
    demand = (
        base +
        price_effect +
        weekly_pattern +
        monthly_pattern +
        holiday_boost +
        summer_boost +
        trend +
        promotion_boost +
        noise
    )
    demand = np.maximum(demand, 0)  # Non-negative
    demand = demand.astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': date_range,
        'price': prices,
        'demand': demand,
        'day_of_week': day_of_week,
        'month': month,
        'quarter': quarter,
        'day_of_year': day_of_year,
        'is_weekend': is_weekend,
        'is_holiday_season': is_holiday_season,
        'is_summer': is_summer,
        'is_promotion': is_promotion,
    })
    
    # Add derived features
    df['revenue'] = df['price'] * df['demand']
    df['price_segment'] = pd.cut(
        df['price'],
        bins=[MIN_PRICE-1, 30, 60, MAX_PRICE+1],
        labels=['Low', 'Medium', 'High']
    )
    
    # Save if requested
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Data saved to: {output_path}")
        print(f"Shape: {df.shape}")
        print(f"\nSummary statistics:")
        print(df.describe())
    
    return df


def generate_dataset(output_path: str = None, n_samples: int = 5000) -> pd.DataFrame:
    """
    Convenience function to generate and save dataset.
    
    Args:
        output_path: Where to save the CSV
        n_samples: Number of samples
        
    Returns:
        Generated DataFrame
    """
    if output_path is None:
        output_path = DATA_DIR / "synthetic_sales.csv"
    
    return generate_synthetic_sales_data(
        n_samples=n_samples,
        output_path=output_path
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic sales data")
    parser.add_argument(
        "--output",
        type=str,
        default=str(DATA_DIR / "synthetic_sales.csv"),
        help="Output CSV file path"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5000,
        help="Number of samples to generate"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2022-01-01",
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2023-12-31",
        help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Synthetic Data Generation")
    print("=" * 60)
    
    df = generate_synthetic_sales_data(
        n_samples=args.samples,
        start_date=args.start_date,
        end_date=args.end_date,
        output_path=args.output,
        seed=args.seed
    )
    
    print("\nData generation complete!")
    print("=" * 60)
    
    # Show sample
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nPrice segment distribution:")
    print(df['price_segment'].value_counts())
