"""
Price-dependent context for demand inference in RL simulation.

At training time, discount_rate and transaction_count come from historical aggregates.
At simulation time, they are derived from the chosen price so the LightGBM sees a
coherent (price, discount, volume) tuple instead of fixed category averages.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

# Bounds for inferred simulation features
MIN_DISCOUNT_RATE = 0.0
MAX_DISCOUNT_RATE = 0.45
MIN_TRANSACTION_COUNT = 1.0
MAX_TRANSACTION_COUNT = 25.0


def _power_law_discount(
    price: float,
    median_price: float,
    mean_discount: float,
    exponent: float = 0.55,
) -> float:
    """Higher listed price -> lower discount rate (simple elasticity-style rule)."""
    if median_price <= 0:
        return float(mean_discount)
    ratio = median_price / max(float(price), 1.0)
    return float(np.clip(mean_discount * (ratio ** exponent), MIN_DISCOUNT_RATE, MAX_DISCOUNT_RATE))


def _power_law_transaction_count(
    price: float,
    median_price: float,
    mean_txn: float,
    exponent: float = 0.35,
) -> float:
    """Slightly fewer transactions when price is far above category median."""
    if median_price <= 0:
        return float(mean_txn)
    ratio = median_price / max(float(price), 1.0)
    return float(
        np.clip(mean_txn * (ratio ** exponent), MIN_TRANSACTION_COUNT, MAX_TRANSACTION_COUNT)
    )


def fit_category_price_rules(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Fit linear rules: feature ~ log1p(price) per category from processed daily rows.

    Returns dict[category] with keys discount_slope, discount_intercept,
    txn_slope, txn_intercept, plus median_price and means for fallback.
    """
    rules: Dict[str, Dict[str, float]] = {}
    for cat, sub in df.groupby("category"):
        sub = sub.copy()
        log_p = np.log1p(sub["price"].astype(float))
        dr = sub["avg_discount_rate"].astype(float)
        txn = sub["transaction_count"].astype(float)

        median_price = float(sub["price"].median())
        mean_dr = float(dr.mean())
        mean_txn = float(txn.mean())

        if len(sub) >= 8 and log_p.std() > 1e-6:
            dr_coef = np.polyfit(log_p, dr, 1)
            txn_coef = np.polyfit(log_p, txn, 1)
            discount_slope, discount_intercept = float(dr_coef[0]), float(dr_coef[1])
            txn_slope, txn_intercept = float(txn_coef[0]), float(txn_coef[1])
        else:
            discount_slope, discount_intercept = 0.0, mean_dr
            txn_slope, txn_intercept = 0.0, mean_txn

        rules[str(cat)] = {
            "median_price": median_price,
            "mean_discount_rate": mean_dr,
            "mean_transaction_count": mean_txn,
            "discount_slope": discount_slope,
            "discount_intercept": discount_intercept,
            "txn_slope": txn_slope,
            "txn_intercept": txn_intercept,
        }
    return rules


def infer_discount_rate(
    price: float,
    category: str,
    price_rules: Dict[str, Dict[str, float]],
    global_stats: Dict[str, float],
) -> float:
    stats = price_rules.get(category, {})
    median = stats.get("median_price", global_stats.get("median_price", price))
    mean_dr = stats.get("mean_discount_rate", global_stats.get("mean_discount_rate", 0.08))

    if not stats:
        return _power_law_discount(price, median, mean_dr)

    log_p = np.log1p(max(float(price), 0.0))
    raw = stats["discount_slope"] * log_p + stats["discount_intercept"]
    if not np.isfinite(raw):
        raw = _power_law_discount(price, median, mean_dr)
    return float(np.clip(raw, MIN_DISCOUNT_RATE, MAX_DISCOUNT_RATE))


def infer_transaction_count(
    price: float,
    category: str,
    price_rules: Dict[str, Dict[str, float]],
    global_stats: Dict[str, float],
) -> float:
    stats = price_rules.get(category, {})
    median = stats.get("median_price", global_stats.get("median_price", price))
    mean_txn = stats.get("mean_transaction_count", global_stats.get("mean_transaction_count", 2.0))

    if not stats:
        return _power_law_transaction_count(price, median, mean_txn)

    log_p = np.log1p(max(float(price), 0.0))
    raw = stats["txn_slope"] * log_p + stats["txn_intercept"]
    if not np.isfinite(raw):
        raw = _power_law_transaction_count(price, median, mean_txn)
    return float(np.clip(raw, MIN_TRANSACTION_COUNT, MAX_TRANSACTION_COUNT))


def demand_to_sales_units(demand: float, inventory: int) -> int:
    """Convert fractional expected demand to integer sales (stochastic rounding via ceil)."""
    if demand <= 0:
        return 0
    requested = int(np.ceil(demand))
    return min(max(0, requested), max(0, inventory))


def fit_category_demand_elasticity(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Fit log-log elasticity of demand vs price per category from historical rows.

    Used to calibrate simulator predictions when the global model under-reacts to price.
    """
    rules: Dict[str, Dict[str, float]] = {}
    for cat, sub in df.groupby("category"):
        sub = sub.copy()
        sub = sub[(sub["price"].astype(float) > 0) & (sub["demand"].astype(float) >= 0)]
        median_price = float(sub["price"].median()) if len(sub) else 1.0
        mean_demand = float(sub["demand"].mean()) if len(sub) else 1.0

        if len(sub) >= 10:
            log_p = np.log1p(sub["price"].astype(float))
            log_d = np.log1p(sub["demand"].astype(float))
            if log_p.std() > 1e-6:
                coef = np.polyfit(log_p, log_d, 1)
                elasticity = float(np.clip(coef[0], -1.5, 0.05))
            else:
                elasticity = -0.35
        else:
            elasticity = -0.35

        rules[str(cat)] = {
            "median_price": median_price,
            "mean_demand": mean_demand,
            "demand_elasticity": elasticity,
        }
    return rules


def apply_elasticity_calibration(
    raw_pred: float,
    price: float,
    category: str,
    elasticity_rules: Dict[str, Dict[str, float]],
    global_stats: Dict[str, float],
    blend: float = 0.55,
) -> float:
    """
    Softly adjust predicted demand using empirical price elasticity.

    blend=0 leaves prediction unchanged; blend=1 applies full (price/median)^beta.
    """
    if raw_pred <= 0 or blend <= 0:
        return max(0.0, raw_pred)

    stats = elasticity_rules.get(category, {})
    median = stats.get("median_price", global_stats.get("median_price", price))
    beta = stats.get("demand_elasticity", -0.35)
    if median <= 0:
        return max(0.0, raw_pred)

    ratio = max(float(price), 1.0) / median
    factor = ratio ** beta
    # dampen extreme adjustments
    factor = float(np.clip(factor, 0.25, 2.5))
    adjusted = raw_pred * (factor ** blend)
    return max(0.0, float(adjusted))
