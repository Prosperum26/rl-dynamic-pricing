"""
Evaluation metrics for demand forecasting.
"""

from __future__ import annotations

import numpy as np


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAPE on rows where actual demand > 0."""
    mask = y_true > 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def weighted_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted MAPE: sum(|error|) / sum(actual)."""
    denom = np.sum(y_true)
    if denom <= 0:
        return float("nan")
    return float(np.sum(np.abs(y_true - y_pred)) / denom)


def compute_demand_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """
    Compute regression metrics for demand forecasts.

    Predictions are clipped at zero before metrics.
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0.0, None)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    mask = y_true > 0
    r2_positive = float(r2_score(y_true[mask], y_pred[mask])) if mask.sum() >= 2 else float("nan")
    mape = mean_absolute_percentage_error(y_true, y_pred)
    wmape = weighted_mape(y_true, y_pred)

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "r2_demand_positive": r2_positive,
        "mape": mape,
        "wmape": wmape,
        "n_samples": int(len(y_true)),
        "n_positive_demand": int(mask.sum()),
    }


def format_metrics(metrics: dict[str, float]) -> str:
    """Pretty-print metrics for training logs."""
    lines = [
        f"  RMSE:  {metrics['rmse']:.3f}",
        f"  MAE:   {metrics['mae']:.3f}",
        f"  R2:    {metrics['r2']:.4f}",
        f"  R2 (>0 demand): {metrics['r2_demand_positive']:.4f}",
        f"  MAPE:  {metrics['mape']:.2%}" if not np.isnan(metrics["mape"]) else "  MAPE:  n/a",
        f"  WMAPE: {metrics['wmape']:.2%}" if not np.isnan(metrics["wmape"]) else "  WMAPE: n/a",
    ]
    return "\n".join(lines)
