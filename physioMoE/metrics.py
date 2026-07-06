"""Regression metrics for NASA-TLX prediction, shared by training and evaluation."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from physioMoE.config import NASA_TLX_DIMENSIONS


def compute_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    """predictions, targets: (N, 6) arrays -> flat dict of per-dimension and overall metrics."""
    metrics: dict[str, float] = {}

    for i, dim in enumerate(NASA_TLX_DIMENSIONS):
        y_pred, y_true = predictions[:, i], targets[:, i]
        metrics[f"{dim}_mae"] = mean_absolute_error(y_true, y_pred)
        metrics[f"{dim}_rmse"] = mean_squared_error(y_true, y_pred) ** 0.5
        metrics[f"{dim}_r2"] = r2_score(y_true, y_pred) if len(y_true) > 1 else float("nan")

    metrics["overall_mae"] = mean_absolute_error(targets, predictions)
    metrics["overall_rmse"] = mean_squared_error(targets, predictions) ** 0.5
    return metrics
