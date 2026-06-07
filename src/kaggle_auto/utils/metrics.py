"""Competition metric implementations."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return mean_squared_error(y_true, y_pred, squared=False)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return mean_absolute_error(y_true, y_pred)


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return mean_squared_error(y_true, y_pred)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_pred.dtype == float and y_pred.max() <= 1:
        y_pred = (y_pred > 0.5).astype(int)
    return accuracy_score(y_true, y_pred)


def auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return roc_auc_score(y_true, y_pred)


def logloss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return log_loss(y_true, y_pred)


def f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_pred.dtype == float and y_pred.max() <= 1:
        y_pred = (y_pred > 0.5).astype(int)
    return f1_score(y_true, y_pred, average="macro")


def weighted_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted Pearson correlation (common in crypto competitions)."""
    correlation = np.corrcoef(y_true, y_pred)[0, 1]
    return correlation if not np.isnan(correlation) else 0.0


METRICS = {
    "rmse": (rmse, "minimize"),
    "mae": (mae, "minimize"),
    "mse": (mse, "minimize"),
    "accuracy": (accuracy, "maximize"),
    "auc": (auc, "maximize"),
    "log_loss": (logloss, "minimize"),
    "logloss": (logloss, "minimize"),
    "f1": (f1, "maximize"),
    "weighted_pearson": (weighted_pearson, "maximize"),
    "r2": (r2_score, "maximize"),
}


def get_metric_fn(name: str) -> tuple[callable, str]:
    """Get metric function and direction by name.

    Returns (metric_fn, direction) where direction is "minimize" or "maximize".
    """
    if name not in METRICS:
        raise ValueError(f"Unknown metric: {name}. Available: {list(METRICS.keys())}")
    return METRICS[name]
