"""Ensemble methods: blending, stacking, weighted average."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .base import BaseModel


class Ensemble:
    """Combine multiple models for better predictions."""

    def __init__(self, models: list[BaseModel] | None = None, weights: list[float] | None = None):
        self.models = models or []
        self.weights = weights

    def add_model(self, model: BaseModel, weight: float = 1.0):
        self.models.append(model)
        if self.weights is None:
            self.weights = [1.0] * len(self.models)
        else:
            self.weights.append(weight)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Weighted average prediction."""
        if not self.models:
            raise RuntimeError("No models in ensemble")

        weights = self._normalize_weights()
        predictions = np.zeros(len(X))

        for model, w in zip(self.models, weights):
            predictions += w * model.predict(X)

        return predictions

    def optimize_weights(
        self,
        oof_predictions: list[np.ndarray],
        y_true: np.ndarray,
        metric_fn: callable,
        minimize_metric: bool = True,
    ) -> list[float]:
        """Find optimal ensemble weights via Nelder-Mead optimization."""
        n_models = len(oof_predictions)

        def objective(weights):
            weights = np.abs(weights)
            weights = weights / weights.sum()
            blended = sum(w * p for w, p in zip(weights, oof_predictions))
            score = metric_fn(y_true, blended)
            return score if minimize_metric else -score

        initial = np.ones(n_models) / n_models
        result = minimize(
            objective,
            initial,
            method="Nelder-Mead",
            options={"maxiter": 1000, "xatol": 1e-6},
        )

        optimal = np.abs(result.x)
        optimal = optimal / optimal.sum()
        self.weights = optimal.tolist()
        return self.weights

    def blend_oof(self, oof_predictions: list[np.ndarray]) -> np.ndarray:
        """Blend OOF predictions with current weights."""
        weights = self._normalize_weights()
        return sum(w * p for w, p in zip(weights, oof_predictions))

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        meta = {
            "n_models": len(self.models),
            "weights": self.weights,
            "model_names": [m.name for m in self.models],
        }
        with open(path / "ensemble_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        for i, model in enumerate(self.models):
            model.save(path / f"model_{i}")

    def _normalize_weights(self) -> list[float]:
        if self.weights is None:
            n = len(self.models)
            return [1.0 / n] * n
        total = sum(self.weights)
        if total == 0:
            n = len(self.weights)
            return [1.0 / n] * n
        return [w / total for w in self.weights]
