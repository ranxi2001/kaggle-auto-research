"""Optuna-based hyperparameter optimization."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import optuna

from ..models.base import BaseModel
from ..models.cv import CrossValidator
from .search_space import get_search_space, suggest_param

optuna.logging.set_verbosity(optuna.logging.WARNING)


class OptunaTuner:
    """Hyperparameter tuning with Optuna."""

    def __init__(
        self,
        model_cls: type[BaseModel],
        task: str = "regression",
        cv_strategy: str = "stratified_kfold",
        n_splits: int = 5,
        seed: int = 42,
        direction: str = "minimize",
    ):
        self.model_cls = model_cls
        self.task = task
        self.cv_strategy = cv_strategy
        self.n_splits = n_splits
        self.seed = seed
        self.direction = direction
        self.study: optuna.Study | None = None
        self.best_params: dict | None = None

    def tune(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        n_trials: int = 50,
        search_space: dict | None = None,
        timeout: int | None = None,
    ) -> dict:
        """Run hyperparameter optimization."""
        space = search_space or get_search_space(self.model_cls.name)

        self.study = optuna.create_study(direction=self.direction)

        def objective(trial):
            params = {}
            for name, spec in space.items():
                params[name] = suggest_param(trial, name, spec)

            cv = CrossValidator(
                strategy=self.cv_strategy,
                n_splits=self.n_splits,
                seed=self.seed,
            )

            result = cv.run(
                model_cls=self.model_cls,
                X=X, y=y,
                params=params,
                task=self.task,
            )

            return result["mean_score"]

        self.study.optimize(objective, n_trials=n_trials, timeout=timeout)

        self.best_params = self.study.best_params
        return {
            "best_params": self.best_params,
            "best_score": self.study.best_value,
            "n_trials": len(self.study.trials),
            "trials_summary": self._summarize_trials(),
        }

    def get_param_importance(self) -> dict:
        """Get hyperparameter importance from study."""
        if self.study is None:
            return {}
        try:
            importance = optuna.importance.get_param_importances(self.study)
            return importance
        except Exception:
            return {}

    def save_results(self, path: Path) -> None:
        """Save tuning results to disk."""
        path.mkdir(parents=True, exist_ok=True)

        if self.best_params:
            with open(path / "best_params.json", "w") as f:
                json.dump(self.best_params, f, indent=2)

        if self.study:
            trials_data = []
            for trial in self.study.trials:
                trials_data.append({
                    "number": trial.number,
                    "value": trial.value,
                    "params": trial.params,
                    "state": trial.state.name,
                })
            with open(path / "trials.json", "w") as f:
                json.dump(trials_data, f, indent=2)

    def _summarize_trials(self) -> dict:
        if self.study is None:
            return {}

        values = [t.value for t in self.study.trials if t.value is not None]
        if not values:
            return {}

        return {
            "best": min(values) if self.direction == "minimize" else max(values),
            "worst": max(values) if self.direction == "minimize" else min(values),
            "mean": np.mean(values),
            "std": np.std(values),
        }
