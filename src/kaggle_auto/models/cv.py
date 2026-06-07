"""Cross-validation strategies."""

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    GroupKFold,
    TimeSeriesSplit,
)

from .base import BaseModel


class CrossValidator:
    """Run cross-validation with any model."""

    STRATEGIES = {
        "kfold": KFold,
        "stratified_kfold": StratifiedKFold,
        "group_kfold": GroupKFold,
        "time_series_split": TimeSeriesSplit,
    }

    def __init__(
        self,
        strategy: str = "stratified_kfold",
        n_splits: int = 5,
        seed: int = 42,
    ):
        self.strategy = strategy
        self.n_splits = n_splits
        self.seed = seed

    def run(
        self,
        model_cls: type[BaseModel],
        X: pd.DataFrame,
        y: np.ndarray,
        params: dict | None = None,
        groups: np.ndarray | None = None,
        task: str = "regression",
    ) -> dict:
        """Run CV and return fold scores + OOF predictions."""
        splitter = self._get_splitter()
        split_args = self._get_split_args(X, y, groups)

        oof_preds = np.zeros(len(X))
        fold_scores = []
        models = []

        for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(*split_args)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = model_cls(params=params, task=task)
            metrics = model.fit(X_train, y_train, X_val, y_val)

            val_pred = model.predict(X_val)
            oof_preds[val_idx] = val_pred

            fold_score = metrics.get("val_score", 0.0)
            fold_scores.append(fold_score)
            models.append(model)

        return {
            "fold_scores": fold_scores,
            "mean_score": np.mean(fold_scores),
            "std_score": np.std(fold_scores),
            "oof_preds": oof_preds,
            "models": models,
        }

    def _get_splitter(self):
        splitter_cls = self.STRATEGIES.get(self.strategy, KFold)

        if self.strategy == "group_kfold":
            return splitter_cls(n_splits=self.n_splits)
        elif self.strategy == "time_series_split":
            return splitter_cls(n_splits=self.n_splits)
        else:
            return splitter_cls(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.seed,
            )

    def _get_split_args(self, X, y, groups):
        if self.strategy == "group_kfold":
            return (X, y, groups)
        return (X, y)
