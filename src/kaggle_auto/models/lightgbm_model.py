"""LightGBM model wrapper."""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from .base import BaseModel


class LightGBMModel(BaseModel):
    """LightGBM wrapper with unified interface."""

    name = "lightgbm"

    def __init__(self, params: dict | None = None, task: str = "regression"):
        self.task = task
        self.params = params or self._default_params()
        self.model: lgb.Booster | None = None
        self.feature_names: list[str] = []

    def _default_params(self) -> dict:
        base = {
            "verbosity": -1,
            "n_estimators": 1000,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "random_state": 42,
        }
        if self.task == "classification":
            base["objective"] = "binary"
            base["metric"] = "binary_logloss"
        else:
            base["objective"] = "regression"
            base["metric"] = "rmse"
        return base

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame | None = None,
        y_val: np.ndarray | None = None,
        params: dict | None = None,
    ) -> dict:
        if params:
            self.params.update(params)

        self.feature_names = X_train.columns.tolist()

        fit_params = {k: v for k, v in self.params.items()
                      if k not in ["n_estimators", "random_state"]}

        if self.task == "classification":
            model = lgb.LGBMClassifier(
                n_estimators=self.params.get("n_estimators", 1000),
                random_state=self.params.get("random_state", 42),
                **fit_params,
            )
        else:
            model = lgb.LGBMRegressor(
                n_estimators=self.params.get("n_estimators", 1000),
                random_state=self.params.get("random_state", 42),
                **fit_params,
            )

        callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]

        if X_val is not None and y_val is not None:
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=callbacks,
            )
        else:
            model.fit(X_train, y_train)

        self.model = model

        metrics = {"best_iteration": getattr(model, "best_iteration_", -1)}
        if X_val is not None and y_val is not None:
            val_pred = self.predict(X_val)
            metrics["val_score"] = self._compute_metric(y_val, val_pred)

        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted")
        if self.task == "classification" and hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)[:, 1]
        return self.model.predict(X)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "model.pkl", "wb") as f:
            pickle.dump(self.model, f)
        with open(path / "params.json", "w") as f:
            json.dump(self.params, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "LightGBMModel":
        with open(path / "params.json") as f:
            params = json.load(f)
        instance = cls(params=params)
        with open(path / "model.pkl", "rb") as f:
            instance.model = pickle.load(f)
        return instance

    def get_feature_importance(self) -> pd.DataFrame:
        if self.model is None:
            return pd.DataFrame(columns=["feature", "importance"])
        importance = self.model.feature_importances_
        return pd.DataFrame({
            "feature": self.feature_names,
            "importance": importance,
        }).sort_values("importance", ascending=False).reset_index(drop=True)

    def _compute_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if self.task == "classification":
            from sklearn.metrics import log_loss
            return log_loss(y_true, y_pred)
        else:
            from sklearn.metrics import r2_score
            return r2_score(y_true, y_pred)
