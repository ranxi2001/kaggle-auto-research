"""XGBoost model wrapper."""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from .base import BaseModel


class XGBoostModel(BaseModel):
    """XGBoost wrapper with unified interface."""

    name = "xgboost"

    def __init__(self, params: dict | None = None, task: str = "regression"):
        self.task = task
        self.params = params or self._default_params()
        self.model: xgb.XGBModel | None = None
        self.feature_names: list[str] = []

    def _default_params(self) -> dict:
        base = {
            "n_estimators": 1000,
            "learning_rate": 0.05,
            "max_depth": 6,
            "min_child_weight": 1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "tree_method": "hist",
            "verbosity": 0,
        }
        if self.task == "classification":
            base["objective"] = "binary:logistic"
            base["eval_metric"] = "logloss"
        else:
            base["objective"] = "reg:squarederror"
            base["eval_metric"] = "rmse"
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

        if self.task == "classification":
            model = xgb.XGBClassifier(**self.params)
        else:
            model = xgb.XGBRegressor(**self.params)

        fit_kwargs = {}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["verbose"] = False

        model.fit(X_train, y_train, **fit_kwargs)
        self.model = model

        metrics = {"best_iteration": getattr(model, "best_iteration", -1)}
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
            json.dump(self.params, f, indent=2, default=str)

    @classmethod
    def load(cls, path: Path) -> "XGBoostModel":
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
            from sklearn.metrics import mean_squared_error
            return mean_squared_error(y_true, y_pred, squared=False)
