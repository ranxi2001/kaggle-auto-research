"""Base model interface."""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd


class BaseModel(ABC):
    """Unified interface for all competition models."""

    name: str = "base"

    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame | None = None,
        y_val: np.ndarray | None = None,
        params: dict | None = None,
    ) -> dict:
        """Train the model. Returns metrics dict."""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""

    @abstractmethod
    def save(self, path: Path) -> None:
        """Save model to disk."""

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "BaseModel":
        """Load model from disk."""

    def get_feature_importance(self) -> pd.DataFrame:
        """Return feature importance as DataFrame with columns [feature, importance]."""
        return pd.DataFrame(columns=["feature", "importance"])
