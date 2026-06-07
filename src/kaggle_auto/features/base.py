"""Base feature interface."""

from abc import ABC, abstractmethod

import pandas as pd


class BaseFeature(ABC):
    """Base class for feature generators."""

    name: str = "base_feature"
    dependencies: list[str] = []

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute features and return DataFrame with new columns."""

    def __repr__(self) -> str:
        return f"<Feature:{self.name}>"
