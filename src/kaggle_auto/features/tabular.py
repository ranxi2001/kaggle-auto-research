"""Tabular feature generators."""

import numpy as np
import pandas as pd

from .base import BaseFeature
from .registry import register


@register
class BasicStats(BaseFeature):
    """Row-level basic statistics across numeric columns."""

    name = "basic_stats"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric = df.select_dtypes(include=[np.number])
        result = pd.DataFrame(index=df.index)
        if numeric.empty:
            return result
        result["row_mean"] = numeric.mean(axis=1)
        result["row_std"] = numeric.std(axis=1)
        result["row_max"] = numeric.max(axis=1)
        result["row_min"] = numeric.min(axis=1)
        result["row_null_count"] = df.isnull().sum(axis=1)
        return result


@register
class Interactions(BaseFeature):
    """Pairwise interactions between top numeric features."""

    name = "interactions"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric = df.select_dtypes(include=[np.number])
        cols = numeric.columns[:10]
        result = pd.DataFrame(index=df.index)

        for i in range(len(cols)):
            for j in range(i + 1, min(i + 3, len(cols))):
                c1, c2 = cols[i], cols[j]
                result[f"{c1}_x_{c2}"] = numeric[c1] * numeric[c2]
                denom = numeric[c2].replace(0, np.nan)
                result[f"{c1}_div_{c2}"] = numeric[c1] / denom

        return result


@register
class TargetEncoding(BaseFeature):
    """Target encoding for categorical columns (placeholder: uses frequency encoding)."""

    name = "target_encoding"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        categorical = df.select_dtypes(include=["object", "category"])
        result = pd.DataFrame(index=df.index)

        for col in categorical.columns[:20]:
            freq = df[col].value_counts(normalize=True)
            result[f"{col}_freq"] = df[col].map(freq)

        return result


@register
class Aggregations(BaseFeature):
    """Group-based aggregations for categorical x numeric pairs."""

    name = "aggregations"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:5]
        cat_cols = df.select_dtypes(include=["object", "category"]).columns[:3]
        result = pd.DataFrame(index=df.index)

        for cat in cat_cols:
            for num in numeric_cols:
                group = df.groupby(cat)[num]
                result[f"{cat}_{num}_mean"] = group.transform("mean")
                result[f"{cat}_{num}_std"] = group.transform("std")

        return result


@register
class NullIndicator(BaseFeature):
    """Binary indicators for missing values."""

    name = "null_indicator"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        null_cols = df.columns[df.isnull().any()]
        result = pd.DataFrame(index=df.index)
        for col in null_cols:
            result[f"{col}_is_null"] = df[col].isnull().astype(int)
        return result
