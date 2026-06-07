"""Data profiling: types, nulls, distributions, correlations."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    null_count: int
    null_pct: float
    unique_count: int
    cardinality_pct: float
    mean: float | None = None
    std: float | None = None
    min_val: float | None = None
    max_val: float | None = None
    skew: float | None = None
    top_values: list[tuple] = field(default_factory=list)


@dataclass
class DataProfile:
    n_rows: int
    n_cols: int
    memory_mb: float
    columns: list[ColumnProfile]
    numeric_cols: list[str]
    categorical_cols: list[str]
    datetime_cols: list[str]
    high_null_cols: list[str]
    high_cardinality_cols: list[str]
    constant_cols: list[str]
    duplicate_rows: int
    target_stats: dict | None = None


class DataProfiler:
    """Profile a DataFrame to understand data characteristics."""

    def __init__(self, null_threshold: float = 0.3, cardinality_threshold: float = 0.95):
        self.null_threshold = null_threshold
        self.cardinality_threshold = cardinality_threshold

    def profile(self, df: pd.DataFrame, target_col: str | None = None) -> DataProfile:
        """Generate a complete data profile."""
        columns = []
        numeric_cols = []
        categorical_cols = []
        datetime_cols = []
        high_null_cols = []
        high_cardinality_cols = []
        constant_cols = []

        for col in df.columns:
            cp = self._profile_column(df[col], len(df))
            columns.append(cp)

            if cp.null_pct > self.null_threshold:
                high_null_cols.append(col)
            if cp.cardinality_pct > self.cardinality_threshold:
                high_cardinality_cols.append(col)
            if cp.unique_count <= 1:
                constant_cols.append(col)

            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_cols.append(col)
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                datetime_cols.append(col)
            else:
                categorical_cols.append(col)

        target_stats = None
        if target_col and target_col in df.columns:
            target_stats = self._target_analysis(df[target_col])

        return DataProfile(
            n_rows=len(df),
            n_cols=len(df.columns),
            memory_mb=df.memory_usage(deep=True).sum() / 1024 / 1024,
            columns=columns,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            datetime_cols=datetime_cols,
            high_null_cols=high_null_cols,
            high_cardinality_cols=high_cardinality_cols,
            constant_cols=constant_cols,
            duplicate_rows=df.duplicated().sum(),
            target_stats=target_stats,
        )

    def detect_leakage(self, df: pd.DataFrame, target_col: str) -> list[str]:
        """Detect potential target leakage (features highly correlated with target)."""
        if target_col not in df.columns:
            return []
        if not pd.api.types.is_numeric_dtype(df[target_col]):
            return []

        numeric = df.select_dtypes(include=[np.number])
        if target_col not in numeric.columns:
            return []

        corr = numeric.corr()[target_col].abs().sort_values(ascending=False)
        leaky = corr[(corr > 0.95) & (corr.index != target_col)]
        return leaky.index.tolist()

    def detect_distribution_shift(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> list[dict]:
        """Detect train/test distribution shift for numeric columns."""
        shifts = []
        common_cols = set(train_df.columns) & set(test_df.columns)

        for col in common_cols:
            if not pd.api.types.is_numeric_dtype(train_df[col]):
                continue
            train_mean = train_df[col].mean()
            test_mean = test_df[col].mean()
            train_std = train_df[col].std()

            if train_std == 0:
                continue

            shift_score = abs(train_mean - test_mean) / train_std
            if shift_score > 1.0:
                shifts.append({
                    "column": col,
                    "train_mean": train_mean,
                    "test_mean": test_mean,
                    "shift_score": shift_score,
                })

        return sorted(shifts, key=lambda x: x["shift_score"], reverse=True)

    def _profile_column(self, series: pd.Series, n_rows: int) -> ColumnProfile:
        null_count = series.isna().sum()
        unique_count = series.nunique()

        profile = ColumnProfile(
            name=series.name,
            dtype=str(series.dtype),
            null_count=int(null_count),
            null_pct=null_count / n_rows if n_rows > 0 else 0,
            unique_count=int(unique_count),
            cardinality_pct=unique_count / n_rows if n_rows > 0 else 0,
        )

        if pd.api.types.is_numeric_dtype(series):
            profile.mean = series.mean()
            profile.std = series.std()
            profile.min_val = series.min()
            profile.max_val = series.max()
            profile.skew = series.skew()
        else:
            vc = series.value_counts().head(5)
            profile.top_values = list(zip(vc.index.tolist(), vc.values.tolist()))

        return profile

    def _target_analysis(self, series: pd.Series) -> dict:
        stats = {"name": series.name, "dtype": str(series.dtype)}

        is_classification = (
            not pd.api.types.is_numeric_dtype(series) or series.nunique() <= 20
        )

        if is_classification:
            stats["type"] = "classification"
            vc = series.value_counts()
            stats["n_classes"] = len(vc)
            stats["class_balance"] = {str(k): int(v) for k, v in vc.items()}
            stats["imbalance_ratio"] = vc.max() / vc.min() if vc.min() > 0 else float("inf")
        else:
            stats["type"] = "regression"
            stats["mean"] = series.mean()
            stats["std"] = series.std()
            stats["min"] = series.min()
            stats["max"] = series.max()

        return stats
