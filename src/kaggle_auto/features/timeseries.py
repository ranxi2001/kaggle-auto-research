"""Time series / crypto feature generators."""

import numpy as np
import pandas as pd

from .base import BaseFeature
from .registry import register


@register
class LagFeatures(BaseFeature):
    """Lag features for time series data."""

    name = "lag_features"

    def __init__(self, lags: list[int] | None = None, cols: list[str] | None = None):
        self.lags = lags or [1, 2, 3, 5, 10]
        self.cols = cols

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric = df.select_dtypes(include=[np.number])
        cols = self.cols or numeric.columns[:5].tolist()
        result = pd.DataFrame(index=df.index)

        for col in cols:
            if col not in df.columns:
                continue
            for lag in self.lags:
                result[f"{col}_lag_{lag}"] = df[col].shift(lag)

        return result


@register
class RollingStats(BaseFeature):
    """Rolling window statistics."""

    name = "rolling_stats"

    def __init__(self, windows: list[int] | None = None, cols: list[str] | None = None):
        self.windows = windows or [5, 10, 20, 50]
        self.cols = cols

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric = df.select_dtypes(include=[np.number])
        cols = self.cols or numeric.columns[:5].tolist()
        result = pd.DataFrame(index=df.index)

        for col in cols:
            if col not in df.columns:
                continue
            for w in self.windows:
                rolling = df[col].rolling(window=w, min_periods=1)
                result[f"{col}_roll_mean_{w}"] = rolling.mean()
                result[f"{col}_roll_std_{w}"] = rolling.std()
                result[f"{col}_roll_min_{w}"] = rolling.min()
                result[f"{col}_roll_max_{w}"] = rolling.max()

        return result


@register
class TechnicalIndicators(BaseFeature):
    """Common technical indicators for crypto/financial data."""

    name = "technical_indicators"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(index=df.index)

        close_col = self._find_column(df, ["close", "price", "target"])
        if close_col is None:
            return result

        close = df[close_col]

        # RSI
        result[f"{close_col}_rsi_14"] = self._rsi(close, 14)

        # MACD
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        result[f"{close_col}_macd"] = ema12 - ema26
        result[f"{close_col}_macd_signal"] = result[f"{close_col}_macd"].ewm(span=9).mean()

        # Bollinger Bands
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        result[f"{close_col}_bb_upper"] = sma20 + 2 * std20
        result[f"{close_col}_bb_lower"] = sma20 - 2 * std20
        result[f"{close_col}_bb_width"] = (result[f"{close_col}_bb_upper"] - result[f"{close_col}_bb_lower"]) / sma20

        # Moving averages
        for period in [5, 10, 20, 50]:
            result[f"{close_col}_sma_{period}"] = close.rolling(period).mean()
            result[f"{close_col}_ema_{period}"] = close.ewm(span=period).mean()

        # Returns
        result[f"{close_col}_return_1"] = close.pct_change(1)
        result[f"{close_col}_return_5"] = close.pct_change(5)

        # Volatility
        result[f"{close_col}_volatility_10"] = close.pct_change().rolling(10).std()
        result[f"{close_col}_volatility_20"] = close.pct_change().rolling(20).std()

        return result

    @staticmethod
    def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
        for c in candidates:
            matching = [col for col in df.columns if c in col.lower()]
            if matching:
                return matching[0]
        return None


@register
class Volatility(BaseFeature):
    """Volatility-based features for crypto/financial data."""

    name = "volatility"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(index=df.index)

        high_col = self._find(df, ["high"])
        low_col = self._find(df, ["low"])
        close_col = self._find(df, ["close", "price"])

        if high_col and low_col:
            result["true_range"] = df[high_col] - df[low_col]
            result["atr_14"] = result["true_range"].rolling(14).mean()

        if close_col:
            returns = df[close_col].pct_change()
            result["realized_vol_5"] = returns.rolling(5).std() * np.sqrt(252)
            result["realized_vol_20"] = returns.rolling(20).std() * np.sqrt(252)
            result["vol_of_vol"] = result["realized_vol_20"].rolling(20).std()

        volume_col = self._find(df, ["volume"])
        if volume_col:
            result["volume_sma_10"] = df[volume_col].rolling(10).mean()
            result["volume_ratio"] = df[volume_col] / result["volume_sma_10"]

        return result

    @staticmethod
    def _find(df: pd.DataFrame, candidates: list[str]) -> str | None:
        for c in candidates:
            matching = [col for col in df.columns if c in col.lower()]
            if matching:
                return matching[0]
        return None
