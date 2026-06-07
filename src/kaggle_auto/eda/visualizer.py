"""Auto-generate EDA visualizations."""

from pathlib import Path

import numpy as np
import pandas as pd


class Visualizer:
    """Generate EDA plots and save to files."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_target_distribution(self, df: pd.DataFrame, target_col: str) -> Path:
        """Plot target variable distribution."""
        import plotly.express as px

        series = df[target_col].dropna()
        if pd.api.types.is_numeric_dtype(series) and series.nunique() > 20:
            fig = px.histogram(series, nbins=50, title=f"Target Distribution: {target_col}")
        else:
            vc = series.value_counts()
            fig = px.bar(x=vc.index.astype(str), y=vc.values, title=f"Target Distribution: {target_col}")

        path = self.output_dir / "target_distribution.html"
        fig.write_html(str(path))
        return path

    def plot_correlation_matrix(self, df: pd.DataFrame, top_n: int = 30) -> Path:
        """Plot correlation heatmap for numeric columns."""
        import plotly.express as px

        numeric = df.select_dtypes(include=[np.number])
        if len(numeric.columns) > top_n:
            variance = numeric.var().sort_values(ascending=False)
            numeric = numeric[variance.head(top_n).index]

        corr = numeric.corr()
        fig = px.imshow(
            corr,
            title="Feature Correlation Matrix",
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
        )

        path = self.output_dir / "correlation_matrix.html"
        fig.write_html(str(path))
        return path

    def plot_missing_values(self, df: pd.DataFrame) -> Path:
        """Plot missing value percentages."""
        import plotly.express as px

        null_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
        null_pct = null_pct[null_pct > 0]

        if null_pct.empty:
            return self._write_empty_plot("No missing values found")

        fig = px.bar(
            x=null_pct.values,
            y=null_pct.index,
            orientation="h",
            title="Missing Values (%)",
        )

        path = self.output_dir / "missing_values.html"
        fig.write_html(str(path))
        return path

    def plot_feature_vs_target(
        self, df: pd.DataFrame, feature_col: str, target_col: str
    ) -> Path:
        """Plot a single feature against target."""
        import plotly.express as px

        subset = df[[feature_col, target_col]].dropna()

        if pd.api.types.is_numeric_dtype(subset[feature_col]):
            fig = px.scatter(
                subset, x=feature_col, y=target_col,
                title=f"{feature_col} vs {target_col}",
                opacity=0.5,
            )
        else:
            fig = px.box(
                subset, x=feature_col, y=target_col,
                title=f"{feature_col} vs {target_col}",
            )

        path = self.output_dir / f"feat_{feature_col}_vs_target.html"
        fig.write_html(str(path))
        return path

    def plot_numeric_distributions(self, df: pd.DataFrame, cols: list[str] | None = None) -> Path:
        """Plot distributions for numeric columns."""
        import plotly.express as px
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go

        if cols is None:
            cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cols = cols[:20]

        n_cols = min(4, len(cols))
        n_rows = (len(cols) + n_cols - 1) // n_cols

        fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=cols)

        for i, col in enumerate(cols):
            row = i // n_cols + 1
            col_idx = i % n_cols + 1
            fig.add_trace(
                go.Histogram(x=df[col].dropna(), name=col, showlegend=False),
                row=row, col=col_idx,
            )

        fig.update_layout(height=300 * n_rows, title_text="Numeric Distributions")
        path = self.output_dir / "numeric_distributions.html"
        fig.write_html(str(path))
        return path

    def plot_time_series(self, df: pd.DataFrame, time_col: str, value_cols: list[str]) -> Path:
        """Plot time series for crypto/temporal data."""
        import plotly.express as px

        subset = df[[time_col] + value_cols].copy()
        subset[time_col] = pd.to_datetime(subset[time_col])
        subset = subset.sort_values(time_col)

        melted = subset.melt(id_vars=[time_col], value_vars=value_cols)
        fig = px.line(melted, x=time_col, y="value", color="variable", title="Time Series")

        path = self.output_dir / "time_series.html"
        fig.write_html(str(path))
        return path

    def _write_empty_plot(self, message: str) -> Path:
        path = self.output_dir / "empty.html"
        path.write_text(f"<html><body><p>{message}</p></body></html>")
        return path
