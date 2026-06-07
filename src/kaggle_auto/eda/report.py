"""Generate EDA summary reports."""

from pathlib import Path

import pandas as pd

from .profiler import DataProfile, DataProfiler


class EDAReport:
    """Generate markdown EDA report from profiling results."""

    def __init__(self, workspace_path: Path):
        self.workspace = workspace_path
        self.reports_dir = workspace_path / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame | None = None,
        target_col: str | None = None,
    ) -> Path:
        """Generate full EDA report."""
        profiler = DataProfiler()
        profile = profiler.profile(train_df, target_col)

        lines = [
            "# EDA Summary Report",
            "",
            "## Dataset Overview",
            "",
            f"- **Rows**: {profile.n_rows:,}",
            f"- **Columns**: {profile.n_cols}",
            f"- **Memory**: {profile.memory_mb:.1f} MB",
            f"- **Duplicate rows**: {profile.duplicate_rows:,}",
            "",
            f"- **Numeric columns**: {len(profile.numeric_cols)}",
            f"- **Categorical columns**: {len(profile.categorical_cols)}",
            f"- **Datetime columns**: {len(profile.datetime_cols)}",
            "",
        ]

        if profile.target_stats:
            lines.extend(self._format_target_section(profile.target_stats))

        lines.extend(self._format_quality_section(profile))
        lines.extend(self._format_column_details(profile))

        if test_df is not None:
            shifts = profiler.detect_distribution_shift(train_df, test_df)
            lines.extend(self._format_shift_section(shifts))

        if target_col:
            leaky = profiler.detect_leakage(train_df, target_col)
            if leaky:
                lines.extend([
                    "## Potential Target Leakage",
                    "",
                    "The following columns have >0.95 correlation with target:",
                    "",
                ])
                for col in leaky:
                    lines.append(f"- `{col}`")
                lines.append("")

        output_path = self.reports_dir / "eda_summary.md"
        output_path.write_text("\n".join(lines))
        return output_path

    def _format_target_section(self, stats: dict) -> list[str]:
        lines = [
            "## Target Variable",
            "",
            f"- **Name**: `{stats['name']}`",
            f"- **Type**: {stats['type']}",
        ]

        if stats["type"] == "regression":
            lines.extend([
                f"- **Mean**: {stats['mean']:.4f}",
                f"- **Std**: {stats['std']:.4f}",
                f"- **Range**: [{stats['min']:.4f}, {stats['max']:.4f}]",
            ])
        else:
            lines.extend([
                f"- **Classes**: {stats['n_classes']}",
                f"- **Imbalance ratio**: {stats['imbalance_ratio']:.2f}",
            ])

        lines.append("")
        return lines

    def _format_quality_section(self, profile: DataProfile) -> list[str]:
        lines = ["## Data Quality", ""]

        if profile.high_null_cols:
            lines.append(f"**High null columns** (>{self._pct(0.3)}): "
                        f"{', '.join(f'`{c}`' for c in profile.high_null_cols)}")
            lines.append("")

        if profile.constant_cols:
            lines.append(f"**Constant columns** (can drop): "
                        f"{', '.join(f'`{c}`' for c in profile.constant_cols)}")
            lines.append("")

        if profile.high_cardinality_cols:
            lines.append(f"**High cardinality** (>95%): "
                        f"{', '.join(f'`{c}`' for c in profile.high_cardinality_cols)}")
            lines.append("")

        if not (profile.high_null_cols or profile.constant_cols or profile.high_cardinality_cols):
            lines.append("No major quality issues detected.")
            lines.append("")

        return lines

    def _format_column_details(self, profile: DataProfile) -> list[str]:
        lines = ["## Column Details", "", "| Column | Type | Null% | Unique | Notes |",
                 "|--------|------|-------|--------|-------|"]

        for col in profile.columns:
            notes = []
            if col.null_pct > 0.3:
                notes.append("high null")
            if col.unique_count <= 1:
                notes.append("constant")
            if col.skew and abs(col.skew) > 2:
                notes.append(f"skew={col.skew:.1f}")

            lines.append(
                f"| `{col.name}` | {col.dtype} | {col.null_pct*100:.1f}% | "
                f"{col.unique_count} | {', '.join(notes)} |"
            )

        lines.append("")
        return lines

    def _format_shift_section(self, shifts: list[dict]) -> list[str]:
        if not shifts:
            return ["## Train/Test Distribution", "", "No significant shift detected.", ""]

        lines = [
            "## Train/Test Distribution Shift",
            "",
            "| Column | Train Mean | Test Mean | Shift Score |",
            "|--------|-----------|----------|-------------|",
        ]

        for s in shifts[:10]:
            lines.append(
                f"| `{s['column']}` | {s['train_mean']:.4f} | "
                f"{s['test_mean']:.4f} | {s['shift_score']:.2f} |"
            )

        lines.append("")
        return lines

    @staticmethod
    def _pct(val: float) -> str:
        return f"{val*100:.0f}%"
