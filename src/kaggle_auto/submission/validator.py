"""Pre-submission validation."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class ValidationResult:
    is_valid: bool
    errors: list[str]
    warnings: list[str]


class SubmissionValidator:
    """Validate submission format before uploading."""

    def validate(
        self,
        submission_path: Path,
        sample_submission_path: Path,
    ) -> ValidationResult:
        """Validate submission against sample_submission format."""
        errors = []
        warnings = []

        if not submission_path.exists():
            return ValidationResult(False, ["Submission file does not exist"], [])

        if not sample_submission_path.exists():
            return ValidationResult(False, ["Sample submission file does not exist"], [])

        try:
            sub = pd.read_csv(submission_path)
            sample = pd.read_csv(sample_submission_path)
        except Exception as e:
            return ValidationResult(False, [f"Failed to read CSV: {e}"], [])

        if list(sub.columns) != list(sample.columns):
            errors.append(
                f"Column mismatch. Expected: {list(sample.columns)}, "
                f"Got: {list(sub.columns)}"
            )

        if len(sub) != len(sample):
            errors.append(
                f"Row count mismatch. Expected: {len(sample)}, Got: {len(sub)}"
            )

        if sub.isnull().any().any():
            null_cols = sub.columns[sub.isnull().any()].tolist()
            errors.append(f"Null values found in columns: {null_cols}")

        id_col = sample.columns[0]
        if id_col in sub.columns and id_col in sample.columns:
            if not sub[id_col].equals(sample[id_col]):
                if set(sub[id_col]) != set(sample[id_col]):
                    errors.append(f"ID column '{id_col}' values don't match sample")
                else:
                    warnings.append(f"ID column '{id_col}' order differs from sample")

        pred_cols = sample.columns[1:]
        for col in pred_cols:
            if col in sub.columns:
                if pd.api.types.is_numeric_dtype(sub[col]):
                    if sub[col].min() < -1e10 or sub[col].max() > 1e10:
                        warnings.append(f"Column '{col}' has extreme values")
                    if np.isinf(sub[col]).any():
                        errors.append(f"Column '{col}' contains infinity")

        if sub.duplicated(subset=[id_col]).any() if id_col in sub.columns else False:
            errors.append("Duplicate IDs found")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )
