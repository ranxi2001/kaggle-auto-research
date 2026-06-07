"""Submit predictions to Kaggle."""

import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import WorkspaceConfig
from ..research.kaggle_api import KaggleAPI
from .validator import SubmissionValidator, ValidationResult
from .tracker import ScoreTracker


class Submitter:
    """Handle prediction generation and submission."""

    def __init__(self, workspace_path: Path, config: WorkspaceConfig):
        self.workspace = workspace_path
        self.config = config
        self.api = KaggleAPI()
        self.validator = SubmissionValidator()
        self.tracker = ScoreTracker(workspace_path)

    def generate_submission(
        self,
        predictions: np.ndarray,
        model_version: str = "v001",
    ) -> Path:
        """Generate submission CSV from predictions."""
        sample_path = self.workspace / self.config.data.sample_submission
        sample = pd.read_csv(sample_path)

        sub = sample.copy()
        pred_col = sub.columns[1]
        sub[pred_col] = predictions

        sub_name = f"sub_{self._next_sub_id()}_{model_version}.csv"
        sub_path = self.workspace / "submissions" / sub_name
        sub_path.parent.mkdir(parents=True, exist_ok=True)
        sub.to_csv(sub_path, index=False)

        return sub_path

    def validate(self, submission_path: Path) -> ValidationResult:
        """Validate submission format."""
        sample_path = self.workspace / self.config.data.sample_submission
        return self.validator.validate(submission_path, sample_path)

    def submit(
        self,
        submission_path: Path,
        message: str = "",
        model_version: str = "",
        features_version: str = "",
        cv_score: float | None = None,
        force: bool = False,
    ) -> dict:
        """Submit to Kaggle with validation and tracking."""
        validation = self.validate(submission_path)
        if not validation.is_valid:
            return {
                "success": False,
                "errors": validation.errors,
            }

        if not force and cv_score is not None:
            best = self.tracker.get_best_score()
            threshold = self.config.submission.best_threshold
            if best is not None:
                direction = self.config.competition.metric_direction
                if direction == "maximize" and cv_score < best + threshold:
                    return {
                        "success": False,
                        "errors": [f"CV score {cv_score:.6f} does not improve over best {best:.6f} by threshold {threshold}"],
                    }
                elif direction == "minimize" and cv_score > best - threshold:
                    return {
                        "success": False,
                        "errors": [f"CV score {cv_score:.6f} does not improve over best {best:.6f} by threshold {threshold}"],
                    }

        daily_count = self.tracker.get_daily_submission_count()
        if daily_count >= self.config.submission.max_daily and not force:
            return {
                "success": False,
                "errors": [f"Daily submission limit reached ({daily_count}/{self.config.submission.max_daily})"],
            }

        slug = self.config.competition.slug if hasattr(self.config.competition, 'slug') else self.config.competition.name
        result = self.api.submit(slug, submission_path, message)

        self.tracker.record_submission(
            submission_path=submission_path,
            model_version=model_version,
            features_version=features_version,
            cv_score=cv_score,
        )

        return {
            "success": True,
            "message": result,
            "warnings": validation.warnings,
        }

    def _next_sub_id(self) -> str:
        existing = list((self.workspace / "submissions").glob("sub_*.csv"))
        return f"{len(existing) + 1:03d}"
