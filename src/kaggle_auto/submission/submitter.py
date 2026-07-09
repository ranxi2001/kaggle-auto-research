"""Submit predictions to Kaggle with strict budget management."""

import json
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import WorkspaceConfig
from ..research.kaggle_api import KaggleAPI
from .validator import SubmissionValidator, ValidationResult
from .tracker import ScoreTracker


class SubmissionBudget:
    """Track and enforce daily submission budget.

    Strategy: reserve submissions for high-confidence candidates only.
    Default budget: 2 per day (out of Kaggle's typical 5-10 limit).
    """

    def __init__(self, workspace_path: Path, max_daily: int = 2):
        self.path = workspace_path / ".state" / "submission_budget.json"
        self.max_daily = max_daily
        self._load()

    def _load(self):
        if self.path.exists():
            self.data = json.loads(self.path.read_text())
        else:
            self.data = {"submissions": [], "reserved": []}

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2, default=str))

    def today_count(self) -> int:
        today = str(date.today())
        return sum(1 for s in self.data["submissions"] if s["date"] == today)

    def remaining_today(self) -> int:
        return max(0, self.max_daily - self.today_count())

    def can_submit(self) -> bool:
        return self.remaining_today() > 0

    def record(self, submission_path: str, cv_score: float | None, message: str):
        self.data["submissions"].append({
            "date": str(date.today()),
            "path": submission_path,
            "cv_score": cv_score,
            "message": message,
            "timestamp": time.time(),
        })
        self._save()

    def reserve(self, submission_path: str, cv_score: float, reason: str):
        """Queue a candidate for submission (doesn't actually submit)."""
        normalized_path = str(Path(submission_path).resolve())
        entry = {
            "path": normalized_path,
            "cv_score": cv_score,
            "reason": reason,
            "date_reserved": str(date.today()),
        }
        for idx, reserved in enumerate(self.data.get("reserved", [])):
            if str(Path(reserved.get("path", "")).resolve()) == normalized_path:
                self.data["reserved"][idx] = entry
                self._save()
                return
        self.data["reserved"].append({
            **entry,
        })
        self._save()

    def get_reserved(self) -> list[dict]:
        return self.data.get("reserved", [])

    def pop_best_reserved(self, direction: str = "maximize") -> dict | None:
        reserved = self.get_reserved()
        if not reserved:
            return None
        if direction == "maximize":
            best = max(reserved, key=lambda x: x.get("cv_score", 0))
        else:
            best = min(reserved, key=lambda x: x.get("cv_score", float("inf")))
        self.data["reserved"].remove(best)
        self._save()
        return best


class Submitter:
    """Handle prediction generation and submission with budget protection."""

    def __init__(self, workspace_path: Path, config: WorkspaceConfig):
        self.workspace = workspace_path
        self.config = config
        self.api: KaggleAPI | None = None
        self.validator = SubmissionValidator()
        self.tracker = ScoreTracker(workspace_path)
        self.budget = SubmissionBudget(
            workspace_path,
            max_daily=config.submission.max_daily,
        )

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
        """Submit to Kaggle with strict budget enforcement.

        Default behavior (force=False):
        1. Validates format
        2. Checks CV score beats previous best by threshold
        3. Checks daily budget (NOT Kaggle's limit — our own stricter limit)
        4. Only then submits

        With force=True:
        - Skips threshold check but STILL enforces daily budget
        - Use only for diverse/exploratory submissions you intentionally chose
        """
        validation = self.validate(submission_path)
        if not validation.is_valid:
            return {
                "success": False,
                "errors": validation.errors,
            }

        # Budget check — ALWAYS enforced, even with force=True
        if not self.budget.can_submit():
            self.budget.reserve(
                str(submission_path), cv_score or 0.0,
                reason=f"Budget exhausted. {message}",
            )
            return {
                "success": False,
                "queued": True,
                "errors": [
                    f"Daily budget exhausted ({self.budget.today_count()}/{self.budget.max_daily}). "
                    f"Submission queued for tomorrow. Use `kar submit --flush` to submit reserved."
                ],
                "remaining_today": 0,
            }

        # Threshold check (skipped with force=True)
        if not force and cv_score is not None:
            best = self.tracker.get_best_score()
            threshold = self.config.submission.best_threshold
            if best is not None:
                direction = self.config.competition.metric_direction
                if direction == "maximize" and cv_score < best + threshold:
                    self.budget.reserve(
                        str(submission_path), cv_score,
                        reason=f"Below threshold. CV={cv_score:.4f}, best={best:.4f}",
                    )
                    return {
                        "success": False,
                        "queued": True,
                        "errors": [
                            f"CV {cv_score:.4f} doesn't beat best {best:.4f} by threshold {threshold}. "
                            f"Queued instead of wasting submission."
                        ],
                        "remaining_today": self.budget.remaining_today(),
                    }
                elif direction == "minimize" and cv_score > best - threshold:
                    self.budget.reserve(
                        str(submission_path), cv_score,
                        reason=f"Below threshold. CV={cv_score:.4f}, best={best:.4f}",
                    )
                    return {
                        "success": False,
                        "queued": True,
                        "errors": [
                            f"CV {cv_score:.4f} doesn't beat best {best:.4f} by threshold {threshold}. "
                            f"Queued instead of wasting submission."
                        ],
                        "remaining_today": self.budget.remaining_today(),
                    }

        # Actually submit
        slug = self.config.competition.slug if hasattr(self.config.competition, 'slug') else self.config.competition.name
        if self.api is None:
            self.api = KaggleAPI()
        result = self.api.submit(slug, submission_path, message)

        self.budget.record(str(submission_path), cv_score, message)
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
            "remaining_today": self.budget.remaining_today(),
        }

    def submit_reserved(self, n: int = 1) -> list[dict]:
        """Submit the best N reserved candidates."""
        results = []
        direction = self.config.competition.metric_direction
        for _ in range(n):
            if not self.budget.can_submit():
                break
            candidate = self.budget.pop_best_reserved(direction)
            if not candidate:
                break
            path = Path(candidate["path"])
            if not path.exists():
                results.append({"success": False, "errors": [f"File not found: {path}"]})
                continue
            result = self.submit(
                path,
                message=f"Reserved: {candidate['reason']}",
                cv_score=candidate.get("cv_score"),
                force=True,
            )
            results.append(result)
        return results

    def status(self) -> dict:
        """Get submission budget status."""
        today = date.today()
        return {
            "today": str(today),
            "next_reset_date": str(today + timedelta(days=1)),
            "submitted_today": self.budget.today_count(),
            "remaining_today": self.budget.remaining_today(),
            "max_daily": self.budget.max_daily,
            "reserved_queue": len(self.budget.get_reserved()),
            "reserved": self.budget.get_reserved(),
        }

    def _next_sub_id(self) -> str:
        existing = list((self.workspace / "submissions").glob("sub_*.csv"))
        return f"{len(existing) + 1:03d}"
