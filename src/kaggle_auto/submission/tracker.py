"""Score and submission tracking."""

import json
from datetime import datetime, date
from pathlib import Path


class ScoreTracker:
    """Track submission history and scores."""

    def __init__(self, workspace_path: Path):
        self.workspace = workspace_path
        self.history_file = workspace_path / "submissions" / "history.json"
        self._ensure_file()

    def record_submission(
        self,
        submission_path: Path,
        model_version: str = "",
        features_version: str = "",
        cv_score: float | None = None,
        lb_score: float | None = None,
        rank: int | None = None,
    ) -> None:
        """Record a submission in history."""
        history = self._load()

        entry = {
            "id": f"sub_{len(history) + 1:03d}",
            "timestamp": datetime.now().isoformat(),
            "file": submission_path.name,
            "model_version": model_version,
            "features_version": features_version,
            "cv_score": cv_score,
            "lb_score": lb_score,
            "rank": rank,
        }

        history.append(entry)
        self._save(history)

    def update_lb_score(self, submission_id: str, lb_score: float, rank: int | None = None) -> None:
        """Update LB score for a submission (after Kaggle returns it)."""
        history = self._load()
        for entry in history:
            if entry["id"] == submission_id:
                entry["lb_score"] = lb_score
                if rank is not None:
                    entry["rank"] = rank
                break
        self._save(history)

    def get_best_score(self) -> float | None:
        """Get best CV score from history."""
        history = self._load()
        scores = [e["cv_score"] for e in history if e.get("cv_score") is not None]
        return min(scores) if scores else None

    def get_best_lb_score(self) -> float | None:
        """Get best LB score from history."""
        history = self._load()
        scores = [e["lb_score"] for e in history if e.get("lb_score") is not None]
        return min(scores) if scores else None

    def get_daily_submission_count(self) -> int:
        """Count submissions made today."""
        history = self._load()
        today = date.today().isoformat()
        return sum(1 for e in history if e["timestamp"].startswith(today))

    def get_history(self) -> list[dict]:
        """Get full submission history."""
        return self._load()

    def get_cv_lb_gap(self) -> list[dict]:
        """Analyze CV vs LB score gap over time."""
        history = self._load()
        gaps = []
        for entry in history:
            if entry.get("cv_score") is not None and entry.get("lb_score") is not None:
                gaps.append({
                    "id": entry["id"],
                    "cv_score": entry["cv_score"],
                    "lb_score": entry["lb_score"],
                    "gap": abs(entry["cv_score"] - entry["lb_score"]),
                    "timestamp": entry["timestamp"],
                })
        return gaps

    def get_score_trend(self) -> dict:
        """Get score improvement trend."""
        history = self._load()
        cv_scores = [e["cv_score"] for e in history if e.get("cv_score") is not None]

        if len(cv_scores) < 2:
            return {"trend": "insufficient_data", "scores": cv_scores}

        improvements = [cv_scores[i] - cv_scores[i-1] for i in range(1, len(cv_scores))]
        recent_improvements = improvements[-3:] if len(improvements) >= 3 else improvements

        if all(imp >= 0 for imp in recent_improvements):
            trend = "stagnating"
        elif all(imp < 0 for imp in recent_improvements):
            trend = "improving"
        else:
            trend = "mixed"

        return {
            "trend": trend,
            "scores": cv_scores,
            "last_improvement": improvements[-1] if improvements else 0,
            "avg_recent_improvement": sum(recent_improvements) / len(recent_improvements),
        }

    def _load(self) -> list[dict]:
        if not self.history_file.exists():
            return []
        with open(self.history_file) as f:
            return json.load(f)

    def _save(self, history: list[dict]) -> None:
        with open(self.history_file, "w") as f:
            json.dump(history, f, indent=2)

    def _ensure_file(self) -> None:
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.history_file.exists():
            self._save([])
