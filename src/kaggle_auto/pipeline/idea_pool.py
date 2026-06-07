"""Idea Pool: persistent store of improvement ideas for tree-search iteration.

Inspired by Qgentic-AI's idea pool concept. Ideas come from:
- Research (top solution patterns)
- Error analysis (what's failing)
- Feature importance (what's working)
- Agent suggestions

Each idea has a priority score and tracks whether it's been tried.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class Idea:
    """A single improvement idea."""

    id: str
    category: Literal["feature", "model", "preprocessing", "ensemble", "postprocess"]
    title: str
    description: str
    priority: float = 0.5
    source: str = ""
    tried: bool = False
    result: str = ""
    metric_delta: float | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "category": self.category,
            "title": self.title,
            "description": self.description,
            "priority": self.priority,
            "source": self.source,
            "tried": self.tried,
            "result": self.result,
            "metric_delta": self.metric_delta,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Idea":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class IdeaPool:
    """Manages a pool of improvement ideas with priority ranking."""

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.pool_file = workspace / ".state" / "idea_pool.json"
        self.ideas: list[Idea] = []
        self._load()

    def add(
        self,
        category: str,
        title: str,
        description: str,
        priority: float = 0.5,
        source: str = "",
    ) -> Idea:
        """Add a new idea to the pool."""
        # Avoid duplicates
        for existing in self.ideas:
            if existing.title.lower() == title.lower():
                existing.priority = max(existing.priority, priority)
                self._save()
                return existing

        idea_id = f"idea_{len(self.ideas):03d}"
        idea = Idea(
            id=idea_id,
            category=category,
            title=title,
            description=description,
            priority=priority,
            source=source,
        )
        self.ideas.append(idea)
        self._save()
        return idea

    def get_next(self, n: int = 1) -> list[Idea]:
        """Get the top N untried ideas by priority."""
        untried = [i for i in self.ideas if not i.tried]
        untried.sort(key=lambda x: x.priority, reverse=True)
        return untried[:n]

    def mark_tried(self, idea_id: str, result: str = "", metric_delta: float | None = None) -> None:
        """Mark an idea as tried with its result."""
        for idea in self.ideas:
            if idea.id == idea_id:
                idea.tried = True
                idea.result = result
                idea.metric_delta = metric_delta
                self._save()
                return

    def get_successful(self) -> list[Idea]:
        """Get ideas that improved the metric."""
        return [i for i in self.ideas if i.tried and i.metric_delta is not None and i.metric_delta < 0]

    def get_failed(self) -> list[Idea]:
        """Get ideas that didn't improve."""
        return [i for i in self.ideas if i.tried and (i.metric_delta is None or i.metric_delta >= 0)]

    def seed_from_research(self, research_report_path: Path) -> int:
        """Extract ideas from research report and seed the pool."""
        if not research_report_path.exists():
            return 0

        content = research_report_path.read_text()
        added = 0

        # Extract common patterns from research
        idea_templates = [
            ("feature", "Target encoding with smoothing", "Use target encoding with leave-one-out to reduce overfitting", 0.7),
            ("feature", "Feature interactions (multiply/divide)", "Create pairwise feature interactions for top important features", 0.6),
            ("feature", "Frequency encoding for categoricals", "Replace categories with their frequency counts", 0.5),
            ("model", "LightGBM with dart boosting", "Try dart boosting type for better generalization", 0.4),
            ("model", "XGBoost with tuned parameters", "Switch to XGBoost with competition-proven hyperparams", 0.5),
            ("ensemble", "Weighted average of LGB + XGB", "Ensemble LightGBM and XGBoost with optimized weights", 0.8),
            ("preprocessing", "Outlier clipping at 1st/99th percentile", "Clip extreme values to reduce noise", 0.4),
            ("preprocessing", "Log transform skewed features", "Apply log1p to features with skew > 2", 0.5),
            ("postprocess", "Threshold optimization", "Optimize classification threshold on OOF predictions", 0.6),
        ]

        for cat, title, desc, pri in idea_templates:
            if any(keyword in content.lower() for keyword in title.lower().split()[:2]):
                self.add(cat, title, desc, priority=pri, source="research")
                added += 1

        # Always seed basic ideas
        if not self.ideas:
            for cat, title, desc, pri in idea_templates:
                self.add(cat, title, desc, priority=pri, source="default")
                added += 1

        return added

    def seed_from_analysis(self, analysis: dict) -> int:
        """Generate ideas based on model analysis."""
        added = 0

        zero_feats = analysis.get("zero_importance_features", [])
        if len(zero_feats) > 3:
            self.add(
                "feature",
                "Remove zero-importance features",
                f"Drop {len(zero_feats)} features with zero importance: {zero_feats[:5]}",
                priority=0.7,
                source="analysis",
            )
            added += 1

        concentration = analysis.get("feature_concentration", 0)
        if concentration > 0.7:
            self.add(
                "feature",
                "Diversify feature sources",
                "Top features dominate — add new feature types (lag, rolling, poly)",
                priority=0.6,
                source="analysis",
            )
            added += 1

        stability = analysis.get("cv_stability", 0)
        if stability > 0.05:
            self.add(
                "model",
                "Reduce model complexity",
                f"CV instability {stability:.3f} — lower num_leaves or increase min_child_samples",
                priority=0.8,
                source="analysis",
            )
            added += 1

        return added

    def summary(self) -> str:
        """Get a text summary of the idea pool."""
        total = len(self.ideas)
        tried = sum(1 for i in self.ideas if i.tried)
        successful = len(self.get_successful())

        lines = [
            f"Idea Pool: {total} ideas ({tried} tried, {successful} successful)",
            "",
            "Top untried ideas:",
        ]
        for idea in self.get_next(5):
            lines.append(f"  [{idea.priority:.1f}] [{idea.category}] {idea.title}")

        if self.get_successful():
            lines.append("\nSuccessful ideas:")
            for idea in self.get_successful():
                lines.append(f"  ✓ {idea.title} (Δ={idea.metric_delta:.6f})")

        return "\n".join(lines)

    def _load(self) -> None:
        if self.pool_file.exists():
            with open(self.pool_file) as f:
                data = json.load(f)
            self.ideas = [Idea.from_dict(d) for d in data]

    def _save(self) -> None:
        self.pool_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.pool_file, "w") as f:
            json.dump([i.to_dict() for i in self.ideas], f, indent=2)
