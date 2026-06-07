"""Pipeline state persistence and iteration tracking."""

import json
from datetime import datetime
from pathlib import Path


class PipelineState:
    """Manage pipeline execution state for checkpoint/resume."""

    def __init__(self, workspace_path: Path):
        self.workspace = workspace_path
        self.state_dir = workspace_path / ".state"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.state_dir / "pipeline_state.json"
        self.iteration_file = self.state_dir / "iteration_history.json"

    def save_stage_result(self, stage: str, result: dict) -> None:
        """Record completion of a pipeline stage."""
        state = self._load_state()
        state["last_stage"] = stage
        state["last_updated"] = datetime.now().isoformat()
        state["stages_completed"] = state.get("stages_completed", [])
        if stage not in state["stages_completed"]:
            state["stages_completed"].append(stage)
        state[f"stage_{stage}"] = result
        self._save_state(state)

    def get_last_stage(self) -> str | None:
        """Get the last completed stage."""
        state = self._load_state()
        return state.get("last_stage")

    def get_completed_stages(self) -> list[str]:
        """Get all completed stages."""
        state = self._load_state()
        return state.get("stages_completed", [])

    def is_stage_done(self, stage: str) -> bool:
        return stage in self.get_completed_stages()

    def reset(self) -> None:
        """Reset pipeline state (start fresh)."""
        self._save_state({})

    def record_iteration(
        self,
        round_num: int,
        action: str,
        cv_score: float | None = None,
        lb_score: float | None = None,
        delta_cv: float | None = None,
        notes: str = "",
        node_id: str = "",
        minimize: bool = True,
    ) -> None:
        """Record an iteration in the improvement loop."""
        history = self._load_iterations()

        entry = {
            "round": round_num,
            "action": action,
            "cv_score": cv_score,
            "lb_score": lb_score,
            "delta_cv": delta_cv,
            "notes": notes,
            "node_id": node_id,
            "timestamp": datetime.now().isoformat(),
        }

        history["iterations"].append(entry)

        if cv_score is not None:
            is_better = (
                history["best_cv"] is None
                or (minimize and cv_score < history["best_cv"])
                or (not minimize and cv_score > history["best_cv"])
            )
            if is_better:
                history["best_cv"] = cv_score
                history["stale_rounds"] = 0
            else:
                history["stale_rounds"] = history.get("stale_rounds", 0) + 1

        if lb_score is not None:
            if history["best_lb"] is None or lb_score < history["best_lb"]:
                history["best_lb"] = lb_score

        self._save_iterations(history)

    def get_iterations(self) -> dict:
        """Get iteration history."""
        return self._load_iterations()

    def get_stale_count(self) -> int:
        """Get number of iterations without improvement."""
        history = self._load_iterations()
        return history.get("stale_rounds", 0)

    def _load_state(self) -> dict:
        if self.state_file.exists():
            with open(self.state_file) as f:
                return json.load(f)
        return {}

    def _save_state(self, state: dict) -> None:
        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2)

    def _load_iterations(self) -> dict:
        if self.iteration_file.exists():
            with open(self.iteration_file) as f:
                return json.load(f)
        return {
            "iterations": [],
            "best_cv": None,
            "best_lb": None,
            "stale_rounds": 0,
        }

    def _save_iterations(self, history: dict) -> None:
        with open(self.iteration_file, "w") as f:
            json.dump(history, f, indent=2)
