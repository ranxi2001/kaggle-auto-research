"""Workspace manager: create and manage isolated competition workspaces."""

import shutil
from pathlib import Path

from .config import (
    CompetitionConfig,
    DataConfig,
    ModelConfig,
    PipelineConfig,
    SubmissionConfig,
    WorkspaceConfig,
    save_config,
)

PROJECT_ROOT = Path(__file__).parent.parent.parent
WORKSPACES_DIR = PROJECT_ROOT / "workspaces"
TEMPLATES_DIR = PROJECT_ROOT / "templates"


WORKSPACE_DIRS = [
    "data/raw",
    "data/processed",
    "data/features",
    "models",
    "submissions",
    "reports",
    "notebooks",
    "scripts",
    "logs/pipeline_runs",
    ".state",
]


def init_workspace(
    name: str,
    competition_type: str = "tabular",
    url: str = "",
    metric: str = "",
) -> Path:
    """Initialize a new competition workspace from template."""
    workspace_path = WORKSPACES_DIR / name

    if workspace_path.exists():
        raise FileExistsError(f"Workspace already exists: {workspace_path}")

    workspace_path.mkdir(parents=True)
    for d in WORKSPACE_DIRS:
        (workspace_path / d).mkdir(parents=True, exist_ok=True)

    template_dir = TEMPLATES_DIR / competition_type
    if template_dir.exists():
        _copy_template(template_dir, workspace_path)

    metric_defaults = {
        "tabular": ("rmse", "minimize"),
        "crypto": ("weighted_pearson", "maximize"),
        "llm": ("log_loss", "minimize"),
    }
    default_metric, default_direction = metric_defaults.get(
        competition_type, ("rmse", "minimize")
    )

    cv_defaults = {
        "tabular": "stratified_kfold",
        "crypto": "time_series_split",
        "llm": "group_kfold",
    }

    config = WorkspaceConfig(
        competition=CompetitionConfig(
            name=name,
            url=url,
            type=competition_type,
            metric=metric or default_metric,
            metric_direction=default_direction,
        ),
        data=DataConfig(),
        model=ModelConfig(cv_strategy=cv_defaults.get(competition_type, "kfold")),
        submission=SubmissionConfig(),
        pipeline=PipelineConfig(),
    )

    save_config(config, workspace_path)
    return workspace_path


def list_workspaces() -> list[Path]:
    """List all existing workspaces."""
    if not WORKSPACES_DIR.exists():
        return []
    return [p for p in WORKSPACES_DIR.iterdir() if p.is_dir() and (p / "config.yaml").exists()]


def get_workspace(name: str) -> Path:
    """Get workspace path by name, raise if not exists."""
    workspace_path = WORKSPACES_DIR / name
    if not workspace_path.exists():
        raise FileNotFoundError(f"Workspace not found: {name}")
    return workspace_path


def _copy_template(template_dir: Path, workspace_path: Path) -> None:
    """Copy template files into workspace."""
    for item in template_dir.rglob("*"):
        if item.is_file():
            rel_path = item.relative_to(template_dir)
            dest = workspace_path / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, dest)
