"""Path resolution helpers."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
WORKSPACES_DIR = PROJECT_ROOT / "workspaces"


def get_workspace_path(name: str) -> Path:
    """Get workspace path by competition name."""
    path = WORKSPACES_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Workspace not found: {name}")
    return path


def get_latest_model(workspace: Path) -> Path | None:
    """Get latest model directory in workspace."""
    models_dir = workspace / "models"
    if not models_dir.exists():
        return None
    versions = sorted(models_dir.glob("v*"))
    return versions[-1] if versions else None


def get_latest_features(workspace: Path) -> Path | None:
    """Get latest feature file in workspace."""
    features_dir = workspace / "data" / "features"
    if not features_dir.exists():
        return None
    versions = sorted(features_dir.glob("v*.parquet"))
    return versions[-1] if versions else None
