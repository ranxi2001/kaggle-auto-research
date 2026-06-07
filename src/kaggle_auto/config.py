"""Competition configuration schema and loader."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml


@dataclass
class CompetitionConfig:
    name: str
    url: str = ""
    type: Literal["tabular", "crypto", "llm"] = "tabular"
    metric: str = "rmse"
    metric_direction: Literal["maximize", "minimize"] = "minimize"
    deadline: str = ""


@dataclass
class DataConfig:
    train: str = "data/raw/train.csv"
    test: str = "data/raw/test.csv"
    sample_submission: str = "data/raw/sample_submission.csv"
    target_column: str = "target"
    id_column: str = "id"


@dataclass
class ModelConfig:
    primary: str = "lightgbm"
    cv_strategy: str = "stratified_kfold"
    cv_folds: int = 5
    seed: int = 42


@dataclass
class SubmissionConfig:
    auto_submit: bool = False
    best_threshold: float = 0.01
    max_daily: int = 5


@dataclass
class PipelineConfig:
    stages: list[str] = field(
        default_factory=lambda: ["research", "eda", "features", "train", "evaluate", "submit"]
    )
    iteration_limit: int = 10


@dataclass
class WorkspaceConfig:
    competition: CompetitionConfig
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    submission: SubmissionConfig = field(default_factory=SubmissionConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)


def load_config(workspace_path: Path) -> WorkspaceConfig:
    """Load workspace config from config.yaml."""
    config_file = workspace_path / "config.yaml"
    if not config_file.exists():
        raise FileNotFoundError(f"Config not found: {config_file}")

    with open(config_file) as f:
        raw = yaml.safe_load(f)

    return WorkspaceConfig(
        competition=CompetitionConfig(**raw.get("competition", {})),
        data=DataConfig(**raw.get("data", {})),
        model=ModelConfig(**raw.get("model", {})),
        submission=SubmissionConfig(**raw.get("submission", {})),
        pipeline=PipelineConfig(**raw.get("pipeline", {})),
    )


def save_config(config: WorkspaceConfig, workspace_path: Path) -> None:
    """Save workspace config to config.yaml."""
    from dataclasses import asdict

    config_file = workspace_path / "config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(asdict(config), f, default_flow_style=False, allow_unicode=True)
