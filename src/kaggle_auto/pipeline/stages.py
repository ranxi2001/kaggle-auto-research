"""Pipeline stage implementations."""

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import WorkspaceConfig
from .runner import stage


@stage("research")
def run_research(workspace: Path, config: WorkspaceConfig) -> dict:
    """Run competition research stage."""
    from ..research import KaggleAPI, CompetitionScraper

    slug = config.competition.name
    scraper = CompetitionScraper(slug)

    notebooks = scraper.get_top_notebooks(limit=10)

    competition_info = {
        "title": config.competition.name,
        "category": config.competition.type,
        "evaluation_metric": config.competition.metric,
        "reward": "",
        "deadline": config.competition.deadline,
        "team_count": 0,
    }

    report_path = scraper.generate_research_report(
        competition_info=competition_info,
        notebooks=notebooks,
        output_path=workspace / "reports" / "research_notes.md",
    )

    return {
        "status": "completed",
        "report_path": str(report_path),
        "notebooks_found": len(notebooks),
    }


@stage("eda")
def run_eda(workspace: Path, config: WorkspaceConfig) -> dict:
    """Run EDA stage."""
    from ..eda import DataProfiler, EDAReport

    train_path = workspace / config.data.train
    if not train_path.exists():
        return {"status": "skipped", "reason": f"Train data not found: {train_path}"}

    if str(train_path).endswith(".parquet"):
        train_df = pd.read_parquet(train_path)
    else:
        train_df = pd.read_csv(train_path)

    test_path = workspace / config.data.test
    test_df = None
    if test_path.exists():
        if str(test_path).endswith(".parquet"):
            test_df = pd.read_parquet(test_path)
        else:
            test_df = pd.read_csv(test_path)

    report = EDAReport(workspace)
    report_path = report.generate(
        train_df=train_df,
        test_df=test_df,
        target_col=config.data.target_column,
    )

    return {
        "status": "completed",
        "report_path": str(report_path),
        "n_rows": len(train_df),
        "n_cols": len(train_df.columns),
    }


@stage("features")
def run_features(workspace: Path, config: WorkspaceConfig) -> dict:
    """Run feature engineering stage."""
    from ..features import build, list_features

    train_path = workspace / config.data.train
    if not train_path.exists():
        return {"status": "skipped", "reason": "Train data not found"}

    if str(train_path).endswith(".parquet"):
        train_df = pd.read_parquet(train_path)
    else:
        train_df = pd.read_csv(train_path)

    available = list_features()
    features_dir = workspace / "data" / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    type_features = {
        "tabular": ["basic_stats", "interactions", "target_encoding", "null_indicator"],
        "crypto": ["lag_features", "rolling_stats", "technical_indicators", "volatility"],
        "llm": ["text_stats", "token_count"],
    }

    to_build = type_features.get(config.competition.type, ["basic_stats"])
    to_build = [f for f in to_build if f in available]

    if not to_build:
        return {"status": "skipped", "reason": "No applicable features found"}

    result_df = build(to_build, train_df)
    new_cols = [c for c in result_df.columns if c not in train_df.columns]

    version = 1
    while (features_dir / f"v{version:03d}.parquet").exists():
        version += 1

    output_path = features_dir / f"v{version:03d}.parquet"
    result_df.to_parquet(output_path)

    return {
        "status": "completed",
        "features_built": to_build,
        "new_columns": len(new_cols),
        "version": f"v{version:03d}",
        "output_path": str(output_path),
    }


@stage("train")
def run_train(workspace: Path, config: WorkspaceConfig) -> dict:
    """Run model training stage."""
    from ..models import LightGBMModel, CrossValidator

    features_dir = workspace / "data" / "features"
    feature_files = sorted(features_dir.glob("v*.parquet"))

    if feature_files:
        df = pd.read_parquet(feature_files[-1])
    else:
        train_path = workspace / config.data.train
        if not train_path.exists():
            return {"status": "skipped", "reason": "No data available"}
        if str(train_path).endswith(".parquet"):
            df = pd.read_parquet(train_path)
        else:
            df = pd.read_csv(train_path)

    target_col = config.data.target_column
    if target_col not in df.columns:
        return {"status": "skipped", "reason": f"Target column '{target_col}' not found"}

    y = df[target_col].values
    X = df.drop(columns=[target_col, config.data.id_column], errors="ignore")
    X = X.select_dtypes(include=[np.number])

    if X.empty:
        return {"status": "skipped", "reason": "No numeric features available"}

    task = "classification" if df[target_col].nunique() <= 20 else "regression"

    cv = CrossValidator(
        strategy=config.model.cv_strategy,
        n_splits=config.model.cv_folds,
        seed=config.model.seed,
    )

    result = cv.run(
        model_cls=LightGBMModel,
        X=X, y=y,
        task=task,
    )

    models_dir = workspace / "models"
    version = 1
    while (models_dir / f"v{version:03d}").exists():
        version += 1
    model_path = models_dir / f"v{version:03d}"

    result["models"][0].save(model_path)
    np.save(model_path / "oof_preds.npy", result["oof_preds"])

    import json
    with open(model_path / "cv_scores.json", "w") as f:
        json.dump({
            "fold_scores": result["fold_scores"],
            "mean_score": result["mean_score"],
            "std_score": result["std_score"],
        }, f, indent=2)

    importance = result["models"][0].get_feature_importance()
    importance.to_csv(model_path / "importance.csv", index=False)

    # Generate test predictions using all fold models (average)
    test_path = workspace / config.data.test
    if test_path.exists():
        if str(test_path).endswith(".parquet"):
            test_df = pd.read_parquet(test_path)
        else:
            test_df = pd.read_csv(test_path)

        test_X = test_df.drop(columns=[config.data.id_column], errors="ignore")
        test_X = test_X.select_dtypes(include=[np.number])
        # Align test columns with train columns
        missing_cols = [c for c in X.columns if c not in test_X.columns]
        for c in missing_cols:
            test_X[c] = 0
        test_X = test_X[X.columns]

        # Average predictions from all fold models
        preds = np.zeros(len(test_X))
        for model in result["models"]:
            preds += model.predict(test_X)
        preds /= len(result["models"])

        np.save(model_path / "test_preds.npy", preds)

    return {
        "status": "completed",
        "cv_mean": result["mean_score"],
        "cv_std": result["std_score"],
        "fold_scores": result["fold_scores"],
        "model_version": f"v{version:03d}",
        "model_path": str(model_path),
        "task": task,
    }


@stage("evaluate")
def run_evaluate(workspace: Path, config: WorkspaceConfig) -> dict:
    """Evaluate latest model and summarize results."""
    models_dir = workspace / "models"
    model_dirs = sorted(models_dir.glob("v*"))

    if not model_dirs:
        return {"status": "skipped", "reason": "No trained models found"}

    latest = model_dirs[-1]
    cv_file = latest / "cv_scores.json"

    if not cv_file.exists():
        return {"status": "skipped", "reason": "No CV scores found"}

    import json
    with open(cv_file) as f:
        scores = json.load(f)

    return {
        "status": "completed",
        "model_version": latest.name,
        "cv_mean": scores["mean_score"],
        "cv_std": scores["std_score"],
        "fold_scores": scores["fold_scores"],
    }


@stage("submit")
def run_submit(workspace: Path, config: WorkspaceConfig) -> dict:
    """Generate and submit predictions."""
    from ..submission import Submitter

    models_dir = workspace / "models"
    model_dirs = sorted(models_dir.glob("v*"))

    if not model_dirs:
        return {"status": "skipped", "reason": "No trained models found"}

    latest = model_dirs[-1]
    preds_path = latest / "test_preds.npy"

    if not preds_path.exists():
        return {"status": "skipped", "reason": "No test predictions found"}

    preds = np.load(preds_path)

    # For classification tasks, convert probabilities to class labels
    import json
    cv_file = latest / "cv_scores.json"
    task = "regression"
    if cv_file.exists():
        with open(cv_file) as f:
            scores = json.load(f)

    # Detect task type from target
    train_path = workspace / config.data.train
    if train_path.exists():
        if str(train_path).endswith(".parquet"):
            train_df = pd.read_parquet(train_path, columns=[config.data.target_column])
        else:
            train_df = pd.read_csv(train_path, usecols=[config.data.target_column])
        if train_df[config.data.target_column].nunique() <= 20:
            task = "classification"
            preds = (preds >= 0.5).astype(int)

    submitter = Submitter(workspace, config)
    model_version = latest.name
    sub_path = submitter.generate_submission(preds, model_version=model_version)

    validation = submitter.validate(sub_path)
    if not validation.is_valid:
        return {
            "status": "failed",
            "errors": validation.errors,
            "submission_path": str(sub_path),
        }

    if not config.submission.auto_submit:
        return {
            "status": "completed",
            "action": "generated",
            "submission_path": str(sub_path),
            "message": "Submission file generated. Set auto_submit: true to submit automatically.",
        }

    # Actually submit to Kaggle
    cv_score = scores.get("mean_score") if cv_file.exists() else None
    result = submitter.submit(
        sub_path,
        message=f"Pipeline auto-submit {model_version}",
        model_version=model_version,
        cv_score=cv_score,
    )

    return {
        "status": "completed" if result["success"] else "failed",
        "submission_path": str(sub_path),
        "submit_result": result,
    }


def register_all_stages():
    """Ensure all stages are registered (import side-effect)."""
    pass
