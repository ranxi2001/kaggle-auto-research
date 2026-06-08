"""Pipeline stage implementations."""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ..config import WorkspaceConfig
from .runner import stage


@stage("research")
def run_research(workspace: Path, config: WorkspaceConfig) -> dict:
    """Run competition research stage."""
    from ..research import KaggleAPI, CompetitionScraper

    slug = config.competition.name
    scraper = CompetitionScraper(slug)

    notebooks = scraper.get_top_notebooks(limit=10)
    report_path = workspace / "reports" / "research_notes.md"

    if not notebooks and report_path.exists() and report_path.stat().st_size > 0:
        return {
            "status": "completed",
            "report_path": str(report_path),
            "notebooks_found": 0,
            "message": "Kept existing research report because no notebooks were found.",
        }

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
        output_path=report_path,
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

    max_rows = 100_000
    train_path = workspace / config.data.train
    if not train_path.exists():
        return {"status": "skipped", "reason": f"Train data not found: {train_path}"}

    if str(train_path).endswith(".parquet"):
        train_df, train_total_rows = _read_table_sample(train_path, max_rows=max_rows)
    else:
        train_df = pd.read_csv(train_path, nrows=max_rows)
        train_total_rows = None

    test_path = workspace / config.data.test
    test_df = None
    test_total_rows = None
    if test_path.exists():
        if str(test_path).endswith(".parquet"):
            test_df, test_total_rows = _read_table_sample(test_path, max_rows=max_rows)
        else:
            test_df = pd.read_csv(test_path, nrows=max_rows)

    report = EDAReport(workspace)
    report_path = report.generate(
        train_df=train_df,
        test_df=test_df,
        target_col=config.data.target_column,
    )
    _append_sample_note(report_path, max_rows, train_total_rows, test_total_rows)

    return {
        "status": "completed",
        "report_path": str(report_path),
        "n_rows": train_total_rows or len(train_df),
        "n_cols": len(train_df.columns),
        "sample_rows": len(train_df),
    }


def _read_table_sample(path: Path, max_rows: int) -> tuple[pd.DataFrame, int]:
    parquet_file = pq.ParquetFile(path)
    total_rows = parquet_file.metadata.num_rows
    if total_rows <= max_rows:
        return pd.read_parquet(path), total_rows

    batches = []
    rows_read = 0
    for batch in parquet_file.iter_batches(batch_size=min(max_rows, 25_000)):
        batches.append(batch.to_pandas())
        rows_read += batch.num_rows
        if rows_read >= max_rows:
            break
    df = pd.concat(batches, ignore_index=True).head(max_rows)
    return df, total_rows


def _append_sample_note(
    report_path: Path,
    max_rows: int,
    train_total_rows: int | None,
    test_total_rows: int | None,
) -> None:
    notes = [
        "",
        "## Profiling Scope",
        "",
        f"- EDA sampled at most {max_rows:,} rows per split for local speed and memory safety.",
    ]
    if train_total_rows is not None:
        notes.append(f"- Train total rows: {train_total_rows:,}")
    if test_total_rows is not None:
        notes.append(f"- Test total rows: {test_total_rows:,}")
    with open(report_path, "a") as f:
        f.write("\n".join(notes))
        f.write("\n")


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
        "titanic": ["titanic_custom"],
    }

    to_build = type_features.get(config.competition.type, ["basic_stats"])
    to_build = [f for f in to_build if f in available]

    if not to_build:
        return {"status": "skipped", "reason": "No applicable features found"}

    # Build features WITHOUT target to prevent leakage
    target_col = config.data.target_column
    build_df = train_df.drop(columns=[target_col], errors="ignore")
    result_df = build(to_build, build_df)
    # Re-attach target for downstream train stage
    if target_col in train_df.columns:
        result_df[target_col] = train_df[target_col].values
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

        # Apply same feature engineering to test data
        from ..features import build, list_features
        available = list_features()
        type_features = {
            "tabular": ["basic_stats", "interactions", "null_indicator"],
            "crypto": ["lag_features", "rolling_stats", "technical_indicators", "volatility"],
            "llm": ["text_stats", "token_count"],
            "titanic": ["titanic_custom"],
        }
        feat_names = type_features.get(config.competition.type, ["basic_stats"])
        feat_names = [f for f in feat_names if f in available]

        test_feat_df = test_df.drop(columns=[config.data.id_column], errors="ignore")
        if feat_names:
            try:
                test_feat_df = build(feat_names, test_feat_df)
            except Exception:
                pass
        test_X = test_feat_df.select_dtypes(include=[np.number])

        # Align test columns with train columns
        missing_cols = [c for c in X.columns if c not in test_X.columns]
        for c in missing_cols:
            test_X[c] = 0
        extra_cols = [c for c in test_X.columns if c not in X.columns]
        test_X = test_X.drop(columns=extra_cols, errors="ignore")
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
    """Generate submission file and optionally submit (budget-protected)."""
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
    scores = {}
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

    # Report budget status
    budget_status = submitter.status()
    cv_score = scores.get("mean_score")

    if not config.submission.auto_submit:
        return {
            "status": "completed",
            "action": "generated_only",
            "submission_path": str(sub_path),
            "cv_score": cv_score,
            "budget": budget_status,
            "message": (
                f"Submission file generated (CV={cv_score:.4f}). "
                f"Budget: {budget_status['remaining_today']}/{budget_status['max_daily']} remaining today. "
                f"Use `kar submit <name>` to submit manually."
            ),
        }

    # Auto-submit (budget-protected — will queue if exhausted)
    result = submitter.submit(
        sub_path,
        message=f"Pipeline auto-submit {model_version} CV={cv_score:.4f}" if cv_score else f"Pipeline auto-submit {model_version}",
        model_version=model_version,
        cv_score=cv_score,
    )

    return {
        "status": "completed" if result.get("success") else ("queued" if result.get("queued") else "failed"),
        "submission_path": str(sub_path),
        "submit_result": result,
        "budget": budget_status,
    }


def register_all_stages():
    """Ensure all stages are registered (import side-effect)."""
    pass
