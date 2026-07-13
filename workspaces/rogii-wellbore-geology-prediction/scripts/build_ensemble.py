#!/usr/bin/env python3
"""Build a versioned OOF-selected ensemble from two ROGII model runs."""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd


def rmse(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(target - prediction))))


def next_model_dir(workspace: Path) -> Path:
    root = workspace / "models"
    versions = [int(path.name[1:]) for path in root.glob("v[0-9][0-9][0-9]")]
    path = root / f"v{max(versions, default=0) + 1:03d}"
    path.mkdir()
    return path


def git_commit(project_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def load_oof(model_dir: Path, label: str) -> pd.DataFrame:
    rows = pd.read_parquet(model_dir / "oof_rows.parquet")
    prediction = np.load(model_dir / "oof_preds.npy")
    if len(rows) != len(prediction):
        raise ValueError(f"{label} OOF length mismatch")
    return rows.assign(**{label: prediction})


def build(workspace: Path, left_version: str, right_version: str, right_weight: float) -> Path:
    started = time.time()
    left_dir = workspace / "models" / left_version
    right_dir = workspace / "models" / right_version
    model_dir = next_model_dir(workspace)
    run_path = model_dir / "run.json"
    run = {
        "version": model_dir.name,
        "status": "running",
        "parent_run": right_version,
        "template": "rogii-oof-linear-ensemble",
        "command": (
            f"uv run python scripts/build_ensemble.py {left_version} {right_version} "
            f"--right-weight {right_weight}"
        ),
        "git_commit": git_commit(workspace.parents[1]),
        "params": {
            "source_model_versions": [left_version, right_version],
            "weights": [1.0 - right_weight, right_weight],
        },
        "random_seed": 42,
        "cv_splitter": "inherited well GroupKFold OOF",
        "metric": "rmse",
        "metric_direction": "minimize",
        "target_column": "TVT",
        "id_column": "id",
        "data_fingerprint": "",
        "runtime_seconds": None,
        "error": None,
    }
    run_path.write_text(json.dumps(run, indent=2), encoding="utf-8")
    try:
        left = load_oof(left_dir, left_version)
        right = load_oof(right_dir, right_version)
        columns = ["_id", "_target", right_version]
        if "fold" in right:
            columns.append("fold")
        merged = left.merge(
            right[columns],
            on="_id",
            how="inner",
            validate="one_to_one",
            suffixes=("", "_right"),
        )
        if len(merged) != len(left) or len(merged) != len(right):
            raise ValueError("Source OOF ID sets do not match")
        if not np.allclose(merged["_target"], merged["_target_right"]):
            raise ValueError("Source OOF targets do not match")
        prediction = (
            (1.0 - right_weight) * merged[left_version].to_numpy(dtype=float)
            + right_weight * merged[right_version].to_numpy(dtype=float)
        )
        target = merged["_target"].to_numpy(dtype=float)
        np.save(model_dir / "oof_preds.npy", prediction)
        row_columns = ["_id", "_well_id", "_row_index", "_target"]
        if "fold" in merged:
            row_columns.append("fold")
        merged[row_columns].to_parquet(model_dir / "oof_rows.parquet", index=False)

        if "fold" in merged:
            fold_scores = [
                rmse(
                    target[merged["fold"].to_numpy() == fold],
                    prediction[merged["fold"].to_numpy() == fold],
                )
                for fold in sorted(merged["fold"].unique())
            ]
        else:
            fold_scores = []
        scores = {
            "metric": "rmse",
            "direction": "minimize",
            "fold_scores": fold_scores,
            "mean": float(np.mean(fold_scores)) if fold_scores else None,
            "std": float(np.std(fold_scores)) if fold_scores else None,
            "overall": rmse(target, prediction),
            "valid_rows": len(merged),
            "valid_wells": int(merged["_well_id"].nunique()),
        }
        (model_dir / "cv_scores.json").write_text(
            json.dumps(scores, indent=2), encoding="utf-8"
        )
        left_features = (left_dir / "feature_list.txt").read_text(encoding="utf-8").splitlines()
        right_features = (right_dir / "feature_list.txt").read_text(encoding="utf-8").splitlines()
        features = sorted(set(left_features + right_features))
        (model_dir / "feature_list.txt").write_text(
            "\n".join(features) + "\n", encoding="utf-8"
        )
        right_importance = pd.read_csv(right_dir / "importance.csv")
        if not right_importance.empty:
            right_importance["importance"] *= right_weight
        right_importance.to_csv(model_dir / "importance.csv", index=False)
        model = {
            "model": "linear_ensemble",
            "source_model_versions": [left_version, right_version],
            "weights": [1.0 - right_weight, right_weight],
        }
        with (model_dir / "model.pkl").open("wb") as handle:
            pickle.dump(model, handle)

        left_test = np.load(left_dir / "test_preds.npy")
        right_test = np.load(right_dir / "test_preds.npy")
        if len(left_test) != len(right_test):
            raise ValueError("Source test prediction lengths do not match")
        test_prediction = (1.0 - right_weight) * left_test + right_weight * right_test
        np.save(model_dir / "test_preds.npy", test_prediction)
        sample = pd.read_csv(workspace / "data" / "raw" / "sample_submission.csv")
        if len(sample) != len(test_prediction) or list(sample.columns) != ["id", "tvt"]:
            raise ValueError("Sample submission does not match ensemble test predictions")
        submission = sample.copy()
        submission["tvt"] = test_prediction
        if not np.isfinite(test_prediction).all():
            raise ValueError("Ensemble test predictions contain non-finite values")
        submission_path = workspace / "submissions" / f"sub_{model_dir.name}_ensemble.csv"
        if submission_path.exists():
            raise FileExistsError(f"Refusing to overwrite {submission_path}")
        submission.to_csv(submission_path, index=False)
        metadata = {
            "source_model_versions": [left_version, right_version],
            "ensemble_weights": [1.0 - right_weight, right_weight],
            "local_cv_score": scores["overall"],
            "fold_mean_score": scores["mean"],
            "metric": "rmse",
            "generated_command": run["command"],
            "generated_at": pd.Timestamp.now("UTC").isoformat(),
            "dry_run": {
                "valid": True,
                "checks": ["sample ID order", "finite predictions", "row count"],
            },
            "kaggle_submission_id": None,
            "lb_score": None,
            "rank": None,
            "submitted": False,
            "submission_mode": "notebook",
        }
        submission_path.with_suffix(".json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        right_run = json.loads((right_dir / "run.json").read_text(encoding="utf-8"))
        run.update(
            {
                "status": "completed",
                "data_fingerprint": right_run.get("data_fingerprint", ""),
                "runtime_seconds": round(time.time() - started, 3),
            }
        )
    except Exception as exc:
        run.update(
            {
                "status": "failed",
                "runtime_seconds": round(time.time() - started, 3),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        run_path.write_text(json.dumps(run, indent=2), encoding="utf-8")
        raise
    run_path.write_text(json.dumps(run, indent=2), encoding="utf-8")
    return model_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left_version")
    parser.add_argument("right_version")
    parser.add_argument("--right-weight", type=float, required=True)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()
    print(
        build(
            args.workspace.resolve(),
            args.left_version,
            args.right_version,
            args.right_weight,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
