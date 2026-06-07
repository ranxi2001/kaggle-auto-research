"""Error analysis and iteration diagnostics."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


class IterationAnalyzer:
    """Analyze model results to guide next iteration decisions."""

    def __init__(self, workspace: Path):
        self.workspace = workspace

    def analyze_latest(self) -> dict:
        """Analyze the latest model's performance for iteration guidance."""
        models_dir = self.workspace / "models"
        model_dirs = sorted(models_dir.glob("v*"))
        if not model_dirs:
            return {"status": "no_models"}

        latest = model_dirs[-1]
        analysis = {"model_version": latest.name}

        # Load CV scores
        cv_file = latest / "cv_scores.json"
        if cv_file.exists():
            with open(cv_file) as f:
                scores = json.load(f)
            analysis["cv_mean"] = scores["mean_score"]
            analysis["cv_std"] = scores["std_score"]
            analysis["fold_scores"] = scores["fold_scores"]
            analysis["cv_stability"] = scores["std_score"] / max(scores["mean_score"], 1e-10)

        # Load feature importance
        imp_file = latest / "importance.csv"
        if imp_file.exists():
            imp = pd.read_csv(imp_file)
            analysis["top_features"] = imp.head(10)["feature"].tolist()
            analysis["zero_importance_features"] = imp[imp["importance"] == 0]["feature"].tolist()
            analysis["feature_concentration"] = (
                imp.head(5)["importance"].sum() / max(imp["importance"].sum(), 1)
            )

        # OOF error analysis
        oof_file = latest / "oof_preds.npy"
        if oof_file.exists():
            oof_preds = np.load(oof_file)
            analysis["oof_stats"] = {
                "mean": float(np.mean(oof_preds)),
                "std": float(np.std(oof_preds)),
                "min": float(np.min(oof_preds)),
                "max": float(np.max(oof_preds)),
            }

        return analysis

    def compare_models(self) -> list[dict]:
        """Compare all model versions."""
        models_dir = self.workspace / "models"
        comparisons = []

        for model_dir in sorted(models_dir.glob("v*")):
            cv_file = model_dir / "cv_scores.json"
            if not cv_file.exists():
                continue
            with open(cv_file) as f:
                scores = json.load(f)
            comparisons.append({
                "version": model_dir.name,
                "cv_mean": scores["mean_score"],
                "cv_std": scores["std_score"],
                "patch": scores.get("patch", "baseline"),
                "model_type": scores.get("model_type", "unknown"),
                "features": scores.get("features", []),
            })

        return sorted(comparisons, key=lambda x: x["cv_mean"])

    def get_recommendations(self) -> list[str]:
        """Generate recommendations for next iteration based on analysis."""
        analysis = self.analyze_latest()
        recs = []

        if analysis.get("cv_stability", 0) > 0.1:
            recs.append("High CV variance — consider more folds or simpler model")

        zero_feats = analysis.get("zero_importance_features", [])
        if len(zero_feats) > 5:
            recs.append(f"Remove {len(zero_feats)} zero-importance features to reduce noise")

        concentration = analysis.get("feature_concentration", 0)
        if concentration > 0.8:
            recs.append("Top 5 features dominate — add diverse feature types")

        comparisons = self.compare_models()
        if len(comparisons) >= 3:
            scores = [c["cv_mean"] for c in comparisons[-3:]]
            if max(scores) - min(scores) < 1e-6:
                recs.append("Last 3 models identical — try aggressive changes (model switch, new feature type)")

        if not recs:
            recs.append("Model performing well — try ensemble or submit")

        return recs
