"""Build ensemble from best iteration models."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, mean_squared_error

from ..models.ensemble import Ensemble


class EnsembleBuilder:
    """Build an ensemble from the top models found during iteration."""

    def __init__(self, workspace: Path):
        self.workspace = Path(workspace).resolve()
        self.models_dir = self.workspace / "models"

    def find_top_models(self, n: int = 3, minimize: bool = True) -> list[dict]:
        """Find the top N models by CV score."""
        candidates = []

        for model_dir in sorted(self.models_dir.glob("v*")):
            cv_file = model_dir / "cv_scores.json"
            oof_file = model_dir / "oof_preds.npy"
            if not cv_file.exists():
                continue

            with open(cv_file) as f:
                scores = json.load(f)

            candidates.append({
                "version": model_dir.name,
                "path": model_dir,
                "cv_mean": scores["mean_score"],
                "cv_std": scores["std_score"],
                "model_type": scores.get("model_type", "lightgbm"),
                "has_oof": oof_file.exists(),
            })

        candidates.sort(key=lambda x: x["cv_mean"], reverse=not minimize)
        return candidates[:n]

    def build_ensemble(
        self,
        target: np.ndarray,
        top_n: int = 3,
        minimize: bool = True,
    ) -> dict:
        """Build an optimized ensemble from top models.

        Returns dict with ensemble predictions, weights, and score.
        """
        top_models = self.find_top_models(n=top_n, minimize=minimize)

        # Collect OOF predictions
        oof_list = []
        valid_models = []
        for model_info in top_models:
            oof_file = model_info["path"] / "oof_preds.npy"
            if not oof_file.exists():
                continue
            oof = np.load(oof_file)
            if len(oof) == len(target):
                oof_list.append(oof)
                valid_models.append(model_info)

        if len(oof_list) < 2:
            return {"status": "skipped", "reason": "Need at least 2 models with OOF predictions"}

        # Detect task
        is_classification = len(np.unique(target)) <= 20

        # Optimize weights
        if is_classification:
            metric_fn = log_loss
        else:
            metric_fn = lambda y, p: mean_squared_error(y, p, squared=False)

        ensemble = Ensemble()
        weights = ensemble.optimize_weights(
            oof_predictions=oof_list,
            y_true=target,
            metric_fn=metric_fn,
            minimize_metric=True,
        )

        # Calculate ensemble score
        blended = ensemble.blend_oof(oof_list)
        ensemble_score = metric_fn(target, blended)

        # Compare with best single model
        best_single = min(m["cv_mean"] for m in valid_models) if minimize else max(m["cv_mean"] for m in valid_models)

        # Save ensemble
        ensemble_dir = self.models_dir / "ensemble"
        ensemble_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "models": [
                {"version": m["version"], "weight": w, "cv_mean": m["cv_mean"]}
                for m, w in zip(valid_models, weights)
            ],
            "ensemble_score": ensemble_score,
            "best_single_score": best_single,
            "improvement": best_single - ensemble_score if minimize else ensemble_score - best_single,
        }

        with open(ensemble_dir / "ensemble_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        np.save(ensemble_dir / "ensemble_oof.npy", blended)

        # Generate ensemble test predictions if available
        test_preds_list = []
        for model_info in valid_models:
            test_file = model_info["path"] / "test_preds.npy"
            if test_file.exists():
                test_preds_list.append(np.load(test_file))

        if len(test_preds_list) == len(valid_models):
            ensemble_test = sum(w * p for w, p in zip(weights, test_preds_list))
            np.save(ensemble_dir / "test_preds.npy", ensemble_test)
            meta["has_test_preds"] = True
        else:
            meta["has_test_preds"] = False

        return {
            "status": "completed",
            "ensemble_score": ensemble_score,
            "best_single_score": best_single,
            "improvement": meta["improvement"],
            "weights": weights,
            "models_used": [m["version"] for m in valid_models],
        }
