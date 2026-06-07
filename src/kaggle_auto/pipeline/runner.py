"""Pipeline runner: orchestrate competition stages with tree-search iteration."""

import json
from pathlib import Path
from typing import Callable

from ..config import WorkspaceConfig, load_config
from .state import PipelineState
from .journal import Journal


StageFunc = Callable[[Path, WorkspaceConfig], dict]

_STAGES: dict[str, StageFunc] = {}


def stage(name: str):
    """Decorator to register a pipeline stage."""
    def decorator(func: StageFunc) -> StageFunc:
        _STAGES[name] = func
        return func
    return decorator


class PipelineRunner:
    """Run competition pipeline with checkpoint/resume and tree-search iteration."""

    def __init__(self, workspace_path: Path):
        self.workspace = workspace_path
        self.config = load_config(workspace_path)
        self.state = PipelineState(workspace_path)
        self.journal = Journal(workspace_path)

    def run(self, from_stage: str | None = None, full: bool = False) -> dict:
        """Run pipeline from a given stage (or from beginning)."""
        if full:
            self.state.reset()

        stages = self.config.pipeline.stages
        if from_stage:
            if from_stage not in stages:
                return {"error": f"Unknown stage: {from_stage}"}
            idx = stages.index(from_stage)
            stages = stages[idx:]
        else:
            completed = self.state.get_completed_stages()
            stages = [s for s in stages if s not in completed]

        if not stages:
            return {"status": "all_stages_completed"}

        results = {}
        for stage_name in stages:
            if stage_name not in _STAGES:
                results[stage_name] = {"status": "skipped", "reason": "not implemented"}
                continue

            result = _STAGES[stage_name](self.workspace, self.config)
            results[stage_name] = result
            self.state.save_stage_result(stage_name, result)

            if result.get("status") == "completed":
                self.journal.add_node(
                    plan=f"Stage: {stage_name}",
                    stage="draft",
                )

        return results

    def iterate(self, n_iterations: int = 5) -> dict:
        """Run improvement iterations using tree-search strategy.

        Each iteration:
        1. Plan — select a strategy (feature/hparam/model variant)
        2. Execute — run training with the variant applied
        3. Evaluate — record metric, update tree
        4. Search — pick next node to expand based on metrics
        """
        from .strategies import IterationPlanner
        from .idea_pool import IdeaPool
        from .analyzer import IterationAnalyzer
        from ..features import list_features

        available_features = list_features()
        planner = IterationPlanner(
            available_features=available_features,
            competition_type=self.config.competition.type,
        )

        # Initialize idea pool and seed it
        pool = IdeaPool(self.workspace)
        research_path = self.workspace / "reports" / "research_notes.md"
        pool.seed_from_research(research_path)

        analyzer = IterationAnalyzer(self.workspace)
        analysis = analyzer.analyze_latest()
        if analysis.get("cv_mean"):
            pool.seed_from_analysis(analysis)

        minimize = self.config.competition.metric_direction == "minimize"
        results = []

        # Determine current baseline
        type_features = {
            "tabular": ["basic_stats", "interactions", "target_encoding", "null_indicator"],
            "crypto": ["lag_features", "rolling_stats", "technical_indicators", "volatility"],
            "llm": ["text_stats", "token_count"],
        }
        current_features = type_features.get(self.config.competition.type, ["basic_stats"])
        current_features = [f for f in current_features if f in available_features]
        current_model = self.config.model.primary

        for i in range(n_iterations):
            # Plan candidates
            patches = planner.plan_next(
                current_features=current_features,
                current_model=current_model,
                iteration=i + 1,
                best_score=self.journal.get_best_node(minimize=minimize).metric_value
                if self.journal.get_best_node(minimize=minimize) else None,
                stale_rounds=self.state.get_stale_count(),
            )

            if not patches:
                results.append({"iteration": i + 1, "status": "no_patches"})
                continue

            # Pick one patch (tree search: expand best node's most promising child)
            patch = patches[0]

            # Determine parent node
            parent_node = self.journal.search_policy()
            stage = "draft" if parent_node is None else (
                "debug" if parent_node and parent_node.is_buggy else "improve"
            )

            node = self.journal.add_node(
                plan=f"[{patch.name}] {patch.description}",
                parent_id=parent_node.id if parent_node else None,
                stage=stage,
            )

            # Execute with patch
            train_result = self._run_variant(patch)

            if train_result.get("status") == "completed":
                cv_score = train_result.get("cv_mean")
                self.journal.update_node(
                    node.id,
                    metric_value=cv_score,
                    model_version=train_result.get("model_version", ""),
                    features_version=train_result.get("features_version", ""),
                )
                self.state.record_iteration(
                    round_num=i + 1,
                    action=patch.description,
                    cv_score=cv_score,
                    node_id=node.id,
                    minimize=minimize,
                )

                # Update current state if this is the new best
                best = self.journal.get_best_node(minimize=minimize)
                if best and best.id == node.id:
                    if patch.feature_names:
                        current_features = patch.feature_names
                    if patch.model_type:
                        current_model = patch.model_type
            else:
                self.journal.update_node(
                    node.id,
                    is_buggy=True,
                    error_message=train_result.get("reason", "Unknown error"),
                )

            results.append({
                "iteration": i + 1,
                "node_id": node.id,
                "patch": patch.name,
                "description": patch.description,
                "stage": stage,
                "cv_score": train_result.get("cv_mean"),
                "status": train_result.get("status"),
            })

            if self.state.get_stale_count() >= 3:
                results.append({"status": "early_stop", "reason": "3 rounds without improvement"})
                break

        best_node = self.journal.get_best_node(minimize=minimize)
        return {
            "iterations_run": len([r for r in results if "iteration" in r]),
            "best_score": best_node.metric_value if best_node else None,
            "best_model": best_node.model_version if best_node else None,
            "results": results,
            "tree_summary": self.journal.get_tree_summary(),
            "idea_pool_summary": pool.summary(),
            "recommendations": analyzer.get_recommendations(),
        }

    def _run_variant(self, patch) -> dict:
        """Run a single experiment variant (feature/model/hyperparam change)."""
        import numpy as np
        import pandas as pd
        from ..features import build, list_features

        # Load data
        features_dir = self.workspace / "data" / "features"
        train_path = self.workspace / self.config.data.train

        if not train_path.exists():
            return {"status": "skipped", "reason": "No train data"}

        if str(train_path).endswith(".parquet"):
            raw_df = pd.read_parquet(train_path)
        else:
            raw_df = pd.read_csv(train_path)

        target_col = self.config.data.target_column
        if target_col not in raw_df.columns:
            return {"status": "skipped", "reason": f"Target '{target_col}' not found"}

        # Build features based on patch
        feature_names = patch.feature_names
        if feature_names is None:
            type_features = {
                "tabular": ["basic_stats", "interactions", "target_encoding", "null_indicator"],
                "crypto": ["lag_features", "rolling_stats", "technical_indicators", "volatility"],
                "llm": ["text_stats", "token_count"],
            }
            feature_names = type_features.get(self.config.competition.type, ["basic_stats"])

        available = list_features()
        feature_names = [f for f in feature_names if f in available]

        try:
            df = build(feature_names, raw_df) if feature_names else raw_df
        except Exception as e:
            return {"status": "failed", "reason": f"Feature build error: {e}"}

        y = df[target_col].values
        X = df.drop(columns=[target_col, self.config.data.id_column], errors="ignore")
        X = X.select_dtypes(include=[np.number])

        if X.empty:
            return {"status": "skipped", "reason": "No numeric features"}

        task = "classification" if raw_df[target_col].nunique() <= 20 else "regression"

        # Select model
        model_type = patch.model_type or self.config.model.primary
        if model_type == "xgboost":
            from ..models import XGBoostModel as ModelCls
        else:
            from ..models import LightGBMModel as ModelCls

        # Run CV
        from ..models import CrossValidator
        cv = CrossValidator(
            strategy=self.config.model.cv_strategy,
            n_splits=self.config.model.cv_folds,
            seed=self.config.model.seed,
        )

        try:
            result = cv.run(
                model_cls=ModelCls,
                X=X, y=y,
                task=task,
                params=patch.model_params or None,
            )
        except Exception as e:
            return {"status": "failed", "reason": f"Training error: {e}"}

        # Save model
        models_dir = self.workspace / "models"
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
                "patch": patch.name,
                "features": feature_names,
                "model_type": model_type,
                "params": patch.model_params,
            }, f, indent=2)

        return {
            "status": "completed",
            "cv_mean": result["mean_score"],
            "cv_std": result["std_score"],
            "fold_scores": result["fold_scores"],
            "model_version": f"v{version:03d}",
            "features_version": ",".join(feature_names),
            "model_type": model_type,
        }

    def get_status(self) -> dict:
        """Get current pipeline status."""
        return {
            "completed_stages": self.state.get_completed_stages(),
            "last_stage": self.state.get_last_stage(),
            "iterations": self.state.get_iterations(),
            "best_node": self.journal.get_best_node(),
            "total_experiments": len(self.journal.nodes),
            "tree_summary": self.journal.get_tree_summary(),
        }
