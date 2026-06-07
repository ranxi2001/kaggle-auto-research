"""Iteration strategies: generate concrete experiment variants for tree-search."""

import random
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExperimentPatch:
    """A concrete change to apply in one iteration."""

    name: str
    description: str
    feature_names: list[str] | None = None
    model_type: str | None = None
    model_params: dict[str, Any] = field(default_factory=dict)
    preprocessing: dict[str, Any] = field(default_factory=dict)


class FeatureExploration:
    """Generate feature subset variants."""

    def __init__(self, available_features: list[str], competition_type: str = "tabular"):
        self.available = available_features
        self.comp_type = competition_type

    def generate_variants(self, current_features: list[str], n: int = 3) -> list[ExperimentPatch]:
        patches = []

        # Variant 1: Add features not yet used
        unused = [f for f in self.available if f not in current_features]
        if unused:
            for feat in unused[:n]:
                patches.append(ExperimentPatch(
                    name=f"add_{feat}",
                    description=f"Add feature generator: {feat}",
                    feature_names=current_features + [feat],
                ))

        # Variant 2: Remove one feature (ablation)
        if len(current_features) > 1:
            for feat in random.sample(current_features, min(2, len(current_features))):
                patches.append(ExperimentPatch(
                    name=f"ablate_{feat}",
                    description=f"Remove feature generator: {feat} (ablation study)",
                    feature_names=[f for f in current_features if f != feat],
                ))

        return patches[:n]


class HyperparamExploration:
    """Generate hyperparameter variants."""

    LIGHTGBM_SPACE = {
        "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.15],
        "num_leaves": [15, 31, 63, 127],
        "max_depth": [-1, 5, 7, 10],
        "min_child_samples": [5, 10, 20, 50],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "reg_alpha": [0, 0.01, 0.1, 1.0],
        "reg_lambda": [0, 0.01, 0.1, 1.0],
        "n_estimators": [500, 1000, 2000, 3000],
    }

    XGBOOST_SPACE = {
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "max_depth": [3, 5, 6, 7, 9],
        "min_child_weight": [1, 3, 5, 10],
        "subsample": [0.6, 0.7, 0.8, 0.9],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
        "reg_alpha": [0, 0.01, 0.1, 1.0],
        "reg_lambda": [0, 0.1, 1.0, 5.0],
        "n_estimators": [500, 1000, 2000],
    }

    def generate_variants(self, model_type: str = "lightgbm", n: int = 3) -> list[ExperimentPatch]:
        space = self.LIGHTGBM_SPACE if model_type == "lightgbm" else self.XGBOOST_SPACE
        patches = []

        for i in range(n):
            params = {}
            # Randomly sample 3-5 hyperparameters to change
            keys = random.sample(list(space.keys()), min(4, len(space)))
            for key in keys:
                params[key] = random.choice(space[key])

            patches.append(ExperimentPatch(
                name=f"hparam_v{i+1}",
                description=f"Hyperparam variant: {', '.join(f'{k}={v}' for k, v in params.items())}",
                model_type=model_type,
                model_params=params,
            ))

        return patches


class ModelSwitch:
    """Generate model type variants."""

    MODELS = ["lightgbm", "xgboost"]

    def generate_variants(self, current_model: str) -> list[ExperimentPatch]:
        patches = []
        for model in self.MODELS:
            if model != current_model:
                patches.append(ExperimentPatch(
                    name=f"switch_to_{model}",
                    description=f"Switch model from {current_model} to {model}",
                    model_type=model,
                ))
        return patches


class IterationPlanner:
    """Plan the next iteration based on current state and history."""

    def __init__(self, available_features: list[str], competition_type: str = "tabular"):
        self.feat_explorer = FeatureExploration(available_features, competition_type)
        self.hparam_explorer = HyperparamExploration()
        self.model_switcher = ModelSwitch()

    def plan_next(
        self,
        current_features: list[str],
        current_model: str,
        iteration: int,
        best_score: float | None = None,
        stale_rounds: int = 0,
    ) -> list[ExperimentPatch]:
        """Generate candidate experiments for next iteration.

        Strategy:
        - Early iterations: explore features
        - Middle iterations: tune hyperparams
        - Stale: try model switch or aggressive changes
        """
        patches = []

        if iteration <= 3:
            patches.extend(self.feat_explorer.generate_variants(current_features, n=2))
            patches.extend(self.hparam_explorer.generate_variants(current_model, n=1))
        elif stale_rounds >= 2:
            patches.extend(self.model_switcher.generate_variants(current_model))
            patches.extend(self.hparam_explorer.generate_variants(current_model, n=2))
        else:
            patches.extend(self.hparam_explorer.generate_variants(current_model, n=2))
            patches.extend(self.feat_explorer.generate_variants(current_features, n=1))

        return patches
