"""Analyze top solutions and extract patterns."""

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SolutionPattern:
    approach: str
    models_used: list[str] = field(default_factory=list)
    features_used: list[str] = field(default_factory=list)
    cv_strategy: str = ""
    score: float | None = None
    key_insight: str = ""


class SolutionAnalyzer:
    """Analyze pulled notebooks and extract reusable patterns."""

    COMMON_MODELS = [
        "lightgbm", "lgbm", "xgboost", "xgb", "catboost",
        "random_forest", "randomforest",
        "lstm", "gru", "transformer",
        "tabnet", "deberta", "bert",
        "linear_regression", "logistic_regression",
        "svm", "knn", "ridge", "lasso",
    ]

    COMMON_FEATURES = [
        "target_encoding", "label_encoding", "one_hot",
        "rolling_mean", "rolling_std", "lag",
        "pca", "svd", "umap",
        "tfidf", "word2vec", "embedding",
        "interaction", "polynomial",
        "rsi", "macd", "bollinger", "ema", "sma",
    ]

    CV_PATTERNS = [
        "stratifiedkfold", "kfold", "groupkfold",
        "timeseriessplit", "repeatedkfold",
        "train_test_split",
    ]

    def analyze_notebook(self, notebook_path: Path) -> SolutionPattern:
        """Analyze a single pulled notebook for patterns."""
        content = notebook_path.read_text(errors="ignore").lower()

        models = self._detect_models(content)
        features = self._detect_features(content)
        cv = self._detect_cv(content)
        approach = self._infer_approach(models, features)

        return SolutionPattern(
            approach=approach,
            models_used=models,
            features_used=features,
            cv_strategy=cv,
        )

    def analyze_directory(self, solutions_dir: Path) -> list[SolutionPattern]:
        """Analyze all notebooks in a directory."""
        patterns = []
        for f in solutions_dir.glob("**/*.py"):
            patterns.append(self.analyze_notebook(f))
        for f in solutions_dir.glob("**/*.ipynb"):
            patterns.append(self.analyze_notebook(f))
        return patterns

    def summarize_patterns(self, patterns: list[SolutionPattern]) -> dict:
        """Summarize common patterns across solutions."""
        from collections import Counter

        model_counts = Counter()
        feature_counts = Counter()
        cv_counts = Counter()

        for p in patterns:
            for m in p.models_used:
                model_counts[m] += 1
            for f in p.features_used:
                feature_counts[f] += 1
            if p.cv_strategy:
                cv_counts[p.cv_strategy] += 1

        return {
            "top_models": model_counts.most_common(5),
            "top_features": feature_counts.most_common(10),
            "cv_strategies": cv_counts.most_common(3),
            "total_solutions_analyzed": len(patterns),
        }

    def _detect_models(self, content: str) -> list[str]:
        found = []
        for model in self.COMMON_MODELS:
            if model in content:
                found.append(model)
        return found

    def _detect_features(self, content: str) -> list[str]:
        found = []
        for feat in self.COMMON_FEATURES:
            if feat in content:
                found.append(feat)
        return found

    def _detect_cv(self, content: str) -> str:
        for cv in self.CV_PATTERNS:
            if cv in content:
                return cv
        return ""

    def _infer_approach(self, models: list[str], features: list[str]) -> str:
        if any(m in ["lstm", "gru", "transformer"] for m in models):
            return "deep_learning"
        if any(m in ["lightgbm", "lgbm", "xgboost", "xgb", "catboost"] for m in models):
            if len(models) > 1:
                return "gradient_boosting_ensemble"
            return "gradient_boosting"
        if any(m in ["deberta", "bert"] for m in models):
            return "nlp_transformer"
        return "traditional_ml"
