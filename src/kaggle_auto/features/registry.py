"""Feature registry: register, resolve dependencies, and build feature pipelines."""

import pandas as pd

from .base import BaseFeature

_REGISTRY: dict[str, type[BaseFeature]] = {}


def register(cls: type[BaseFeature]) -> type[BaseFeature]:
    """Decorator to register a feature class."""
    _REGISTRY[cls.name] = cls
    return cls


def get(name: str) -> type[BaseFeature]:
    """Get a registered feature class by name."""
    if name not in _REGISTRY:
        raise KeyError(f"Feature not registered: {name}. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[name]


def list_features() -> list[str]:
    """List all registered feature names."""
    return list(_REGISTRY.keys())


def build(feature_names: list[str], df: pd.DataFrame) -> pd.DataFrame:
    """Build features by composing registered feature generators."""
    built: set[str] = set()
    parts: list[pd.DataFrame] = [df]

    def _build_one(name: str) -> None:
        if name in built:
            return
        feat_cls = get(name)
        for dep in feat_cls.dependencies:
            _build_one(dep)
        feat = feat_cls()
        current = pd.concat(parts, axis=1)
        new_cols = feat.compute(current)
        parts.append(new_cols)
        built.add(name)

    for name in feature_names:
        _build_one(name)

    return pd.concat(parts, axis=1)
