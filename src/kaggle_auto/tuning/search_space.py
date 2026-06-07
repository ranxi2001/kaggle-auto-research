"""Default hyperparameter search spaces per model type."""


SEARCH_SPACES = {
    "lightgbm": {
        "learning_rate": ("float_log", 0.005, 0.3),
        "num_leaves": ("int", 15, 127),
        "max_depth": ("int", 3, 12),
        "min_child_samples": ("int", 5, 100),
        "subsample": ("float", 0.5, 1.0),
        "colsample_bytree": ("float", 0.3, 1.0),
        "reg_alpha": ("float_log", 1e-8, 10.0),
        "reg_lambda": ("float_log", 1e-8, 10.0),
        "n_estimators": ("int", 100, 3000),
    },
    "xgboost": {
        "learning_rate": ("float_log", 0.005, 0.3),
        "max_depth": ("int", 3, 12),
        "min_child_weight": ("int", 1, 50),
        "subsample": ("float", 0.5, 1.0),
        "colsample_bytree": ("float", 0.3, 1.0),
        "reg_alpha": ("float_log", 1e-8, 10.0),
        "reg_lambda": ("float_log", 1e-8, 10.0),
        "n_estimators": ("int", 100, 3000),
        "gamma": ("float_log", 1e-8, 5.0),
    },
    "catboost": {
        "learning_rate": ("float_log", 0.005, 0.3),
        "depth": ("int", 4, 10),
        "l2_leaf_reg": ("float_log", 1e-3, 10.0),
        "bagging_temperature": ("float", 0.0, 1.0),
        "random_strength": ("float", 0.0, 10.0),
        "iterations": ("int", 100, 3000),
    },
}


def get_search_space(model_name: str) -> dict:
    """Get default search space for a model."""
    return SEARCH_SPACES.get(model_name, SEARCH_SPACES["lightgbm"])


def suggest_param(trial, name: str, spec: tuple):
    """Suggest a parameter value from an Optuna trial."""
    param_type = spec[0]

    if param_type == "int":
        return trial.suggest_int(name, spec[1], spec[2])
    elif param_type == "float":
        return trial.suggest_float(name, spec[1], spec[2])
    elif param_type == "float_log":
        return trial.suggest_float(name, spec[1], spec[2], log=True)
    elif param_type == "categorical":
        return trial.suggest_categorical(name, spec[1])
    else:
        raise ValueError(f"Unknown param type: {param_type}")
