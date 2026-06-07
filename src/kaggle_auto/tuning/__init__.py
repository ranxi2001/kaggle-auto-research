from .search_space import get_search_space


def OptunaTuner(*args, **kwargs):
    from .optuna_tuner import OptunaTuner as _cls
    return _cls(*args, **kwargs)


__all__ = ["OptunaTuner", "get_search_space"]
