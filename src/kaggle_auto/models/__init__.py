from .base import BaseModel
from .ensemble import Ensemble
from .cv import CrossValidator


def LightGBMModel(*args, **kwargs):
    from .lightgbm_model import LightGBMModel as _cls
    return _cls(*args, **kwargs)


def XGBoostModel(*args, **kwargs):
    from .xgboost_model import XGBoostModel as _cls
    return _cls(*args, **kwargs)


__all__ = ["BaseModel", "LightGBMModel", "XGBoostModel", "Ensemble", "CrossValidator"]
