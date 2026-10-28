from typing import Any

from source.autoregression import HARModel
from source.models.lightgbm_model import LightGBMModel
from source.models.random_forest_model import RandomForestModel
from source.models.xgboost_model import XGBoostModel

MODELS = {
    "har": HARModel,
    "rf": RandomForestModel,
    "xgb": XGBoostModel,
    "lgbm": LightGBMModel,
}


def get_model(name: str, params: dict[str, Any] | None = None):
    """
    Создать модель по названию из реестра.

    Parameters
    ----------
    name : str
        Название модели (ключ MODELS: 'har', 'rf', 'xgb', 'lgbm')
    params : dict[str, Any] | None
        Гиперпараметры, передаются в конструктор модели как есть

    Returns
    -------
    BaseModel | BaseARModel
        Новый экземпляр модели

    Raises
    ------
    ValueError
        Если модели с таким названием нет в реестре

    Notes
    -----
    Чтобы добавить модель, напишите класс и добавьте его в MODELS.
    """

    if name not in MODELS:
        raise ValueError(f"Неизвестная модель: '{name}'. Доступные: {list(MODELS)}")

    return MODELS[name](**(params or {}))
