import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict
from source.utils.logger import Logger


class BaseARModel(ABC):
    """
    Базовый класс для авторегрессионных моделей.

    Parameters
    ----------
    name : str
        Название модели ('HAR', 'ARIMA')
    lags : List[int]
        Лаги для авторегрессии
    """

    def __init__(self, name: str, lags: List[int] = [1, 5, 22]):

        self.name = name
        self.lags = lags
        self.model = None
        self.is_trained = False
        self._features_names = None
        self.logger = Logger(name)

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseARModel":
        """
        Обучить модель.

        Parameters
        ----------
        X : pd.DataFrame
            Признаки
        y : pd.Series
            Целевая переменная

        Returns
        -------
        BaseModel
            self
        """

        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Сделать прогноз.

        Parameters
        ----------
        X : pd.DataFrame
            Признаки

        Returns
        -------
        np.ndarray
            Прогнозы
        """

        pass

    def get_params(self) -> Dict:
        """
        Вернуть параметры модели.

        Returns
        -------
        Dict
            Словарь с параметрами (пустой, если модель не создана)
        """

        return {"name": self.name, "lags": self.lags}

    def _validate_input(self, X: pd.DataFrame) -> None:
        """
        Проверить входные данные.

        Parameters
        ----------
        X : pd.DataFrame
            Признаки

        Raises
        ------
        TypeError
            Если X не pd.DataFrame
        ValueError
            Если X пустой
        """

        if not isinstance(X, pd.DataFrame):
            self.logger.error(f"Ожидается DataFrame, получено {type(X)}")
            raise TypeError(f"Ожидается DataFrame, получено {type(X)}!")

        if X.empty:
            self.logger.error("Признаки пустые!")
            raise ValueError("Признаки пустые!")

    def _check_is_trained(self) -> None:
        """
        Проверить, что модель обучена.

        Raises
        ------
        ValueError
            Если модель не обучена
        """
        if not self.is_trained:
            self.logger.error(f"Модель {self.name} не обучена")
            raise ValueError(f"Модель {self.name} не обучена. Вызовите fit().")

    def __repr__(self) -> str:
        """
        Строковое представление модели.

        Returns
        -------
        str
            Например: 'HAR : (name = HAR, status = trained)'
        """
        status = "trained" if self.is_trained else "not trained"
        return f"{self.__class__.__name__} : (name = {self.name}, status = {status})"
