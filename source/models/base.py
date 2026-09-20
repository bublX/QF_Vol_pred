import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, List

class BaseModel(ABC):
    """
    Базовый абстрактный класс для моделей машинного обучения

    Parameters
    ----------
    name : str
        Название конкретной модели ('RandomForest')

    Attributes
    ----------
    name : str
        Название модели
    model : object
        Внутренняя модель (xgboost, lightgbm и тд)
    is_trained : bool
        Индикатор обученности модели
    _feature_names  
    """
    def __init__(
        self,
        name : str
    ):
        self.name = name
        self.model = None
        self.is_trained = False
        self._feature_names = None

    @abstractmethod
    def fit(
        self, 
        X : pd.DataFrame,
        y : pd.Series
    ) -> 'BaseModel':
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
    def predict(
        self,
        X : pd.DataFrame
    ) -> np.ndarray:
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

    def get_params(self) -> dict:
        """
        Вернуть параметры модели.
        
        Returns
        -------
        Dict
            Словарь с параметрами (пустой, если модель не создана)
        """

        if self.model is None:
            return {}
        
        return self.model.get_params()

    def get_feature_names(self) -> Optional[List]:
        """
        Вернуть имена признаков.
        
        Returns
        -------
        Optional[List]
            Список имён признаков, если модель не обучена - None
        """
        return self._feature_names

    def _validate_input(
        self, 
        X : pd.DataFrame
    ) -> None:
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

        if not isinstance (X, pd.DataFrame):
            raise TypeError(f'Waiting pd.DataFrame, but actually get {type(X)}!')
        
        if X.empty: 
            raise ValueError('Features are empty!')

    def _check_is_trained(self) -> None:
        """
        Проверить, что модель обучена.
        
        Raises
        ------
        ValueError
            Если модель не обучена
        """
        if not self.is_trained:
            raise ValueError(f'Model {self.name} is not trained. Call fit() to train.')
        
    def __repr__(self) -> str:
        """
        Строковое представление модели.
        
        Returns
        -------
        str
            Например: 'RandomForestModel : (name = RandomForest, status = trained)'
        """
        status = 'trained' if self.is_trained else 'not trained'
        return f'{self.__class__.__name__} : (name = {self.name}, status = {status})'