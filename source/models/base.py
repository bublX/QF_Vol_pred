import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, List

class BaseModel(ABC):
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
        pass

    @abstractmethod
    def predict(
        self,
        X : pd.DataFrame
    ) -> np.ndarray:
        pass

    def get_params(self) -> dict:
        if self.model is None:
            return {}
        return self.model.get_params()

    def get_feature_names(self) -> Optional[List]:
        return self._feature_names

    def _validate_input(
        self, 
        X : pd.DataFrame
    ) -> None:
        if not isinstance (X, pd.DataFrame):
            raise TypeError(f'Waiting pd.DataFrame, but actually get {type(X)}!')
        if X.empty: 
            raise ValueError('Features are empty!')

    def _check_is_trained(self) -> None:
        if not self.is_trained:
            raise ValueError(f'Model {self.name} is not trained. Call fit() to train.')
        
    def __repr__(self) -> str:
        status = 'trained' if self.is_trained else 'not trained'
        return f'{self.__class__.__name__} : (name = {self.name}, status = {status})'