import pandas as pd
import numpy as np
from typing import List, Tuple
import xgboost as xgb
from source.models.base import BaseModel

class XGBoostModel(BaseModel):
    """
    XGBoost для прогнозирования RV.
    
    Наследует BaseModel. Реализует fit/predict для XGBRegressor.
    Поддерживает early stopping через eval_set.
    
    Parameters
    ----------
    n_estimators : int
        Количество деревьев
    max_depth : int
        Максимальная глубина
    learning_rate : float
        Скорость обучения
    reg_alpha : float
        L1-регуляризация
    reg_lambda : float
        L2-регуляризация
    early_stopping_rounds : int
        Раундов без улучшения до остановки
    subsample : float
        Доля объектов для обучения каждого дерева
    colsample_bytree : float
        Доля признаков для каждого дерева
    random_state : int
        Seed для воспроизводимости
    """
    def __init__(
        self,
        n_estimators : int = 1000,
        max_depth : int = 6,
        learning_rate : float = 0.05,
        reg_alpha : float = 0.1,
        reg_lambda : float = 1,
        early_stopping_rounds : int = 50,
        subsample : float = 0.8,
        colsample_bytree : float = 0.8,
        random_state : int = 42,
        **kwargs
    ):
        super().__init__('XGBoost')
        self.early_stopping_rounds = early_stopping_rounds
        self.model = xgb.XGBRegressor(
            n_estimators = n_estimators,
            max_depth = max_depth,
            learning_rate = learning_rate,
            reg_alpha = reg_alpha,            
            reg_lambda = reg_lambda,
            subsample = subsample, 
            colsample_bytree = colsample_bytree, 
            random_state = random_state,
            n_jobs = -1,
            **kwargs 
        )

    def fit(
        self,
        X : pd.DataFrame,
        y : pd.Series,
        eval_set : List[Tuple[pd.DataFrame, pd.Series]] = None,
        verbose : bool = False
    ) -> 'XGBoostModel':
        """
        Обучить модель.
        
        Parameters
        ----------
        X : pd.DataFrame
            Признаки
        y : pd.Series
            Целевая переменная
        eval_set : List[Tuple[pd.DataFrame, pd.Series]], optional
            Валидационная выборка для early stopping
        verbose : bool
            Печатать ли прогресс обучения
        
        Notes
        -----
        Если eval_set не передан — early stopping отключается.
        См. BaseModel.fit для общих параметров.
        """

        self._validate_input(X)
        self._feature_names = X.columns.to_list()
        if eval_set is not None and eval_set[0][0] is not None:
            self.model.set_params(early_stopping_rounds = self.early_stopping_rounds)
            self.model.fit(X, y, 
                eval_set = eval_set, 
                verbose = verbose
                )
        else:
            self.model.set_params(early_stopping_rounds = None)
            self.model.fit(X, y)

        self.is_trained = True
        return self

    def predict(
        self,
        X : pd.DataFrame
    ) -> np.ndarray:
        """
        Сделать прогноз.
        
        Notes
        -----
        См. BaseModel.predict.
        """

        self._check_is_trained()
        self._validate_input(X)
        return self.model.predict(X)

    def features_importance(self) -> dict:
        """
        Вернуть важность признаков.
        
        Returns
        -------
        dict
            {признак: важность}
        
        Raises
        ------
        ValueError
            Если модель не обучена
        """

        self._check_is_trained()
        
        if self._feature_names is None:
            raise ValueError('No features names')
        else:
            return dict(zip(self._feature_names, self.model.feature_importances_))