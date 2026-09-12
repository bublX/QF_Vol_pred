import pandas as pd
import numpy as np
from typing import List, Tuple
import xgboost as xgb
from source.models.base import BaseModel

class XGBoostModel(BaseModel):
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
        callbacks = [xgb.callback.EarlyStopping(rounds = early_stopping_rounds, save_best = True)]
        self.model = xgb.XGBRegressor(
            n_estimators = n_estimators,
            max_depth = max_depth,
            learning_rate = learning_rate,
            reg_alpha = reg_alpha,            
            reg_lambda = reg_lambda, 
            callbacks = callbacks,
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
        
        self._validate_input(X)
        self._feature_names = X.columns.to_list()
        if eval_set is not None:
            self.model.fit(X, y, 
                eval_set = eval_set, 
                verbose = verbose
                )
        else:
            self.model.fit(X, y)

        self.is_trained = True
        return self

    def predict(
        self,
        X : pd.DataFrame
    ) -> np.ndarray:
        self._check_is_trained()
        self._validate_input(X)
        return self.model.predict(X)

    def features_importance(self) -> dict:
        self._check_is_trained()
        if self._feature_names is None:
            raise ValueError('No features names')
        else:
            return dict(zip(self._feature_names, self.model.feature_importances_))