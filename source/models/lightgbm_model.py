import pandas as pd
import numpy as np
from typing import List, Tuple
from lightgbm import LGBMRegressor, early_stopping
from source.models.base import BaseModel

class LightGBMModel(BaseModel):
    def __init__(
        self,
        n_estimators : int = 1000,
        max_depth : int = 31,
        learning_rate : float = 0.05,
        min_child_samples : int = 20,
        reg_alpha : float = 0.1,
        reg_lambda : float = 1,
        subsample : float = 0.8, 
        subsample_freq : int = 5,
        random_state : int = 42, 
        **kwargs
    ):
        super().__init__('LightGBM')

        self.model = LGBMRegressor(
            n_estimators = n_estimators,
            max_depth = max_depth,
            learning_rate = learning_rate,
            min_child_samples = min_child_samples,
            reg_alpha = reg_alpha,
            reg_lambda = reg_lambda,
            subsample = subsample, 
            subsample_freq = subsample_freq,
            random_state = random_state,
            verbosity = -1,
            n_jobs = -1,
            **kwargs
        )

    def fit(
        self,
        X : pd.DataFrame,
        y : pd.Series,
        eval_set : List[Tuple[pd.DataFrame, pd.Series]] = None,
        callbacks : List = None
    ) -> 'LightGBMModel':

        self._validate_input(X)
        self._feature_names = X.columns.to_list()

        if eval_set is not None:
            self.model.fit(X, y, eval_set = eval_set, callbacks = callbacks)
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