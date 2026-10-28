import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import Dict, List, Optional
from source.autoregression.base_ar import BaseARModel


class HARModel(BaseARModel):
    """
    HAR-модель для прогнозирования RV. Наследует BaseARModel.

    Реализует fit/predict для HAR.
    """

    def __init__(self, lags: List[int] = [1, 5, 22]):
        super().__init__("HAR", lags)
        self._coef: Optional[np.ndarray] = None
        self._intercept: Optional[float] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HARModel":
        """
        Обучение HAR через OLS.

        Returns
        -------
        HARModel
            Обученная модель

        Notes
        -----
        См. class BaseARModel.fit.
        """

        self._validate_input(X)
        self._features_names = X.columns.to_list()

        X = sm.add_constant(X)
        self.model = sm.OLS(y, X).fit()

        self._intercept = float(self.model.params.iloc[0])
        self._coef = self.model.params.iloc[1:].to_numpy()

        self.is_trained = True
        self.logger.success(f"HAR обучен!")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Сделать прогноз.

        Returns
        -------
        np.ndarray
            Прогнозы

        Notes
        -----
            См. class BaseARModel.predict.
        """

        self._check_is_trained()
        self._validate_input(X)

        X = sm.add_constant(X)
        return self.model.predict(X).values

    def get_params(self) -> Dict:
        """
        Вернуть параметры модели.

        Returns
        -------
        Dict
            Словарь с параметрами (intercept, coef)
        """

        params = super().get_params()
        if self._intercept is not None and self._coef is not None:
            params["intercept"] = float(self._intercept)
            params["coef"] = self._coef.tolist()

        return params
