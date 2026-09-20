import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from source.models.base import BaseModel

class RandomForestModel(BaseModel):
    """
    Random Forest для прогнозирования RV.
    
    Наследует BaseModel. Реализует fit/predict для RandomForestRegressor.
    
    Parameters
    ----------
    n_estimators : int
        Количество деревьев
    max_depth : int
        Максимальная глубина
    min_samples_split : int
        Минимальное число объектов для разибения узла
    max_features : str
        Максимальное число признаков (по умолчанию sqrt(n))
    random_state : int
        Seed для воспроизводимости
    """
    def __init__(
        self,
        n_estimators : int = 100, 
        max_depth : int = 10, 
        min_samples_split : int = 10, 
        max_features : str = 'sqrt',
        random_state : int = 42,
        **kwargs
    ):
        super().__init__('RandomForest')

        self.model = RandomForestRegressor(
            n_estimators = n_estimators,
            max_depth = max_depth,
            min_samples_split = min_samples_split,
            max_features = max_features,
            random_state = random_state,
            n_jobs = -1,
            **kwargs
        )

    def fit(
        self,
        X : pd.DataFrame,
        y : pd.Series
    ) -> 'RandomForestModel':
        """
        Обучить модель.
        
        Notes
        -----
        См. BaseModel.fit для общих параметров.
        Random Forest не использует eval_set.
        """

        self._validate_input(X)
        self._feature_names = X.columns.to_list()

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