import pandas as pd
import numpy as np
import os
from typing import List, Tuple
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from source.models.base import BaseModel
from contextlib import redirect_stdout


class LightGBMModel(BaseModel):
    """
    LightGBM для прогнозирования волатильности sqrt(RV) в текущем пайплайне.

    Наследует BaseModel. Реализует fit/predict для LGBMRegressor.
    Обучается на переданном y и возвращает прогнозы в его шкале;
    преобразование таргета выполняется вне модели.

    Parameters
    ----------
    n_estimators : int
        Количество деревьев
    max_depth : int
        Максимальная глубина
    learning_rate : float
        Скорость обучения
    min_child_samples : int
        Минимальное число объектов в листе
    reg_alpha : float
        L1-регуляризация
    reg_lambda : float
        L2-регуляризация
    subsample : float
        Доля объектов для обучения
    subsample_freq : int
        Частота подвыборки
    colsample_bytree : float
        Доля признаков для каждого дерева
    random_state : int
        Seed для воспроизводимости
    verbose : bool
        Индикатор для вывода логирования обучения
    Notes
    -----
    Если eval_set не передан — early stopping отключается.
    Вывод early stopping подавляется через sys.stdout.
    """

    def __init__(
        self,
        n_estimators: int = 1000,
        max_depth: int = 31,
        learning_rate: float = 0.05,
        min_child_samples: int = 20,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1,
        subsample: float = 0.8,
        subsample_freq: int = 5,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__("LightGBM")
        self.verbose = verbose
        self.model = LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            min_child_samples=min_child_samples,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            subsample=subsample,
            subsample_freq=subsample_freq,
            random_state=random_state,
            verbosity=-1,
            n_jobs=-1,
            colsample_bytree=colsample_bytree,
            **kwargs,
        )
        self.best_n_estimators = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: List[Tuple[pd.DataFrame, pd.Series]] | None = None,
        early_stopping_rounds: int = 50,
    ) -> "LightGBMModel":
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
        early_stopping_rounds : int
            Раундов без улучшения до остановки

        Notes
        -----
        Если eval_set не передан — early stopping отключается.
        Вывод early stopping подавляется при помощи параметра verbose.
        best_n_estimators — число деревьев после early stopping (без валидации — n_estimators)
        См. BaseModel.fit для общих параметров.
        """

        self._validate_input(X)
        self._feature_names = X.columns.to_list()

        if eval_set is not None and eval_set[0][0] is not None:
            X_valid, y_valid = eval_set[0]
            callbacks = [
                early_stopping(early_stopping_rounds),
                log_evaluation(period=-1),
            ]
            if self.verbose:
                self.model.fit(
                    X, y, eval_X=X_valid, eval_y=y_valid, callbacks=callbacks
                )
                self.best_n_estimators = self.model.best_iteration_
            else:
                with open(os.devnull, "w") as f, redirect_stdout(f):
                    self.model.fit(
                        X, y, eval_X=X_valid, eval_y=y_valid, callbacks=callbacks
                    )
                    self.best_n_estimators = self.model.best_iteration_
        else:
            self.model.fit(X, y)
            self.best_n_estimators = self.model.n_estimators

        self.is_trained = True

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Сделать прогноз.

        Notes
        -----
        См. BaseModel.predict.
        """

        self._check_is_trained()
        self._validate_input(X)

        return np.asarray(self.model.predict(X))

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
            raise ValueError("No features names")
        else:
            return dict(zip(self._feature_names, self.model.feature_importances_))
