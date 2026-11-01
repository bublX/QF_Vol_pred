import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
)
from typing import Dict, Optional


class MetricsCalculator:
    """
    Калькулятор метрик для оценки прогнозов sqrt{RV}.

    Parameters
    ----------
    metrics : list, optional
        Список метрик для расчёта.
        По умолчанию ['MSE', 'RMSE', 'MAE', 'R2', 'QLIKE', 'MAPE'].

    Notes
    -----
    Доступные метрики:
    - MSE — Mean Squared Error
    - RMSE — Root Mean Squared Error
    - MAE — Mean Absolute Error
    - R2 — коэффициент детерминации
    - QLIKE — Quasi-likelihood loss function
    - MAPE — Mean Absolute Percentage Error
    Все метрики считаются на шкале σ, кроме QLIKE: она на дисперсии (σ²), отрицательный прогноз σ заменяется на 0
    """

    def __init__(self, metrics: Optional[list] = None):
        self.metrics = metrics or ["MSE", "RMSE", "MAE", "R2", "QLIKE", "MAPE"]

    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Рассчитать метрики для одного тикера.

        Parameters
        ----------
        y_true : np.ndarray
            Истинные значения
        y_pred : np.ndarray
            Прогнозы

        Returns
        -------
        Dict[str, float]
            Словарь с метриками {метрика: значение}

        Raises
        ------
        ValueError
            Если размеры y_true и y_pred не совпадают
        """

        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()

        if len(y_true) != len(y_pred):
            raise ValueError(f"Shapes are different: {len(y_true)} and {len(y_pred)}")

        results = {}

        if "MSE" in self.metrics:
            results["MSE"] = mean_squared_error(y_true, y_pred)

        if "RMSE" in self.metrics:
            results["RMSE"] = np.sqrt(mean_squared_error(y_true, y_pred))

        if "MAE" in self.metrics:
            results["MAE"] = mean_absolute_error(y_true, y_pred)

        if "R2" in self.metrics:
            results["R2"] = r2_score(y_true, y_pred)

        if "QLIKE" in self.metrics:
            results["QLIKE"] = self.qlike(y_true**2, np.maximum(y_pred, 0) ** 2)

        if "MAPE" in self.metrics:
            results["MAPE"] = mean_absolute_percentage_error(y_true, y_pred)

        return results

    def calculate_all(
        self, y_true_dict: Dict[str, np.ndarray], y_pred_dict: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, float]]:
        """
        Рассчитать метрики для нескольких тикеров.

        Parameters
        ----------
        y_true_dict : Dict[str, np.ndarray]
            {тикер: y_true}
        y_pred_dict : Dict[str, np.ndarray]
            {тикер: y_pred}

        Returns
        -------
        Dict[str, Dict[str, float]]
            {тикер: {метрика: значение}}
        """

        results = {}

        for ticker in y_true_dict:
            results[ticker] = self.calculate(y_true_dict[ticker], y_pred_dict[ticker])

        return results

    @staticmethod
    def qlike(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        QLIKE = mean(y / p - ln(y / p) - 1), 0 при идеальном прогнозе.

        Notes
        -----
        Считается на шкале дисперсии (RV), y_true > 0. Занижение прогноза
        штрафуется сильнее завышения. Прогноз p <= 0 заменяется на 1e-8:
        штраф огромный, но конечный (иначе nan).
        """

        ratio = y_true / np.maximum(y_pred, 1e-8)
        return float(np.mean(ratio - np.log(ratio) - 1))
