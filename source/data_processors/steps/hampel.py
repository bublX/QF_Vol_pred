import numpy as np
import pandas as pd

from typing import Tuple
from source.data_processors.steps.base import ProcessingStep
from source.utils.logger import Logger


class HampelFilter(ProcessingStep):
    """
    Обработка доходностей при помощи фильтра Хампеля
    внутри каждого дня отдельно
    """

    def __init__(self, window_size: int = 5, sigma: float = 3.0, const: float = 1.4826):
        if (
            not isinstance(window_size, int)
            or isinstance(window_size, bool)
            or window_size < 1
        ):
            raise ValueError("window_size должен быть положительным целым")

        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("sigma должен быть конечным положительным числом")
        if not np.isfinite(const) or const <= 0:
            raise ValueError("const должен быть конечным положительным числом")

        self.window_size = window_size
        self.sigma = sigma
        self.const = const
        self.logger = Logger("HampelFilter")

    @staticmethod
    def _filter_hampel(
        series: np.ndarray,
        window_size: int = 11,
        sigma: float = 3.0,
        const: float = 1.4826,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Фильтр Хампеля для обнаружения выбросов.

        Parameters
        ----------
        series : np.ndarray
            Временной ряд
        window_size : int
            Полуширина окна (окно = 2w + 1)
        sigma : float
            Порог в сигмах
        const : float
            Константа нормализации MAD

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (cleaned, flags) — очищенный ряд и флаги для логирования
        """

        series = np.asarray(series, dtype=float)
        series_length = len(series)
        cleaned = series.copy()
        flags = np.zeros(series_length, dtype=bool)

        for i in range(series_length):
            if np.isnan(series[i]):
                continue
            low, high = max(0, i - window_size), min(series_length, i + window_size + 1)
            window = series[low:high]
            r_median = np.nanmedian(window)
            r_mad = np.nanmedian(np.abs(window - r_median))

            if r_mad == 0:
                continue

            if np.abs(series[i] - r_median) > sigma * r_mad * const:
                cleaned[i] = r_median
                flags[i] = True

        return cleaned, flags

    def transform(self, data: pd.DataFrame, ticker: str = "") -> pd.DataFrame:
        """
        Применить фильтр Хампеля к доходностям.

        Parameters
        ----------
        data : pd.DataFrame
            Данные тикера (столбцы date типа datetime и числовой returns)
        ticker : str
            Название тикера

        Returns
        -------
        pd.DataFrame
            DataFrame с очищенными доходностями
        """

        data = data.copy()

        if data.empty:
            return data

        returns = data["returns"].to_numpy(dtype=float, copy=True)
        flags = np.zeros(len(returns), dtype=bool)
        daily_indices = data.groupby(data["date"].dt.date).indices.values()
        for idx in daily_indices:
            returns[idx], flags[idx] = self._filter_hampel(
                series=returns[idx],
                window_size=self.window_size,
                sigma=self.sigma,
                const=self.const,
            )

        data["returns"] = returns
        n = int(flags.sum())

        self.logger.info(f"{ticker}: Hampel filter - {n} замен ({n / len(flags):.3%})")
        return data
