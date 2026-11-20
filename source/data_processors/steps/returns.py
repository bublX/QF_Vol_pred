import numpy as np
import pandas as pd

from source.data_processors.steps.base import ProcessingStep
from source.utils.logger import Logger


class LogReturns(ProcessingStep):
    """
    Расчёт внутридневных логарифмических доходностей в процентах.

    Notes
    -----
    Вход: таблица одного тикера с колонками date и close.

    Доходность рассчитывается между соседними наблюдениями
    одного дня. Первое наблюдение каждого дня получает NaN.
    Ночной разрыв между днями не учитывается.
    """

    def __init__(self):
        self.logger = Logger("LogReturns")

    def transform(
        self,
        data: pd.DataFrame,
        ticker: str = "",
    ) -> pd.DataFrame:
        """
        Вернуть копию таблицы с рассчитанным столбцом returns.

        Parameters
        ----------
        data : pd.DataFrame
            Датасет тикера с датой и ценой закрытия
        ticker : str
            Название тикера

        Returns
        -------
        pd.DataFrame
            Таблица с рассчитанным столбцом returns
        """
        data = data.copy()
        previous_close = data.groupby(data["date"].dt.date)["close"].shift(1)

        data["returns"] = np.log(data["close"] / previous_close) * 100

        self.logger.info(f"{ticker}: доходности рассчитаны")
        return data
