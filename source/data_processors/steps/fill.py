import pandas as pd

from source.data_processors.steps.base import ProcessingStep
from source.utils.logger import Logger


class ForwardFillWithinDay(ProcessingStep):
    """
    Заполнение пропусков цены последним известным значением внутри дня.

    Notes
    -----
    Вход: таблица одного тикера с колонками date и close.

    Цена не переносится между днями. Строки без известной цены
    после заполнения удаляются, индекс сбрасывается.
    Исходная таблица не изменяется.
    """

    def __init__(self):
        self.logger = Logger("ForwardFillWithinDay")

    def transform(
        self,
        data: pd.DataFrame,
        ticker: str = "",
    ) -> pd.DataFrame:
        """
        Вернуть копию таблицы после заполнения цен внутри дня.

        Parameters
        ----------
        data : pd.DataFrame
            Датасет тикера с датой и ценой закрытия
        ticker : str
            Название тикера

        Returns
        -------
        pd.DataFrame
            Таблица с заполненными пропусками
        """
        data = data.copy()

        missing_before = int(data["close"].isna().sum())
        day = data["date"].dt.date
        data["close"] = data.groupby(day)["close"].ffill()
        missing_after = int(data["close"].isna().sum())

        data = data.dropna(subset=["close"]).reset_index(drop=True)

        self.logger.info(
            f"{ticker}: ffill внутри дня — "
            f"{missing_before - missing_after} баров; "
            f"удалено строк без цены — {missing_after}"
        )

        return data
