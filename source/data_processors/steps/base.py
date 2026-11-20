from abc import ABC, abstractmethod
import pandas as pd


class ProcessingStep(ABC):
    """
    Шаг преобразования временного ряда для одного тикера.
    """

    @abstractmethod
    def transform(self, data: pd.DataFrame, ticker: str = "") -> pd.DataFrame:
        """
        Применить преобразование и вернуть результат.

        Parameters
        ----------
        data: pd.DataFrame
            Таблица одного тикера с колонкой date.
            Необходимые дополнительные столбцы может определять сам шаг.
        ticker : str
            Название тикера для диагностики.

        Returns
        -------
        pd.DataFrame
            Преобразованная таблица. Входные данные не изменяются.
        """
        pass
