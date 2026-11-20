import numpy as np
import pandas as pd

from source.data_processors.steps.base import ProcessingStep
from source.utils.logger import Logger


class ClipReturns(ProcessingStep):
    """
    Ограничение абсолютной величины доходностей.

    Parameters
    ----------
    max_return : float
        Конечный положительный порог в процентных пунктах.
        Например, 5 означает границы от -5% до +5%.
    """

    def __init__(self, max_return: float = 5.0):
        if not np.isfinite(max_return) or max_return <= 0:
            raise ValueError("max_return должен быть конечным положительным числом")

        self.max_return = max_return
        self.logger = Logger("ClipReturns")

    def transform(
        self,
        data: pd.DataFrame,
        ticker: str = "",
    ) -> pd.DataFrame:
        """
        Ограничить числовой столбец returns.

        Возвращает копию таблицы. Строки, индекс, остальные
        столбцы и пропуски сохраняются.
        """
        data = data.copy()
        extreme = data["returns"].abs() > self.max_return
        data["returns"] = data["returns"].clip(
            lower=-self.max_return,
            upper=self.max_return,
        )

        count = int(extreme.sum())
        if count:
            self.logger.warning(
                f"{ticker}: обрезано {count} доходностей до ±{self.max_return}%"
            )

        return data
