from typing import Any, cast

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
from matplotlib.figure import Figure


class DistributionCalculator:
    """
    Калькулятор для расчета вероятностных статистик и
    визуализации распределений.

    Parameters
    ----------
    quantiles : list
        Список со значениями квантилей.
        Значения задаются долями от 0 до 1. По умолчанию используются
        0.01, 0.05, 0.25, 0.5, 0.75, 0.95 и 0.99.

    Notes
    -----
    Бесконечные значения заменяются на NaN. При расчёте характеристик
    пропуски учитываются в excluded и исключаются из соответствующих
    статистик. Нулевые значения сохраняются.
    """

    def __init__(self, quantiles: list = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]):
        self.quantiles = quantiles

    @staticmethod
    def _replace_infinite(data: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
        """
        Заменить бесконечности на NaN, сохранив строки и тип данных.

        Parameters
        ----------
        data : pd.Series | pd.DataFrame
            Столбец данных для рассчета характеристик
            или таблица с числовыми столбцами и уникальными названиями.
        """
        return data.replace([np.inf, -np.inf], np.nan)
    
    def calculate(self, data: pd.Series | pd.DataFrame) -> pd.DataFrame:
        """
        Рассчет статистик отдельно для каждого числового столбца.

        Parameters
        ----------
        data : pd.Series | pd.DataFrame
            Столбец данных для рассчета характеристик
            или таблица с числовыми столбцами и уникальными названиями.

        Returns
        -------
        pd.DataFrame
            Возвращает таблицу характеристик для столбца
            или нескольких столбцов

        Notes
        -----
        std — выборочное стандартное отклонение с ddof=1. Эксцесс
        определяется относительно нормального распределения, для которого
        он равен нулю. Для постоянного ряда skew и excess возвращаются
        как NaN; недостаток наблюдений также может давать NaN в статистиках.

        count учитывает только конечные значения. Если в столбце их нет,
        count равен нулю, а zero_share и описательные статистики — NaN.
        """
        if not isinstance(data, (pd.Series, pd.DataFrame)):
            raise TypeError("Ожидался pandas Series или DataFrame")

        frame = data.to_frame() if isinstance(data, pd.Series) else data

        if frame.empty:
            raise ValueError("Таблица не содержит данных для рассчета!")
        if not frame.columns.is_unique:
            raise ValueError("Названия столбцов не уникальны!")
        if any(dtype.kind not in "iuf" for dtype in frame.dtypes):
            raise TypeError("Ожидаются только числовые столбцы!")

        frame = self._replace_infinite(frame)
        result = frame.describe(percentiles=self.quantiles).T
        result["count"] = result["count"].astype(int)

        varying = frame.nunique() > 1
        result["skew"] = frame.skew().where(varying)
        result["excess"] = frame.kurt().where(varying)
        result["excluded"] = len(frame) - frame.count()
        result["zero_share"] = frame.eq(0).sum() / frame.count()

        return result

    def plot(
        self,
        data: pd.Series | pd.DataFrame,
        bins: int = 100,
        kde: bool = True,
        title: str = "",
        xlabel: str = "",
    ) -> Figure:
        """
        Построение гистограммы плотности распределения случайной величины.

        Parameters
        ----------
        data : pd.Series | pd.DataFrame
            Столбец данных для построения гистограммы
            в виде одного числового Series
        bins : int
            Количество бинов в гистограмме
            в виде положительного целого числа
        kde : bool
            Построение сглаженной оценки плотности поверх гистограммы
        title : str
            Заголовок рисунка
        xlabel : str
            Подпись оси X

        Returns
        -------
        Figure
            Рисунок с гистограммой плотности
        """
        if not isinstance(data, pd.Series):
            raise TypeError("Для графика необходим один pandas Series")
        if data.dtype.kind not in "iuf":
            raise TypeError("Переданная таблица содержит нечисловые значения!")
        if not isinstance(bins, int) or isinstance(bins, bool) or bins < 1:
            raise ValueError

        values = data.replace([np.inf, -np.inf], np.nan).dropna()
        if values.empty:
            raise ValueError("Отсутствуют значения для графика")

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(
            x=values, bins=bins, stat="density", kde=kde and values.nunique() > 1, ax=ax
        )
        ax.set_title(title)
        ax.set_xlabel(xlabel or str(data.name or "Value"))
        ax.set_ylabel("Density")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        return fig

    def plot_qq(
        self, data: pd.Series, title: str = "Q-Q plot", ylabel: str = ""
    ) -> Figure:
        """
        Сравнение квантилей ряда с нормальным распределением.

        Parameters
        ----------
        data : pd.Series
            Числовой ряд. NaN и бесконечные значения исключаются;
            нулевые значения сохраняются.
        title : str
            Заголовок рисунка. По умолчанию "Q-Q plot".
        ylabel : str
            Подпись оси Y. Если не задана, используется имя Series
            либо строка "Observed values".

        Returns
        -------
        Figure
            Рисунок с теоретическими квантилями нормального распределения
            по оси X, упорядоченными наблюдениями по оси Y
            и опорной прямой, подобранной методом наименьших квадратов.

        Raises
        ------
        TypeError
            Если data не является числовым Series.
        ValueError
            Если после очистки осталось меньше двух наблюдений
            или все конечные значения одинаковы.

        Notes
        -----
        Построение выполняется через scipy.stats.probplot с dist="norm"
        и подбором опорной прямой. В терминологии SciPy это normal
        probability plot. График служит визуальной диагностикой формы
        распределения и не является статистическим тестом нормальности.

        Значения по оси Y сохраняют единицы исходного ряда.
        Метод не вызывает plt.show() и не сохраняет рисунок в файл.
        """
        if not isinstance(data, pd.Series):
            raise TypeError("Для Q-Q графика нужен один pandas Series!")
        if data.dtype.kind not in "iuf":
            raise TypeError("Переданная таблица содержит нечисловые значения!")
        values = self._replace_infinite(data).dropna()
        if len(values) < 2 or values.nunique() < 2:
            raise ValueError("Нужны хотя бы два различных значения!")

        fig, ax = plt.subplots(figsize=(7, 7))
        sample = np.asarray(values.to_numpy(), dtype=np.float64)
        stats.probplot(x=sample, dist="norm", plot=cast(Any, ax))
        ax.set_title(title)
        ax.set_xlabel("Theoretical quantiles of normal distribution")
        ax.set_ylabel(ylabel or str(data.name or "Observed values"))
        ax.grid(alpha=0.25)
        fig.tight_layout()
        return fig
