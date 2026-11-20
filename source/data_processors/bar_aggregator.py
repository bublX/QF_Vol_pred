import pandas as pd
import datetime as dt
from math import ceil
from source.utils.logger import Logger
from typing import cast


class BarAggregator:
    """
    Агрегация цен закрытия в данный минутный интервал. Позволяет конструировать интервалы
    желаемой длительности.

    Parameters
    ----------
    source_interval : int
        Исходный интервал для аггрегирования
    interval : int
        Желаемый интервал после агрегирования
    min_bar_coverage : float
        Минимальная доля исходных свечей внутри итогового бара
    session_start : str
        Начало сессии в формате HH:MM, включительно.
    session_end : str
        Конец сессии в формате HH:MM, исключительно.
        Поддерживается одна сессия внутри календарного дня.

    Notes
    -----
    Исходный интервал обязан быть делителем желаемого интервала, иначе
    агрегирование невозможно. Выход содержит полную сетку баров, а интервалы
    с недостаточным покрытием получают NaN.

    Examples
    --------
    >>> aggregator = BarAggregator(
    ...     interval=15,
    ...     session_start="10:00",
    ...     session_end="18:50",
    ... )
    >>> bars = aggregator.transform(prices)
    prices - исходный DataFrame
    """

    def __init__(
        self,
        source_interval: int = 1,
        interval: int = 10,
        min_bar_coverage: float = 0.0,
        session_start: str = "10:00",
        session_end: str = "18:50",
    ):
        if source_interval <= 0 or interval < source_interval:
            raise ValueError(
                "Интервалы должны быть положительными; итоговый не меньше исходного"
            )
        if interval % source_interval != 0:
            raise ValueError("Итоговый интервал должен быть кратен исходному")
        if not 0 <= min_bar_coverage <= 1:
            raise ValueError("min_bar_coverage должен быть в диапазоне [0, 1]")

        self.session_start = dt.datetime.strptime(session_start, "%H:%M").time()
        self.session_end = dt.datetime.strptime(session_end, "%H:%M").time()

        start_minutes = self.session_start.hour * 60 + self.session_start.minute
        if start_minutes % source_interval != 0:
            raise ValueError("Начало сессии должно совпадать с сеткой исходных свечей")
        end_minutes = self.session_end.hour * 60 + self.session_end.minute
        if end_minutes <= start_minutes:
            raise ValueError(
                "Конец сессии должен быть позже начала в пределах одного дня"
            )

        self.start_offset = pd.Timedelta(minutes=start_minutes)
        self.end_offset = pd.Timedelta(minutes=end_minutes)
        self.expected_bars_per_day = (end_minutes - start_minutes) // interval
        if self.expected_bars_per_day < 1:
            raise ValueError("В сессии должен помещаться хотя бы один полный бар")

        self.source_interval = source_interval
        self.interval = interval
        self.min_bar_coverage = min_bar_coverage
        self.logger = Logger("BarAggregator")

    def _filter_trading_hours(
        self, ticker_df: pd.DataFrame, ticker: str = ""
    ) -> pd.DataFrame:
        """
        Оставить только бары в торговые часы (session_start–session_end) и только будние дни.

        Parameters
        ----------
        ticker_df : pd.DataFrame
            Данные тикера
        ticker : str
            Название тикера

        Returns
        -------
        pd.DataFrame
            Отфильтрованный DataFrame
        """

        ticker_df = ticker_df.copy()

        weekday = ticker_df["date"].dt.day_of_week < 5
        time = ticker_df["date"].dt.time
        mask = (time >= self.session_start) & (time < self.session_end) & weekday
        ticker_df = cast(pd.DataFrame, ticker_df.loc[mask]).reset_index(drop=True)

        self.logger.info(f"{ticker}: оставлены бары в торговые часы и будние дни")
        return ticker_df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Возвращает таблицу date, '{ticker}_close' после агрегации.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame с колонками date, {ticker}_close

        Returns
        -------
        pd.DataFrame
            DataFrame с выровненными слотами (10:50:00 - 10:59:59 -> 10:59:59)

        Notes
        -----
        Если у тикера в слоте несколько свечей, то берется последняя по времени:
        её close будет считаться ценой закрытия слота.
        """

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date", kind="stable").reset_index(drop=True)
        df = self._filter_trading_hours(df)

        closes = [col for col in df.columns if col.endswith("_close")]
        if df.empty:
            return cast(pd.DataFrame, df[["date", *closes]])

        day = df["date"].dt.normalize()
        source_freq = pd.Timedelta(minutes=self.source_interval)
        source_slots = (day + ((df["date"] - day) // source_freq) * source_freq).rename(
            "date"
        )

        source_groups = df.groupby(source_slots)[closes]
        duplicates = (source_groups.count() > 1).sum()
        duplicates = duplicates[duplicates > 0]
        if not duplicates.empty:
            self.logger.warning(
                f"Дубликаты исходных свечей: {duplicates.to_dict()}; "
                "используем последнюю цену по времени."
            )

        df = source_groups.last().reset_index()

        day = df["date"].dt.normalize()
        days = day.drop_duplicates()
        target_freq = pd.Timedelta(minutes=self.interval)
        session_start = day + self.start_offset
        session_end = day + self.end_offset

        slot_start = (
            session_start + ((df["date"] - session_start) // target_freq) * target_freq
        )
        slot_end = slot_start + target_freq
        fits_session = slot_end <= session_end

        df = df.loc[fits_session].copy()
        df["date"] = slot_end.loc[fits_session] - pd.Timedelta(seconds=1)

        groups = df.groupby("date")[closes]
        required = max(
            1, ceil((self.interval // self.source_interval) * self.min_bar_coverage)
        )
        aligned = groups.last().where(groups.count() >= required)

        offsets = pd.timedelta_range(
            start=self.start_offset + target_freq - pd.Timedelta(seconds=1),
            periods=self.expected_bars_per_day,
            freq=target_freq,
        )
        grid = pd.DatetimeIndex(
            [d + offset for d in days for offset in offsets], name="date"
        )

        result = aligned.reindex(grid).reset_index()
        self.logger.info(
            f"Агрегация {self.source_interval} → {self.interval} минут: "
            f"{len(result)} слотов."
        )
        return result
