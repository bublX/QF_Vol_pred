import pandas as pd
from source.utils.logger import Logger
from source.data_processors.bar_aggregator import BarAggregator
from source.data_processors.steps.base import ProcessingStep
from typing import List, Dict
from pathlib import Path
from math import ceil


class DataProcessor:
    """
    Обработчик полученных данных с MOEX ISS API.

    Parameters
    ----------
    input_dir : str
        Директория для чтения
    output_dir : str
        Директория для сохранения
    min_day_coverage : float
        Минимальная доля заполненных итоговых баров за сессию,
        до заполнения пропусков. Значение в диапазоне (0, 1].
    min_bar_coverage : float
        Минимальная доля исходных свечей внутри итогового бара.
        Значение в диапазоне [0, 1]. Даже при 0 требуется
        хотя бы одна исходная свеча.
    source_interval : int
        Длительность исходных свечей в минутах
    interval : int
        Интервал желаемой длины свечи в минутах.
        Должен быть кратным source_interval.
    session_start : str
        Начало сессии в формате HH:MM, включительно.
        Должно совпадать с сеткой исходных свечей.
    session_end : str
        Конец сессии в формате HH:MM, невключительно.
        Должен быть позже session_start в пределах одного дня.
    steps : list[ProcessingStep] | None
        Дополнительные шаги обработки доходностей в порядке выполнения.
        None и пустой список отключают дополнительные шаги.
    price_steps : list[ProcessingStep] | None
        Дополнительные шаги до расчета доходностей.
        None и пустой список отключают дополнительные шаги.
    returns_calculator : ProcessingStep
        Обязательный шаг расчёта доходностей. Передаётся по имени.
        Выполняется после price_steps и перед steps.
        Должен формировать столбец returns.
    Raises
    ------
    ValueError
        Некорректные интервалы, границы сессии или доли покрытия.
        Также возникает, если в сессии помещается меньше двух
        полных итоговых баров.

    Notes
    -----
    Агрегация выполняется через BarAggregator. Неполный последний
    интервал сессии отбрасывается.

    Минимальное число заполненных баров за день вычисляется как
    max(2, ceil(expected_bars_per_day * min_day_coverage)).
    Количество баров определяется границами сессии и interval.

    Подготовка → проверка покрытия дня → price_steps →
    returns_calculator → steps → удаление строк без close или returns.
    """

    def __init__(
        self,
        input_dir: str = "data/raw/",
        output_dir: str = "data/processed/",
        min_day_coverage: float = 0.75,
        min_bar_coverage: float = 0.0,
        source_interval: int = 1,
        interval: int = 10,
        session_start: str = "10:00",
        session_end: str = "18:50",
        steps: list[ProcessingStep] | None = None,
        price_steps: list[ProcessingStep] | None = None,
        *,
        returns_calculator: ProcessingStep,
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.missing_days = {}
        self.logger = Logger("DataProcessor")

        self._results = {}

        self.aggregator = BarAggregator(
            source_interval=source_interval,
            interval=interval,
            min_bar_coverage=min_bar_coverage,
            session_start=session_start,
            session_end=session_end,
        )

        self.returns_calculator = returns_calculator
        self.price_steps = tuple(price_steps) if price_steps is not None else ()

        if not 0 < min_day_coverage <= 1:
            raise ValueError("min_day_coverage должен быть в диапазоне (0, 1]")

        expected_bars = self.aggregator.expected_bars_per_day
        if expected_bars < 2:
            raise ValueError("Для расчёта RV нужны хотя бы два бара за сессию")

        self.expected_bars_per_day = expected_bars
        self.min_day_coverage = min_day_coverage
        self.min_bars_per_day = max(2, ceil(expected_bars * min_day_coverage))

        self.steps = tuple(steps) if steps is not None else ()

    def process_all(
        self, input_file: str = "all_prices.csv"
    ) -> Dict[str, pd.DataFrame]:
        """
        Обработать все тикеры из файла.

        Parameters
        ----------
        input_file : str
            Имя файла с сырыми данными

        Returns
        -------
        Dict[str, pd.DataFrame]
            Словарь {тикер: обработанный DataFrame}
        """

        self.logger.info(f"Обработка {input_file}")

        df_all = self._load_data(input_file)
        df_all = self._align_to_slots(df_all)
        tickers = self._get_tickers_names(df_all)

        for ticker in tickers:
            self.logger.step("PROCESS", f"Обрабатывается {ticker}")
            df_ticker = self._get_ticker_data(df_all, ticker)
            df_processed_ticker = self.process_ticker(df_ticker, ticker)
            self._results[ticker] = df_processed_ticker
            self._save_rv(df_processed_ticker, ticker)
            self.logger.success(f"{ticker}: RV сохранен")

        self._save_missing_days()
        self._save_rv_matrix(tickers)
        self._save_profitabilities_matrix(tickers)
        self.logger.success("Матрицы сохранены")

        return self._results

    def process_ticker(self, ticker_df: pd.DataFrame, ticker: str = "") -> pd.DataFrame:
        """
        Обработать данные одного тикера.

        Parameters
        ----------
        ticker_df : pd.DataFrame
            Данные тикера (date, close)
        ticker : str
            Название тикера

        Returns
        -------
        pd.DataFrame
            Обработанный DataFrame с колонками date, close, returns
        """

        ticker_df = self._prepare_data(ticker_df, ticker)
        ticker_df = self._filter_valid_dates(ticker_df, ticker)
        for step in self.price_steps:
            ticker_df = step.transform(ticker_df, ticker)

        ticker_df = self.returns_calculator.transform(ticker_df, ticker)

        for step in self.steps:
            ticker_df = step.transform(ticker_df, ticker)

        ticker_df = ticker_df.dropna(subset=["close", "returns"]).reset_index(drop=True)

        return ticker_df

    def _load_data(self, input_file: str) -> pd.DataFrame:
        """
        Загрузить данные из csv-файла.

        Parameters
        ----------
        input_file : str
            Имя файла

        Returns
        -------
        pd.DataFrame
            Загруженные данные

        Raises
        ------
        FileNotFoundError
            Если файл не найден
        ValueError
            Если формат файла не CSV
        """

        path = self.input_dir / input_file

        if not path.exists():
            self.logger.error(f"Файл не найден: {path}")
            raise FileNotFoundError(f"Файл не найден: {path}")

        if path.suffix == ".csv":
            df = pd.read_csv(path)
        else:
            self.logger.error(f"Неверный формат: {path.suffix}")
            raise ValueError(f"Неверный формат: {path.suffix}")

        return df

    def _align_to_slots(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Агрегирование исходных цен через BarAggregator

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame с колонками date, {ticker}_close

        Returns
        -------
        pd.DataFrame
            Таблица с метками конца итоговых слотов минус одна секунда.
            Например, при interval=10 и session_start="10:00" интервал
            [10:50:00, 11:00:00) получает метку 10:59:59.

        Notes
        -----
        BarAggregator выбирает последнее непустое значение close по времени
        отдельно для каждого тикера и проверяет покрытие итогового бара.
        Сетка строится от session_start; последний неполный интервал
        сессии отбрасывается.
        """
        return self.aggregator.transform(df)

    @staticmethod
    def _get_tickers_names(df: pd.DataFrame) -> List[str]:
        """
        Извлечь названия тикеров из колонок DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame с колонками date, {ticker}_close

        Returns
        -------
        List[str]
            Список тикеров
        """

        ticker_cols = [col for col in df.columns if col.endswith("_close")]
        tickers = [col.replace("_close", "") for col in ticker_cols]

        return tickers

    @staticmethod
    def _get_ticker_data(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Извлечь данные одного тикера.

        Parameters
        ----------
        df : pd.DataFrame
            Общий DataFrame
        ticker : str
            Название тикера

        Returns
        -------
        pd.DataFrame
            DataFrame с колонками date, close
        """

        ticker_df = df[["date", f"{ticker}_close"]].copy()
        ticker_df = ticker_df.rename(columns={f"{ticker}_close": "close"})

        return ticker_df

    def _prepare_data(self, ticker_df: pd.DataFrame, ticker: str = "") -> pd.DataFrame:
        """
        Подготовить данные: преобразовать date, отсортировать.

        Parameters
        ----------
        ticker_df : pd.DataFrame
            Данные тикера
        ticker : str
            Название тикера

        Returns
        -------
        pd.DataFrame
            Подготовленный DataFrame
        """

        ticker_df = ticker_df.copy()

        ticker_df["date"] = pd.to_datetime(ticker_df["date"])
        ticker_df = ticker_df.sort_values("date").reset_index(drop=True)

        self.logger.info(
            f"{ticker}: данные подготовлены (дата преобразована, отсортировано)"
        )
        return ticker_df

    def _filter_valid_dates(
        self, ticker_df: pd.DataFrame, ticker: str = ""
    ) -> pd.DataFrame:
        """
        Оставить только дни с достаточным числом баров.

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

        counts = ticker_df.groupby(ticker_df["date"].dt.date)["close"].count()
        valid_dates = counts[counts >= self.min_bars_per_day].index
        missing_dates = counts[counts < self.min_bars_per_day]
        self.missing_days[ticker] = missing_dates
        if not missing_dates.empty:
            self.logger.warning(
                f"{ticker}: пропущено {len(missing_dates)} дней с < {self.min_bars_per_day} баров"
            )
        if valid_dates.empty:
            self.logger.error(f"{ticker}: нет дней с >= {self.min_bars_per_day} баров")
            raise ValueError(f"{ticker}: нет дней с >= {self.min_bars_per_day} баров")

        mask = ticker_df["date"].dt.date.isin(valid_dates)
        ticker_df = ticker_df[mask].reset_index(drop=True)

        self.logger.info(
            f"{ticker}: оставлено {len(valid_dates)} дней с >= {self.min_bars_per_day} баров ({len(ticker_df)} строк)"
        )
        return ticker_df

    def _save_missing_days(self) -> None:
        """
        Сохранить пропущенные дни в CSV.

        Notes
        -----
        Дни, в которые биржа была закрыта целиком, не попадают, так как в этом случае нет ни одной строки.
        Сохраняются только дни, в которые было меньше min_bars_per_day баров.
        """

        rows = [
            {"ticker": ticker, "date": date, "bars": bars}
            for ticker, missing in self.missing_days.items()
            for date, bars in missing.items()
        ]
        missing_days = pd.DataFrame(rows, columns=["ticker", "date", "bars"])
        missing_days.to_csv(self.output_dir / "missing_days.csv", index=False)

        self.logger.info(
            f"Исключённые дни: {len(missing_days)} тикер-дней, "
            f"{missing_days['date'].nunique()} дат -> missing_days.csv"
        )

    def _save_rv(self, ticker_df: pd.DataFrame, ticker: str) -> None:
        """
        Рассчитать RV и сохранить в CSV.

        Parameters
        ----------
        ticker_df : pd.DataFrame
            Данные тикера
        ticker : str
            Название тикера
        """

        rv = (
            ticker_df.groupby(ticker_df["date"].dt.date)["returns"]
            .apply(lambda x: (x**2).sum())
            .reset_index()
        )
        rv.columns = ["date", "RV"]
        rv["date"] = pd.to_datetime(rv["date"])

        path = self.output_dir / f"RV/{ticker}_RV.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        rv.to_csv(path, index=False)

    def _save_rv_matrix(self, tickers: List[str]) -> None:
        """
        Объединить RV всех тикеров в одну таблицу (только те, где есть данные у всех тикеров).

        Parameters
        ----------
        tickers : List[str]
            Список тикеров
        """

        rv_list = []

        for ticker in tickers:
            rv = pd.read_csv(self.output_dir / f"RV/{ticker}_RV.csv")
            rv = rv.rename(columns={"RV": f"{ticker}_RV"})
            rv_list.append(rv)

        rv_matrix = rv_list[0]
        for rv in rv_list[1:]:
            rv_matrix = pd.merge(rv_matrix, rv, on="date", how="inner")

        rv_matrix = rv_matrix.sort_values("date").reset_index(drop=True)
        all_dates = pd.concat([rv["date"] for rv in rv_list]).nunique()
        dropped = all_dates - len(rv_matrix)
        if dropped:
            self.logger.warning(
                f"rv_matrix: исключено {dropped} дат, где RV есть не у всех тикеров (см. missing_days.csv)"
            )

        rv_matrix.to_csv(self.output_dir / "rv_matrix.csv", index=False)

    def _save_profitabilities_matrix(self, tickers: List[str]) -> None:
        """
        Объединить доходности всех тикеров в одну таблицу.

        Parameters
        ----------
        tickers : List[str]
            Список тикеров
        """

        profit_list = []

        for ticker in tickers:
            profits = self._results[ticker][["date", "returns"]].copy()
            profits = profits.rename(columns={"returns": f"{ticker}_returns"})
            profit_list.append(profits)

        profit_matrix = profit_list[0]
        for profit in profit_list[1:]:
            profit_matrix = pd.merge(profit_matrix, profit, on="date", how="inner")

        profit_matrix.to_csv(self.output_dir / "profit_matrix.csv", index=False)
