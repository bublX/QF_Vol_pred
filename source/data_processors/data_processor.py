import pandas as pd
import numpy as np
from source.utils.logger import Logger
from typing import Tuple, List, Dict
from pathlib import Path


class DataProcessor:
    """
    Обработчик полученных данных с MOEX ISS API.

    Parameters
    ----------
    input_dir : str
        Директория для чтения
    output_dir : str
        Директория для сохранения
    min_bars_per_day : int
        Минимальное допустимое число сделок в день
    max_return : float
        Максимальное допустимое значение доходностей в процентах
    hampel_const : float
        Константа для фильтра Хампеля
    hampel_window : int
        Полуширина окна в барах
    hampel_sigma : float
        Порог для фильтра в сигмах
    interval : int
        Интервал длины свечи в минутах
    """

    def __init__(
        self,
        input_dir: str = "data/raw/",
        output_dir: str = "data/processed/",
        min_bars_per_day: int = 40,
        max_return: float = 5.0,
        hampel_const: float = 1.4826,
        hampel_window: int = 5,
        hampel_sigma: float = 3.0,
        interval: int = 10,
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.min_bars_per_day = min_bars_per_day
        self.max_return = max_return
        self.hampel_const = hampel_const
        self.hampel_window = hampel_window
        self.hampel_sigma = hampel_sigma
        self.interval = interval
        self.logger = Logger("DataProcessor")

        self._results = {}

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

        ticker_df = self._prepare_data(ticker_df)
        ticker_df = self._filter_trading_hours(ticker_df)
        ticker_df = self._filter_valid_dates(ticker_df)
        ticker_df = self._calculate_returns(ticker_df)
        ticker_df = self._filter_returns_hampel(ticker_df, ticker)
        ticker_df = self._clip_extreme_returns(ticker_df, ticker)

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
        Выровнять свечи тикеров по слотам интервала (например, 10 минут): одна строка на слот.

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
        df = df.sort_values("date").reset_index(drop=True)
        freq = pd.Timedelta(minutes=self.interval)
        df["date"] = df["date"].dt.floor(freq=freq) + freq - pd.Timedelta(seconds=1)

        closes = [col for col in df.columns if col.endswith("_close")]
        duplicates = (df.groupby("date")[closes].count() > 1).sum()
        duplicates = duplicates[duplicates > 0]
        if not duplicates.empty:
            self.logger.warning(
                f"Найдены дубликаты по слотам: {duplicates.to_dict()}, берем только последнюю."
            )

        aligned = df.groupby("date", as_index=False).last()
        self.logger.info(
            f"Выравнивание по {self.interval} минут: {len(df)} строк - {len(aligned)} слотов."
        )

        return aligned

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
        ticker_df = ticker_df.dropna()

        return ticker_df

    @staticmethod
    def _prepare_data(ticker_df: pd.DataFrame) -> pd.DataFrame:
        """
        Подготовить данные: преобразовать date, отсортировать.

        Parameters
        ----------
        ticker_df : pd.DataFrame
            Данные тикера

        Returns
        -------
        pd.DataFrame
            Подготовленный DataFrame
        """

        ticker_df = ticker_df.copy()

        ticker_df["date"] = pd.to_datetime(ticker_df["date"])
        ticker_df = ticker_df.sort_values("date").reset_index(drop=True)

        return ticker_df

    def _filter_trading_hours(
        self,
        ticker_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Оставить только бары в торговые часы (10:00–18:50) и только будние дни.

        Parameters
        ----------
        ticker_df : pd.DataFrame
            Данные тикера

        Returns
        -------
        pd.DataFrame
            Отфильтрованный DataFrame
        """

        ticker_df = ticker_df.copy()

        hour = ticker_df["date"].dt.hour
        minute = ticker_df["date"].dt.minute

        weekday = ticker_df["date"].dt.day_of_week < 5
        mask = (hour >= 10) & ((hour < 18) | ((hour == 18) & (minute <= 50))) & weekday
        ticker_df = ticker_df[mask].reset_index(drop=True)

        return ticker_df

    def _filter_valid_dates(self, ticker_df: pd.DataFrame) -> pd.DataFrame:
        """
        Оставить только дни с достаточным числом баров.

        Parameters
        ----------
        ticker_df : pd.DataFrame
            Данные тикера

        Returns
        -------
        pd.DataFrame
            Отфильтрованный DataFrame
        """

        ticker_df = ticker_df.copy()

        counts = ticker_df.groupby(ticker_df["date"].dt.date).size()
        valid_dates = counts[counts >= self.min_bars_per_day].index

        mask = ticker_df["date"].dt.date.isin(valid_dates)
        ticker_df = ticker_df[mask].reset_index(drop=True)

        return ticker_df

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

    def _filter_returns_hampel(
        self, ticker_df: pd.DataFrame, ticker: str = ""
    ) -> pd.DataFrame:
        """
        Применить фильтр Хампеля к доходностям.

        Parameters
        ----------
        ticker_df : pd.DataFrame
            Данные тикера
        ticker : str
            Название тикера

        Returns
        -------
        pd.DataFrame
            DataFrame с очищенными доходностями
        """

        ticker_df = ticker_df.copy()
        returns = ticker_df["returns"].to_numpy(dtype=float, copy=True)
        flags = np.zeros(len(returns), dtype=bool)
        daily_indices = ticker_df.groupby(ticker_df["date"].dt.date).indices.values()
        for idx in daily_indices:
            returns[idx], flags[idx] = self._filter_hampel(
                series=returns[idx],
                window_size=self.hampel_window,
                sigma=self.hampel_sigma,
                const=self.hampel_const,
            )

        ticker_df["returns"] = returns
        n = int(flags.sum())
        self.logger.info(f"{ticker}: Hampel filter - {n} замен ({n / len(flags):.3%})")
        return ticker_df

    def _calculate_returns(self, ticker_df: pd.DataFrame) -> pd.DataFrame:
        """
        Рассчитать логарифмические доходности.

        Parameters
        ----------
        ticker_df : pd.DataFrame
            Данные тикера

        Returns
        -------
        pd.DataFrame
            DataFrame с колонкой returns
        """

        ticker_df = ticker_df.copy()
        ticker_df["returns"] = (
            np.log(
                ticker_df["close"]
                / ticker_df.groupby(ticker_df["date"].dt.date)["close"].shift(1)
            )
            * 100
        )

        return ticker_df

    def _clip_extreme_returns(
        self, ticker_df: pd.DataFrame, ticker: str = ""
    ) -> pd.DataFrame:
        """
        Винсоризация экстремальных доходностей (|returns| > max_return).

        Parameters
        ----------
        ticker_df : pd.DataFrame
            Данные тикера
        ticker : str
            Название тикера

        Returns
        -------
        pd.DataFrame
            DataFrame с обрезанными доходностями
        """

        ticker_df = ticker_df.copy()
        ticker_df = ticker_df.dropna().reset_index(drop=True)
        extreme_values = ticker_df["returns"].abs() > self.max_return
        ticker_df["returns"] = ticker_df["returns"].clip(
            -self.max_return, self.max_return
        )
        n = int(extreme_values.sum())
        if n != 0:
            self.logger.warning(
                f"{ticker}: обрезано {n} доходностей до ±{self.max_return}%"
            )
        return ticker_df

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
        Объединить RV всех тикеров в одну таблицу.

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
            rv_matrix = pd.merge(rv_matrix, rv, on="date", how="outer")

        rv_matrix = rv_matrix.sort_values("date").reset_index(drop=True)
        rv_matrix = rv_matrix.ffill()
        rv_matrix = rv_matrix.dropna().reset_index(drop=True)

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
