import requests
import pandas as pd
import time
from functools import reduce
from typing import List, Optional, Dict
from pathlib import Path
from source.utils.logger import Logger


class DataLoader:
    """
    Загрузчик данных с MOEX ISS API.

    Parameters
    ----------
    tickers : List[str]
        Список тикеров для загрузки
    start_year : int
        Начальный год
    end_year : int
        Конечный год
    interval : int
        Интервал между сделками
    delay : float
        Пауза между HTTP-запросами
    save_dir : str
        Директория для сохранения
    filename : str
        Имя файла
    base_url : str
        Ссылка для парсинга
    """

    def __init__(
        self,
        tickers: List[str],
        start_year: int = 2014,
        end_year: int = 2025,
        interval: int = 10,
        delay: float = 0.01,
        save_dir: str = "data/raw/",
        filename: str = "all_prices.csv",
        # base_url : str = 'https://iss.moex.com/iss/engines/stock/markets/shares/securities'
        base_url: str = "https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities",
    ):
        self.tickers = tickers
        self.start_year = start_year
        self.end_year = end_year
        self.interval = interval
        self.delay = delay
        self.base_url = base_url
        self.logger = Logger("DataLoader")

        self._session = None
        self._data: Dict[str, pd.DataFrame] = {}
        self._save_dir = Path(save_dir)

        self.save_path = self._save_dir / filename

    def load_all(self, save_individual: bool = False) -> pd.DataFrame:
        """
        Загрузить все тикеры и объединить в DataFrame.

        Parameters
        ----------
        save_individual : bool
            Индикатор для сохранения тикера

        Returns
        -------
        pd.DataFrame
            Объединенный DataFrame с колонками date, {ticker}_close
        """

        self.logger.info(f"Загрузка {len(self.tickers)} тикеров")
        self._init_session()

        for ticker in self.tickers:
            self.logger.step("LOADING", f"Загружается {ticker}")
            df_ticker = self._load_ticker(ticker)
            self._data[ticker] = df_ticker
            if save_individual:
                self._save_ticker(df_ticker, ticker)
            self.logger.success(f"{ticker}: {len(df_ticker):,} строк")

        self._close_session()
        merged_df = self._merge_data()

        if self.save_path:
            self._save(merged_df)

        return merged_df

    def _init_session(self) -> None:
        """
        Создать HTTP-сессию для запросов к MOEX.

        Notes
        -----
        Ускоряет запросы
        """

        self._session = requests.Session()

    def _close_session(self) -> None:
        """
        Закрыть HTTP-сессию.

        Notes
        -----
        Вызывается после завершения загрузки, чтобы освободить ресурсы.
        """

        if self._session:
            self._session.close()
            self._session = None

    def _load_ticker(self, ticker: str) -> pd.DataFrame:
        """
        Загрузить один тикер.

        Parameters
        ----------
        ticker : str
            Название тикера (например, 'VTBR')

        Returns
        -------
        pd.DataFrame
            DataFrame с колонками date, {ticker}_close

        Notes
        -----
        Загрузка идет постранично: по 500 записей за запрос
        """

        all_data = []
        columns = None
        url = f"{self.base_url}/{ticker}/candles.json"

        for year in range(self.start_year, self.end_year + 1):
            params = {
                "from": f"{year}-01-01",
                "till": f"{year}-12-31",
                "interval": self.interval,
                "start": 0,
            }

            year_data = self._fetch_year(url, params, columns)
            if columns is None and year_data:
                columns = year_data["columns"]

            if year_data["data"]:
                all_data.extend(year_data["data"])

        df = self._to_dataframe(all_data, columns, ticker)

        return df

    def _fetch_year(self, url: str, params: dict, columns: Optional[List[str]]) -> dict:
        """
        Загрузить один год данных с пагинацией и retry-логикой.

        Parameters
        ----------
        url : str
            URL для запроса
        params : dict
            Параметры запроса (from, till, interval, start)
        columns : Optional[List[str]]
            Названия колонок (None при первом запросе)

        Returns
        -------
        dict
            {'columns': [...], 'data': [...]}

        Raises
        ------
        RuntimeError
            Если нет интернета или слишком много retry
        ValueError
            Если HTTP-ошибка, не связанная с 429
        """

        year_data = []
        year_columns = None
        max_tries = 3
        retry_count = 0
        while True:
            time.sleep(self.delay)
            try:
                response = self._session.get(url, params=params, timeout=30)
                response.raise_for_status()

            except requests.exceptions.ConnectionError:
                self.logger.error("Нет интернет соединения!")
                raise RuntimeError("Нет интернет соединения!") from None

            except requests.exceptions.HTTPError as excep:
                if excep.response.status_code == 429:
                    retry_count += 1
                    self.logger.warning(f"HTTP 429: попытка {retry_count}/{max_tries}")
                    if retry_count > max_tries:
                        self.logger.error(f"429: превышено {max_tries} попыток")
                        raise RuntimeError(
                            f"Много запросов 429 ошибка после {max_tries} попыток"
                        ) from excep
                    time.sleep(5)
                    continue
                else:
                    self.logger.error(f"HTTP ошибка: {excep}")
                    raise ValueError(f"HTTP ошибка: {excep}") from excep

            except requests.exceptions.Timeout:
                retry_count += 1
                self.logger.warning(f"Timeout: пробую снова {retry_count}/{max_tries}")
                if retry_count > max_tries:
                    self.logger.error(f"Timeout после {max_tries} попыток")
                    raise RuntimeError(f"Timeout после {max_tries} попыток")
                self.logger.warning(
                    f"Timeout. Попытка {retry_count}/{max_tries} через 2 секунды"
                )
                time.sleep(2)
                continue

            except requests.exceptions.RequestException as excep:
                retry_count += 1
                if retry_count > max_tries:
                    self.logger.error(
                        f"Ошибка запроса после {max_tries} попыток: {excep}"
                    )
                    raise RuntimeError(f"Ошибка после {max_tries} попыток") from excep
                self.logger.warning(
                    f"Ошибка запроса: {excep}. Попытка {retry_count}/{max_tries}"
                )
                time.sleep(2)
                continue

            data = response.json()
            candles = data["candles"]
            if not candles["data"]:
                break

            if year_columns is None:
                year_columns = candles["columns"]

            year_data.extend(candles["data"])
            params["start"] += len(candles["data"])

            if len(candles["data"]) < 500:
                break

        return {"columns": year_columns, "data": year_data}

    def _to_dataframe(
        self, data: List, columns: List[str], ticker: str
    ) -> pd.DataFrame:
        """
        Преобразовать сырые данные MOEX в DataFrame

        Parameters
        ----------
        data : List
            Список списков с данными (open, close, ..)
        columns : List[str]
            Названия колонок из ответа API
        ticker : str
            Название тикера

        Returns
        -------
        pd.DataFrame
            DataFrame с колонками date, {ticker}_close
        """

        if not data or columns is None:
            self.logger.warning(f"{ticker}: данные не найдены")
            return pd.DataFrame(columns=["date", f"{ticker}_close"])
        df = pd.DataFrame(data, columns=columns)
        df = df[["end", "close"]].rename(
            columns={"end": "date", "close": f"{ticker}_close"}
        )
        df["date"] = pd.to_datetime(df["date"])
        df = df.drop_duplicates(subset="date", keep="last")
        df = df.sort_values("date").reset_index(drop=True)
        return df

    def _merge_data(self) -> pd.DataFrame:
        """
        Объединить данные всех тикеров в один DataFrame.

        Returns
        -------
        pd.DataFrame
            Объединённый DataFrame с колонками date, {ticker}_close

        Raises
        ------
        ValueError
            Если данные пустые или не хватает колонок
        """

        if not self._data:
            self.logger.error("Нет данных для объединения")
            raise ValueError("Нет данных для объединения")

        dfs = list(self._data.values())

        merged = reduce(
            lambda left, right: pd.merge(left, right, on="date", how="outer"), dfs
        )
        merged = merged.sort_values("date").reset_index(drop=True)

        columns = ["date"] + [f"{tmp}_close" for tmp in self.tickers]

        missing_cols = [col for col in columns if col not in merged.columns]

        if missing_cols:
            self.logger.error(f"Не хватает колонок: {missing_cols}")
            raise ValueError(
                f"Не хватает колонок: {missing_cols}. Доступные колонки: {list(merged.columns)}"
            )
        merged = merged[columns]

        return merged

    def _save(self, df: pd.DataFrame) -> None:
        """
        Сохранить объединенный DataFrame в csv формат

        Parameters
        ----------
        df : pd.DataFrame
            Данные для сохранения
        """

        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.save_path, index=False)

    def _save_ticker(self, ticker_df: pd.DataFrame, ticker: str) -> None:
        """
        Сохранить данные одного тикера в отдельный csv

        Parameters
        ----------
        ticker_df : pd.DataFrame
            Данные для тикера
        ticker : str
            Название тикера
        """
        path = self._save_dir / f"{ticker}.csv"
        ticker_df.to_csv(path, index=False)
