import pandas as pd
import numpy as np
from typing import Tuple, Optional, List, Dict
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
    fat_finger_window : int
        Окно для обработки fat-finger ошибок
    fat_finger_threshold : float
        Порог для обработки fat-finger ошибок
    max_return : float 
        Максимальное допустимое значение доходностей
    fat_finger_type : str 
        Способ обработки fat-finger ошибок (standart или hampel)
    fat_finger_sigma : float
        Sigma значение для фильтра Хампеля
    hampel_const : float 
        Константа для фильтра Хампеля 
    use_rv_filter : bool 
        Флаг для использования фильтра Хампеля

    Notes
    -----
    Фильтр `standart` заменяет выбросы на медиану окна.
    Фильтр `hampel` использует MAD для адаптивного порога.
    """
    def __init__(
        self,
        input_dir : str = 'data/raw/',
        output_dir : str = 'data/processed/',
        min_bars_per_day : int = 40,
        fat_finger_window : int = 11,
        fat_finger_threshold : float = 0.03,
        max_return : float = 15.0,
        fat_finger_type : str = 'standart',
        fat_finger_sigma : float = 3.0,
        hampel_const : float = 1.4826,
        use_rv_filter : bool = False
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.min_bars_per_day = min_bars_per_day
        self.fat_finger_window = fat_finger_window
        self.fat_finger_threshold = fat_finger_threshold
        self.max_return = max_return
        self.fat_finger_type = fat_finger_type
        self.fat_finger_sigma = fat_finger_sigma
        self.hampel_const = hampel_const
        self.use_rv_filter = use_rv_filter

        self._results = {}

    def process_all(
        self,
        input_file : str = 'all_prices.csv'
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

        df_all = self._load_data(input_file)
        tickers = self._get_tickers_names(df_all)

        for ticker in tickers:
            df_ticker = self._get_ticker_data(df_all, ticker)
            df_processed_ticker = self.process_ticker(df_ticker, ticker)
            self._results[ticker] = df_processed_ticker
            self._save_rv(df_processed_ticker, ticker)
            
        self._save_rv_matrix(tickers)
        self._save_profitabilities_matrix(tickers)

        return self._results

    def process_ticker(
        self,
        ticker_df : pd.DataFrame,
        save_path : Optional[str] = None
    ) -> pd.DataFrame:
        """
        Обработать данные одного тикера.
        
        Parameters
        ----------
        ticker_df : pd.DataFrame
            Данные тикера (date, close)
        save_path : Optional[str]
            Путь для сохранения (при желании)
        
        Returns
        -------
        pd.DataFrame
            Обработанный DataFrame с колонками date, close, processed_close, returns
        """

        ticker_df = self._prepare_data(ticker_df)
        ticker_df = self._filter_trading_hours(ticker_df)
        ticker_df = self._filter_valid_dates(ticker_df)
        ticker_df = self._detect_fat_finger(ticker_df)
        ticker_df = self._calculate_returns(ticker_df)
        ticker_df = self._filter_extreme_returns(ticker_df)

        return ticker_df
    
    def _load_data(
        self,
        input_file : str
    ) -> pd.DataFrame:
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
            raise FileNotFoundError(f'File is not found: {path}')
        
        if path.suffix == '.csv':
            df = pd.read_csv(path)
        else:
            raise ValueError(f'Format is incorrect: {path.suffix}')

        return df

    @staticmethod
    def _get_tickers_names(
        df : pd.DataFrame
    ) -> List[str]:
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
        
        ticker_cols = [col for col in df.columns if col.endswith('_close')]
        tickers = [col.replace('_close', '') for col in ticker_cols]

        return tickers

    @staticmethod
    def _get_ticker_data(
        df : pd.DataFrame,
        ticker : str
    ) -> pd.DataFrame:
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
            DataFrame с колонками date, close, time
        """

        ticker_df = df[['date', f'{ticker}_close']].copy()
        ticker_df = ticker_df.rename(columns = {f'{ticker}_close' : 'close'})
        ticker_df['time'] = pd.to_datetime(ticker_df['date'])
        ticker_df.dropna(inplace = True)
    
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

        ticker_df['date'] = pd.to_datetime(ticker_df['date']).dt.date
        ticker_df = ticker_df.sort_values('date').reset_index(drop = True)

        return ticker_df

    def _filter_trading_hours(
        self,
        ticker_df : pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Оставить только бары в торговые часы (10:00–18:50).
        
        Parameters
        ----------
        ticker_df : pd.DataFrame
            Данные тикера
        
        Returns
        -------
        pd.DataFrame
            Отфильтрованный DataFrame
        
        Raises
        ------
        ValueError
            Если нет колонки time
        """

        ticker_df = ticker_df.copy()

        if 'time' not in ticker_df.columns:
            raise ValueError('Column "time" is not in data!')

        time = pd.to_datetime(ticker_df['time'])
        hour = time.dt.hour
        minute = time.dt.minute

        mask = (hour < 18) | ((hour == 18) & (minute <= 50))
        ticker_df = ticker_df[mask].reset_index(drop = True)

        return ticker_df
    
    def _filter_valid_dates(
        self,
        ticker_df : pd.DataFrame
    ) -> pd.DataFrame:
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

        counts = ticker_df.groupby('date').size()
        valid_dates = counts[counts >= self.min_bars_per_day].index
        ticker_df = ticker_df[ticker_df['date'].isin(valid_dates)]
        ticker_df.reset_index(inplace = True, drop = True)

        return ticker_df

    def _detect_fat_finger(
        self,
        ticker_df : pd.DataFrame
    ) -> pd.DataFrame:
        """
        Обнаружить и исправить выбросы fat-finger.
        
        Parameters
        ----------
        ticker_df : pd.DataFrame
            Данные тикера
        
        Returns
        -------
        pd.DataFrame
            DataFrame с колонками processed_close, fat_finger_flag
        
        Raises
        ------
        ValueError
            Если неизвестный тип фильтра

        Notes
        -----
        Поддерживает несколько способов обработки (standart, hampel)
        """
        
        ticker_df = ticker_df.copy()

        if self.fat_finger_type == 'standart':
            results = ticker_df.groupby('date')['close'].apply(
                lambda x : self._detect_fat_finger_single(
                    x.values, 
                    self.fat_finger_window, 
                    self.fat_finger_threshold)
                )

        elif self.fat_finger_type == 'hampel':
            results = ticker_df.groupby('date')['close'].apply(
                            lambda x : self._filter_hampel(
                                x.values,
                                self.fat_finger_window,
                                self.fat_finger_sigma,
                                self.hampel_const)
                            )
        else:
            raise ValueError(f'Unknown filter type {self.fat_finger_type}')
        ticker_df['processed_close'] = np.concatenate([result[0] for result in results])
        ticker_df['fat_finger_flag'] = np.concatenate([result[1] for result in results])

        number_of_fat_fingers = ticker_df['fat_finger_flag'].sum()

        if number_of_fat_fingers:
            print(f'{number_of_fat_fingers} fat finger values deleted')

        return ticker_df

    @staticmethod
    def _detect_fat_finger_single(
        prices : np.ndarray,
        window_size : int,
        threshold : float
    ) -> Tuple[np.ndarray, np.ndarray]:

        """
        Обнаружить выбросы в одном массиве цен (стандартный метод).
        
        Parameters
        ----------
        prices : np.ndarray
            Массив цен
        window_size : int
            Размер окна
        threshold : float
            Порог отклонения
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (cleaned, flags) — очищенные цены и флаги
        """

        prices = np.asarray(prices, dtype = float)
        length = len(prices)
        prices_processed = prices.copy()
        flags = np.zeros(length, dtype = bool)

        for idx in range(window_size, length - window_size):
            window = prices[idx - window_size : idx + window_size + 1]
            med = np.median(window)

            if med == 0:
                continue
            diff = np.abs(prices[idx] - med) / med

            if diff > threshold:
                flags[idx] = True
                prices_processed[idx] = med

        return prices_processed, flags

    @staticmethod
    def _filter_hampel(
        series : np.ndarray,
        window_size : int = 11,
        sigma : float = 3.0,
        const : float = 1.4826
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Фильтр Хампеля для обнаружения выбросов.
        
        Parameters
        ----------
        series : np.ndarray
            Временной ряд
        window_size : int
            Размер окна
        sigma : float
            Порог в сигмах
        const : float
            Константа нормализации MAD
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (cleaned, flags) — очищенный ряд и флаги для логирования
        """
        
        series = np.asarray(series, dtype = float)
        series_length = len(series)
        cleaned = series.copy()
        flags = np.zeros(series_length, dtype = bool)

        for i in range(window_size, series_length - window_size):
            window = series[i - window_size : i + window_size + 1]
            r_median = np.median(window)
            r_mad = np.median(np.abs(window - r_median))

            if r_mad == 0:
                continue

            if np.abs(series[i] - r_median) > sigma * r_mad * const:
                cleaned[i] = r_median
                flags[i] = True

        return cleaned, flags

    def _calculate_returns(
        self, 
        ticker_df : pd.DataFrame
    ) -> pd.DataFrame:
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
        ticker_df['returns'] = np.log(ticker_df['processed_close'] / ticker_df.groupby('date')['processed_close'].shift(1)) * 100

        return ticker_df

    def _filter_extreme_returns(
        self, 
        ticker_df : pd.DataFrame
    ) -> pd.DataFrame:
        """
        Удалить экстремальные доходности (|returns| > max_return).
        
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
        ticker_df = ticker_df[ticker_df['returns'].abs() <= self.max_return]
        ticker_df = ticker_df.dropna().reset_index(drop = True)

        return ticker_df

    def _save_rv(
        self,
        ticker_df : pd.DataFrame,
        ticker : str
    ) -> None:
        """
        Рассчитать RV и сохранить в CSV.
        
        Parameters
        ----------
        ticker_df : pd.DataFrame
            Данные тикера
        ticker : str
            Название тикера
        
        Notes
        -----
        Если use_rv_filter = True, применяется фильтр Хампеля на RV.
        """

        rv = ticker_df.groupby('date')['returns'].apply(lambda x : (x ** 2).sum()).reset_index()
        rv.columns = ['date', 'RV']

        if self.use_rv_filter:
            rv['RV'] = self._filter_hampel(
                rv['RV'].values,
                self.fat_finger_window,
                self.fat_finger_sigma,
                self.hampel_const
            )[0]

        path = self.output_dir / f'RV/{ticker}_RV.csv'
        path.parent.mkdir(parents = True, exist_ok = True)
        rv.to_csv(path, index = False)

    def _save_rv_matrix(
        self,
        tickers : List[str]
    ) -> None:
        """
        Объединить RV всех тикеров в одну таблицу.
        
        Parameters
        ----------
        tickers : List[str]
            Список тикеров
        """
        
        rv_list = []

        for ticker in tickers:
            rv = pd.read_csv(self.output_dir / f'RV/{ticker}_RV.csv')
            rv = rv.rename(columns = {'RV' : f'{ticker}_RV'})
            rv_list.append(rv)

        rv_matrix = rv_list[0]
        for rv in rv_list[1:]:
            rv_matrix = pd.merge(rv_matrix, rv, on = 'date', how = 'inner')

        rv_matrix.to_csv(self.output_dir / 'rv_matrix.csv', index = False)

    def _save_profitabilities_matrix(
        self,
        tickers : List[str]
    ) -> None:
        """
        Объединить доходности всех тикеров в одну таблицу.
        
        Parameters
        ----------
        tickers : List[str]
            Список тикеров
        """

        profit_list = []

        for ticker in tickers:
            profits = self._results[ticker][['time', 'returns']].copy()
            profits = profits.rename(columns = {'returns' : f'{ticker}_returns'})
            profit_list.append(profits)

        profit_matrix = profit_list[0]
        for profit in profit_list[1:]:
            profit_matrix = pd.merge(profit_matrix, profit, on = 'time', how = 'outer')

        profit_matrix = profit_matrix.sort_values('time').reset_index(drop = True)
        profit_matrix = profit_matrix.ffill()
        profit_matrix = profit_matrix.dropna().reset_index(drop = True) 
        profit_matrix.to_csv(self.output_dir / 'profit_matrix.csv', index = False)