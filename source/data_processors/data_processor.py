import pandas as pd
import numpy as np
from typing import Tuple, Optional
from pathlib import Path

class DataProcessor:
    def __init__(
            self,
            input_dir : str = 'data/raw/',
            output_dir : str = 'data/processed/',
            min_bars_per_day : int = 40,
            fat_finger_window : int = 11,
            fat_finger_threshold : float = 0.03,
            max_return : float = 5.0,
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.min_bars_per_day = min_bars_per_day
        self.fat_finger_window = fat_finger_window
        self.fat_finger_threshold = fat_finger_threshold
        self.max_return = max_return

        self._results = {}

    def process_all(
            self,
            input_file : str = 'all_prices.csv'
    ) -> Dict[str, pd.DataFrame]:
        df_all = self._load_data(input_file)

        tickers = self._get_tickers_names(df_all)

        for ticker in tickers:
            df_ticker = self._get_ticker_data(df_all, ticker)
            df_processed = self.process_ticker(df_ticker, ticker)
            self._results[ticker] = df_processed

        self._save_rv_matrix(tickers)
        self._save_profitabilities_matrix(tickers)

        return self._results

    def process_ticker(
            self,
            ticker_df : pd.DataFrame,
            ticker : str,
            save_path : Optional[str] = None
        ) -> pd.DataFrame:

        ticker_df = self._prepare_data(ticker_df)
        ticker_df = self._filter_valid_dates(ticker_df)
        ticker_df = self._detect_fat_finger(ticker_df)
        ticker_df = self._calculate_returns(ticker_df)
        ticker_df = self._filter_extreme_returns(ticker_df)

    def _load_data(
            self,
            input_file : str
    ) -> pd.DataFrame:

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
        ticker_cols = [col for col in df.columns if col.endswith('_close')]
        tickers = [col.replace('_close', '') for col in ticker_cols]

        return tickers

    @staticmethod
    def _get_ticker_data(
            df : pd.DataFrame,
            ticker : str
    ) -> pd.DataFrame:

        ticker_df = df[['date', f'{ticker}_close']].copy()
        ticker_df = ticker_df.rename(columns = {f'{ticker}_close' : 'close'})
        ticker_df.dropna(inplace = True)

        return ticker_df

    @staticmethod
    def _prepare_data(
            ticker_df : pd.DataFrame,
    ) -> pd.DataFrame:
        ticker_df = ticker_df.copy()

        if 'time' in df.columns:
            ticker_df['time'] = pd.to_datetime(ticker_df['time'])
            ticker_df['date'] = ticker_df['time'].dt.date
        else:
            ticker_df['date'] = pd.to_datetime(ticker_df['date']).dt.date

        if 'time' in df.columns:
            ticker_df = ticker_df.sort_values('time').reset_index(drop = True)
        else:
            ticker_df = ticker_df.sort_values('date').reset_index(drop = True)

        return ticker_df

    def _filter_valid_days(
            self,
            ticker_df : pd.DataFrame
    ) -> pd.DataFrame:
        ticker_df = ticker_df.copy()

        counts = ticker_df.groupby('date').size()
        valid_dates = counts[counts >= self.min_bars_per_day].index
        ticker_df = ticker_df[ticker_df['date'].isin(valid_dates)]
        ticker_df.reset_index(inplace = True, drop = True)

        return ticker_df

    # aggregate detection to all tickers
    # def _detect_fat_finger

    @staticmethod
    def _detect_fat_finger_single(
            prices : np.ndarray,
            window_size : int,
            threshold : float
    ) -> tuple:

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

    #def _calculate_returns(self, df):

    #def _filter_extreme_returns(self, df):

    #def _save_rv_matrix(self):

    #def _save_profitabilities_matrix(self):

    def _save(
            self,
            df: pd.DataFrame,
            save_path : Path,
            filename : str
        ):
            path = save_path / filename
            path.parent.mkdir(parents = True, exist_ok = True)
            df.to_csv(path, index = False)