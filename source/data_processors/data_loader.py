import requests
import pandas as pd
from functools import reduce
from typing import List, Optional, Dict
from pathlib import Path
import time

class DataLoader:
    def __init__(self,
        tickers : List[str],
        start_year : int = 2014,
        end_year : int = 2025,
        interval : int = 10,
        delay : float = 0.2,
        save_dir : str = 'data/raw/',
        filename: str = 'all_prices.csv',
        base_url : str = 'https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities'
    ):
        self.tickers = tickers
        self.start_year = start_year
        self.end_year = end_year
        self.interval = interval
        self.delay = delay
        self.base_url = base_url

        self._session = None
        self._data : Dict[str, pd.DataFrame] = {}
        self._save_dir = Path(save_dir)
        
        self.save_path = self._save_dir / filename

    def load_all(
        self,
        save_individual : bool = False
    ) -> pd.DataFrame:
        self._init_session()

        for ticker in self.tickers:
            df_ticker = self._load_ticker(ticker)
            self._data[ticker] = df_ticker
            if save_individual:
                self._save_ticker(df_ticker, ticker)
        self._close_session()
        merged_df = self._merge_data()

        if self.save_path:
            self._save(merged_df)

        return merged_df

    def _init_session(self) -> None:
        self._session = requests.Session()

    def _close_session(self) -> None:
        if self._session:
            self._session.close()
            self._session = None

    def _load_ticker(
        self,
        ticker : str
    ) -> pd.DataFrame:
        all_data = []
        columns = None
        url = f'{self.base_url}/{ticker}/candles.json'

        for year in range(self.start_year, self.end_year + 1):
            params = {
                'from' : f'{year}-01-01',
                'till' : f'{year}-12-31',
                'interval' : self.interval,
                'start' : 0
            }

            year_data = self._fetch_year(url, params, columns)
            if columns is None and year_data:
                columns = year_data['columns']

            if year_data['data']:
                all_data.extend(year_data['data'])

        df = self._to_dataframe(all_data, columns, ticker)

        return df

    def _fetch_year(
        self,
        url : str,
        params : dict,
        columns : Optional[List[str]]
    ) -> dict:
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
                raise RuntimeError('No Internet connection!') from None

            except requests.exceptions.HTTPError as excep:
                if excep.response.status_code == 429:
                    retry_count += 1
                    if retry_count > max_tries:
                        raise RuntimeError(f'Too many 429 errors after {max_tries} retries') from excep
                    print(f'HTTP error 429. Retry {retry_count}/{max_tries} in 5 seconds')
                    time.sleep(5)
                    continue
                raise ValueError(f'HTTP error: {excep}') from excep

            except requests.exceptions.Timeout:
                retry_count += 1
                if retry_count > max_tries:
                    raise RuntimeError(f'Timeout after {max_tries} retries')
                print(f'Timeout. Retry {retry_count}/{max_tries} in 2s...')
                time.sleep(2)
                continue

            except requests.exceptions.RequestException as excep:
                retry_count += 1
                if retry_count > max_tries:
                    raise RuntimeError(f'Failed after {max_tries} retries') from excep
                print(f'Request error: {excep}. Retry {retry_count}/{max_tries} in 2 seconds')
                time.sleep(2)
                continue

            data = response.json()
            candles = data['candles']
            if not candles['data']:
                break

            if year_columns is None:
                year_columns = candles['columns']

            year_data.extend(candles['data'])
            params['start'] += len(candles['data'])

            if len(candles['data']) < 500:
                break
        
        return {'columns' : year_columns, 'data' : year_data}

    @staticmethod
    def _to_dataframe(
        data : List,
        columns : List[str],
        ticker : str
    ) -> pd.DataFrame:
        if not data or columns is None:
            print(f'{ticker}: data is not found')
            return pd.DataFrame(columns = ['date', f'{ticker}_close'])
        df = pd.DataFrame(data, columns = columns)
        df = df[['end', 'close']].rename(columns = {'end' : 'date', 'close' : f'{ticker}_close'})
        df['date'] = pd.to_datetime(df['date'])
        df = df.drop_duplicates(subset = 'date', keep = 'last')
        df = df.sort_values('date').reset_index(drop = True)
        return df

    def _merge_data(self) -> pd.DataFrame:
        if not self._data:
            raise ValueError('No data to merge')

        dfs = list(self._data.values())

        merged = reduce(lambda left, right: pd.merge(left, right, on = 'date', how = 'outer'), dfs)
        merged = merged.sort_values('date').reset_index(drop = True)

        columns = ['date'] + [f'{tmp}_close' for tmp in self.tickers]

        missing_cols = [col for col in columns if col not in merged.columns]

        if missing_cols:
            raise ValueError(f'Required columns not found: {missing_cols}. Available columns: {list(merged.columns)}')
        merged = merged[columns]

        return merged

    def _save(
        self,
        df : pd.DataFrame
    ):
        self.save_path.parent.mkdir(parents = True, exist_ok = True)
        df.to_csv(self.save_path, index = False)

    def _save_ticker(
        self,
        df : pd.DataFrame,
        ticker : str
    ):
        path = self.save_dir / f'{ticker}.csv'
        df.to_csv(path, index = False)