import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*Precision loss.*")

from source.utils.logger import Logger
from typing import List
from scipy.stats import skew


class FeatureEngineer:
    """
    Генератор признаков для моделей прогнозирования RV.

    Parameters
    ----------
    num_lags : List[int]
        Значения для лаговых признаков (по умолчанию [1, 2, 5, 10])
    windows : List[int]
        Окна для скользящих статистик (по умолчанию [5, 22])
    ma_windows : List[int]
        Окна для скользящих средних (по умолчанию [20, 50])
    vol_ratio_windows : List[int]
        Окна для vol_ratio (по умолчанию [10, 15, 20])
    momentum_window : int
        Окно для momentum (по умолчанию 20)
    rsi_window : int
        Окно для RSI (по умолчанию 14)

    Notes
    -----
    Признаки генерируются для каждого тикера отдельно.
    Все признаки имеют префикс {ticker}_.
    """

    def __init__(
        self,
        num_lags: List[int] = [1, 2, 5, 10],
        windows: List[int] = [5, 22],
        ma_windows: List[int] = [20, 50],
        vol_ratio_windows: List[int] = [10, 15, 20],
        momentum_window: int = 20,
        rsi_window: int = 14,
    ):
        self.ma_windows = ma_windows
        self.momentum_window = momentum_window
        self.rsi_window = rsi_window
        self.vol_ratio_windows = vol_ratio_windows
        self.num_lags = num_lags
        self.windows = windows
        self.logger = Logger("FeatureEngineer")

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Создать все признаки для всех тикеров.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame с колонками date, {ticker}_RV

        Returns
        -------
        pd.DataFrame
            DataFrame с признаками для всех тикеров

        Notes
        -----
        Удаляются строки с NaN, возникшие из-за rolling().
        """

        self.logger.info("Генерация признаков")

        df = df.copy()  # ???
        feature_dfs = [df[["date"]]]
        tickers = [col.replace("_RV", "") for col in df.columns if col.endswith("_RV")]

        for ticker in tickers:
            self.logger.step("FEATURES", f"Генерируются признаки для {ticker}")
            ticker_rv = df[f"{ticker}_RV"]
            ticker_features = self._create_ticker_features(ticker_rv, ticker)
            feature_dfs.append(ticker_features)

        result = pd.concat(feature_dfs, axis=1)
        result = result.dropna().reset_index(drop=True)
        self.logger.success(f"Признаки: {result.shape}")

        return result

    def _create_ticker_features(
        self, ticker_rv: pd.Series, ticker: str
    ) -> pd.DataFrame:
        """
        Создать признаки для одного тикера.

        Parameters
        ----------
        ticker_rv : pd.Series
            Ряд RV для тикера
        ticker : str
            Название тикера

        Returns
        -------
        pd.DataFrame
            DataFrame с признаками тикера
        """

        features = {}

        features.update(self._create_har_features(ticker_rv, ticker))
        features.update(self._create_lag_features(ticker_rv, ticker))
        features.update(self._create_technical_features(ticker_rv, ticker))
        features.update(self._create_stat_features(ticker_rv, ticker))
        features.update(self._create_rsi_features(ticker_rv, ticker))

        return pd.DataFrame(features)

    def _create_har_features(self, ticker_rv: pd.Series, ticker: str) -> pd.DataFrame:
        """
        Создать HAR-признаки.

        Parameters
        ----------
        ticker_rv : pd.Series
            Ряд RV для тикера
        ticker : str
            Название тикера

        Returns
        -------
        pd.DataFrame
            DataFrame с HAR-признаками
        """

        return pd.DataFrame(
            {
                # f'{ticker}_rv_1' : np.sqrt(ticker_rv),
                # f'{ticker}_rv_5' : np.sqrt(ticker_rv.rolling(5).mean()),
                # f'{ticker}_rv_22': np.sqrt(ticker_rv.rolling(22).mean())
                f"{ticker}_rv_1": ticker_rv,
                f"{ticker}_rv_5": ticker_rv.rolling(5).mean(),
                f"{ticker}_rv_22": ticker_rv.rolling(22).mean(),
            }
        )

    def _create_lag_features(self, ticker_rv: pd.Series, ticker: str) -> dict:
        """
        Создать лаговые признаки.

        Parameters
        ----------
        ticker_rv : pd.Series
            Ряд RV для тикера
        ticker : str
            Название тикера

        Returns
        -------
        dict
            Словарь с признаками lag_{n}
        """

        features = {}

        for num in self.num_lags:
            features[f"{ticker}_lag_{num}"] = ticker_rv.shift(num)

        return features

    def _create_technical_features(self, ticker_rv: pd.Series, ticker: str) -> dict:
        """
        Создать технические признаки (MA, momentum, vol_ratio).

        Parameters
        ----------
        ticker_rv : pd.Series
            Ряд RV для тикера
        ticker : str
            Название тикера

        Returns
        -------
        dict
            Словарь с техническими признаками
        """

        features = {}

        for num in self.ma_windows:
            features[f"{ticker}_MA_{num}"] = ticker_rv.rolling(window=num).mean()
        features[f"{ticker}_momentum"] = ticker_rv / ticker_rv.rolling(
            window=self.momentum_window
        ).mean().replace(0, np.nan)

        for day in self.vol_ratio_windows:
            features[f"{ticker}_vol_ratio_{day}"] = (
                ticker_rv / ticker_rv.rolling(day).mean()
            )

        return features

    def _create_stat_features(self, ticker_rv: pd.Series, ticker: str) -> dict:
        """
        Создать статистические признаки (mean, std, min, max, skew).

        Parameters
        ----------
        ticker_rv : pd.Series
            Ряд RV для тикера
        ticker : str
            Название тикера

        Returns
        -------
        dict
            Словарь со статистическими признаками

        Notes
        -----
        `skew` считается с `min_periods=num` (полное окно).
        """

        features = {}

        for num in self.windows:
            features[f"{ticker}_rv_mean_{num}"] = ticker_rv.rolling(num).mean()
            features[f"{ticker}_rv_std_{num}"] = ticker_rv.rolling(num).std()
            features[f"{ticker}_rv_min_{num}"] = ticker_rv.rolling(num).min()
            features[f"{ticker}_rv_max_{num}"] = ticker_rv.rolling(num).max()
            features[f"{ticker}_rv_skew_{num}"] = ticker_rv.rolling(
                num, min_periods=num
            ).apply(skew, raw=True)

        return features

    def _create_rsi_features(self, ticker_rv: pd.Series, ticker: str) -> dict:
        """
        Создать RSI-признак.

        Parameters
        ----------
        ticker_rv : pd.Series
            Ряд RV для тикера
        ticker : str
            Название тикера

        Returns
        -------
        dict
            Словарь с признаком rsi
        """

        features = {}

        diff = ticker_rv.diff()
        increase = (diff.where(diff > 0, 0)).rolling(window=self.rsi_window).mean()
        decrease = (-diff.where(diff < 0, 0)).rolling(window=self.rsi_window).mean()
        rs = increase / decrease.replace(0, np.nan)
        features[f"{ticker}_rsi"] = 100 - (100 / (1 + rs))

        return features

    def create_targets(self, df: pd.DataFrame, tickers: list) -> pd.DataFrame:
        """
        Создать целевую переменную (RV на следующий день).

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame с колонками date, {ticker}_RV

        Returns
        -------
        pd.DataFrame
            DataFrame с колонками date, {ticker}_target
        """

        result = df[["date"]].copy()

        for ticker in tickers:
            # result[f'{ticker}_target'] = np.sqrt(df[f'{ticker}_RV'].shift(-1))
            result[f"{ticker}_target"] = df[f"{ticker}_RV"].shift(-1)

        result = result.dropna().reset_index(drop=True)
        return result

    def merge_features_targets(
        self, df_features: pd.DataFrame, df_targets: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Объединить признаки и таргеты по дате.

        Parameters
        ----------
        df_features : pd.DataFrame
            Признаки
        df_targets : pd.DataFrame
            Таргеты

        Returns
        -------
        pd.DataFrame
            Объединённый DataFrame
        """

        return pd.merge(df_features, df_targets, on="date", how="inner")

    def create_har_features(self, df: pd.DataFrame, tickers: list) -> pd.DataFrame:
        """
        Создать HAR-признаки для HAR моделей.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame с колонками date, {ticker}_RV

        Returns
        -------
        pd.DataFrame
            DataFrame с колонками date и HAR-признаками
        """
        result = df[["date"]].copy()

        for ticker in tickers:
            ticker_rv = df[f"{ticker}_RV"]
            har_features = self._create_har_features(ticker_rv, ticker)
            result = result.join(har_features)

        result = result.dropna().reset_index(drop=True)

        return result
