import pandas as pd
from typing import List, Tuple, Optional

class DataSplitter:
    """
    Преобразователь данных перед обучением моделей машинного обучения

    Parameters
    ----------
    train_end : str
        Конец обучающей выборки (включительно)
    valid_end : Optional[str]
        Конец валидационной выборки.
        Если None — валидация не используется.
    
    Notes
    -----
    Данные делятся строго по времени, без перемешивания.
    Это важно для временных рядов, чтобы избежать утечки данных.
    Поддерживается возможность отключения создания валидационной выборки.
    """
    def __init__(
        self,
        train_end : str = '2021-12-31',
        valid_end : Optional[str] = '2023-12-31'
    ):
        self.train_end = train_end
        self.valid_end = valid_end

    def split(
        self,
        df : pd.DataFrame,
        ticker : str,
        use_validation : bool = True 
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame, 
               pd.Series, Optional[pd.Series], pd.Series]:
        """
        Разделить данные одного тикера на train/valid/test.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame с признаками и таргетом
        ticker : str
            Название тикера
        use_validation : bool
            Использовать ли валидационную выборку
        
        Returns
        -------
        Tuple
            Если use_validation=True:
                (X_train, X_valid, X_test, y_train, y_valid, y_test)
            Если use_validation=False:
                (X_train, None, X_test, y_train, None, y_test)
        """

        feature_cols = [col for col in df.columns if col.startswith(ticker) and not col.endswith('_target')]
        target_col = f'{ticker}_target'

        train_mask = df['date'] <= self.train_end

        if use_validation and self.valid_end:
            validation_mask = (df['date'] > self.train_end) & (df['date'] <= self.valid_end) 
            test_mask = df['date'] > self.valid_end
        else:
            test_mask = df['date'] > self.train_end

        X_train = df.loc[train_mask, feature_cols]
        X_test = df.loc[test_mask, feature_cols]

        y_train = df.loc[train_mask, target_col]
        y_test = df.loc[test_mask, target_col]

        if use_validation and self.valid_end:
            X_valid = df.loc[validation_mask, feature_cols]
            y_valid = df.loc[validation_mask, target_col]
        else:
            X_valid = None
            y_valid = None

        return X_train, X_valid, X_test, y_train, y_valid, y_test

    def split_all(
        self,
        df : pd.DataFrame,
        tickers : List[str],
        use_validation : bool = True
    ) -> dict:
        """
        Разделить данные для всех тикеров.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame с признаками и таргетами
        tickers : List[str]
            Список тикеров
        use_validation : bool
            Использовать ли валидационную выборку
        
        Returns
        -------
        dict
            {тикер: (X_train, X_valid, X_test, y_train, y_valid, y_test)}
        """

        result = {}
        
        for ticker in tickers: 
            result[ticker] = self.split(df, ticker, use_validation)

        return result