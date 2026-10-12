import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from source.features.features_engineering import FeatureEngineer
from source.features.data_splitter import DataSplitter
from source.autoregression import HARModel
from source.utils.metrics import MetricsCalculator
from source.utils.logger import Logger

def main():
    logger = Logger('TrainHAR')
    logger.info('Обучение HAR')

    tickers = ['VTBR', 'SBER', 'LKOH', 'ROSN', 'NVTK', 'TATN', 'GAZP', 'GMKN', 'NLMK', 'MGNT'] 
    
    rv_matrix = pd.read_excel('data/processed/RV_clear_V2.xlsx')
    engineer = FeatureEngineer()
    features = engineer.create_har_features(rv_matrix, tickers)
    targets = engineer.create_targets(rv_matrix, tickers)

    df = pd.merge(features, targets, on = 'date', how = 'inner')

    splitter = DataSplitter(train_end = '2021-12-31',
                            valid_end = None)

    splits = splitter.split_all(df, tickers, use_validation = False)

    calculator = MetricsCalculator()
    y_true_dict = {} 
    y_pred_dict = {}

    for ticker, (X_train, X_valid, X_test, y_train, y_valid, y_test) in splits.items():
        logger.step('Train', f'HAR для {ticker}')

        model = HARModel()
        model.fit(X_train, y_train)

        y_true_dict[ticker] = y_test.values
        y_pred_dict[ticker] = model.predict(X_test)

        logger.success(f"{ticker}: обучено")

    results = calculator.calculate_all(y_true_dict, y_pred_dict)
    
    summary = []

    for ticker in tickers:
        summary.append({
            'Ticker' : ticker,
            'R2' : results[ticker]['R2'],
            'RMSE' : results[ticker]['RMSE'],
            'MAE' : results[ticker]['MAE'],
            'MSE' : results[ticker]['MSE']
        })

    df_results = pd.DataFrame(summary)
    print(df_results.round(4))
    print(df_results[['R2', 'RMSE', 'MAE', 'MSE']].mean())
    
    Path('reports/tables').mkdir(parents = True, exist_ok = True)
    df_results.to_csv('reports/tables/har_results.csv', index = False)
    logger.success('Результаты HAR сохранены!')
    
if __name__ == '__main__':
    main()
    