import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from source.features.features_engineering import FeatureEngineer
from source.features.data_splitter import DataSplitter
from source.models import RandomForestModel, XGBoostModel, LightGBMModel
from source.utils.metrics import MetricsCalculator

def main():
    rv_matrix = pd.read_csv('data/processed/rv_matrix.csv')

    engineer = FeatureEngineer()
    features = engineer.create_features(rv_matrix)
    targets = engineer.create_targets(rv_matrix)

    df = engineer.merge_features_targets(features, targets)

    tickers = ['VTBR', 'SBER', 'LKOH', 'ROSN', 'NVTK',
               'TATN', 'GAZP', 'GMKN', 'NLMK', 'MGNT']

    splitter = DataSplitter(
        train_end = '2021-12-31',
        valid_end = '2023-12-31'
    )

    splits = splitter.split_all(df, tickers, use_validation = True)

    calculator = MetricsCalculator()

    y_true_dict = {}
    y_pred_rf = {}
    y_pred_xgb = {}
    y_pred_lgbm = {}

    for ticker, (X_train, X_valid, X_test, y_train, y_valid, y_test) in splits.items():
        y_true_dict[ticker] = y_test.values

        rf = RandomForestModel(n_estimators = 100, max_depth = 10)
        rf.fit(X_train, y_train)
        y_pred_rf[ticker] = rf.predict(X_test)

        xgb = XGBoostModel(n_estimators = 1000, max_depth = 6)
        xgb.fit(X_train, y_train, eval_set = [(X_valid, y_valid)])
        y_pred_xgb[ticker] = xgb.predict(X_test)

        lgbm = LightGBMModel(n_estimators = 1000, max_depth = 31)
        lgbm.fit(X_train, y_train, eval_set = [(X_valid, y_valid)], early_stopping_rounds = 50)
        y_pred_lgbm[ticker] = lgbm.predict(X_test)

    results_rf = calculator.calculate_all(y_true_dict, y_pred_rf)
    results_xgb = calculator.calculate_all(y_true_dict, y_pred_xgb)
    results_lgbm = calculator.calculate_all(y_true_dict, y_pred_lgbm)

    summary = []
    for ticker in tickers:
        summary.append({
            'Ticker': ticker,
            'RF_R2':    results_rf[ticker]['R2'],
            'RF_RMSE':  results_rf[ticker]['RMSE'],
            'RF_MAE':   results_rf[ticker]['MAE'],
            'XGB_R2':   results_xgb[ticker]['R2'],
            'XGB_RMSE': results_xgb[ticker]['RMSE'],
            'XGB_MAE':  results_xgb[ticker]['MAE'],
            'LGBM_R2':   results_lgbm[ticker]['R2'],
            'LGBM_RMSE': results_lgbm[ticker]['RMSE'],
            'LGBM_MAE':  results_lgbm[ticker]['MAE'],
        })

    df_results = pd.DataFrame(summary)
    print(df_results.round(4))
    print(df_results.drop(columns = ['Ticker']).mean())

if __name__ == '__main__':
    main()
