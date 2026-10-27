import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from source.features.features_engineering import FeatureEngineer
from source.features.data_splitter import DataSplitter
from source.models import RandomForestModel
from source.utils.metrics import MetricsCalculator
from source.utils.logger import Logger


def main():
    logger = Logger("TrainRF")
    logger.info("Обучение Random Forest")

    rv_matrix = pd.read_csv("data/processed/rv_matrix.csv")
    tickers = [
        "VTBR",
        "SBER",
        "LKOH",
        "ROSN",
        "NVTK",
        "TATN",
        "GAZP",
        "GMKN",
        "NLMK",
        "MGNT",
    ]

    engineer = FeatureEngineer()
    features = engineer.create_features(rv_matrix)
    targets = engineer.create_targets(rv_matrix, tickers)

    df = engineer.merge_features_targets(features, targets)

    splitter = DataSplitter(train_end="2021-12-31", valid_end=None)

    splits = splitter.split_all(df, tickers, use_validation=False)

    calculator = MetricsCalculator()
    y_true_dict = {}
    y_pred_dict = {}

    for ticker, (X_train, X_valid, X_test, y_train, y_valid, y_test) in splits.items():
        logger.step("Train", f"Random Forest для {ticker}")

        model = RandomForestModel(n_estimators=100, max_depth=10)
        model.fit(X_train, y_train)

        y_true_dict[ticker] = y_test.values
        y_pred_dict[ticker] = model.predict(X_test)

        logger.success(f"{ticker}: обучено")

    results = calculator.calculate_all(y_true_dict, y_pred_dict)

    summary = []

    for ticker in tickers:
        summary.append(
            {
                "Ticker": ticker,
                "R2": results[ticker]["R2"],
                "RMSE": results[ticker]["RMSE"],
                "MAE": results[ticker]["MAE"],
                "MSE": results[ticker]["MSE"],
            }
        )

    df_results = pd.DataFrame(summary)
    print(df_results.round(3))
    print(df_results[["R2", "RMSE", "MAE", "MSE"]].mean().round(3))

    Path("reports/tables").mkdir(parents=True, exist_ok=True)
    df_results.to_csv("reports/tables/rf_results.csv", index=False)
    logger.success("Результаты Random Forest сохранены!")


if __name__ == "__main__":
    main()
