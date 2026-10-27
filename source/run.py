import argparse
from pathlib import Path

import pandas as pd

from source.config import Config, load_config
from source.features.data_splitter import DataSplitter
from source.features.features_engineering import FeatureEngineer
from source.models.registry import get_model
from source.utils.logger import Logger
from source.utils.metrics import MetricsCalculator


def run(config: Config) -> pd.DataFrame:
    """
    Обучить модель для каждого тикера и сохранить метрики на тесте.

    Parameters
    ----------
    config : Config
        Конфиг эксперимента

    Returns
    -------
    pd.DataFrame
        Таблица метрик: колонка Ticker и по колонке на метрику

    Notes
    -----
    Результаты сохраняются в {output.reports_dir}/{output.name}_results.csv,
    если output.name не задан, используется model.name.
    """

    name = config.output.name or config.model.name
    logger = Logger(f"Train{name.upper()}")
    logger.info(f"Обучение {config.model.name}")

    tickers = config.data.tickers
    rv_matrix = pd.read_csv(Path(config.processing.processed_dir) / "rv_matrix.csv")

    features_cfg = config.features
    engineer = FeatureEngineer(
        num_lags=features_cfg.num_lags,
        windows=features_cfg.windows,
        ma_windows=features_cfg.ma_windows,
        vol_ratio_windows=features_cfg.vol_ratio_windows,
        momentum_window=features_cfg.momentum_window,
        rsi_window=features_cfg.rsi_window,
    )
    if features_cfg.feature_set == "har":
        features = engineer.create_har_features(rv_matrix, tickers)
    else:
        features = engineer.create_features(rv_matrix)
    targets = engineer.create_targets(rv_matrix, tickers)

    df = engineer.merge_features_targets(features, targets)

    splitter = DataSplitter(
        train_end=config.split.train_end, valid_end=config.split.valid_end
    )
    splits = splitter.split_all(df, tickers, use_validation=config.split.use_validation)

    metrics = [metric.upper() for metric in config.evaluation.metrics]
    calculator = MetricsCalculator(metrics)
    y_true_dict = {}
    y_pred_dict = {}

    for ticker, (X_train, X_valid, X_test, y_train, y_valid, y_test) in splits.items():
        model = get_model(config.model.name, config.model.params)
        logger.step("Train", f"{model.name} для {ticker}")

        if config.split.use_validation:
            model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
        else:
            model.fit(X_train, y_train)

        y_true_dict[ticker] = y_test.values
        y_pred_dict[ticker] = model.predict(X_test)

        logger.success(f"{ticker}: обучено")

    results = calculator.calculate_all(y_true_dict, y_pred_dict)

    df_results = pd.DataFrame(
        [
            {
                "Ticker": ticker,
                **{metric: results[ticker][metric] for metric in metrics},
            }
            for ticker in tickers
        ]
    )
    logger.info(f"Метрики по тикерам:\n{df_results.round(3)}")
    logger.info(f"Средние метрики:\n{df_results[metrics].mean().round(3)}")

    reports_dir = Path(config.output.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(reports_dir / f"{name}_results.csv", index=False)
    logger.success(f"Результаты сохранены: {reports_dir / f'{name}_results.csv'}")

    return df_results


def main() -> None:
    """
    Точка входа: uv run python -m source.run --config configs/<name>.yaml
    """

    parser = argparse.ArgumentParser(description="Обучение модели по YAML-конфигу")
    parser.add_argument(
        "--config", required=True, help="Путь к YAML-конфигу эксперимента"
    )
    args = parser.parse_args()

    run(load_config(args.config))


if __name__ == "__main__":
    main()
