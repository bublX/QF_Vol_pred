import argparse
import sys
from pathlib import Path

import pandas as pd

from source.config import Config, load_config
from source.data_processors.data_loader import DataLoader
from source.data_processors.data_processor import DataProcessor
from source.data_processors.dataset_preparer import DatasetPreparer
from source.features.data_splitter import DataSplitter
from source.features.features_engineering import FeatureEngineer
from source.models.registry import get_model
from source.utils.logger import Logger
from source.utils.metrics import MetricsCalculator

COMMAND = "uv run python main.py --config <конфиг>"


def prepare(config: Config) -> None:
    """
    Подготовить CSV из архива данных.

    Parameters
    ----------
    config : Config
        Конфиг эксперимента
    """
    data_cfg = config.data
    preparer = DatasetPreparer(
        archive_path=data_cfg.archive_path,
        output_path=Path(data_cfg.raw_dir) / data_cfg.raw_file,
        tickers=data_cfg.tickers,
    )
    path = preparer.prepare()
    logger = Logger("DatasetPreparer")
    logger.success(f"Данные готовы: {path}")


def load(config: Config) -> None:
    """
    Загрузить цены с MOEX и сохранить в {data.raw_dir}/{data.raw_file}.

    Parameters
    ----------
    config : Config
        Конфиг эксперимента

    Notes
    -----
    Делает сетевые запросы к MOEX ISS API, может занять много времени.
    """

    data_cfg = config.data
    loader = DataLoader(
        tickers=data_cfg.tickers,
        start_year=data_cfg.start_year,
        end_year=data_cfg.end_year,
        interval=data_cfg.source_interval,
        delay=data_cfg.delay,
        save_dir=data_cfg.raw_dir,
        filename=data_cfg.raw_file,
        base_url=data_cfg.base_url,
    )
    loader.load_all(save_individual=True)


def process(config: Config) -> None:
    """
    Очистить сырые цены и рассчитать RV, результат в {processing.processed_dir}.

    Parameters
    ----------
    config : Config
        Конфиг эксперимента

    Raises
    ------
    FileNotFoundError
        Если нет файла с сырыми ценами
    """

    raw_path = Path(config.data.raw_dir) / config.data.raw_file
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Нет сырых данных: {raw_path}. Сначала загрузите их: {COMMAND} --stage load"
        )

    proc_cfg = config.processing
    processor = DataProcessor(
        input_dir=config.data.raw_dir,
        output_dir=proc_cfg.processed_dir,
        min_day_coverage=proc_cfg.min_day_coverage,
        min_bar_coverage=proc_cfg.min_bar_coverage,
        max_return=proc_cfg.max_return,
        hampel_window=proc_cfg.hampel_window,
        hampel_sigma=proc_cfg.hampel_sigma,
        hampel_const=proc_cfg.hampel_const,
        source_interval=config.data.source_interval,
        interval=config.data.interval,
    )

    processor.process_all(config.data.raw_file)


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

    Raises
    ------
    FileNotFoundError
        Если нет {processing.processed_dir}/rv_matrix.csv
    """

    rv_path = Path(config.processing.processed_dir) / "rv_matrix.csv"
    if not rv_path.exists():
        raw_path = Path(config.data.raw_dir) / config.data.raw_file
        hint = f"{COMMAND} --stage process"
        if not raw_path.exists():
            hint = f"{COMMAND} --stage load, затем {hint}"
        raise FileNotFoundError(
            f"Нет обработанных данных: {rv_path}. Сначала выполните: {hint}"
        )

    model_name = config.model.name
    assert model_name is not None
    name = config.output.name or model_name
    logger = Logger(f"Train{name.upper()}")
    logger.info(f"Обучение {config.model.name}")

    tickers = config.data.tickers
    rv_matrix = pd.read_csv(rv_path)

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
        model = get_model(model_name, config.model.params)
        logger.step("Train", f"{model.name} для {ticker}")

        if config.split.use_validation:
            model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
            if config.split.refit:
                n_trees = model.best_n_estimators
                logger.info(
                    f"{ticker}: early_stopping - {n_trees} деревьев, refit на train + valid"
                )
                params = {**config.model.params, "n_estimators": n_trees}
                model = get_model(model_name, params)
                model.fit(pd.concat([X_train, X_valid]), pd.concat([y_train, y_valid]))
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
    Точка входа (вызывается из main.py): uv run python main.py --config configs/<name>.yaml [--stage ...]

    Этапы: load — загрузка с MOEX, process — расчёт RV, train — обучение (по умолчанию).
    """

    parser = argparse.ArgumentParser(
        description="Пайплайн прогнозирования RV по YAML-конфигу"
    )
    parser.add_argument(
        "--config", required=True, help="Путь к YAML-конфигу эксперимента"
    )
    parser.add_argument(
        "--stage",
        choices=["prepare", "load", "process", "train"],
        default="train",
        help="Этап пайплайна (по умолчанию train)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    stages = {"prepare": prepare, "load": load, "process": process, "train": run}
    try:
        stages[args.stage](config)
    except FileNotFoundError as error:
        sys.exit(f"Ошибка: {error}")


if __name__ == "__main__":
    main()
