import types
import typing
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Union

import yaml

from source.models.registry import MODELS

DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
)

AVAILABLE_METRICS = ["mse", "rmse", "mae", "r2", "qlike", "mape"]

FEATURE_SETS = ["full", "har"]


@dataclass
class DataConfig:
    """
    Параметры подготовки CSV из архива, загрузки с MOEX и интервалов свечей.

    archive_path используется DatasetPreparer. При загрузке DataLoader
    получает source_interval как код интервала свечей MOEX ISS; для
    BarAggregator source_interval и interval задают длительности исходных
    и итоговых свечей в минутах. raw_dir и raw_file определяют путь к CSV.
    """

    tickers: list[str]
    start_year: int
    end_year: int
    archive_path: str
    source_interval: int
    interval: int
    delay: float
    base_url: str
    raw_dir: str
    raw_file: str


@dataclass
class ProcessingConfig:
    """
    Параметры очистки данных и расчёта RV (см. DataProcessor).
    """

    min_day_coverage: float
    min_bar_coverage: float
    processed_dir: str
    session_start: str
    session_end: str
    steps: list[dict[str, Any]]
    price_steps: list[dict[str, Any]]
    returns: dict[str, Any]


@dataclass
class FeaturesConfig:
    """
    Параметры генерации признаков (см. FeatureEngineer).

    feature_set: 'full' — create_features, 'har' — create_har_features.
    """

    feature_set: str
    num_lags: list[int]
    windows: list[int]
    ma_windows: list[int]
    vol_ratio_windows: list[int]
    momentum_window: int
    rsi_window: int


@dataclass
class SplitConfig:
    """
    Параметры разбиения на выборки (см. DataSplitter).
    """

    train_end: str
    valid_end: str | None
    refit: bool
    use_validation: bool


@dataclass
class ModelConfig:
    """
    Название модели и её гиперпараметры.
    """

    name: str | None
    params: dict[str, Any]


@dataclass
class EvaluationConfig:
    """
    Список метрик для оценки прогнозов.
    """

    metrics: list[str]


@dataclass
class OutputConfig:
    """
    Куда сохранять результаты.
    """

    reports_dir: str
    name: str | None


@dataclass
class Config:
    """
    Полный конфиг эксперимента.
    """

    data: DataConfig
    processing: ProcessingConfig
    features: FeaturesConfig
    split: SplitConfig
    model: ModelConfig
    evaluation: EvaluationConfig
    output: OutputConfig


def load_config(
    path: str | Path, default_path: str | Path = DEFAULT_CONFIG_PATH
) -> Config:
    """
    Загрузить конфиг эксперимента поверх конфига по умолчанию.

    Parameters
    ----------
    path : Union[str, Path]
        Путь к YAML-файлу эксперимента
    default_path : Union[str, Path]
        Путь к YAML-файлу со значениями по умолчанию

    Returns
    -------
    Config
        Проверенный конфиг

    Raises
    ------
    FileNotFoundError
        Если файл не найден
    TypeError
        Если конфиг или его секция не словарь
    ValueError
        Если в конфиге неизвестные или пропущенные ключи, неверные типы
        или недопустимые значения

    Notes
    -----
    Вложенные словари объединяются, списки и остальные значения заменяются целиком.
    """

    values = _deep_merge(_read_yaml(Path(default_path)), _read_yaml(Path(path)))
    config = _build(Config, values, "")

    if config.model.name is None:
        raise ValueError("'model.name' не задан: укажите модель в файле эксперимента")
    if config.model.name not in MODELS:
        raise ValueError(
            f"Неизвестная модель в 'model.name': '{config.model.name}'. "
            f"Доступные: {list(MODELS)}"
        )

    if config.features.feature_set not in FEATURE_SETS:
        raise ValueError(
            f"Неизвестный 'features.feature_set': '{config.features.feature_set}'. "
            f"Доступные: {FEATURE_SETS}"
        )

    unknown_metrics = [
        m for m in config.evaluation.metrics if m not in AVAILABLE_METRICS
    ]
    if unknown_metrics:
        raise ValueError(
            f"Неизвестные метрики в 'evaluation.metrics': {unknown_metrics}. "
            f"Доступные: {AVAILABLE_METRICS}"
        )

    return config


def _read_yaml(path: Path) -> dict:
    """
    Прочитать YAML-файл в словарь (пустой файл даёт пустой словарь).
    """

    if not path.exists():
        raise FileNotFoundError(f"Файл конфига не найден: {path}")

    with open(path, encoding="utf-8") as f:
        values = yaml.safe_load(f)

    if values is None:
        return {}
    if not isinstance(values, dict):
        raise TypeError(
            f"Конфиг {path} должен быть словарём, получено {type(values).__name__}"
        )

    return values


def _deep_merge(base: dict, override: dict) -> dict:
    """
    Рекурсивно объединить словари: значения из override имеют приоритет.
    """

    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def _build(cls: type, values: Any, section: str) -> Any:
    """
    Создать dataclass из словаря, проверив ключи и типы.

    Parameters
    ----------
    cls : type
        Класс dataclass
    values : Any
        Значения из YAML
    section : str
        Путь к секции для сообщений об ошибках ('' для корня)
    """

    where = f"'{section}'" if section else "корне конфига"
    if not isinstance(values, dict):
        raise TypeError(
            f"В {where} ожидается словарь, получено {type(values).__name__}"
        )

    hints = typing.get_type_hints(cls)
    names = [f.name for f in fields(cls)]

    unknown = sorted(set(values) - set(names))
    if unknown:
        raise ValueError(f"Неизвестные ключи в {where}: {unknown}. Допустимые: {names}")

    missing = [name for name in names if name not in values]
    if missing:
        raise ValueError(f"Пропущены ключи в {where}: {missing}")

    kwargs = {}
    for name in names:
        key = f"{section}.{name}" if section else name
        tp = hints[name]
        if isinstance(tp, type) and is_dataclass(tp):
            kwargs[name] = _build(tp, values[name], key)
        else:
            if not _matches(values[name], tp):
                raise ValueError(
                    f"'{key}': ожидается {_type_name(tp)}, "
                    f"получено {values[name]!r} ({type(values[name]).__name__})"
                )
            kwargs[name] = values[name]

    return cls(**kwargs)


def _matches(value: Any, tp: Any) -> bool:
    """
    Проверить, что значение соответствует аннотации типа.

    Notes
    -----
    Поддерживаются int, float, str, bool, list[...], dict[...], X | None.
    Для float допускается int (в YAML `5` читается как int).
    """

    origin = typing.get_origin(tp)

    if origin in (Union, types.UnionType):
        return any(_matches(value, arg) for arg in typing.get_args(tp))
    if origin is list:
        item_type = typing.get_args(tp)[0]
        return isinstance(value, list) and all(_matches(v, item_type) for v in value)
    if origin is dict:
        return isinstance(value, dict)
    if tp is type(None):
        return value is None
    if tp is float:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if tp is int:
        return isinstance(value, int) and not isinstance(value, bool)

    return isinstance(value, tp)


def _type_name(tp: Any) -> str:
    """
    Читаемое имя типа для сообщений об ошибках.
    """

    if typing.get_origin(tp) is None:
        return tp.__name__
    return str(tp).replace("typing.", "")
