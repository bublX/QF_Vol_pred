import inspect

import pytest

from source.config import DEFAULT_CONFIG_PATH, load_config
from source.data_processors.data_processor import DataProcessor
from source.features.features_engineering import FeatureEngineer


def write(tmp_path, text):
    path = tmp_path / "exp.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_override_merges_with_defaults(tmp_path):
    path = write(
        tmp_path,
        """
data:
  tickers: [SBER]
split:
  valid_end: null
  use_validation: false
model:
  name: rf
  params: {n_estimators: 100, max_depth: 10}
""",
    )
    config = load_config(path)

    assert config.data.tickers == ["SBER"]
    assert config.data.start_year == 2014
    assert config.split.train_end == "2021-12-31"
    assert config.split.valid_end is None
    assert config.split.use_validation is False
    assert config.model.params == {"n_estimators": 100, "max_depth": 10}


def test_unknown_key(tmp_path):
    path = write(tmp_path, 'model: {name: rf}\nsplit: {train_endd: "2020-01-01"}\n')
    with pytest.raises(
        ValueError, match=r"Неизвестные ключи в 'split': \['train_endd'\]"
    ):
        load_config(path)


def test_wrong_type(tmp_path):
    # Дата без кавычек читается YAML как date, а не str
    path = write(tmp_path, "model: {name: rf}\nsplit: {train_end: 2020-01-01}\n")
    with pytest.raises(ValueError, match="'split.train_end': ожидается str"):
        load_config(path)


def test_model_name_required(tmp_path):
    path = write(tmp_path, "split: {use_validation: false}\n")
    with pytest.raises(ValueError, match="'model.name' не задан"):
        load_config(path)


def test_unknown_metric(tmp_path):
    path = write(tmp_path, "model: {name: rf}\nevaluation: {metrics: [rmse, qlike]}\n")
    with pytest.raises(ValueError, match="qlike"):
        load_config(path)


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def _defaults(cls):
    return {
        name: param.default
        for name, param in inspect.signature(cls.__init__).parameters.items()
        if param.default is not inspect.Parameter.empty
    }


def test_features_defaults_match_feature_engineer(tmp_path):
    config = load_config(write(tmp_path, "model: {name: rf}\n"))
    defaults = _defaults(FeatureEngineer)
    for name, value in vars(config.features).items():
        if name == "feature_set":
            continue
        assert value == defaults[name], name


def test_processing_defaults_match_data_processor(tmp_path):
    # max_return = 5 (значение из бывшего scripts/processing.py), а не 15.0 из класса
    config = load_config(write(tmp_path, "model: {name: rf}\n"))
    defaults = _defaults(DataProcessor)
    for name in [
        "min_bars_per_day",
        "fat_finger_window",
        "fat_finger_threshold",
        "fat_finger_type",
        "fat_finger_sigma",
        "hampel_const",
        "use_rv_filter",
    ]:
        assert getattr(config.processing, name) == defaults[name], name


def test_default_file_exists():
    assert DEFAULT_CONFIG_PATH.exists()


def test_unknown_model_name(tmp_path):
    path = write(tmp_path, "model: {name: garch}\n")
    with pytest.raises(ValueError, match="Неизвестная модель в 'model.name': 'garch'"):
        load_config(path)


def test_unknown_feature_set(tmp_path):
    path = write(tmp_path, "model: {name: rf}\nfeatures: {feature_set: tiny}\n")
    with pytest.raises(ValueError, match="feature_set"):
        load_config(path)


@pytest.mark.parametrize("name", ["har", "rf", "xgb", "lgbm"])
def test_experiment_configs_load(name):
    config = load_config(DEFAULT_CONFIG_PATH.parent / f"{name}.yaml")
    assert config.model.name == name
