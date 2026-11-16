import numpy as np
import pandas as pd
import pytest

from source.config import load_config
from source.models.registry import get_model
from source.run import load, process, run


@pytest.fixture
def config_path(tmp_path, monkeypatch):
    # Logger пишет логи в reports/logs/ относительно текущей директории
    monkeypatch.chdir(tmp_path)

    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2021-01-01", "2022-06-30")
    rv_matrix = pd.DataFrame({"date": dates.strftime("%Y-%m-%d")})
    for ticker in ["AAA", "BBB"]:
        rv_matrix[f"{ticker}_RV"] = rng.lognormal(0, 0.5, len(dates))
    (tmp_path / "processed").mkdir()
    rv_matrix.to_csv(tmp_path / "processed" / "rv_matrix.csv", index=False)

    def write(model_yaml, split_yaml="{valid_end: null, use_validation: false}"):
        path = tmp_path / "exp.yaml"
        path.write_text(
            f"""
data: {{tickers: [AAA, BBB]}}
processing: {{processed_dir: {tmp_path / "processed"}}}
split: {split_yaml}
output: {{reports_dir: {tmp_path / "out"}}}
{model_yaml}
""",
            encoding="utf-8",
        )
        return path

    return write


@pytest.mark.parametrize(
    "model_yaml, file_name",
    [
        ("model: {name: har}\nfeatures: {feature_set: har}", "har_results.csv"),
        ("model: {name: rf, params: {n_estimators: 5}}", "rf_results.csv"),
    ],
)
def test_run_saves_results(config_path, tmp_path, model_yaml, file_name):
    df = run(load_config(config_path(model_yaml)))

    saved = pd.read_csv(tmp_path / "out" / file_name)
    metrics = ["R2", "RMSE", "MAE", "MSE", "QLIKE", "MAPE"]
    assert list(saved.columns) == ["Ticker", *metrics]
    assert list(saved["Ticker"]) == ["AAA", "BBB"]
    assert np.allclose(saved[metrics], df[metrics])


def _spy_fits(monkeypatch):
    """
    Подменить get_model в run.py: каждый вызов fit записывается в список.
    """

    fits = []

    def spy_get_model(name, params=None):
        model = get_model(name, params)
        real_fit = model.fit

        def fit(X, y, **kwargs):
            fits.append({"model": model, "rows": len(X), "params": params})
            return real_fit(X, y, **kwargs)

        model.fit = fit
        return model

    monkeypatch.setattr("source.run.get_model", spy_get_model)
    return fits


def _boosting_yaml(name, refit):
    # Синтетика 2021-01..2022-06: train до сентября, valid — IV квартал, тест — 2022
    model_yaml = f"model: {{name: {name}, params: {{n_estimators: 20}}}}"
    split_yaml = (
        "{train_end: '2021-09-30', valid_end: '2021-12-31', "
        f"use_validation: true, refit: {'true' if refit else 'false'}}}"
    )
    return model_yaml, split_yaml


@pytest.mark.parametrize("name", ["xgb", "lgbm"])
def test_run_refits_on_train_and_valid(config_path, monkeypatch, name):
    fits = _spy_fits(monkeypatch)

    run(load_config(config_path(*_boosting_yaml(name, refit=True))))

    assert len(fits) == 4  # 2 тикера × (early stopping + refit)
    for first, refit in (fits[0:2], fits[2:4]):
        assert first["params"]["n_estimators"] == 20
        assert 1 <= first["model"].best_n_estimators <= 20
        assert refit["params"]["n_estimators"] == first["model"].best_n_estimators
        assert refit["rows"] > first["rows"]


@pytest.mark.parametrize("name", ["xgb", "lgbm"])
def test_run_without_refit_fits_once(config_path, monkeypatch, name):
    fits = _spy_fits(monkeypatch)

    run(load_config(config_path(*_boosting_yaml(name, refit=False))))

    assert len(fits) == 2  # по одному fit на тикер


def _write_experiment(tmp_path, data_extra=""):
    path = tmp_path / "exp.yaml"
    path.write_text(
        f"""
model: {{name: rf}}
data: {{tickers: [AAA, BBB], raw_dir: {tmp_path / "raw"},
        raw_file: all_prices.csv{data_extra}}}
processing: {{processed_dir: {tmp_path / "processed"}}}
""",
        encoding="utf-8",
    )
    return load_config(path)


def test_process_creates_rv_matrix(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(0)
    days = pd.bdate_range("2024-01-01", periods=5)
    times = [
        day + pd.Timedelta(minutes=600 + 10 * i) for day in days for i in range(53)
    ]
    raw = pd.DataFrame({"date": times})
    for ticker in ["AAA", "BBB"]:
        raw[f"{ticker}_close"] = 100 * np.exp(
            np.cumsum(rng.normal(0, 0.001, len(times)))
        )
    (tmp_path / "raw").mkdir()
    raw.to_csv(tmp_path / "raw" / "all_prices.csv", index=False)

    process(_write_experiment(tmp_path))

    rv_matrix = pd.read_csv(tmp_path / "processed" / "rv_matrix.csv")
    assert list(rv_matrix.columns) == ["date", "AAA_RV", "BBB_RV"]
    assert len(rv_matrix) == 5


def test_process_without_raw_data_suggests_load(tmp_path):
    with pytest.raises(FileNotFoundError, match="--stage load"):
        process(_write_experiment(tmp_path))


def test_train_without_any_data_suggests_load_and_process(tmp_path):
    with pytest.raises(
        FileNotFoundError, match="--stage load, затем .*--stage process"
    ):
        run(_write_experiment(tmp_path))


def test_train_without_processed_data_suggests_process(tmp_path):
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "all_prices.csv").write_text("date\n")
    with pytest.raises(FileNotFoundError) as error:
        run(_write_experiment(tmp_path))
    assert "--stage process" in str(error.value)
    assert "--stage load" not in str(error.value)


def test_load_passes_config_to_data_loader(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    captured = {}
    monkeypatch.setattr(
        "source.run.DataLoader.load_all",
        lambda self, **kwargs: captured.update(vars(self), **kwargs),
    )

    load(
        _write_experiment(
            tmp_path,
            data_extra=(
                ", start_year: 2019, end_year: 2020, source_interval: 1, interval: 5"
            ),
        )
    )

    assert captured["interval"] == 1
    assert captured["tickers"] == ["AAA", "BBB"]
    assert captured["start_year"] == 2019
    assert captured["end_year"] == 2020
    assert captured["save_path"] == tmp_path / "raw" / "all_prices.csv"
    assert captured["save_individual"] is True
