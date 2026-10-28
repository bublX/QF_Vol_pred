import numpy as np
import pandas as pd
import pytest

from source.config import load_config
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

    def write(model_yaml):
        path = tmp_path / "exp.yaml"
        path.write_text(
            f"""
data: {{tickers: [AAA, BBB]}}
processing: {{processed_dir: {tmp_path / "processed"}}}
split: {{valid_end: null, use_validation: false}}
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
    assert list(saved.columns) == ["Ticker", "R2", "RMSE", "MAE", "MSE"]
    assert list(saved["Ticker"]) == ["AAA", "BBB"]
    assert np.allclose(
        saved[["R2", "RMSE", "MAE", "MSE"]], df[["R2", "RMSE", "MAE", "MSE"]]
    )


def _write_experiment(tmp_path, data_extra=""):
    path = tmp_path / "exp.yaml"
    path.write_text(
        f"""
model: {{name: rf}}
data: {{tickers: [AAA, BBB], raw_dir: {tmp_path / "raw"}{data_extra}}}
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
        "source.run.DataLoader.load_all", lambda self: captured.update(vars(self))
    )

    load(_write_experiment(tmp_path, data_extra=", start_year: 2019, end_year: 2020"))

    assert captured["tickers"] == ["AAA", "BBB"]
    assert captured["start_year"] == 2019
    assert captured["end_year"] == 2020
    assert captured["save_path"] == tmp_path / "raw" / "all_prices.csv"
