import numpy as np
import pandas as pd
import pytest

from source.config import load_config
from source.run import run


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
