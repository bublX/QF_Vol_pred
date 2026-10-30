import numpy as np
import pandas as pd
import pytest

from source.features.features_engineering import FeatureEngineer


@pytest.fixture
def rv_matrix():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2021-01-01", periods=60)
    return pd.DataFrame(
        {"date": dates.strftime("%Y-%m-%d"), "AAA_RV": rng.lognormal(0, 0.5, 60)}
    )


def test_target_is_sqrt_of_next_day_rv(rv_matrix):
    targets = FeatureEngineer().create_targets(rv_matrix, ["AAA"])

    assert len(targets) == len(rv_matrix) - 1  # у последнего дня нет «завтра»
    expected = np.sqrt(rv_matrix["AAA_RV"].shift(-1).dropna())
    assert np.allclose(targets["AAA_target"], expected)


def test_har_features_are_sqrt_of_rv_averages(rv_matrix):
    features = FeatureEngineer().create_har_features(rv_matrix, ["AAA"])

    rv = rv_matrix["AAA_RV"]
    row = features.iloc[0]  # первая строка, где доступно окно 22
    i = rv_matrix.index[rv_matrix["date"] == row["date"]][0]
    assert row["AAA_rv_1"] == pytest.approx(np.sqrt(rv[i]))
    assert row["AAA_rv_5"] == pytest.approx(np.sqrt(rv[i - 4 : i + 1].mean()))
    assert row["AAA_rv_22"] == pytest.approx(np.sqrt(rv[i - 21 : i + 1].mean()))
