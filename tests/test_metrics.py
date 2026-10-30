import numpy as np
import pytest

from source.utils.metrics import MetricsCalculator


def test_qlike_and_mape():
    calculator = MetricsCalculator(["QLIKE", "MAPE"])
    y = np.array([1.0, 2.0, 4.0])

    assert calculator.calculate(y, y) == pytest.approx({"QLIKE": 0, "MAPE": 0})
    # прогноз вдвое выше: y / p = 0.5, ошибка 100%
    assert calculator.calculate(y, 2 * y) == pytest.approx(
        {"QLIKE": 0.5 - np.log(0.5) - 1, "MAPE": 1.0}
    )


def test_qlike_penalizes_underprediction_more():
    y = np.array([1.0, 2.0, 4.0])
    assert MetricsCalculator.qlike(y, y / 2) > MetricsCalculator.qlike(y, 2 * y)


def test_qlike_finite_for_nonpositive_prediction():
    y = np.array([1.0, 2.0])
    assert np.isfinite(MetricsCalculator.qlike(y, np.array([0.0, -1.0])))
