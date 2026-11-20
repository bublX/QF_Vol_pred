import pandas as pd
import pytest
import numpy as np

from source.data_processors.steps.fill import ForwardFillWithinDay
from source.data_processors.data_processor import DataProcessor
from source.data_processors.steps.returns import LogReturns
from source.data_processors.steps.clip import ClipReturns

@pytest.mark.parametrize("with_duplicate", [False, True])
def test_aggregation_and_source_duplicates(tmp_path, monkeypatch, with_duplicate):
    monkeypatch.chdir(tmp_path)
    processor = DataProcessor(
        source_interval=1, interval=5, returns_calculator=LogReturns()
    )
    warnings = []
    monkeypatch.setattr(processor.aggregator.logger, "warning", warnings.append)

    raw = pd.DataFrame(
        {
            "date": [
                "2024-01-09 10:00:59",
                "2024-01-09 10:04:59",
                "2024-01-09 10:05:59",
            ],
            "AAA_close": [100, 104, 105],
        }
    )
    if with_duplicate:
        raw.loc[len(raw)] = ["2024-01-09 10:04:15", 103]

    result = (
        processor._align_to_slots(raw)
        .dropna(subset=["AAA_close"])
        .reset_index(drop=True)
    )

    assert (
        result["date"].tolist()
        == pd.to_datetime(["2024-01-09 10:04:59", "2024-01-09 10:09:59"]).tolist()
    )
    assert result["AAA_close"].tolist() == [104, 105]
    assert len(warnings) == int(with_duplicate)


def test_aggregation_respects_session(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    processor = DataProcessor(
        source_interval=1, interval=7, returns_calculator=LogReturns()
    )

    raw = pd.DataFrame(
        {
            "date": [
                "2024-01-09 09:59:59",
                "2024-01-09 10:00:00",
                "2024-01-09 10:06:59",
                "2024-01-09 18:44:59",
                "2024-01-09 18:45:59",
                "2024-01-09 18:50:00",
            ],
            "AAA_close": [999, 100, 106, 200, 201, 999],
        }
    )

    result = (
        processor._align_to_slots(raw)
        .dropna(subset=["AAA_close"])
        .reset_index(drop=True)
    )

    assert (
        result["date"].tolist()
        == pd.to_datetime(["2024-01-09 10:06:59", "2024-01-09 18:44:59"]).tolist()
    )
    assert result["AAA_close"].tolist() == [106, 200]


@pytest.mark.parametrize(
    "interval, required",
    [(5, 80), (7, 57), (10, 40), (20, 20)],
)
def test_day_coverage_depends_on_interval(tmp_path, monkeypatch, interval, required):
    monkeypatch.chdir(tmp_path)
    processor = DataProcessor(
        interval=interval, min_day_coverage=0.75, returns_calculator=LogReturns()
    )

    dates = [
        pd.Timestamp(day)
        + pd.Timedelta(hours=10, minutes=interval * (i + 1))
        - pd.Timedelta(seconds=1)
        for day in ["2024-01-09", "2024-01-10"]
        for i in range(required)
    ]
    data = pd.DataFrame(
        {
            "date": dates,
            "close": [100.0] * (len(dates) - 1) + [float("nan")],
        }
    )

    result = processor._filter_valid_dates(data, "AAA")

    assert len(result) == required
    assert (result["date"].dt.date == pd.Timestamp("2024-01-09").date()).all()
    assert processor.missing_days["AAA"].to_dict() == {
        pd.Timestamp("2024-01-10").date(): required - 1,
    }


@pytest.mark.parametrize("use_fill", [False, True])
def test_pipeline_with_optional_forward_fill(tmp_path, monkeypatch, use_fill):
    monkeypatch.chdir(tmp_path)

    dates = pd.DatetimeIndex([
        *pd.date_range("2024-01-09 10:09:59", periods=4, freq="10min"),
        *pd.date_range("2024-01-10 10:09:59", periods=4, freq="10min"),
    ])
    data = pd.DataFrame({
        "date": dates,
        "close": [100, np.nan, 121, 133.1, np.nan, 200, 220, 242],
    })
    original = data.copy(deep=True)

    processor = DataProcessor(
        interval=10,
        session_start="10:00",
        session_end="10:40",
        min_day_coverage=0.75,
        price_steps=[ForwardFillWithinDay()] if use_fill else [],
        returns_calculator=LogReturns(),
        steps=[],
    )
    result = processor.process_ticker(data, "AAA")

    if use_fill:
        positions = [1, 2, 3, 6, 7]
        ratios = [1, 1.21, 1.1, 1.1, 1.1]
    else:
        positions = [3, 6, 7]
        ratios = [1.1, 1.1, 1.1]

    assert result["date"].tolist() == dates.take(positions).tolist()
    np.testing.assert_allclose(result["returns"], np.log(ratios) * 100)
    pd.testing.assert_frame_equal(data, original)


def test_day_coverage_is_checked_before_filling(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    dates = pd.DatetimeIndex([
        *pd.date_range("2024-01-09 10:09:59", periods=4, freq="10min"),
        *pd.date_range("2024-01-10 10:09:59", periods=4, freq="10min"),
    ])
    data = pd.DataFrame({
        "date": dates,
        "close": [100, 110, 121, 133.1, 200, np.nan, np.nan, 242],
    })

    processor = DataProcessor(
        interval=10,
        session_start="10:00",
        session_end="10:40",
        min_day_coverage=0.75,
        price_steps=[ForwardFillWithinDay()],
        returns_calculator=LogReturns(),
        steps=[],
    )
    result = processor.process_ticker(data, "AAA")

    assert result["date"].tolist() == dates.take([1, 2, 3]).tolist()
    assert processor.missing_days["AAA"].to_dict() == {
        pd.Timestamp("2024-01-10").date(): 2,
    }


@pytest.mark.parametrize(
    "max_return, expected",
    [(None, np.log(1.1) * 100), (2, 2), (5, 5)],
)
def test_pipeline_with_optional_clipping(tmp_path, monkeypatch, max_return, expected):
    monkeypatch.chdir(tmp_path)

    dates = pd.date_range("2024-01-09 10:09:59", periods=4, freq="10min")
    data = pd.DataFrame({
        "date": dates,
        "close": [100, 110, 100, 110],
    })

    processor = DataProcessor(
        session_start="10:00",
        session_end="10:40",
        price_steps=[],
        returns_calculator=LogReturns(),
        steps=[] if max_return is None else [ClipReturns(max_return=max_return)],
    )
    result = processor.process_ticker(data, "AAA")

    assert result["date"].tolist() == dates[1:].tolist()
    assert result["close"].tolist() == [110, 100, 110]
    np.testing.assert_allclose(
        result["returns"], [expected, -expected, expected]
    )