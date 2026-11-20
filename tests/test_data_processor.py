import pandas as pd
import pytest

from source.data_processors.data_processor import DataProcessor
from source.data_processors.steps.returns import LogReturns


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
