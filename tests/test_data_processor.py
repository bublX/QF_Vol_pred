import pandas as pd
import pytest

from source.data_processors.data_processor import DataProcessor


@pytest.mark.parametrize("with_duplicate", [False, True])
def test_aggregation_and_source_duplicates(tmp_path, monkeypatch, with_duplicate):
    monkeypatch.chdir(tmp_path)
    processor = DataProcessor(source_interval=1, interval=5)
    warnings = []
    monkeypatch.setattr(processor.logger, "warning", warnings.append)

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

    result = processor._align_to_slots(raw)

    assert (
        result["date"].tolist()
        == pd.to_datetime(["2024-01-09 10:04:59", "2024-01-09 10:09:59"]).tolist()
    )
    assert result["AAA_close"].tolist() == [104, 105]
    assert len(warnings) == int(with_duplicate)

def test_aggregation_respects_session(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    processor = DataProcessor(source_interval=1, interval=7)

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

    result = processor._align_to_slots(raw)

    assert result["date"].tolist() == pd.to_datetime(
        ["2024-01-09 10:06:59", "2024-01-09 18:44:59"]
    ).tolist()
    assert result["AAA_close"].tolist() == [106, 200]