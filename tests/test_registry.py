import pytest

from source.models.registry import MODELS, get_model


@pytest.fixture(autouse=True)
def _run_in_tmp(tmp_path, monkeypatch):
    # Logger пишет логи в reports/logs/ относительно текущей директории
    monkeypatch.chdir(tmp_path)


@pytest.mark.parametrize("name", list(MODELS))
def test_get_model_returns_registered_class(name):
    assert isinstance(get_model(name), MODELS[name])


def test_params_are_passed_to_model():
    model = get_model("rf", {"n_estimators": 7, "max_depth": 3})
    assert model.model.n_estimators == 7
    assert model.model.max_depth == 3


def test_unknown_model():
    with pytest.raises(ValueError, match="Неизвестная модель: 'garch'"):
        get_model("garch")
