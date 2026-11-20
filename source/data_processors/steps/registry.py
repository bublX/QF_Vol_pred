from typing import Any

from source.data_processors.steps.base import ProcessingStep
from source.data_processors.steps.clip import ClipReturns
from source.data_processors.steps.hampel import HampelFilter


STEPS = {"hampel": HampelFilter, "clip": ClipReturns}


def get_step(name: str, params: dict[str, Any] | None = None) -> ProcessingStep:
    """Создать новый экземпляр шага по имени и параметрам."""
    if name not in STEPS:
        raise ValueError(f"Неизвестный шаг: '{name}'. Доступные: {list(STEPS)}")

    return STEPS[name](**(params if params is not None else {}))
