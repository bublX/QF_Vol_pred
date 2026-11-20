from typing import Any

from source.data_processors.steps.base import ProcessingStep
from source.data_processors.steps.clip import ClipReturns
from source.data_processors.steps.hampel import HampelFilter
from source.data_processors.steps.fill import ForwardFillWithinDay
from source.data_processors.steps.returns import LogReturns

PRICE_STEPS = {"hampel": HampelFilter, "forward_fill": ForwardFillWithinDay}
STEPS = {"hampel": HampelFilter, "clip": ClipReturns}
RETURN_CALCULATORS = {"log_returns": LogReturns}


def get_step(name: str, params: dict[str, Any] | None = None) -> ProcessingStep:
    """Создать новый экземпляр шага по имени и параметрам."""
    if name not in STEPS:
        raise ValueError(f"Неизвестный шаг: '{name}'. Доступные: {list(STEPS)}")

    return STEPS[name](**(params if params is not None else {}))


def get_price_step(name: str, params: dict[str, Any] | None = None) -> ProcessingStep:
    """Создать шаг обработки цен."""
    if name not in PRICE_STEPS:
        raise ValueError(
            f"Неизвестный шаг обработки цен: '{name}'. Доступные: {list(PRICE_STEPS)}"
        )

    return PRICE_STEPS[name](**(params if params is not None else {}))


def get_returns_calculator(
    name: str, params: dict[str, Any] | None = None
) -> ProcessingStep:
    "Создать шаг расчета доходностей из цен."
    if name not in RETURN_CALCULATORS:
        raise ValueError(
            f"Неизвестный расчет доходностей: {name}. Доступные: {list(RETURN_CALCULATORS)}"
        )

    return RETURN_CALCULATORS[name](**(params if params is not None else {}))
