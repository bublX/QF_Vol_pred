import logging
import sys
from pathlib import Path
from typing import Optional


class Logger:
    """
    Единый логгер для проекта.

    Parameters
    ----------
    name : str
        Имя логгера
    log_file : Optional[str]
        Путь к файлу логов (None - только консоль)
    level : int
        Уровень логирования (по умолчанию INFO)
    """

    _loggers = {}

    def __new__(
        cls, name: str, log_file: Optional[str] = None, level: int = logging.INFO
    ):
        if name in cls._loggers:
            return cls._loggers[name]
        instance = super().__new__(cls)
        cls._loggers[name] = instance
        return instance

    def __init__(
        self, name: str, log_file: Optional[str] = None, level: int = logging.INFO
    ):
        if hasattr(self, "_initialized"):
            return
        self._initizialized = True

        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        formatter = logging.Formatter(
            "%(asctime)s | %(name)-15s | %(levelname)-8s | %(message)s",
            datefmt="%H:%M:%S",
        )

        if not self.logger.handlers:
            console = logging.StreamHandler(sys.stdout)
            console.setFormatter(formatter)
            self.logger.addHandler(console)

            if log_file is None:
                log_file = f"reports/logs/{name.lower()}.log"

            path = Path(log_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def info(self, msg: str):
        """
        Обычное сообщение
        Parameters
        ----------
        msg : str
            Текст выводимого сообщения
        """

        self.logger.info(f" {msg}")

    def success(self, msg: str):
        """
        Успешное завершение.
        Parameters
        ----------
        msg : str
            Текст выводимого сообщения
        """

        self.logger.info(f" {msg}")

    def warning(self, msg: str):
        """
        Предупреждение.

        Parameters
        ----------
        msg : str
            Текст выводимого сообщения
        """

        self.logger.warning(f" {msg}")

    def error(self, msg: str):
        """
        Ошибка.

        Parameters
        ----------
        msg : str
            Текст выводимого сообщения
        """

        self.logger.error(f" {msg}")

    def debug(self, msg: str):
        """
        Отладочное сообщение.

        Parameters
        ----------
        msg : str
            Текст выводимого сообщения
        """

        self.logger.debug(f" {msg}")

    def step(self, step: str, msg: str):
        """
        Шаг обработки.

        Parameters
        ----------
        step : str
            Шаг обработки
        msg : str
            Текст выводимого сообщения
        """

        self.logger.info(f" [{step}] {msg}")
