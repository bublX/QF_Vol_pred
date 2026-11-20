from pathlib import Path
from typing import List
import zipfile
import shutil
import tempfile
import csv


class DatasetPreparer:
    """
    Класс заготовщик, который отвечает за преобразование
    в CSV данных из архива.

    Parameters
    ----------
    archive_path : str | Path
        Путь к ZIP-архиву. Используется, если выходной CSV отсутствует
    output_path : str | Path
        Путь к готовому CSV. Его имя должно совпадать с именем файла
        внутри архива, файл может находиться во вложенном каталоге
    tickers : List[str]
        Тикеры, для которых обязательны столбцы '{ticker}_close'.

    Notes
    -----
    Относительные пути отсчитываются от текущего рабочего каталога.
    Существующий CSV проверяется и используется повторно.
    Если он не проходит проверку, возникает исключение.

    Распаковка выполняется во временный файл. Итоговый CSV появляется
    после успешного копирования и проверки структуры.

    Ожидается CSV в UTF-8 с необязательным BOM и разделителем-запятой.
    Проверяются заголовок и наличие непустой строки данных.
    Корректность дат, цен и полнота истории здесь не проверяются.

    Examples
    --------
    >>> preparer = DatasetPreparer(
    ...     archive_path="data/archives/1m_2014_2025/prices_1m_2014_2025.zip",
    ...     output_path="data/raw/1m_2014_2025/prices_1m_2014_2025.csv",
    ...     tickers=["SBER", "GAZP"],
    ... )
    >>> csv_path = preparer.prepare()
    """

    def __init__(
        self,
        archive_path: str | Path,
        output_path: str | Path,
        tickers: List[str],
    ):
        self.archive_path = Path(archive_path)
        self.output_path = Path(output_path)
        self.tickers = tuple(tickers)

    def prepare(self) -> Path:
        """
        Подготовить CSV и вернуть путь к нему.

        Returns
        -------
        Path
            Путь к подготовленному и проверенному CSV-файлу.

        Raises
        ------
        FileNotFoundError
            Выходной CSV отсутствует, а архив или нужный файл внутри него
            не найден
        ValueError
            В архиве несколько подходящих файлов либо структура CSV
            не соответствует требованиям
        zipfile.BadZipFile
            Архив имеет неверный формат или повреждён
        OSError
            Ошибка доступа, чтения, записи или переименования файла
        """

        if self.output_path.exists():
            self._validate_csv(self.output_path)
            return self.output_path

        if not self.archive_path.exists():
            raise FileNotFoundError(f"Архив не найден: {self.archive_path}")

        with zipfile.ZipFile(self.archive_path, "r") as zf:
            target_name = self.output_path.name
            matching_files = [
                name
                for name in zf.namelist()
                if Path(name).name == target_name and not name.endswith("/")
            ]

            if not matching_files:
                raise FileNotFoundError(
                    f"Файл {target_name} не найден внутри архива {self.archive_path}"
                )

            if len(matching_files) > 1:
                raise ValueError(
                    f"В архиве несколько файлов с именем {target_name}: "
                    f"{matching_files}"
                )
            csv_in_zip = matching_files[0]

            output_dir = self.output_path.parent
            output_dir.mkdir(parents=True, exist_ok=True)

            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=output_dir,
                prefix=f".{self.output_path.stem}_",
                suffix=".tmp",
                delete=False,
            ) as tmp_file:
                tmp_path = Path(tmp_file.name)

            try:
                with zf.open(csv_in_zip) as src, tmp_path.open("wb") as dst:
                    shutil.copyfileobj(src, dst)

                self._validate_csv(tmp_path)

                tmp_path.replace(self.output_path)

            finally:
                tmp_path.unlink(missing_ok=True)

        return self.output_path

    def _validate_csv(self, path: Path) -> None:
        """
        Проверить структуру CSV, при ошибке вызвать ValueError.

        Parameters
        ----------
        path : Path
            Путь к проверяемому CSV

        Raises
        ------
        ValueError
            Нет заголовка, названия столбцов повторяются или отсутствуют
            обязательные столбцы или нет непустных строк данных

        Notes
        -----
        Обязательные столбцы: 'date' и '{ticker}_close' для каждого
        тикера из 'self.tickers'. Чтение завершается после первой
        непустой строки данных; содержимое всей таблицы не проверяется.
        """

        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)

            header = next(reader, None)
            if not header:
                raise ValueError(f"CSV-файл пуст или отсутствует заголовок: {path}")

            if len(header) != len(set(header)):
                raise ValueError(
                    f"В заголовке CSV найдены дублирующиеся колонки: {path}"
                )

            header_set = set(header)
            if "date" not in header_set:
                raise ValueError(
                    f"В CSV отсутствует обязательная колонка 'date': {path}"
                )

            missing_tickers = [
                f"{ticker}_close"
                for ticker in self.tickers
                if f"{ticker}_close" not in header_set
            ]
            if missing_tickers:
                raise ValueError(
                    f"В CSV отсутствуют колонки: {', '.join(missing_tickers)} ({path})"
                )

            has_data = any(any(cell.strip() for cell in row) for row in reader)
            if not has_data:
                raise ValueError(f"В CSV отсутствуют строки с данными: {path}")
