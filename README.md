# QF_Vol_pred

Прогнозирование дневной реализованной волатильности (RV) акций MOEX.

## Установка и запуск

### Требования

- [uv](https://docs.astral.sh/uv/) — менеджер пакетов и окружений (обязателен).
  Python 3.14 uv скачает сам.
- OpenMP — нужен XGBoost и LightGBM.

| ОС      | uv                                                                                                | OpenMP                                                                                                                                                      |
|---------|---------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| macOS   | `brew install uv`                                                                                 | `brew install libomp`                                                                                                                                       |
| Linux   | `curl -LsSf https://astral.sh/uv/install.sh \| sh`                                                | обычно уже есть; если нет: `sudo apt install libgomp1` (Debian/Ubuntu) или `sudo dnf install libgomp` (Fedora)                                              |
| Windows | PowerShell: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 \| iex"` | входит в [Microsoft Visual C++ Redistributable](https://learn.microsoft.com/cpp/windows/latest-supported-vc-redist) (`vcomp140.dll`); обычно уже установлен |

### Установка

```bash
uv sync
```

### Проверка

```bash
uv run pytest           # тесты
uv run ruff check .     # линтер
uv run ruff format .    # форматирование
```

### Запуск

Все команды запускаются из корня проекта.

Комплектный архив минутных цен хранится в `data/archives/1m_2014_2025/`.
Сначала распакуйте его по [инструкции по данным](data/README.md); повторное скачивание не требуется.

```bash
uv run python main.py --config configs/har.yaml --stage process  # -> data/processed/10m_default_2014_2025/
uv run python main.py --config configs/har.yaml                  # обучение (stage train) -> reports/tables/har_results.csv
```

Для скачивания другого набора настройте секцию `data` и выполните `--stage load`.
Этап `process` выполняется для каждого варианта обработки; затем его данные могут использовать разные модели.

Готовые эксперименты: `configs/har.yaml`, `rf.yaml`, `xgb.yaml`, `lgbm.yaml`.
Все значения по умолчанию лежат в `configs/default.yaml`
Файл эксперимента переопределяет только то, что отличается.
Новый эксперимент — новый YAML-файл в `configs/`.
Новая модель — класс в `source/models/` и строка в `MODELS` в `source/models/registry.py`.

## Структура проекта

```text
QF_Vol_pred/
│
├── configs/                         # YAML-конфиги экспериментов
│   ├── default.yaml                 # Все значения по умолчанию
│   └── har.yaml, rf.yaml, ...       # Эксперименты (только отличия)
│
├── source/                          # Главная папка с исходным кодом
│   ├── config.py                    # Загрузка и проверка конфигов
│   ├── run.py                       # Этапы load / process / train (вызывается из main.py)
│   │
│   ├── data_processors/             # Обработка данных
│   │   ├── data_loader.py           # Загрузка данных с MOEX ISS API
│   │   └── data_processor.py        # Очистка, фильтрация, расчёт RV
│   │
│   ├── features/                    # Работа с признаками
│   │   ├── features_engineering.py  # Генерация HAR, лагов, статистик
│   │   └── data_splitter.py         # Разбиение на train/valid/test
│   │
│   ├── autoregression/              # Авторегрессионные модели
│   │   ├── base_ar.py               # Базовый класс
│   │   └── har_model.py             # HAR
│   │
│   ├── models/                      # Модели машинного обучения
│   │   ├── registry.py              # Реестр моделей по названию
│   │   ├── base.py                  # Базовый абстрактный класс
│   │   ├── random_forest_model.py   # Random Forest
│   │   ├── xgboost_model.py         # XGBoost
│   │   └── lightgbm_model.py        # LightGBM
│   │
│   └── utils/                       # Вспомогательные утилиты
│       ├── logger.py                # Единый логгер
│       └── metrics.py               # Метрики (MSE, RMSE, MAE, R2, QLIKE, MAPE)
│
├── main.py                          # Точка входа проекта
│
├── tests/                           # Тесты (uv run pytest)
│
├── data/
│   ├── README.md                    # Подготовка и описание структуры
│   ├── archives/1m_2014_2025/       # Исходный ZIP (в Git)
│   ├── raw/1m_2014_2025/            # Распакованные минутные цены (локально)
│   └── processed/                   # Производные наборы (локально)
│       ├── 10m_default_2014_2025/   # Стандартная обработка
│       ├── 5m_2014_2025/            # configs/har_5m.yaml
│       └── 10m_2014_2025/           # configs/har_10m.yaml
│       
│
├── notebooks/                       # Jupyter ноутбуки
│   ├── 00_test_pipeline.ipynb       # Проверка загрузки
│   ├── 01_visualization.ipynb       # Визуализация
│   └── 02_correlations.ipynb        # Корреляции
│
├── reports/                         # Отчёты и графики
│   ├── tables/                      # Метрики моделей ({name}_results.csv)
│   ├── images/                      # Графики
│   ├── correlations/                # Корреляции 
│   └── logs/                        # Логи работы
│       ├── dataloader.log
│       ├── dataprocessor.log
│       └── featureengineer.log
|
└──  .gitignore                      # Исключения для Git
```
