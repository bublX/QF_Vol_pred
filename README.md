# QF_Vol_pred

Прогнозирование дневной реализованной волатильности (RV) акций MOEX.

## Установка и запуск

### Требования

- [uv](https://docs.astral.sh/uv/) — менеджер пакетов и окружений (обязателен).
  Python 3.14 uv скачает сам (версия задана в `.python-version`).
- OpenMP — нужен XGBoost и LightGBM.

| ОС | uv | OpenMP |
|---|---|---|
| macOS | `brew install uv` | `brew install libomp` |
| Linux | `curl -LsSf https://astral.sh/uv/install.sh \| sh` | обычно уже есть; если нет: `sudo apt install libgomp1` (Debian/Ubuntu) или `sudo dnf install libgomp` (Fedora) |
| Windows | `winget install --id=astral-sh.uv -e` или в PowerShell: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 \| iex"` | входит в [Microsoft Visual C++ Redistributable](https://learn.microsoft.com/cpp/windows/latest-supported-vc-redist) (`vcomp140.dll`); обычно уже установлен |

Команды ниже одинаковы для всех ОС (в PowerShell тоже работают пути через `/`).

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

```bash
uv run python scripts/loading.py      # загрузка цен с MOEX (сеть, долго) -> data/raw/
uv run python scripts/processing.py   # очистка и расчёт RV -> data/processed/
uv run python -m source.run --config configs/har.yaml   # обучение -> reports/tables/har_results.csv
```

Готовые эксперименты: `configs/har.yaml`, `rf.yaml`, `xgb.yaml`, `lgbm.yaml`.
Все значения по умолчанию лежат в `configs/default.yaml`; файл эксперимента
переопределяет только то, что отличается. Новый эксперимент — новый YAML-файл в `configs/`.
Новая модель — класс в `source/models/` и строка в `MODELS` в `source/models/registry.py`.

Скрипты `scripts/train_*.py` пока оставлены для сверки и будут удалены.

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
│   ├── run.py                       # Точка входа: обучение по конфигу
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
│       └── metrics.py               # Метрики (MSE, RMSE, MAE, R2)
│
├── scripts/                         # Скрипты для запуска
│   ├── loading.py                   # Запуск загрузки данных
│   ├── processing.py                # Запуск обработки
│   ├── train_*.py                   # Старые скрипты обучения (заменены source/run.py)
│   └── run_pipeline.py              # Полный пайплайн
│
├── tests/                           # Тесты (uv run pytest)
│
├── data/                            # Данные (не коммитятся)
│   ├── raw/                         # Сырые данные
│   │   └── all_prices.csv           # Цены от DataLoader
│   │
│   └── processed/                   # Обработанные данные
│       ├── RV/                      # RV по каждому тикеру
│       │   ├── SBER_RV.csv
│       │   ├── VTBR_RV.csv
│       │   └── ...
│       ├── rv_matrix.csv            # Все RV в одной таблице
│       └── profit_matrix.csv        # Все доходности
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
