```text
QF_Vol_pred/
│
├── source/                          # Главная папка с исходным кодом
│   │
│   ├── data_processors/             # Обработка данных
│   │   ├── data_loader.py           # Загрузка данных с MOEX ISS API
│   │   └── data_processor.py        # Очистка, фильтрация, расчёт RV
│   │
│   ├── features/                    # Работа с признаками
│   │   ├── features_engineering.py  # Генерация HAR, лагов, статистик
│   │   └── data_splitter.py         # Разбиение на train/valid/test
│   │
│   ├── models/                      # Модели машинного обучения
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
│   ├── training.py                  # Запуск обучения моделей
│   └── run_pipeline.py              # Полный пайплайн
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
│   ├── 00_test_pipeline.ipynb        # Проверка загрузки
│   ├── 01_visualization.ipynb     # Визуализация
│   └── 02_correlations.ipynb                 # Корреляции
│
├── reports/                         # Отчёты и графики
│   ├── images/                     # Графики
│   ├── correlations/             # Корреляции 
│   └── logs/                        # Логи работы
│       ├── dataloader.log
│       ├── dataprocessor.log
│       └── featureengineer.log
|
└──  .gitignore                       # Исключения для Git
```
