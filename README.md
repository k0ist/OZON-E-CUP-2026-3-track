# Ozon E-Cup 2026 --- Customer GMV Prediction

##  Результат

**Private Leaderboard RMSLE: `1.6720521802`**
**Вошли в 26% лучших"

Финальное решение использует ансамбль:

  Модель             Вес
  ---------------- -----
  LightGBM           50%
  CatBoost           35%
  Tabular ResNet     15%

------------------------------------------------------------------------

##  Постановка задачи

Необходимо предсказать суммарный **GMV (Gross Merchandise Value)**
пользователя в Поиске и Каталоге Ozon за следующие 30 дней.

Исходные данные содержат ежедневную статистику активности пользователей:

-   поисковые запросы;
-   действия в каталоге;
-   добавления товаров в корзину;
-   оформления заказов;
-   GMV;
-   переходы между этапами пользовательской воронки.

История:

``` text
01.01.2025 — 13.02.2026
```

Прогнозируемый период:

``` text
14.02.2026 — 15.03.2026
```

Количество пользователей:

``` text
250 000
```

------------------------------------------------------------------------

## Метрика

Используется RMSLE:

$$
RMSLE =
\sqrt{
\frac{1}{n}
\sum_{i=1}^{n}
\left(
\log(1+y_i) -
\log(1+\hat{y}_i)
\right)^2
}
$$

Модель обучается на логарифме target:

``` python
target_log = np.log1p(target)
```

После предсказания:

``` python
prediction = np.expm1(prediction_log)
```

Отрицательные значения обрезаются до нуля.

------------------------------------------------------------------------

#  Пайплайн

``` text
RAW DATA
   │
   ▼
Feature Engineering
   │
   ├── Recency / Lifetime
   ├── Purchase statistics
   ├── Rolling windows
   ├── Behavioral ratios
   ├── EMA
   ├── BTYD
   └── Log features
   │
   ▼
Temporal Cross-Validation
   │
   ├── LightGBM
   ├── CatBoost
   └── Tabular ResNet
   │
   ▼
OOF predictions
   │
   ▼
Ensemble
   │
   ▼
Submission
```

------------------------------------------------------------------------

#  Feature Engineering

## 1. Recency / Lifetime

Рассчитываются:

``` text
recency_days
tenure_days
n_active_days_total
activity_density_total
```

Они описывают свежесть и длительность пользовательской активности.

------------------------------------------------------------------------

## 2. Purchase Features

Для покупок используются:

``` text
recency_purchase_days
mean_purchase_gap
std_purchase_gap
n_purchase_days
purchase_frequency
total_purchase_gmv
first_purchase_days
```

Они позволяют различать регулярных покупателей и пользователей, давно
прекративших покупки.

------------------------------------------------------------------------

## 3. Rolling Windows

Агрегаты строятся на горизонтах:

``` text
7d
14d
30d
60d
90d
180d
365d
```

Примеры:

``` text
gmv_sum_7d
gmv_sum_30d
gmv_sum_90d
gmv_sum_180d
gmv_sum_365d

to_ord_sum_7d
to_ord_sum_30d
to_ord_sum_90d
to_ord_sum_180d
to_ord_sum_365d

n_active_days_7d
n_purchase_days_60d
n_purchase_days_90d
n_purchase_days_180d
n_purchase_days_365d
```

Использование нескольких горизонтов позволяет модели учитывать как
последние дни, так и долгосрочное поведение пользователя.

------------------------------------------------------------------------

## 4. Behavioral Ratios

Строятся признаки:

``` text
gmv_per_active_day
avg_order_value
cart_to_order_rate
search_to_cart_rate
activity_density
```

Также используются отношения между короткими и длинными периодами,
например:

``` text
7d / 30d
30d / 90d
90d / 365d
```

если они доступны в конкретной версии feature pipeline.

------------------------------------------------------------------------

## 5. Log Features

Для уменьшения влияния выбросов используются:

``` text
log1p_to_ord_sum_180d
log1p_to_ord_sum_365d
log1p_n_purchase_days_180d
log1p_n_purchase_days_90d
log1p_btyd_exp_orders_30d
log1p_btyd_exp_gmv
```

Логарифмические признаки хорошо соответствуют природе RMSLE.

------------------------------------------------------------------------

#  BTYD

``` text
BG/NBD
+
Gamma-Gamma
```

Основные признаки:

``` text
btyd_p_alive
btyd_exp_orders_30d
btyd_exp_gmv_30d
```

### `btyd_p_alive`

Вероятность того, что пользователь всё ещё активен.

### `btyd_exp_orders_30d`

Ожидаемое количество покупок в следующие 30 дней.

### `btyd_exp_gmv_30d`

Ожидаемый GMV в следующие 30 дней.

BTYD-признаки оказались среди наиболее важных признаков LightGBM.

------------------------------------------------------------------------

# 📈 EMA

Для учета затухающей активности используются экспоненциально взвешенные
признаки:

``` text
ema_gmv_7d
ema_gmv_30d
ema_gmv_90d
```

Недавние действия получают больший вес.

------------------------------------------------------------------------

# Temporal Cross-Validation

Случайный K-Fold не используется, поскольку он может привести к утечке
информации из будущего.

Используется временная схема:

``` text
ВРЕМЯ ───────────────────────────────────────►

Fold 1:
[ TRAIN ] [ VALID ]

Fold 2:
[ TRAIN -------- ] [ VALID ]

Fold 3:
[ TRAIN ------------- ] [ VALID ]

Fold 4:
[ TRAIN ------------------ ] [ VALID ]

Fold 5:
[ TRAIN ----------------------- ] [ VALID ]
```

Каждый validation fold расположен после соответствующей обучающей
выборки.

Это приближает локальную оценку к реальному сценарию соревнования.

------------------------------------------------------------------------

#  LightGBM

LightGBM является основной моделью.

Используется:

-   regression objective;
-   `target_log`;
-   early stopping;
-   temporal CV;
-   OOF predictions;
-   feature importance.

Пример запуска:

``` bash
python train.py --mode log_target --device cpu
```

Для GPU:

``` bash
python train.py --mode log_target --device gpu
```

Если установленная сборка LightGBM не поддерживает CUDA, pipeline
переключается на CPU.

------------------------------------------------------------------------

#  CatBoost

CatBoost используется как вторая независимая gradient boosting модель.

Основная цель --- получить другой набор ошибок относительно LightGBM и
повысить качество ансамбля.

Запуск:

``` bash
python train_cb.py
```

------------------------------------------------------------------------

#  Tabular ResNet

Для разнообразия ансамбля используется нейросеть для табличных данных.

Архитектура:

``` text
Input
  │
  ▼
Linear
  │
BatchNorm
  │
SiLU
  │
ResNet Block
  │
ResNet Block
  │
Linear
  │
Prediction
```

Residual connection:

``` python
return x + self.block(x)
```

Используются:

-   BatchNorm;
-   SiLU;
-   Dropout;
-   AdamW;
-   Huber Loss;
-   ReduceLROnPlateau;
-   Early Stopping;
-   несколько random seeds.

Предсказания разных seed усредняются.

------------------------------------------------------------------------

#  Ensemble

Финальная формула:

``` text
Prediction =
    0.50 × LightGBM
  + 0.35 × CatBoost
  + 0.15 × Neural Network
```

  Модель              Вес
  ---------------- ------
  LightGBM           0.50
  CatBoost           0.35
  Neural Network     0.15

Разные модели имеют разные индуктивные свойства и частично независимые
ошибки, поэтому ансамбль оказывается устойчивее отдельных моделей.

------------------------------------------------------------------------

#  Feature Importance

Среди наиболее важных признаков LightGBM:

``` text
btyd_exp_orders_30d
to_ord_sum_180d
n_purchase_days_180d
btyd_exp_gmv_30d
btyd_log_exp_orders
n_purchase_days
log1p_to_ord_sum_180d
n_purchase_days_90d
to_ord_sum_365d
btyd_log_exp_gmv
std_purchase_gap
purchase_frequency
ema_gmv_90d
recency_days
```
------------------------------------------------------------------------

#  Структура проекта

``` text
OZON/
│
├── data/
│   ├── fold_0.parquet
│   ├── fold_1.parquet
│   ├── fold_2.parquet
│   ├── fold_3.parquet
│   ├── fold_4.parquet
│   ├── fold_5.parquet
│   ├── test_features.parquet
│   ├── lgbm_feature_importances.csv
│   └── oof/
│       ├── oof_lgbm.csv
│       └── oof_nn.csv
│
├── models/
│   └── model_fold_*.txt
│
├── submissions/
│   └── submission_*.csv
│
├── uploads/
│   ├── train.parquet
│   └── sample_submission.csv
│
├── btyd_features.py
├── build_dataset.py
├── config.py
├── data_loading.py
├── features.py
├── make_synthetic_data.py
├── predict.py
├── time_split.py
├── train.py
├── train_cb.py
├── tune.py
├── Ensemble.ipynb
└── README.md
```

------------------------------------------------------------------------

# 📝 Описание файлов

  Файл                 Назначение
  -------------------- ---------------------------------
  `config.py`          Конфигурация проекта
  `data_loading.py`    Загрузка исходных данных
  `features.py`        Feature Engineering
  `btyd_features.py`   BG/NBD + Gamma-Gamma
  `time_split.py`      Temporal CV
  `build_dataset.py`   Создание CV/test датасетов
  `train.py`           Обучение LightGBM
  `train_cb.py`        Обучение CatBoost
  `tune.py`            Эксперименты с гиперпараметрами
  `predict.py`         Получение предсказаний
  `Ensemble.ipynb`     Blending моделей

------------------------------------------------------------------------

#  Результаты

  Подход                                                   Результат
  --------------------------------- --------------------------------
  LightGBM baseline                                     \~1.71 RMSLE
  Финальный Ensemble                  **1.6720521802 Private RMSLE**
