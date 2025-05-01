"""
Модуль предобработки данных.

Содержит функции для:
- Создания циклических признаков времени (sin/cos).
- Логарифмирования признака 'Amount'.
- Создания трансформера для масштабирования (RobustScaler)
  новых признаков времени и суммы.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer, RobustScaler
from sklearn.compose import ColumnTransformer


def time_features(X):
    """Преобразует 'Time' в циклические sin/cos времени суток."""
    X_transformed = X.copy()
    seconds_in_day = 24 * 60 * 60
    X_transformed["time_sec_day"] = X_transformed["Time"] % seconds_in_day
    X_transformed["time_sin"] = np.sin(
        2 * np.pi * X_transformed["time_sec_day"] / seconds_in_day
    )
    X_transformed["time_cos"] = np.cos(
        2 * np.pi * X_transformed["time_sec_day"] / seconds_in_day
    )
    if "Time" in X_transformed.columns:
        X_transformed = X_transformed.drop(["Time", "time_sec_day"], axis=1)
    elif "time_sec_day" in X_transformed.columns:
        X_transformed = X_transformed.drop(["time_sec_day"], axis=1)

    return X_transformed


def amount_log_feature(X):
    """Применяет log1p к 'Amount'."""
    X_transformed = X.copy()
    if "Amount" in X_transformed.columns:
        X_transformed["Amount_log"] = np.log1p(X_transformed["Amount"])
        # Важно: Удаляем исходный 'Amount'
        X_transformed = X_transformed.drop("Amount", axis=1)
    return X_transformed


time_transformer = FunctionTransformer(time_features, validate=False)
amount_transformer = FunctionTransformer(amount_log_feature, validate=False)


def create_scaling_transformer():
    """
    Создает ColumnTransformer для масштабирования новых признаков
    'Amount_log', 'time_sin', 'time_cos' с помощью RobustScaler.
    Остальные признаки остаются без изменений.
    """
    # Признаки, созданные на предыдущих шагах, которые нужно масштабировать
    numeric_features_to_scale = ["Amount_log", "time_sin", "time_cos"]

    scaler = ColumnTransformer(
        transformers=[("num_scaler", RobustScaler(), numeric_features_to_scale)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    scaler.set_output(transform="pandas")
    return scaler
