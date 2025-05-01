"""
Модуль для создания полного пайплайна машинного обучения.

Объединяет шаги предобработки данных (создание признаков времени,
логарифмирование суммы, масштабирование) с финальной моделью классификации.
Использует Pipeline из imblearn для потенциальной совместимости с семплерами,
хотя в данной конфигурации семплинг не применяется напрямую в пайплайне.
"""

from imblearn.pipeline import Pipeline as ImbPipeline
from src.data_preprocessing import (
    time_transformer,
    amount_transformer,
    create_scaling_transformer,
)


def create_pipeline(model):
    """
    Собирает и возвращает полный пайплайн обработки данных и моделирования.

    Включает шаги:
    1. Создание циклических признаков времени ('time_features').
    2. Логарифмирование признака суммы ('amount_log').
    3. Масштабирование новых признаков с помощью RobustScaler ('scaler').
    4. Применение финальной модели ('classifier').

    Предполагается, что балансировка классов обрабатывается внутри
    переданной модели (например, через параметр class_weight='balanced').

    Args:
        model: Обучаемая модель классификации (например, LGBMClassifier).

    Returns:
        imblearn.pipeline.Pipeline: Собранный пайплайн.
    """
    scaler = create_scaling_transformer()

    pipeline_steps = [
        ("time_features", time_transformer),
        ("amount_log", amount_transformer),
        ("scaler", scaler),
        ("classifier", model),  # Шаг с финальной моделью
    ]

    pipeline = ImbPipeline(steps=pipeline_steps)
    return pipeline
