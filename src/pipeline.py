from imblearn.pipeline import Pipeline as ImbPipeline # Используем для совместимости
from src.data_preprocessing import (
    time_transformer,
    amount_transformer,
    create_scaling_transformer
)

def create_pipeline(model):
    """
    Собирает полный пайплайн: FE -> Scaling -> Model.
    Балансировка предполагается внутри модели (например, class_weight).
    """
    scaler = create_scaling_transformer()

    pipeline_steps = [
        ('time_features', time_transformer),
        ('amount_log', amount_transformer),
        ('scaler', scaler),
        ('classifier', model) # Имя шага классификатора
    ]
    # Используем ImbPipeline для единообразия
    pipeline = ImbPipeline(steps=pipeline_steps)
    return pipeline