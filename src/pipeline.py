"""ML pipeline assembly.

Combines preprocessing steps (cyclic time encoding, log-Amount, scaling)
with the final classifier into a single sklearn-compatible Pipeline.
"""

from imblearn.pipeline import Pipeline as ImbPipeline

from src.data_preprocessing import (
    create_amount_transformer,
    create_scaling_transformer,
    create_time_transformer,
)


def create_pipeline(model) -> ImbPipeline:
    """Assemble and return the full preprocessing + modeling pipeline.

    Steps:
      1. time_features  — cyclic sin/cos encoding of 'Time'
      2. amount_log     — log1p transformation of 'Amount'
      3. scaler         — RobustScaler on Amount_log, time_sin, time_cos
      4. classifier     — the provided model

    Class imbalance is handled by the model (class_weight='balanced' in LGBMClassifier).

    Args:
        model: An unfitted sklearn-compatible classifier.

    Returns:
        imblearn.pipeline.Pipeline
    """
    return ImbPipeline(
        steps=[
            ("time_features", create_time_transformer()),
            ("amount_log", create_amount_transformer()),
            ("scaler", create_scaling_transformer()),
            ("classifier", model),
        ]
    )
