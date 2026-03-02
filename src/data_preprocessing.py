"""Data preprocessing module.

Provides:
  - time_features: cyclic sin/cos encoding of the 'Time' column
  - amount_log_feature: log1p transformation of the 'Amount' column
  - create_scaling_transformer: RobustScaler applied to the newly engineered features
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, RobustScaler


def time_features(X: pd.DataFrame) -> pd.DataFrame:
    """Convert 'Time' (seconds since first transaction) to cyclic sin/cos of the 24-hour period."""
    X_out = X.copy()
    seconds_in_day = 24 * 60 * 60
    X_out["time_sec_day"] = X_out["Time"] % seconds_in_day
    X_out["time_sin"] = np.sin(2 * np.pi * X_out["time_sec_day"] / seconds_in_day)
    X_out["time_cos"] = np.cos(2 * np.pi * X_out["time_sec_day"] / seconds_in_day)
    drop_cols = [c for c in ("Time", "time_sec_day") if c in X_out.columns]
    return X_out.drop(drop_cols, axis=1)


def amount_log_feature(X: pd.DataFrame) -> pd.DataFrame:
    """Apply log1p to 'Amount' and drop the original column."""
    X_out = X.copy()
    if "Amount" in X_out.columns:
        X_out["Amount_log"] = np.log1p(X_out["Amount"])
        X_out = X_out.drop("Amount", axis=1)
    return X_out


def create_time_transformer() -> FunctionTransformer:
    """Factory for a stateless cyclic-time FunctionTransformer."""
    return FunctionTransformer(time_features)


def create_amount_transformer() -> FunctionTransformer:
    """Factory for a stateless log-Amount FunctionTransformer."""
    return FunctionTransformer(amount_log_feature)


def create_scaling_transformer() -> ColumnTransformer:
    """Build a ColumnTransformer that applies RobustScaler to engineered features.

    Scales Amount_log, time_sin, time_cos. All other columns (V1–V28) pass through unchanged.
    """
    numeric_features_to_scale = ["Amount_log", "time_sin", "time_cos"]
    scaler = ColumnTransformer(
        transformers=[("num_scaler", RobustScaler(), numeric_features_to_scale)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    scaler.set_output(transform="pandas")
    return scaler
