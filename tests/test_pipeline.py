"""Smoke tests for the full ML pipeline."""

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier

from src.pipeline import create_pipeline


def _make_data(n: int = 120, seed: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    """Synthetic dataset that matches the expected column structure."""
    rng = np.random.default_rng(seed)
    cols = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
    X = pd.DataFrame(rng.standard_normal((n, len(cols))), columns=cols)
    X["Time"] = np.abs(X["Time"]) * 86400      # non-negative seconds
    X["Amount"] = np.abs(X["Amount"]) * 100    # non-negative amounts
    y = pd.Series(rng.integers(0, 2, n), name="Class")
    return X, y


@pytest.fixture(scope="module")
def fitted_pipeline():
    X, y = _make_data()
    model = LGBMClassifier(n_estimators=10, verbosity=-1, random_state=42)
    pipe = create_pipeline(model)
    pipe.fit(X, y)
    return pipe, X, y


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------

def test_pipeline_has_four_steps():
    model = LGBMClassifier(n_estimators=5, verbosity=-1)
    pipe = create_pipeline(model)
    assert len(pipe.steps) == 4


def test_pipeline_step_names():
    model = LGBMClassifier(n_estimators=5, verbosity=-1)
    pipe = create_pipeline(model)
    names = [name for name, _ in pipe.steps]
    assert names == ["time_features", "amount_log", "scaler", "classifier"]


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

def test_predict_shape(fitted_pipeline):
    pipe, X, _ = fitted_pipeline
    preds = pipe.predict(X)
    assert preds.shape == (len(X),)


def test_predict_binary_labels(fitted_pipeline):
    pipe, X, _ = fitted_pipeline
    preds = pipe.predict(X)
    assert set(preds).issubset({0, 1})


def test_predict_proba_shape(fitted_pipeline):
    pipe, X, _ = fitted_pipeline
    proba = pipe.predict_proba(X)
    assert proba.shape == (len(X), 2)


def test_predict_proba_valid_range(fitted_pipeline):
    pipe, X, _ = fitted_pipeline
    proba = pipe.predict_proba(X)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_predict_proba_rows_sum_to_one(fitted_pipeline):
    pipe, X, _ = fitted_pipeline
    proba = pipe.predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Preprocessing: Time and Amount columns must be consumed
# ---------------------------------------------------------------------------

def test_time_column_removed_after_transform(fitted_pipeline):
    pipe, X, _ = fitted_pipeline
    preprocessor_steps = pipe.steps[:-1]
    from sklearn.pipeline import Pipeline as SkPipeline
    preprocessor = SkPipeline(preprocessor_steps)
    X_tf = preprocessor.transform(X)
    if hasattr(X_tf, "columns"):
        assert "Time" not in X_tf.columns
        assert "Amount" not in X_tf.columns


def test_engineered_features_present_after_transform(fitted_pipeline):
    pipe, X, _ = fitted_pipeline
    preprocessor_steps = pipe.steps[:-1]
    from sklearn.pipeline import Pipeline as SkPipeline
    preprocessor = SkPipeline(preprocessor_steps)
    X_tf = preprocessor.transform(X)
    if hasattr(X_tf, "columns"):
        assert "time_sin" in X_tf.columns
        assert "time_cos" in X_tf.columns
        assert "Amount_log" in X_tf.columns
