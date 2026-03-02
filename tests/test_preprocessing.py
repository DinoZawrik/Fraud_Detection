"""Unit tests for data_preprocessing.py."""

import numpy as np
import pandas as pd
import pytest

from src.data_preprocessing import amount_log_feature, time_features


# ---------------------------------------------------------------------------
# time_features
# ---------------------------------------------------------------------------

def _time_df(*times):
    return pd.DataFrame({"Time": list(times)})


def test_time_features_drops_time_column():
    df = _time_df(0.0, 43200.0)
    result = time_features(df)
    assert "Time" not in result.columns
    assert "time_sec_day" not in result.columns


def test_time_features_adds_cyclic_columns():
    result = time_features(_time_df(0.0))
    assert "time_sin" in result.columns
    assert "time_cos" in result.columns


def test_time_features_cyclic_at_midnight():
    """At t=0: sin=0, cos=1."""
    result = time_features(_time_df(0.0))
    assert abs(result["time_sin"].iloc[0]) < 1e-10
    assert abs(result["time_cos"].iloc[0] - 1.0) < 1e-10


def test_time_features_cyclic_at_noon():
    """At t=43200 (noon): sin≈0, cos≈-1."""
    result = time_features(_time_df(43200.0))
    assert abs(result["time_sin"].iloc[0]) < 1e-10
    assert abs(result["time_cos"].iloc[0] + 1.0) < 1e-10


def test_time_features_preserves_other_columns():
    df = pd.DataFrame({"Time": [0.0], "V1": [3.14], "V2": [-1.0]})
    result = time_features(df)
    assert "V1" in result.columns
    assert "V2" in result.columns


def test_time_features_output_shape():
    df = _time_df(0.0, 1000.0, 50000.0)
    result = time_features(df)
    assert len(result) == 3


# ---------------------------------------------------------------------------
# amount_log_feature
# ---------------------------------------------------------------------------

def _amount_df(*amounts):
    return pd.DataFrame({"Amount": list(amounts)})


def test_amount_log_drops_amount_column():
    result = amount_log_feature(_amount_df(100.0))
    assert "Amount" not in result.columns


def test_amount_log_adds_amount_log_column():
    result = amount_log_feature(_amount_df(100.0))
    assert "Amount_log" in result.columns


def test_amount_log_zero():
    """log1p(0) == 0."""
    result = amount_log_feature(_amount_df(0.0))
    assert abs(result["Amount_log"].iloc[0]) < 1e-10


def test_amount_log_values():
    """log1p(e - 1) ≈ 1.0."""
    result = amount_log_feature(_amount_df(np.e - 1))
    assert abs(result["Amount_log"].iloc[0] - 1.0) < 1e-10


def test_amount_log_preserves_other_columns():
    df = pd.DataFrame({"Amount": [50.0], "V1": [2.0], "V14": [-0.5]})
    result = amount_log_feature(df)
    assert "V1" in result.columns
    assert "V14" in result.columns


def test_amount_log_no_amount_column_is_noop():
    """If there is no 'Amount' column the function returns the frame unchanged."""
    df = pd.DataFrame({"V1": [1.0, 2.0]})
    result = amount_log_feature(df)
    assert list(result.columns) == ["V1"]
    assert len(result) == 2
