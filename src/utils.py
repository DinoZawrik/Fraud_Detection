"""Utility helpers for saving and loading project artifacts."""

import logging
import os

import joblib
import pandas as pd

logger = logging.getLogger(__name__)


def save_joblib(data, file_path: str) -> None:
    """Serialize an object to disk with joblib."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(data, file_path)
        logger.info("Saved: %s", file_path)
    except Exception:
        logger.exception("Failed to save %s", file_path)


def load_joblib(file_path: str):
    """Deserialize an object from disk with joblib. Returns None if missing."""
    if not os.path.exists(file_path):
        logger.warning("File not found: %s", file_path)
        return None
    try:
        data = joblib.load(file_path)
        logger.info("Loaded: %s", file_path)
        return data
    except Exception:
        logger.exception("Failed to load %s", file_path)
        return None


# Aliases for pipeline I/O
save_pipeline = save_joblib
load_pipeline = load_joblib


def load_data(file_path: str) -> pd.DataFrame | None:
    """Load a CSV dataset. Returns None if the file is missing or unreadable."""
    if not os.path.exists(file_path):
        logger.warning("Data file not found: %s", file_path)
        return None
    try:
        df = pd.read_csv(file_path)
        logger.info("Loaded dataset: %s  shape=%s", file_path, df.shape)
        return df
    except Exception as e:
        logger.exception("Failed to load data from %s: %s", file_path, e)
        return None
