"""Project configuration: paths, model hyperparameters, threshold settings."""

import os

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
DATA_PATH = os.path.join(BASE_DIR, "data", "creditcard.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_FILENAME = "lgbm_fe_recall_optimized.joblib"
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

METRICS_FILENAME = "test_metrics.joblib"
METRICS_SAVE_PATH = os.path.join(MODEL_DIR, METRICS_FILENAME)

SHAP_EXPLAINER_FILENAME = "shap_explainer.joblib"
SHAP_EXPLAINER_SAVE_PATH = os.path.join(MODEL_DIR, SHAP_EXPLAINER_FILENAME)

SHAP_VALUES_FILENAME = "shap_values_sample.joblib"
SHAP_VALUES_SAVE_PATH = os.path.join(MODEL_DIR, SHAP_VALUES_FILENAME)

FEATURE_NAMES_FILENAME = "feature_names_transformed.joblib"
FEATURE_NAMES_SAVE_PATH = os.path.join(MODEL_DIR, FEATURE_NAMES_FILENAME)

TRANSFORMED_DATA_FILENAME = "X_shap_sample_transformed.joblib"
TRANSFORMED_DATA_SAVE_PATH = os.path.join(MODEL_DIR, TRANSFORMED_DATA_FILENAME)

# --- Modeling ---
TARGET_COLUMN = "Class"
TEST_SIZE = 0.25        # fraction of full dataset held out as test
VALIDATION_SIZE = 0.33  # fraction of training remainder used for validation
RANDOM_STATE = 42

# LightGBM hyperparameters — tuned via Optuna (100 trials, optimizing PR AUC on validation set)
LGBM_PARAMS = {
    "class_weight": "balanced",
    "n_estimators": 332,
    "learning_rate": 0.06048731265187917,
    "num_leaves": 39,
    "max_depth": 11,
    "reg_alpha": 0.363629602379294,
    "reg_lambda": 0.008,
    "colsample_bytree": 0.6976502088991097,
    "subsample": 0.9717820827209607,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "objective": "binary",
    "metric": "aucpr",
    "verbosity": -1,
}

OPTIMAL_THRESHOLD = 0.5      # classification threshold
MIN_PRECISION_TARGET = 0.85  # minimum precision for Recall-optimized threshold

# --- SHAP ---
SHAP_SAMPLE_SIZE = 1000  # number of test samples used for SHAP computation
