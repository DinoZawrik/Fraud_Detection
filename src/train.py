"""Training script for the Fraud Detection model.

Steps:
  1. Load data
  2. Prepare and split data (train / validation / test)
  3. Build and train the LightGBM pipeline
  4. Evaluate on the test set with the configured threshold
  5. Save the pipeline and metrics
  6. Compute and save SHAP explainer, SHAP values, and transformed features
"""

import logging
import os
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline as SklearnPipeline

from src.config import (
    DATA_PATH,
    LGBM_PARAMS,
    MODEL_DIR,
    MODEL_SAVE_PATH,
    METRICS_SAVE_PATH,
    OPTIMAL_THRESHOLD,
    RANDOM_STATE,
    SHAP_EXPLAINER_SAVE_PATH,
    SHAP_SAMPLE_SIZE,
    SHAP_VALUES_SAVE_PATH,
    FEATURE_NAMES_SAVE_PATH,
    TARGET_COLUMN,
    TEST_SIZE,
    TRANSFORMED_DATA_SAVE_PATH,
    VALIDATION_SIZE,
)
from src.pipeline import create_pipeline
from src.utils import load_data, save_joblib, save_pipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=UserWarning, module="shap")
warnings.filterwarnings("ignore", category=FutureWarning)


def find_optimal_threshold(pipeline, X_val, y_val, target_metric: str = "f1") -> float:
    """Find the best classification threshold on the validation set.

    Args:
        pipeline: Trained sklearn pipeline.
        X_val: Validation features.
        y_val: Validation labels.
        target_metric: One of 'f1', 'recall', 'precision'.

    Returns:
        Optimal threshold (float). Falls back to 0.5 on error.
    """
    logger.info("--- Threshold optimisation (target: max %s) ---", target_metric)
    try:
        y_proba_val = pipeline.predict_proba(X_val)[:, 1]
        precs, recs, ths = precision_recall_curve(y_val, y_proba_val)
        ths = np.append(ths, 1.0)

        if target_metric == "f1":
            scores = np.divide(
                2 * precs * recs,
                precs + recs,
                out=np.zeros_like(precs),
                where=(precs + recs) != 0,
            )
        elif target_metric == "recall":
            scores = recs
        elif target_metric == "precision":
            scores = precs
        else:
            logger.warning("Unknown metric '%s'. Falling back to F1.", target_metric)
            scores = np.divide(
                2 * precs * recs,
                precs + recs,
                out=np.zeros_like(precs),
                where=(precs + recs) != 0,
            )

        opt_idx = min(int(np.argmax(scores)), len(ths) - 1)
        opt_thresh = ths[opt_idx]
        f1_at_opt = float(
            np.divide(
                2 * precs[opt_idx] * recs[opt_idx],
                precs[opt_idx] + recs[opt_idx],
                out=np.float64(0.0),
                where=(precs[opt_idx] + recs[opt_idx]) != 0,
            )
        )
        logger.info(
            "Optimal threshold (%s): %.4f  →  F1=%.3f  Precision=%.3f  Recall=%.3f",
            target_metric,
            opt_thresh,
            f1_at_opt,
            precs[opt_idx],
            recs[opt_idx],
        )
        return opt_thresh
    except Exception:
        logger.exception("Threshold optimisation failed — using 0.5")
        return 0.5


def train_and_evaluate() -> None:
    """Run the full training pipeline and save all artifacts to models/."""

    # 1. Load data
    logger.info("--- 1. Loading data ---")
    df = load_data(DATA_PATH)
    if df is None:
        return

    # 2. Prepare features and target
    logger.info("--- 2. Preparing data ---")
    try:
        X = df.drop(TARGET_COLUMN, axis=1)
        y = df[TARGET_COLUMN]
    except KeyError:
        logger.error("Target column '%s' not found in dataset.", TARGET_COLUMN)
        return

    # 3. Train / Validation / Test split
    logger.info("--- 3. Splitting data ---")
    try:
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full,
            y_train_full,
            test_size=VALIDATION_SIZE,
            random_state=RANDOM_STATE,
            stratify=y_train_full,
        )
        logger.info(
            "Train: %s  |  Val: %s  |  Test: %s",
            X_train.shape,
            X_val.shape,
            X_test.shape,
        )
    except Exception:
        logger.exception("Data split failed.")
        return

    # 4. Build pipeline
    logger.info("--- 4. Building pipeline ---")
    if LGBM_PARAMS.get("class_weight") != "balanced":
        logger.warning("'class_weight' != 'balanced' — high Recall may be compromised.")
    lgbm_clf = lgb.LGBMClassifier(**LGBM_PARAMS)
    pipeline = create_pipeline(lgbm_clf)
    logger.info("Pipeline steps: %s", [name for name, _ in pipeline.steps])

    # 5. Train
    logger.info("--- 5. Training ---")
    try:
        pipeline.fit(X_train, y_train)
        logger.info("Training complete.")
    except Exception:
        logger.exception("Training failed.")
        return

    # 6. Threshold (from config)
    optimal_threshold = OPTIMAL_THRESHOLD
    logger.info("--- 6. Using threshold from config: %.4f ---", optimal_threshold)

    # 7. Evaluate on test set
    logger.info("--- 7. Evaluating on test set (threshold=%.4f) ---", optimal_threshold)
    metrics: dict = {}
    try:
        y_proba_test = pipeline.predict_proba(X_test)[:, 1]
        y_pred_test = (y_proba_test >= optimal_threshold).astype(int)

        print("\nClassification Report (test set):")
        print(
            classification_report(
                y_test,
                y_pred_test,
                digits=3,
                target_names=["Legit (0)", "Fraud (1)"],
            )
        )

        metrics["recall_fraud"] = recall_score(y_test, y_pred_test, pos_label=1)
        metrics["precision_fraud"] = precision_score(y_test, y_pred_test, pos_label=1)
        metrics["f1_fraud"] = f1_score(y_test, y_pred_test, pos_label=1)
        metrics["roc_auc"] = roc_auc_score(y_test, y_proba_test)
        metrics["pr_auc"] = average_precision_score(y_test, y_proba_test)
        metrics["cm"] = confusion_matrix(y_test, y_pred_test).tolist()

        logger.info("Key metrics (test set):")
        for k, v in metrics.items():
            if k != "cm":
                logger.info("  %-20s %.4f", k, v)
    except Exception:
        logger.exception("Evaluation failed.")
        return

    # 8. Save pipeline and metrics
    logger.info("--- 8. Saving artifacts ---")
    os.makedirs(MODEL_DIR, exist_ok=True)
    save_pipeline(pipeline, MODEL_SAVE_PATH)
    save_joblib(metrics, METRICS_SAVE_PATH)

    # 9. SHAP
    logger.info("--- 9. Computing SHAP values ---")
    try:
        if len(X_test) > SHAP_SAMPLE_SIZE:
            X_shap_raw, _, _, _ = train_test_split(
                X_test,
                y_test,
                train_size=SHAP_SAMPLE_SIZE,
                random_state=RANDOM_STATE,
                stratify=y_test,
            )
        else:
            X_shap_raw = X_test.copy()

        # Apply preprocessing steps only (exclude classifier)
        preprocessor = SklearnPipeline(pipeline.steps[:-1])
        preprocessor.fit(X_train, y_train)
        X_shap_tf = preprocessor.transform(X_shap_raw)

        if not isinstance(X_shap_tf, pd.DataFrame):
            try:
                names = preprocessor.get_feature_names_out()
                X_shap_tf = pd.DataFrame(
                    X_shap_tf, columns=names, index=X_shap_raw.index
                )
            except Exception:
                logger.warning("Could not retrieve feature names after transform.")
                X_shap_tf = pd.DataFrame(X_shap_tf, index=X_shap_raw.index)

        feature_names_tf = list(X_shap_tf.columns)
        logger.info("Features after preprocessing: %d", len(feature_names_tf))
        save_joblib(feature_names_tf, FEATURE_NAMES_SAVE_PATH)

        lgbm_model = pipeline.named_steps["classifier"]
        logger.info("Building SHAP TreeExplainer...")
        explainer = shap.TreeExplainer(lgbm_model, X_shap_tf)

        logger.info("Computing SHAP values for %d samples...", len(X_shap_tf))
        shap_values_obj = explainer(X_shap_tf)

        save_joblib(explainer, SHAP_EXPLAINER_SAVE_PATH)
        save_joblib(shap_values_obj, SHAP_VALUES_SAVE_PATH)
        save_joblib(X_shap_tf, TRANSFORMED_DATA_SAVE_PATH)
        logger.info("SHAP artifacts saved.")

    except ImportError:
        logger.warning("SHAP not installed — skipping SHAP step.")
    except Exception:
        logger.exception("SHAP computation failed.")

    logger.info("--- Training and evaluation complete ---")


if __name__ == "__main__":
    train_and_evaluate()
