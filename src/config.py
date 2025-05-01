import os

# --- Пути ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Корень проекта
DATA_PATH = os.path.join(BASE_DIR, 'data', 'creditcard.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models') # Папка для всех артефактов

# Основная модель/пайплайн
MODEL_FILENAME = 'lgbm_fe_recall_optimized.joblib'
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

# Дополнительные артефакты для дашборда
METRICS_FILENAME = 'test_metrics.joblib'
METRICS_SAVE_PATH = os.path.join(MODEL_DIR, METRICS_FILENAME)

SHAP_EXPLAINER_FILENAME = 'shap_explainer.joblib'
SHAP_EXPLAINER_SAVE_PATH = os.path.join(MODEL_DIR, SHAP_EXPLAINER_FILENAME)

SHAP_VALUES_FILENAME = 'shap_values_sample.joblib' # Для выборки данных
SHAP_VALUES_SAVE_PATH = os.path.join(MODEL_DIR, SHAP_VALUES_FILENAME)

FEATURE_NAMES_FILENAME = 'feature_names_transformed.joblib' # Имена после препроцессинга
FEATURE_NAMES_SAVE_PATH = os.path.join(MODEL_DIR, FEATURE_NAMES_FILENAME)


# src/config.py
# ... другие пути ...
TRANSFORMED_DATA_FILENAME = 'X_shap_sample_transformed.joblib'
TRANSFORMED_DATA_SAVE_PATH = os.path.join(MODEL_DIR, TRANSFORMED_DATA_FILENAME)
# ... остальные параметры ...

# --- Параметры Моделирования ---
TARGET_COLUMN = 'Class'
TEST_SIZE = 0.25 # Доля тестовой выборки
VALIDATION_SIZE = 0.33 # Доля валидационной выборки ОТ ОСТАТКА после test split
RANDOM_STATE = 42

LGBM_PARAMS = {
    'class_weight': 'balanced',
    'n_estimators': 332,
    'learning_rate': 0.06048731265187917,
    'num_leaves': 39,
    'max_depth': 11,
    'reg_alpha': 0.363629602379294,
    'reg_lambda': 0.008, # Пример, замени
    'colsample_bytree': 0.6976502088991097,
    'subsample': 0.9717820827209607,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'objective': 'binary',
    'metric': 'aucpr',
    'verbosity': -1
}

OPTIMAL_THRESHOLD = 0.5 # Пример, ЗАМЕНИ, если оптимизировал порог
MIN_PRECISION_TARGET = 0.85
# --- Параметры SHAP ---
SHAP_SAMPLE_SIZE = 1000 #