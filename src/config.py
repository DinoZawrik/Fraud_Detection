import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
DATA_PATH = os.path.join(BASE_DIR, 'data', 'creditcard.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_FILENAME = 'lgbm_fe_recall_optimized.joblib'
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

TARGET_COLUMN = 'Class'
TEST_SIZE = 0.25
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

# --- Порог Классификации ---
# Используем стандартный 0.5 для .predict() для начала
# Если был найден оптимальный F1/Recall порог, можно указать его здесь
# DECISION_THRESHOLD = 0.XXXX