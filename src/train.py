import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, recall_score, precision_score, f1_score, roc_auc_score, average_precision_score
import lightgbm as lgb
import os
import warnings

# Импортируем компоненты из других модулей
from src.config import DATA_PATH, TARGET_COLUMN, TEST_SIZE, RANDOM_STATE, LGBM_PARAMS, MODEL_SAVE_PATH, MODEL_DIR
from src.utils import load_data, save_pipeline
from src.pipeline import create_pipeline

# Подавляем UserWarning от LGBM/Sklearn при работе без имен признаков в predict
warnings.filterwarnings('ignore', category=UserWarning)

def train_model():
    """Обучает и сохраняет модель."""

    # 1. Загрузка данных
    print("--- Загрузка данных ---")
    df = load_data(DATA_PATH)
    if df is None:
        return

    # 2. Подготовка данных
    print("--- Подготовка данных ---")
    try:
        X = df.drop(TARGET_COLUMN, axis=1)
        y = df[TARGET_COLUMN]
    except KeyError:
        print(f"Ошибка: Целевая колонка '{TARGET_COLUMN}' не найдена в данных.")
        return
    except Exception as e:
        print(f"Ошибка при подготовке данных: {e}")
        return


    # 3. Разделение на обучающую и тестовую выборки
    print("--- Разделение данных ---")
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y
        )
        print(f"Обучающая выборка: {X_train.shape}, Тестовая выборка: {X_test.shape}")
    except Exception as e:
        print(f"Ошибка при разделении данных: {e}")
        return

    # 4. Создание модели и пайплайна
    print("--- Создание модели и пайплайна ---")
    # Убедимся, что используем параметры для высокого Recall (с class_weight='balanced')
    if 'class_weight' not in LGBM_PARAMS or LGBM_PARAMS['class_weight'] != 'balanced':
        print("Предупреждение: Параметр 'class_weight' не установлен в 'balanced' в config.py.")
        print("Для достижения высокого Recall рекомендуется использовать 'class_weight': 'balanced'.")
        # Можно принудительно установить, если нужно:
        # LGBM_PARAMS['class_weight'] = 'balanced'

    lgbm_classifier = lgb.LGBMClassifier(**LGBM_PARAMS)
    pipeline = create_pipeline(lgbm_classifier)
    print("Пайплайн создан.")

    # 5. Обучение модели
    print("--- Обучение модели ---")
    try:
        pipeline.fit(X_train, y_train)
        print("Модель успешно обучена.")
    except Exception as e:
        print(f"Ошибка при обучении модели: {e}")
        return

    # 6. Оценка модели на тестовой выборке (используя порог 0.5 - .predict())
    print("--- Оценка модели на тестовой выборке (порог 0.5) ---")
    try:
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1] # Для AUC метрик

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, digits=3, target_names=['Legit (0)', 'Fraud (1)']))

        recall_fraud = recall_score(y_test, y_pred, pos_label=1)
        precision_fraud = precision_score(y_test, y_pred, pos_label=1)
        f1_fraud = f1_score(y_test, y_pred, pos_label=1)
        roc_auc = roc_auc_score(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)

        print("--- Ключевые метрики ---")
        print(f"Recall (Fraud):    {recall_fraud:.3f}")
        print(f"Precision (Fraud): {precision_fraud:.3f}")
        print(f"F1-Score (Fraud):  {f1_fraud:.3f}")
        print(f"ROC AUC:           {roc_auc:.3f}")
        print(f"PR AUC:            {pr_auc:.3f}")

        # Проверяем, соответствует ли Recall ожиданиям (~0.83)
        if 0.80 < recall_fraud < 0.86: # Примерный диапазон
             print("\nRecall соответствует ожидаемому диапазону (~0.83).")
        else:
             print(f"\nВНИМАНИЕ: Полученный Recall ({recall_fraud:.3f}) отличается от ожидаемого (~0.83).")
             print("          Проверьте гиперпараметры в config.py или данные.")

    except Exception as e:
        print(f"Ошибка при оценке модели: {e}")
        return

    # 7. Сохранение обученного пайплайна
    print("--- Сохранение модели ---")
    os.makedirs(MODEL_DIR, exist_ok=True) # Создаем папку, если ее нет
    save_pipeline(pipeline, MODEL_SAVE_PATH)

    # --- (Опционально) Расчет и сохранение SHAP ---
    # Сюда можно добавить код для расчета SHAP на X_test и сохранения explainer/values
    # ...

    print("--- Процесс обучения завершен ---")

if __name__ == '__main__':
    train_model()