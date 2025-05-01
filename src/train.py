"""
Скрипт для обучения и оценки модели детекции мошенничества.

Выполняет следующие шаги:
1. Загрузка данных.
2. Подготовка и разделение данных на обучающую, валидационную и тестовую выборки.
3. Создание и обучение пайплайна (предобработка + модель LightGBM).
4. Оценка модели на тестовой выборке с использованием порога из конфигурации.
5. Сохранение обученного пайплайна и метрик.
6. Расчет и сохранение SHAP explainer, SHAP values и трансформированных данных
   для использования в дашборде.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
)
from sklearn.pipeline import (
    Pipeline as SklearnPipeline,
)  # Используем для шагов предобработки SHAP
import lightgbm as lgb
import os
import warnings
import joblib
import shap

# Импортируем компоненты из других модулей
from src.config import (
    DATA_PATH,
    TARGET_COLUMN,
    TEST_SIZE,
    VALIDATION_SIZE,
    RANDOM_STATE,
    LGBM_PARAMS,
    MODEL_SAVE_PATH,
    METRICS_SAVE_PATH,
    SHAP_EXPLAINER_SAVE_PATH,
    SHAP_VALUES_SAVE_PATH,
    FEATURE_NAMES_SAVE_PATH,
    SHAP_SAMPLE_SIZE,
    OPTIMAL_THRESHOLD,
    MODEL_DIR,
    TRANSFORMED_DATA_SAVE_PATH,
)  # Добавили импорт пути
from src.utils import load_data, save_pipeline, save_joblib, load_pipeline
from src.pipeline import create_pipeline
from src.data_preprocessing import (
    time_transformer,
    amount_transformer,
    create_scaling_transformer,
)

warnings.filterwarnings("ignore", category=UserWarning, module="shap")
warnings.filterwarnings("ignore", category=FutureWarning)


def find_optimal_threshold(pipeline, X_val, y_val, target_metric="f1"):
    """
    Находит оптимальный порог классификации на валидационной выборке.

    Args:
        pipeline: Обученный пайплайн.
        X_val: Валидационные признаки.
        y_val: Валидационные метки.
        target_metric (str): Метрика для оптимизации ('f1', 'recall', 'precision').

    Returns:
        float: Оптимальный порог. Возвращает 0.5 в случае ошибки.
    """
    print(
        f"--- Оптимизация порога на валидационной выборке (цель: макс {target_metric}) ---"
    )
    try:
        y_proba_val = pipeline.predict_proba(X_val)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y_val, y_proba_val)
        thresholds = np.append(thresholds, 1.0)  # Добавляем 1.0 для полноты

        if target_metric == "f1":
            scores = np.divide(
                2 * precisions * recalls,
                precisions + recalls,
                out=np.zeros_like(precisions),
                where=(precisions + recalls) != 0,
            )
        elif target_metric == "recall":
            scores = recalls
        elif target_metric == "precision":
            scores = precisions
        else:
            print(f"   Неизвестная метрика '{target_metric}'. Используется F1.")
            scores = np.divide(
                2 * precisions * recalls,
                precisions + recalls,
                out=np.zeros_like(precisions),
                where=(precisions + recalls) != 0,
            )

        optimal_idx = np.argmax(scores)
        # Коррекция индекса, если максимум достигается при пороге 1.0
        if optimal_idx >= len(thresholds):
            optimal_idx = len(thresholds) - 1
        optimal_threshold = thresholds[optimal_idx]

        # Рассчитаем F1 для информации в любом случае
        f1_at_optimal = np.divide(
            2 * precisions[optimal_idx] * recalls[optimal_idx],
            precisions[optimal_idx] + recalls[optimal_idx],
            out=0.0,
            where=(precisions[optimal_idx] + recalls[optimal_idx]) != 0,
        )

        print(f"   Найден оптимальный порог ({target_metric}): {optimal_threshold:.4f}")
        print(
            f"   Метрики на Validation при этом пороге: F1={f1_at_optimal:.3f}, Precision={precisions[optimal_idx]:.3f}, Recall={recalls[optimal_idx]:.3f}"
        )
        return optimal_threshold
    except Exception as e:
        print(
            f"   Ошибка при оптимизации порога: {e}. Используется стандартный порог 0.5."
        )
        return 0.5


def train_and_evaluate():
    """
    Выполняет полный цикл обучения модели:
    Загрузка данных -> Разделение -> Создание пайплайна -> Обучение ->
    Оценка на тесте -> Сохранение артефактов (модель, метрики, SHAP).
    """
    # 1. Загрузка данных
    print("--- 1. Загрузка данных ---")
    df = load_data(DATA_PATH)
    if df is None:
        return

    # 2. Подготовка данных
    print("--- 2. Подготовка данных ---")
    try:
        X = df.drop(TARGET_COLUMN, axis=1)
        y = df[TARGET_COLUMN]
    except KeyError:
        print(f"Ошибка: Целевая колонка '{TARGET_COLUMN}' не найдена.")
        return
    except Exception as e:
        print(f"Ошибка при подготовке данных: {e}")
        return

    # 3. Разделение данных: Train / Validation / Test
    print("--- 3. Разделение данных ---")
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
        print(
            f"Train set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}"
        )
    except Exception as e:
        print(f"Ошибка при разделении данных: {e}")
        return

    # 4. Создание модели и пайплайна
    print("--- 4. Создание модели и пайплайна ---")
    if LGBM_PARAMS.get("class_weight") != "balanced":
        print(
            "Предупреждение: 'class_weight' != 'balanced'. Для высокого Recall рекомендуется 'balanced'."
        )
    lgbm_classifier = lgb.LGBMClassifier(**LGBM_PARAMS)
    pipeline = create_pipeline(lgbm_classifier)
    print("Пайплайн создан.")

    # 5. Обучение модели на обучающей выборке
    print("--- 5. Обучение модели ---")
    try:
        pipeline.fit(X_train, y_train)
        print("Модель успешно обучена.")
    except Exception as e:
        print(f"Ошибка при обучении модели: {e}")
        return

    # 6. Определение порога
    # Использование порога из файла конфигурации
    optimal_threshold = OPTIMAL_THRESHOLD
    print(f"--- 6. Используется порог из конфигурации: {optimal_threshold:.4f} ---")

    # 7. Оценка модели на ТЕСТОВОЙ выборке с выбранным порогом
    print(
        f"--- 7. Оценка модели на тестовой выборке (порог {optimal_threshold:.4f}) ---"
    )
    metrics = {}
    try:
        y_proba_test = pipeline.predict_proba(X_test)[:, 1]
        y_pred_test_thresh = (y_proba_test >= optimal_threshold).astype(int)

        print("\nClassification Report (Test Set):")
        print(
            classification_report(
                y_test,
                y_pred_test_thresh,
                digits=3,
                target_names=["Legit (0)", "Fraud (1)"],
            )
        )

        metrics["recall_fraud"] = recall_score(y_test, y_pred_test_thresh, pos_label=1)
        metrics["precision_fraud"] = precision_score(
            y_test, y_pred_test_thresh, pos_label=1
        )
        metrics["f1_fraud"] = f1_score(y_test, y_pred_test_thresh, pos_label=1)
        metrics["roc_auc"] = roc_auc_score(
            y_test, y_proba_test
        )  # AUC не зависит от порога
        metrics["pr_auc"] = average_precision_score(
            y_test, y_proba_test
        )  # AUC не зависит от порога
        metrics["cm"] = confusion_matrix(
            y_test, y_pred_test_thresh
        ).tolist()  # Сохраняем как list

        print("--- Ключевые метрики (Test Set) ---")
        for k, v in metrics.items():
            if k != "cm":
                print(f"{k:<18}: {v:.3f}")
        print(f"{'confusion_matrix':<18}: {metrics['cm']}")

    except Exception as e:
        print(f"Ошибка при оценке модели: {e}")
        return

    # 8. Сохранение артефактов
    print("--- 8. Сохранение артефактов ---")
    os.makedirs(MODEL_DIR, exist_ok=True)
    save_pipeline(pipeline, MODEL_SAVE_PATH)  # Сохраняем пайплайн
    save_joblib(metrics, METRICS_SAVE_PATH)  # Сохраняем метрики

    # 9. Расчет и сохранение SHAP (на примере тестовых данных)
    print("--- 9. Расчет и сохранение SHAP ---")
    try:
        print(f"   Создание выборки для SHAP ({SHAP_SAMPLE_SIZE} примеров)...")
        if len(X_test) > SHAP_SAMPLE_SIZE:
            # Берем стратифицированную выборку из теста для SHAP
            X_test_sample, _, y_test_sample, _ = train_test_split(
                X_test,
                y_test,
                train_size=SHAP_SAMPLE_SIZE,
                random_state=RANDOM_STATE,
                stratify=y_test,
            )
        else:
            X_test_sample = X_test.copy()

        print("   Применение шагов предобработки для SHAP...")
        print("   Применение шагов предобработки для SHAP...")
        # Для SHAP нам нужны данные, прошедшие те же шаги предобработки,
        # что и при обучении модели. Создаем отдельный пайплайн только
        # с шагами предобработки.
        preprocessor_steps = pipeline.steps[:-1]  # Все шаги, кроме классификатора
        preprocessor_pipeline = SklearnPipeline(preprocessor_steps)

        # Обучаем препроцессор на ТРЕЙНЕ (как и основной пайплайн)
        # и трансформируем ВЫБОРКУ из ТЕСТА для SHAP.
        # Важно: Не используем fit_transform на тестовой выборке!
        preprocessor_pipeline.fit(X_train, y_train)
        X_shap_transformed = preprocessor_pipeline.transform(X_test_sample)

        # Убедимся, что результат - DataFrame (некоторые трансформеры могут вернуть numpy)
        if not isinstance(X_shap_transformed, pd.DataFrame):
            # Пытаемся получить имена из последнего шага (обычно scaler)
            try:
                feature_names_out = preprocessor_pipeline.get_feature_names_out()
                X_shap_transformed = pd.DataFrame(
                    X_shap_transformed,
                    columns=feature_names_out,
                    index=X_test_sample.index,
                )
            except Exception as e_fn:
                print(
                    f"   Предупреждение: Не удалось получить имена признаков после трансформации: {e_fn}. SHAP может быть менее интерпретируемым."
                )
                # Если имена не получены, используем стандартные
                X_shap_transformed = pd.DataFrame(
                    X_shap_transformed, index=X_test_sample.index
                )

        # Получаем и сохраняем имена признаков ПОСЛЕ всех трансформаций
        feature_names_transformed = list(X_shap_transformed.columns)
        print(
            f"   Получено {len(feature_names_transformed)} имен признаков после препроцессинга."
        )
        save_joblib(
            feature_names_transformed, FEATURE_NAMES_SAVE_PATH
        )  # Сохраняем имена

        # Создаем SHAP explainer
        lgbm_model = pipeline.named_steps["classifier"]
        print("   Создание SHAP TreeExplainer...")
        # Передаем DataFrame для фона (если возможно)
        explainer = shap.TreeExplainer(lgbm_model, X_shap_transformed)

        # Рассчитываем SHAP values
        print(f"   Расчет SHAP values для {len(X_shap_transformed)} примеров...")
        # shap_values = explainer.shap_values(X_shap_transformed) # Для бинарной классификации shap_values[1] - для класса 1
        shap_values_obj = explainer(
            X_shap_transformed
        )  # Используем объект explainer для получения base_values и т.д.

        print("   Сохранение SHAP explainer и values...")
        save_joblib(explainer, SHAP_EXPLAINER_SAVE_PATH)
        # Сохраняем объект SHAP, который содержит values, base_values и data
        save_joblib(shap_values_obj, SHAP_VALUES_SAVE_PATH)

        # Сохраняем трансформированные данные, использованные для SHAP
        # Путь импортирован из config.py
        save_joblib(X_shap_transformed, TRANSFORMED_DATA_SAVE_PATH)

    except ImportError:
        print("\n   Библиотека SHAP не установлена. Шаг расчета SHAP пропущен.")
    except Exception as e:
        print(f"\n   Ошибка при расчете или сохранении SHAP:")
        import traceback

        traceback.print_exc()  # Печатаем полный traceback для диагностики

    print("\n--- Процесс обучения и оценки завершен ---")


if __name__ == "__main__":
    train_and_evaluate()
