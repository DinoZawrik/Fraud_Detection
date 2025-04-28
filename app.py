import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline # Для применения шагов препроцессинга
import shap
from sklearn.metrics import ConfusionMatrixDisplay

# Импорт утилит и конфигурации
from src.utils import load_pipeline, load_data, load_joblib
from src.config import (MODEL_SAVE_PATH, DATA_PATH, TARGET_COLUMN, OPTIMAL_THRESHOLD,
                        METRICS_SAVE_PATH, SHAP_EXPLAINER_SAVE_PATH,
                        SHAP_VALUES_SAVE_PATH, FEATURE_NAMES_SAVE_PATH)
st.set_page_config(layout="wide")
# --- Функции Загрузки с Кэшированием ---
@st.cache_resource # Кэшируем загрузку "тяжелых" объектов (модель, explainer)
def load_model_assets():
    pipeline = load_pipeline(MODEL_SAVE_PATH)
    explainer = load_joblib(SHAP_EXPLAINER_SAVE_PATH)
    if pipeline is None:
        st.error(f"Критическая ошибка: Не удалось загрузить модель из {MODEL_SAVE_PATH}.")
        st.stop()
    if explainer is None:
        st.warning(f"SHAP Explainer не найден в {SHAP_EXPLAINER_SAVE_PATH}. Объяснения будут недоступны.")
    return pipeline, explainer

@st.cache_data # Кэшируем загрузку данных и других артефактов
def load_other_assets():
    metrics = load_joblib(METRICS_SAVE_PATH)
    shap_values_sample = load_joblib(SHAP_VALUES_SAVE_PATH) # Загружаем объект SHAP values
    feature_names_transformed = load_joblib(FEATURE_NAMES_SAVE_PATH)
    df = load_data(DATA_PATH)
    if df is None:
        st.error(f"Критическая ошибка: Не удалось загрузить данные из {DATA_PATH}.")
        st.stop()
    if metrics is None: st.warning("Файл с метриками не найден. Дашборд может быть неполным.")
    if shap_values_sample is None: st.warning("Файл SHAP values (sample) не найден. Глобальная важность не будет отображена.")
    if feature_names_transformed is None: st.warning("Файл с именами признаков не найден. SHAP может работать некорректно.")

    # Получаем список исходных фичей для полей ввода
    X_cols = [col for col in df.columns if col != TARGET_COLUMN]
    return metrics, shap_values_sample, feature_names_transformed, df, X_cols

# --- Загрузка Ресурсов ---
pipeline, explainer = load_model_assets()
metrics, shap_values_sample, feature_names_transformed, df_full, X_feature_columns = load_other_assets()

# --- Заголовок Приложения ---
st.title("Интерактивный Дашборд: Детекция Мошеннических Транзакций")
st.write("Анализ модели и предсказание статуса транзакций.")

# --- Создание Вкладок ---
tab_predict, tab_dashboard, tab_eda = st.tabs([
    "💳 Предсказание для Транзакции",
    "📊 Дашборд Модели",
    "🔍 Исследование Данных (EDA)"
])

# =====================================
# ВКЛАДКА 1: ПРЕДСКАЗАНИЕ ДЛЯ ТРАНЗАКЦИИ
# =====================================
with tab_predict:
    st.header("Проверка одной транзакции")
    st.write("Введите параметры транзакции для получения предсказания.")

    # Создаем форму для группировки полей ввода и кнопки
    with st.form("transaction_input_form"):
        col1, col2, col3 = st.columns(3)
        input_data = {}
        feature_index = 0
        num_features_per_col = (len(X_feature_columns) + 2) // 3 # Распределяем по колонкам

        # Динамическое создание полей ввода
        for col_widget in [col1, col2, col3]:
            with col_widget:
                for _ in range(num_features_per_col):
                    if feature_index < len(X_feature_columns):
                        feature_name = X_feature_columns[feature_index]
                        # Определяем параметры ввода в зависимости от имени фичи
                        if feature_name == 'Time':
                            input_data[feature_name] = st.number_input(feature_name, value=df_full[feature_name].mean(), step=1000.0, format="%.1f", key=f"input_{feature_name}")
                        elif feature_name == 'Amount':
                            input_data[feature_name] = st.number_input(feature_name, value=df_full[feature_name].mean(), min_value=0.0, step=10.0, format="%.2f", key=f"input_{feature_name}")
                        else: # V1-V28
                            input_data[feature_name] = st.number_input(feature_name, value=0.0, step=0.1, format="%.4f", key=f"input_{feature_name}")
                        feature_index += 1

        submitted = st.form_submit_button("🔍 Проверить статус транзакции")

    if submitted:
        input_df = pd.DataFrame([input_data])
        # Гарантируем правильный порядок столбцов
        input_df = input_df[X_feature_columns]

        st.subheader("Результат Проверки:")
        try:
            probabilities = pipeline.predict_proba(input_df)
            proba_fraud = probabilities[0, 1]
            is_fraud = proba_fraud >= OPTIMAL_THRESHOLD

            if is_fraud: st.error(f"🔴 Транзакция ПОДОЗРИТЕЛЬНА (вероятность: {proba_fraud:.2%}, порог: {OPTIMAL_THRESHOLD:.2f})")
            else: st.success(f"✅ Транзакция ЛЕГИТИМНА (вероятность мошенничества: {proba_fraud:.2%}, порог: {OPTIMAL_THRESHOLD:.2f})")

            # Отображение SHAP для этой транзакции
            if explainer and feature_names_transformed:
                st.subheader("Объяснение Предсказания (SHAP Waterfall)")
                with st.spinner("Расчет SHAP для предсказания..."):
                    try:
                        # Применяем шаги препроцессинга ДО классификатора
                        pipeline_steps_before_classifier = pipeline.steps[:-1]
                        temp_preprocessor_pipeline = Pipeline(pipeline_steps_before_classifier)
                        # Transform, НЕ fit_transform на новых данных!
                        input_transformed = temp_preprocessor_pipeline.transform(input_df)

                        # Создаем DataFrame с правильными именами
                        input_transformed_df = pd.DataFrame(input_transformed, columns=feature_names_transformed, index=input_df.index)

                        # Используем загруженный explainer
                        shap_values_single = explainer(input_transformed_df)

                        # Строим и отображаем waterfall plot
                        fig_waterfall, ax_waterfall = plt.subplots()
                        # Используем объект SHAP, передавая индекс [0] для первого (и единственного) предсказания
                        shap.waterfall_plot(shap_values_single[0], show=False)
                        st.pyplot(fig_waterfall)
                        plt.close(fig_waterfall) # Закрываем фигуру, чтобы не накапливать в памяти

                    except Exception as e_shap:
                        st.warning(f"Не удалось рассчитать SHAP для этой транзакции: {e_shap}")
            else:
                st.info("SHAP explainer или имена признаков не загружены для отображения объяснений.")

        except Exception as e:
            st.error(f"Ошибка при предсказании: {e}")


# =====================================
# ВКЛАДКА 2: ДАШБОРД МОДЕЛИ
# =====================================
with tab_dashboard:
    st.header("Обзор Производительности Модели (на Тестовой выборке)")

    if metrics:
        st.write(f"Метрики рассчитаны с использованием порога: **{OPTIMAL_THRESHOLD:.4f}**")
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Recall (Fraud)", f"{metrics.get('recall_fraud', 'N/A'):.3f}")
        col_m2.metric("Precision (Fraud)", f"{metrics.get('precision_fraud', 'N/A'):.3f}")
        col_m3.metric("F1-Score (Fraud)", f"{metrics.get('f1_fraud', 'N/A'):.3f}")

        col_m4, col_m5 = st.columns(2)
        col_m4.metric("ROC AUC", f"{metrics.get('roc_auc', 'N/A'):.3f}")
        col_m5.metric("PR AUC", f"{metrics.get('pr_auc', 'N/A'):.3f}")

        st.divider()
        st.subheader("Матрица Ошибок (Confusion Matrix)")
        cm_data = metrics.get('cm')
        if cm_data:
            cm = np.array(cm_data)
            fig_cm, ax_cm = plt.subplots()
            disp_cm = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Legit (0)', 'Fraud (1)'])
            disp_cm.plot(ax=ax_cm, cmap='Blues', values_format='d')
            st.pyplot(fig_cm)
            plt.close(fig_cm)
        else:
            st.warning("Данные матрицы ошибок не найдены.")

    else:
        st.warning("Файл с метриками не загружен. Невозможно отобразить производительность.")

    st.divider()
    st.subheader("Глобальная Важность Признаков (SHAP)")
    if shap_values_sample and feature_names_transformed:
         with st.spinner("Отрисовка SHAP summary plot..."):
            try:
                # Нужен DataFrame для summary_plot с именами колонок
                # Получаем его из объекта shap_values_sample
                shap_df = pd.DataFrame(shap_values_sample.values, columns=feature_names_transformed)

                fig_shap, ax_shap = plt.subplots()
                # Передаем DataFrame или объект shap_values
                shap.summary_plot(shap_values_sample.values, features=shap_df, feature_names=feature_names_transformed, plot_type="bar", show=False)
                st.pyplot(fig_shap)
                plt.close(fig_shap)
            except Exception as e_shap_global:
                 st.warning(f"Не удалось отобразить глобальный SHAP plot: {e_shap_global}")
    else:
        st.warning("SHAP values или имена признаков не загружены для отображения глобальной важности.")

# =====================================
# ВКЛАДКА 3: ИССЛЕДОВАНИЕ ДАННЫХ (EDA)
# =====================================
with tab_eda:
    st.header("Исследование Исходных Данных")

    if df_full is not None:
        st.write("Анализ распределений исходных признаков.")
        # Используем исходные колонки, полученные при загрузке
        feature_to_plot = st.selectbox(
            "Выберите признак для анализа:",
            options=X_feature_columns, # Используем список исходных колонок
            key="eda_feature_select"
        )

        if feature_to_plot:
            st.subheader(f"Распределение признака: {feature_to_plot}")
            fig_hist, ax_hist = plt.subplots(1, 2, figsize=(15, 5))
            sns.histplot(df_full[feature_to_plot], kde=True, ax=ax_hist[0])
            ax_hist[0].set_title(f'Общее распределение {feature_to_plot}')
            sns.histplot(data=df_full, x=feature_to_plot, hue=TARGET_COLUMN, kde=True, ax=ax_hist[1], palette="viridis")
            ax_hist[1].set_title(f'Распределение по классу {TARGET_COLUMN}')
            st.pyplot(fig_hist)
            plt.close(fig_hist)

            st.subheader(f"Статистики для {feature_to_plot}")
            st.dataframe(df_full.groupby(TARGET_COLUMN)[feature_to_plot].describe())
    else:
        st.error("Не удалось загрузить данные для EDA.")