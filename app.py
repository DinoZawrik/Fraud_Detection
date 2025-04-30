import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay

# --- Конфигурация страницы ---
st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide", initial_sidebar_state="auto")
# ---------------------------

# Импорт утилит и конфигурации
from src.utils import load_pipeline, load_data, load_joblib
from src.config import (MODEL_SAVE_PATH, DATA_PATH, TARGET_COLUMN, OPTIMAL_THRESHOLD,
                        METRICS_SAVE_PATH, SHAP_EXPLAINER_SAVE_PATH,
                        SHAP_VALUES_SAVE_PATH, FEATURE_NAMES_SAVE_PATH,
                        RANDOM_STATE, TEST_SIZE)

# SHAP (опционально)
try:
    import shap
    SHAP_AVAILABLE = True
    # Инициализация JS для force plot (нужна только один раз)
    shap.initjs()
except ImportError:
    SHAP_AVAILABLE = False

# --- Функции Загрузки с Кэшированием ---
@st.cache_resource
def load_model_assets():
    pipeline = load_pipeline(MODEL_SAVE_PATH)
    explainer = None
    if SHAP_AVAILABLE: explainer = load_joblib(SHAP_EXPLAINER_SAVE_PATH)
    if pipeline is None: st.error(f"Критическая ошибка: Модель не загружена ({MODEL_SAVE_PATH})."); st.stop()
    return pipeline, explainer

@st.cache_data
def load_other_assets():
    metrics = load_joblib(METRICS_SAVE_PATH)
    shap_values_sample = None
    if SHAP_AVAILABLE: shap_values_sample = load_joblib(SHAP_VALUES_SAVE_PATH)
    feature_names_transformed = load_joblib(FEATURE_NAMES_SAVE_PATH)
    df = load_data(DATA_PATH)
    if df is None: st.error(f"Критическая ошибка: Данные не загружены ({DATA_PATH})."); st.stop()

    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]
    _, X_test, _, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    X_cols = list(X.columns)

    # Сохраним индексы для быстрого выбора
    legit_indices = y_test[y_test == 0].index.tolist()
    fraud_indices = y_test[y_test == 1].index.tolist()

    return metrics, shap_values_sample, feature_names_transformed, df, X_cols, X_test, y_test, legit_indices, fraud_indices

# --- Инициализация Session State для хранения примера ---
if 'current_example_data' not in st.session_state:
    st.session_state.current_example_data = None # DataFrame с одним примером
if 'current_example_is_fraud' not in st.session_state:
    st.session_state.current_example_is_fraud = None # True/False
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None # Результат предсказания

# --- Загрузка Ресурсов ---
pipeline, explainer = load_model_assets()
metrics, shap_values_sample, feature_names_transformed, df_full, X_feature_columns, X_test, y_test, legit_indices, fraud_indices = load_other_assets()

# --- Заголовок Приложения ---
st.title("Интерактивный Дашборд: Детекция Мошеннических Транзакций")
st.write("Анализ модели и объяснение предсказаний.")

# --- Создание Вкладок ---
tab_explain, tab_dashboard, tab_eda = st.tabs([
    "🔍 Объяснение Предсказания",
    "📊 Дашборд Модели",
    "📈 Исследование Данных (EDA)"
])


# =====================================
# ВКЛАДКА 1: ОБЪЯСНЕНИЕ ПРЕДСКАЗАНИЯ
# =====================================
with tab_explain:
    st.header("Объяснение Предсказания для Примера")
    st.write("Выберите тип транзакции, чтобы увидеть предсказание модели и объяснение SHAP.")

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("Показать Легитимную", key="btn_legit_explain", use_container_width=True):
            if legit_indices:
                idx = np.random.choice(legit_indices)
                st.session_state.current_example_data = X_test.loc[[idx]]
                st.session_state.current_example_is_fraud = False
                st.session_state.prediction_result = None # Сбросить предыдущий результат
            else: st.warning("Нет легитимных примеров в тестовой выборке.")
    with col_btn2:
        if st.button("Показать Мошенническую", key="btn_fraud_explain", use_container_width=True):
            if fraud_indices:
                idx = np.random.choice(fraud_indices)
                st.session_state.current_example_data = X_test.loc[[idx]]
                st.session_state.current_example_is_fraud = True
                st.session_state.prediction_result = None # Сбросить предыдущий результат
            else: st.warning("Нет мошеннических примеров в тестовой выборке.")

    # Отображаем информацию и SHAP, если пример выбран
    if st.session_state.current_example_data is not None:
        example_data = st.session_state.current_example_data
        is_real_fraud = st.session_state.current_example_is_fraud
        example_idx = example_data.index[0]

        if is_real_fraud:
            status_text = '<span style="color:red;">**Fraud**</span>'
        else:
            status_text = '<span style="color:green;">**Legit**</span>'

        # Затем используем его в f-строке
        st.markdown(f"**Отображен пример с индексом `{example_idx}`. Реальный статус: {status_text}**", unsafe_allow_html=True)
        # Показываем исходные Time и Amount
        st.write(f"Исходные параметры: Time=`{example_data['Time'].iloc[0]:.1f}`, Amount=`{example_data['Amount'].iloc[0]:.2f}`")

        # Делаем предсказание, если его еще нет для этого примера
        if st.session_state.prediction_result is None:
             try:
                probabilities = pipeline.predict_proba(example_data)
                proba_fraud = probabilities[0, 1]
                is_pred_fraud = proba_fraud >= OPTIMAL_THRESHOLD
                st.session_state.prediction_result = {
                    'proba': proba_fraud,
                    'prediction': is_pred_fraud
                }
             except Exception as e:
                 st.error(f"Ошибка при предсказании: {e}")
                 st.session_state.prediction_result = {'error': True}


        # Отображаем результат предсказания
        pred_result = st.session_state.prediction_result
        if pred_result and 'error' not in pred_result:
             st.subheader("Результат Модели:")
             proba_fraud = pred_result['proba']
             is_pred_fraud = pred_result['prediction']
             if is_pred_fraud: st.error(f"🔴 Предсказано: ПОДОЗРИТЕЛЬНАЯ (Вероятность: {proba_fraud:.2%})")
             else: st.success(f"✅ Предсказано: ЛЕГИТИМНАЯ (Вероятность: {proba_fraud:.2%})")

             # Отображение SHAP
             if SHAP_AVAILABLE and explainer and feature_names_transformed:
                st.subheader("Объяснение Предсказания (SHAP)")
                with st.spinner("Расчет SHAP..."):
                    try:
                        pipeline_steps_before_classifier = pipeline.steps[:-1]
                        temp_preprocessor_pipeline = Pipeline(pipeline_steps_before_classifier)
                        input_transformed = temp_preprocessor_pipeline.transform(example_data)
                        input_transformed_df = pd.DataFrame(input_transformed, columns=feature_names_transformed, index=example_data.index)

                        shap_values_single = explainer(input_transformed_df)

                        # Waterfall Plot
                        st.markdown("**Waterfall Plot:** Показывает вклад каждого признака в смещение предсказания от базового значения.")
                        fig_waterfall, ax_waterfall = plt.subplots(figsize=(10, 4))
                        shap.waterfall_plot(shap_values_single[0], show=False, max_display=15)
                        st.pyplot(fig_waterfall, use_container_width=False)
                        plt.close(fig_waterfall)

                        # Force Plot
                        st.markdown("**Force Plot:** Визуализирует 'силы', толкающие предсказание в ту или иную сторону.")
                        # shap.force_plot требует base_value из explainer
                        force_plot_html = shap.force_plot(explainer.expected_value[1], # Base value для класса 1
                                                          shap_values_single.values[0,:,1], # SHAP values для класса 1
                                                          input_transformed_df.iloc[0,:], # Значения признаков
                                                          feature_names=feature_names_transformed,
                                                          matplotlib=False) # Генерируем HTML
                        # Отображаем HTML с помощью components.html
                        st.components.v1.html(force_plot_html.html(), height=150)


                    except Exception as e_shap:
                        st.warning(f"Не удалось рассчитать или отобразить SHAP: {e_shap}")
             else:
                st.info("Объяснения SHAP недоступны (проверьте установку SHAP и наличие артефактов).")

# =====================================
# ВКЛАДКА 2: ДАШБОРД МОДЕЛИ
# =====================================
# (Остается как в предыдущем варианте с исправленными размерами графиков)
with tab_dashboard:
    st.header("Обзор Производительности Модели (на Тестовой выборке)")
    # ... (код метрик, confusion matrix, SHAP summary plot с figsize и use_container_width=False) ...
    if metrics:
        st.markdown(f"*Метрики рассчитаны с использованием порога: **{OPTIMAL_THRESHOLD:.4f}***")
        st.write("")
        row1_col1, row1_col2, row1_col3 = st.columns(3)
        row2_col1, row2_col2, _ = st.columns(3)
        with row1_col1: st.metric("Recall (Fraud)", f"{metrics.get('recall_fraud', 'N/A'):.3f}")
        with row1_col2: st.metric("Precision (Fraud)", f"{metrics.get('precision_fraud', 'N/A'):.3f}")
        with row1_col3: st.metric("F1-Score (Fraud)", f"{metrics.get('f1_fraud', 'N/A'):.3f}")
        with row2_col1: st.metric("ROC AUC", f"{metrics.get('roc_auc', 'N/A'):.3f}", help="Площадь под ROC-кривой")
        with row2_col2: st.metric("PR AUC", f"{metrics.get('pr_auc', 'N/A'):.3f}", help="Площадь под Precision-Recall кривой")
        st.divider()
        st.subheader("Матрица Ошибок")
        cm_data = metrics.get('cm')
        if cm_data:
            cm = np.array(cm_data); fig_cm, ax_cm = plt.subplots(figsize=(6, 4.5))
            disp_cm = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Legit (0)', 'Fraud (1)'])
            disp_cm.plot(ax=ax_cm, cmap='Blues', values_format='d'); st.pyplot(fig_cm, use_container_width=False); plt.close(fig_cm)
        else: st.warning("Нет данных матрицы ошибок.")
    else: st.warning("Нет файла с метриками.")
    st.divider()
    st.subheader("Глобальная Важность Признаков (SHAP)")
    if SHAP_AVAILABLE and shap_values_sample and feature_names_transformed:
         with st.spinner("Отрисовка SHAP..."):
            try:
                shap_df = pd.DataFrame(shap_values_sample.values, columns=feature_names_transformed)
                fig_shap, ax_shap = plt.subplots(figsize=(8, 6))
                shap.summary_plot(shap_values_sample.values, features=shap_df, feature_names=feature_names_transformed, plot_type="bar", show=False, max_display=20)
                st.pyplot(fig_shap, use_container_width=False); plt.close(fig_shap)
            except Exception as e: st.warning(f"Ошибка SHAP plot: {e}")
    else: st.warning("SHAP values или имена признаков не загружены.")


# =====================================
# ВКЛАДКА 3: ИССЛЕДОВАНИЕ ДАННЫХ (EDA)
# =====================================
with tab_eda:
    st.header("Исследование Исходных Данных")
    if df_full is not None:
        st.write("Анализ распределений и взаимосвязей исходных признаков.")

        # Выбор признака для гистограмм
        feature_hist = st.selectbox("Признак для гистограмм:", options=X_feature_columns, key="eda_hist")
        if feature_hist:
            st.subheader(f"Распределение: {feature_hist}")
            fig_h, ax_h = plt.subplots(1, 2, figsize=(14, 4))
            sns.histplot(df_full[feature_hist], kde=True, ax=ax_h[0]); ax_h[0].set_title(f'Общее')
            sns.histplot(data=df_full, x=feature_hist, hue=TARGET_COLUMN, kde=True, ax=ax_h[1], palette="viridis"); ax_h[1].set_title(f'По классу')
            st.pyplot(fig_h, use_container_width=True); plt.close(fig_h)

        st.divider()

        # ---> НОВОЕ: Scatter Plot для двух признаков <---
        st.subheader("Взаимосвязь двух признаков")
        col_sc1, col_sc2 = st.columns(2)
        with col_sc1:
            feature_sc_x = st.selectbox("Выберите признак для оси X:", options=X_feature_columns, key="eda_scatter_x", index=0) # По умолчанию первый
        with col_sc2:
            feature_sc_y = st.selectbox("Выберите признак для оси Y:", options=X_feature_columns, key="eda_scatter_y", index=1) # По умолчанию второй

        if feature_sc_x and feature_sc_y:
            # Используем небольшую выборку для scatter plot, чтобы не тормозило
            sample_size_scatter = min(5000, len(df_full))
            df_sample = df_full.sample(sample_size_scatter, random_state=RANDOM_STATE)

            fig_scatter, ax_scatter = plt.subplots(figsize=(8, 6))
            sns.scatterplot(data=df_sample, x=feature_sc_x, y=feature_sc_y, hue=TARGET_COLUMN, alpha=0.5, palette="viridis", ax=ax_scatter)
            ax_scatter.set_title(f'Взаимосвязь {feature_sc_x} и {feature_sc_y}')
            st.pyplot(fig_scatter, use_container_width=True)
            plt.close(fig_scatter)
        # ---> КОНЕЦ Scatter Plot <---

        st.divider()
        st.subheader(f"Статистики для признаков")
        st.dataframe(df_full.groupby(TARGET_COLUMN)[X_feature_columns].describe().T) # Показываем статистики для всех признаков

    else: st.error("Не удалось загрузить данные для EDA.")