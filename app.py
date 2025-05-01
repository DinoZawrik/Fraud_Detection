"""
Основной скрипт для запуска Streamlit-дашборда.

Отображает производительность модели детекции мошенничества,
позволяет исследовать предсказания на отдельных примерах с помощью SHAP
и выполнять What-if анализ. Также включает вкладку для базового EDA.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    classification_report,
    roc_auc_score,
    average_precision_score,
)
import warnings

# --- Конфигурация страницы ---
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)
# ---------------------------

# Импорт утилит и конфигурации
try:
    from src.utils import load_pipeline, load_data, load_joblib
    from src.config import (
        MODEL_SAVE_PATH,
        DATA_PATH,
        TARGET_COLUMN,
        OPTIMAL_THRESHOLD,
        METRICS_SAVE_PATH,
        SHAP_EXPLAINER_SAVE_PATH,
        SHAP_VALUES_SAVE_PATH,
        FEATURE_NAMES_SAVE_PATH,
        TRANSFORMED_DATA_SAVE_PATH,
        RANDOM_STATE,
        TEST_SIZE,
        VALIDATION_SIZE,
        MIN_PRECISION_TARGET,
    )
except ImportError as e:
    st.error(
        f"Ошибка импорта модулей проекта: {e}. Убедитесь, что запускаете Streamlit из корневой папки проекта и структура папок верна ('src', 'models' и т.д.)."
    )
    st.stop()

# SHAP (опционально)
try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


# --- Функции Загрузки с Кэшированием ---
@st.cache_resource
def load_model_assets():
    """Загружает основные ресурсы модели: пайплайн и SHAP explainer."""
    pipeline = load_pipeline(MODEL_SAVE_PATH)
    explainer = None
    if SHAP_AVAILABLE:
        explainer = load_joblib(SHAP_EXPLAINER_SAVE_PATH)
    if pipeline is None:
        st.error(f"Критическая ошибка: Модель не загружена ({MODEL_SAVE_PATH}).")
        st.stop()
    return pipeline, explainer


@st.cache_data
def load_other_assets():
    """
    Загружает остальные ресурсы: метрики, SHAP values, имена признаков,
    исходные данные и разделенные выборки (test/validation).
    """
    metrics = load_joblib(METRICS_SAVE_PATH)
    shap_values_sample = None
    X_shap_sample_transformed = None
    if SHAP_AVAILABLE:
        shap_values_full_obj = load_joblib(SHAP_VALUES_SAVE_PATH)
        if shap_values_full_obj is not None:
            if hasattr(shap_values_full_obj, "values"):
                shap_values_sample = shap_values_full_obj.values
            else:
                shap_values_sample = shap_values_full_obj
            if hasattr(shap_values_full_obj, "data") and isinstance(
                shap_values_full_obj.data, pd.DataFrame
            ):
                X_shap_sample_transformed = shap_values_full_obj.data
            else:
                X_shap_sample_transformed = load_joblib(TRANSFORMED_DATA_SAVE_PATH)
        if X_shap_sample_transformed is None:
            st.sidebar.warning(f"Данные для SHAP dependence plot не найдены.", icon="⚠️")
        if shap_values_sample is None:
            st.sidebar.warning(f"Файл SHAP values не найден.", icon="⚠️")

    feature_names_transformed = load_joblib(FEATURE_NAMES_SAVE_PATH)
    df = load_data(DATA_PATH)
    if df is None:
        st.error(f"Критическая ошибка: Данные не загружены ({DATA_PATH}).")
        st.stop()

    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    _, X_val, _, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train_full,
    )
    X_cols = list(X.columns)
    legit_indices = y_test[y_test == 0].index.tolist()
    fraud_indices = y_test[y_test == 1].index.tolist()

    if metrics is None:
        st.sidebar.warning(f"Файл метрик не найден.", icon="⚠️")
    if feature_names_transformed is None:
        st.sidebar.warning(f"Имена признаков не найдены.", icon="⚠️")

    return (
        metrics,
        shap_values_sample,
        X_shap_sample_transformed,
        feature_names_transformed,
        df,
        X_cols,
        X_test,
        y_test,
        X_val,
        y_val,
        legit_indices,
        fraud_indices,
    )


# --- Инициализация Session State ---
if "current_example_data" not in st.session_state:
    st.session_state.current_example_data = None
if "current_example_is_fraud" not in st.session_state:
    st.session_state.current_example_is_fraud = None
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "what_if_params" not in st.session_state:
    st.session_state.what_if_params = {}

# --- Загрузка Ресурсов ---
try:
    pipeline, explainer = load_model_assets()
    (
        metrics,
        shap_values_sample,
        X_shap_sample_transformed,
        feature_names_transformed,
        df_full,
        X_feature_columns,
        X_test,
        y_test,
        X_val,
        y_val,
        legit_indices,
        fraud_indices,
    ) = load_other_assets()
    assets_loaded = True
except Exception as e:
    st.error(f"Ошибка при загрузке ресурсов: {e}")
    assets_loaded = False
    st.stop()

# --- Заголовок ---
st.title("Интерактивный Дашборд и Анализ Модели Детекции Мошенничества")

# --- Создание Вкладок ---
tab_explain, tab_dashboard, tab_eda = st.tabs(
    ["🔍 Объяснение и What-If", "📊 Дашборд Модели", "📈 Исследование Данных (EDA)"]
)

# =====================================
# ВКЛАДКА 1: ОБЪЯСНЕНИЕ и WHAT-IF
# =====================================
with tab_explain:
    st.header("Объяснение Предсказания для Примера")
    st.write(
        "Выберите тип транзакции, чтобы увидеть предсказание и объяснение Waterfall Plot."
    )

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button(
            "Показать Легитимную", key="btn_legit_explain", use_container_width=True
        ):
            if legit_indices:
                idx = np.random.choice(legit_indices)
                st.session_state.current_example_data = X_test.loc[[idx]]
                st.session_state.current_example_is_fraud = False
                st.session_state.prediction_result = None
                st.session_state.what_if_params = {}
            else:
                st.warning("Нет легитимных примеров.")
            st.rerun()
    with col_btn2:
        if st.button(
            "Показать Мошенническую", key="btn_fraud_explain", use_container_width=True
        ):
            if fraud_indices:
                idx = np.random.choice(fraud_indices)
                st.session_state.current_example_data = X_test.loc[[idx]]
                st.session_state.current_example_is_fraud = True
                st.session_state.prediction_result = None
                st.session_state.what_if_params = {}
            else:
                st.warning("Нет мошеннических примеров.")
            st.rerun()

    if st.session_state.current_example_data is not None:
        example_data = st.session_state.current_example_data
        is_real_fraud = st.session_state.current_example_is_fraud
        example_idx = example_data.index[0]

        if is_real_fraud:
            status_text = '<span style="color:red;">**Fraud**</span>'
        else:
            status_text = '<span style="color:green;">**Legit**</span>'
        st.markdown(
            f"**Пример `{example_idx}`. Реальный статус: {status_text}**",
            unsafe_allow_html=True,
        )
        st.write(
            f"Time=`{example_data['Time'].iloc[0]:.1f}`, Amount=`{example_data['Amount'].iloc[0]:.2f}`"
        )
        with st.expander("Топ-5 V признаков по модулю для этого примера"):
            st.dataframe(
                example_data.filter(regex="^V")
                .T.iloc[:, 0]
                .abs()
                .nlargest(5)
                .to_frame(name="Value")
            )

        # Предсказание
        pred_result = None
        try:
            probabilities = pipeline.predict_proba(example_data)
            proba_fraud = probabilities[0, 1]
            pred = proba_fraud >= OPTIMAL_THRESHOLD
            pred_result = {"proba": proba_fraud, "prediction": pred}
        except Exception as e:
            pred_result = {"error": True, "message": str(e)}

        # Отображение результата предсказания
        if pred_result:
            if "error" in pred_result:
                st.error(f"Ошибка предсказания: {pred_result['message']}")
            else:
                proba_fraud = pred_result["proba"]
                is_pred_fraud = pred_result["prediction"]
                st.subheader("Результат Модели:")
                pred_text = (
                    f"🔴 Предсказано: ПОДОЗРИТЕЛЬНАЯ (Вероятность: {proba_fraud:.2%})"
                    if is_pred_fraud
                    else f"✅ Предсказано: ЛЕГИТИМНАЯ (Вероятность: {proba_fraud:.2%})"
                )
                st.markdown(f"**{pred_text}** (Порог: {OPTIMAL_THRESHOLD:.2f})")

                # Отображение SHAP
                if SHAP_AVAILABLE and explainer and feature_names_transformed:
                    st.markdown("---")
                    st.subheader("Объяснение Предсказания (SHAP)")
                    with st.spinner("Расчет SHAP..."):
                        try:
                            pipeline_steps_before_classifier = pipeline.steps[:-1]
                            temp_preprocessor_pipeline = Pipeline(
                                pipeline_steps_before_classifier
                            )
                            input_transformed = temp_preprocessor_pipeline.transform(
                                example_data
                            )
                            input_transformed_df = pd.DataFrame(
                                input_transformed,
                                columns=feature_names_transformed,
                                index=example_data.index,
                            )
                            shap_values_single = explainer(input_transformed_df)

                            # Отображение локальной и глобальной важности
                            if shap_values_sample is not None:
                                global_shap_means = np.abs(shap_values_sample).mean(
                                    axis=0
                                )
                                global_shap_df = pd.DataFrame(
                                    {
                                        "feature": feature_names_transformed,
                                        "mean_abs_shap": global_shap_means,
                                    }
                                ).sort_values("mean_abs_shap", ascending=False)
                                local_shap_values = shap_values_single[0].values
                                top_local_feature_idx = np.argmax(
                                    np.abs(local_shap_values)
                                )
                                top_local_feature_name = feature_names_transformed[
                                    top_local_feature_idx
                                ]
                                top_local_feature_shap = local_shap_values[
                                    top_local_feature_idx
                                ]
                                try:
                                    top_global_feature_shap_mean = global_shap_df.loc[
                                        global_shap_df["feature"]
                                        == top_local_feature_name,
                                        "mean_abs_shap",
                                    ].iloc[0]
                                    st.caption(
                                        f"Наиболее влиятельный признак: **{top_local_feature_name}** (локальный SHAP: {top_local_feature_shap:.3f}). Его среднее глобальное влияние: {top_global_feature_shap_mean:.3f}."
                                    )
                                except (
                                    IndexError
                                ):  # Если имя фичи не найдено в глобальных (маловероятно)
                                    st.caption(
                                        f"Наиболее влиятельный признак: **{top_local_feature_name}** (локальный SHAP: {top_local_feature_shap:.3f})."
                                    )

                            # Waterfall Plot
                            st.markdown("**Waterfall Plot:**")
                            fig_waterfall, ax_waterfall = plt.subplots(figsize=(10, 4))
                            shap.waterfall_plot(
                                shap_values_single[0], show=False, max_display=15
                            )
                            st.pyplot(fig_waterfall, use_container_width=False)
                            plt.close(fig_waterfall)

                        except Exception as e_shap:
                            st.warning(f"Ошибка при расчете/отображении SHAP: {e_shap}")

                # --- What-if Анализ ---
                st.markdown("---")
                st.subheader("What-if Анализ")
                st.caption(
                    "Измените значения признаков, чтобы увидеть, как изменится предсказание."
                )
                what_if_data = example_data.copy()
                # Выбираем несколько важных признаков для What-if анализа
                important_features = [
                    "Time",
                    "Amount",
                    "V4",
                    "V12",
                    "V14",
                    "V10",
                    "V17",
                ]
                with st.form("what_if_form"):
                    what_if_cols = st.columns(4)
                    col_idx = 0
                    for feature in important_features:
                        if feature in what_if_data.columns:
                            with what_if_cols[col_idx % 4]:
                                original_value = example_data[feature].iloc[0]
                                default_val = float(
                                    st.session_state.what_if_params.get(
                                        feature, original_value
                                    )
                                )
                                step = (
                                    1000.0
                                    if feature == "Time"
                                    else (10.0 if feature == "Amount" else 0.1)
                                )
                                fmt = (
                                    "%.1f"
                                    if feature == "Time"
                                    else ("%.2f" if feature == "Amount" else "%.4f")
                                )
                                min_val = 0.0 if feature == "Amount" else None
                                edited_val = st.number_input(
                                    f"{feature}:",
                                    value=default_val,
                                    step=step,
                                    format=fmt,
                                    min_value=min_val,
                                    key=f"whatif_{feature}_{example_idx}",
                                )
                                what_if_data[feature] = edited_val
                            col_idx += 1
                    what_if_submitted = st.form_submit_button(
                        "Пересчитать Предсказание (What-If)"
                    )
                if what_if_submitted:
                    try:
                        what_if_proba = pipeline.predict_proba(what_if_data)[0, 1]
                        what_if_pred = what_if_proba >= OPTIMAL_THRESHOLD
                        st.markdown("**Результат What-If:**")
                        what_if_text = (
                            f"🔴 ПОДОЗРИТЕЛЬНАЯ (Вероятность: {what_if_proba:.2%})"
                            if what_if_pred
                            else f"✅ ЛЕГИТИМНАЯ (Вероятность: {what_if_proba:.2%})"
                        )
                        st.info(f"**{what_if_text}**")
                        for feature in important_features:
                            if feature in what_if_data.columns:
                                st.session_state.what_if_params[feature] = what_if_data[
                                    feature
                                ].iloc[0]
                    except Exception as e_whatif:
                        st.error(f"Ошибка при What-If предсказании: {e_whatif}")


# =====================================
# ВКЛАДКА 2: ДАШБОРД МОДЕЛИ
# =====================================
with tab_dashboard:
    st.header("Обзор Производительности Модели (на Тестовой выборке)")
    if metrics:
        st.sidebar.header("Настройки Дашборда")
        threshold_optimize_metric = st.sidebar.selectbox(
            "Оптимизировать порог по:",
            [
                "Из Конфига",
                "Максимум F1",
                f"Макс. Recall (при Precision>={MIN_PRECISION_TARGET:.2f})",
                f"Макс. Precision (при Recall>=0.75)",
            ],
            key="threshold_select",
        )
        display_threshold = OPTIMAL_THRESHOLD
        recalculated_metrics = metrics.copy()
        if threshold_optimize_metric != "Из Конфига":
            if (
                "pipeline" in locals()
                and "X_val" in locals()
                and "y_val" in locals()
                and "X_test" in locals()
                and "y_test" in locals()
            ):
                try:
                    with st.spinner("Пересчет порога..."):
                        y_proba_val = pipeline.predict_proba(X_val)[:, 1]
                        precisions, recalls, thresholds = precision_recall_curve(
                            y_val, y_proba_val
                        )
                        thresholds = np.append(thresholds, 1.0)
                        f1_scores = np.divide(
                            2 * precisions * recalls,
                            precisions + recalls,
                            out=np.zeros_like(precisions),
                            where=(precisions + recalls) != 0,
                        )
                        new_optimal_threshold = OPTIMAL_THRESHOLD
                        if threshold_optimize_metric == "Максимум F1":
                            optimal_idx = np.argmax(f1_scores)
                            new_optimal_threshold = thresholds[optimal_idx]
                        elif "Макс. Recall" in threshold_optimize_metric:
                            valid_idx = np.where(precisions >= MIN_PRECISION_TARGET)[0]
                            if len(valid_idx) > 0:
                                best_recall_idx = valid_idx[
                                    np.argmax(recalls[valid_idx])
                                ]
                                new_optimal_threshold = thresholds[best_recall_idx]
                            else:
                                st.sidebar.warning(
                                    f"Нет порога для Prec >= {MIN_PRECISION_TARGET:.2f}"
                                )
                        elif "Макс. Precision" in threshold_optimize_metric:
                            min_recall_target = 0.75
                            valid_idx = np.where(recalls >= min_recall_target)[0]
                            if len(valid_idx) > 0:
                                best_prec_idx = valid_idx[
                                    np.argmax(precisions[valid_idx])
                                ]
                                new_optimal_threshold = thresholds[best_prec_idx]
                            else:
                                st.sidebar.warning(
                                    f"Нет порога для Recall >= {min_recall_target:.2f}"
                                )
                        display_threshold = new_optimal_threshold
                        st.sidebar.info(f"Выбран порог: {display_threshold:.4f}")
                        y_proba_test = pipeline.predict_proba(X_test)[:, 1]
                        y_pred_test_new_thresh = (
                            y_proba_test >= display_threshold
                        ).astype(int)
                        recalculated_metrics["recall_fraud"] = recall_score(
                            y_test, y_pred_test_new_thresh, pos_label=1
                        )
                        recalculated_metrics["precision_fraud"] = precision_score(
                            y_test, y_pred_test_new_thresh, pos_label=1
                        )
                        recalculated_metrics["f1_fraud"] = f1_score(
                            y_test, y_pred_test_new_thresh, pos_label=1
                        )
                        recalculated_metrics["cm"] = confusion_matrix(
                            y_test, y_pred_test_new_thresh
                        ).tolist()
                except Exception as e_thresh_opt:
                    st.sidebar.error(f"Ошибка оптимизации порога: {e_thresh_opt}")
                    display_threshold = OPTIMAL_THRESHOLD
            else:
                st.sidebar.error("Нет данных для оптимизации.")
                display_threshold = OPTIMAL_THRESHOLD
        st.markdown(f"*Метрики рассчитаны с порогом: **{display_threshold:.4f}***")
        st.write("")
        row1_col1, row1_col2, row1_col3 = st.columns(3)
        row2_col1, row2_col2, _ = st.columns(3)
        with row1_col1:
            st.metric(
                "Recall (Fraud)",
                f"{recalculated_metrics.get('recall_fraud', 'N/A'):.3f}",
            )
        with row1_col2:
            st.metric(
                "Precision (Fraud)",
                f"{recalculated_metrics.get('precision_fraud', 'N/A'):.3f}",
            )
        with row1_col3:
            st.metric(
                "F1-Score (Fraud)", f"{recalculated_metrics.get('f1_fraud', 'N/A'):.3f}"
            )
        with row2_col1:
            st.metric("ROC AUC", f"{recalculated_metrics.get('roc_auc', 'N/A'):.3f}")
        with row2_col2:
            st.metric("PR AUC", f"{recalculated_metrics.get('pr_auc', 'N/A'):.3f}")
        st.divider()
        st.subheader("Матрица Ошибок")
        cm_data = recalculated_metrics.get("cm")
        if cm_data:
            cm = np.array(cm_data)
            fig_cm, ax_cm = plt.subplots(figsize=(6, 4.5))
            disp_cm = ConfusionMatrixDisplay(
                confusion_matrix=cm, display_labels=["Legit (0)", "Fraud (1)"]
            )
            disp_cm.plot(ax=ax_cm, cmap="Blues", values_format="d")
            st.pyplot(fig_cm, use_container_width=False)
            plt.close(fig_cm)
        else:
            st.warning("Нет данных CM.")
    else:
        st.warning("Файл метрик не загружен.")
    st.divider()
    st.subheader("Глобальная Важность Признаков (SHAP)")
    if (
        SHAP_AVAILABLE
        and shap_values_sample is not None
        and feature_names_transformed is not None
        and X_shap_sample_transformed is not None
    ):
        with st.spinner("Отрисовка SHAP..."):
            try:
                shap_df_for_plot = pd.DataFrame(
                    X_shap_sample_transformed, columns=feature_names_transformed
                )
                fig_shap, ax_shap = plt.subplots(figsize=(8, 6))
                shap.summary_plot(
                    shap_values_sample,
                    features=shap_df_for_plot,
                    feature_names=feature_names_transformed,
                    plot_type="bar",
                    show=False,
                    max_display=20,
                )
                st.pyplot(fig_shap, use_container_width=False)
                plt.close(fig_shap)
            except Exception as e:
                st.warning(f"Ошибка SHAP plot: {e}")
    else:
        st.warning("SHAP values/данные/имена не загружены.")
    st.divider()
    st.subheader("SHAP Dependence Plot")
    if (
        SHAP_AVAILABLE
        and shap_values_sample is not None
        and X_shap_sample_transformed is not None
        and feature_names_transformed is not None
    ):
        col_dep1, col_dep2 = st.columns(2)
        with col_dep1:
            feature_for_dependence = st.selectbox(
                "Признак:", options=feature_names_transformed, key="shap_dep_main"
            )
        with col_dep2:
            interaction_options = ["auto"] + feature_names_transformed
            interaction_feature = st.selectbox(
                "Взаимодействие:",
                options=interaction_options,
                key="shap_dep_interaction",
            )
        if feature_for_dependence:
            with st.spinner("Построение Dependence Plot..."):
                try:
                    fig_dep, ax_dep = plt.subplots(figsize=(8, 5))
                    interaction_idx = (
                        interaction_feature if interaction_feature != "auto" else None
                    )
                    shap.dependence_plot(
                        feature_for_dependence,
                        shap_values_sample,
                        X_shap_sample_transformed,
                        feature_names=feature_names_transformed,
                        interaction_index=interaction_idx,
                        ax=ax_dep,
                        show=False,
                    )
                    st.pyplot(fig_dep, use_container_width=True)
                    plt.close(fig_dep)
                except Exception as e_dep:
                    st.warning(f"Ошибка Dependence plot: {e_dep}")
    else:
        st.warning("SHAP values/данные/имена не загружены для Dependence plot.")


# =====================================
# ВКЛАДКА 3: ИССЛЕДОВАНИЕ ДАННЫХ (EDA)
# =====================================
with tab_eda:
    st.header("Исследование Исходных Данных")
    if df_full is not None:
        st.sidebar.header("Фильтры EDA")
        max_amount = int(df_full["Amount"].max())
        amount_range = st.sidebar.slider(
            "Диапазон Сумм (Amount):",
            0,
            max_amount,
            (0, max_amount),
            step=100,
            key="eda_filter_amount",
        )
        df_filtered = df_full[
            (df_full["Amount"] >= amount_range[0])
            & (df_full["Amount"] <= amount_range[1])
        ]
        st.sidebar.info(f"{len(df_filtered)}/{len(df_full)} записей после фильтрации.")
        feature_hist = st.selectbox(
            "Признак для гистограмм:", options=X_feature_columns, key="eda_hist"
        )
        if feature_hist:
            st.subheader(f"Распределение: {feature_hist}")
            if not df_filtered.empty:
                fig_h, ax_h = plt.subplots(1, 2, figsize=(14, 4))
                sns.histplot(df_filtered[feature_hist], kde=True, ax=ax_h[0])
                ax_h[0].set_title(f"Общее (фильтр)")
                sns.histplot(
                    data=df_filtered,
                    x=feature_hist,
                    hue=TARGET_COLUMN,
                    kde=True,
                    ax=ax_h[1],
                    palette="viridis",
                )
                ax_h[1].set_title(f"По классу (фильтр)")
                st.pyplot(fig_h, use_container_width=True)
                plt.close(fig_h)
            else:
                st.warning("Нет данных с фильтрами.")
        st.divider()
        st.subheader("Взаимосвязь двух признаков")
        col_sc1, col_sc2 = st.columns(2)
        with col_sc1:
            feature_sc_x = st.selectbox(
                "Ось X:",
                options=X_feature_columns,
                key="eda_scatter_x",
                index=X_feature_columns.index("V4") if "V4" in X_feature_columns else 0,
            )
        with col_sc2:
            feature_sc_y = st.selectbox(
                "Ось Y:",
                options=X_feature_columns,
                key="eda_scatter_y",
                index=(
                    X_feature_columns.index("V14") if "V14" in X_feature_columns else 1
                ),
            )
        if feature_sc_x and feature_sc_y and not df_filtered.empty:
            sample_size_scatter = min(5000, len(df_filtered))
            df_sample = df_filtered.sample(
                sample_size_scatter, random_state=RANDOM_STATE
            )
            fig_scatter, ax_scatter = plt.subplots(figsize=(8, 6))
            sns.scatterplot(
                data=df_sample,
                x=feature_sc_x,
                y=feature_sc_y,
                hue=TARGET_COLUMN,
                alpha=0.5,
                palette="viridis",
                ax=ax_scatter,
            )
            ax_scatter.set_title(f"{feature_sc_x} vs {feature_sc_y} (фильтр)")
            st.pyplot(fig_scatter, use_container_width=True)
            plt.close(fig_scatter)
        elif df_filtered.empty:
            st.warning("Нет данных для scatter plot.")
        st.divider()
        st.subheader(f"Статистики (для отфильтрованных данных)")
        if not df_filtered.empty:
            st.dataframe(
                df_filtered.groupby(TARGET_COLUMN)[X_feature_columns].describe().T
            )
        else:
            st.warning("Нет данных для статистик.")
    else:
        st.error("Данные для EDA не загружены.")
