"""Streamlit dashboard for the Fraud Detection project.

Three tabs:
  - Prediction & What-If: inspect individual transactions with SHAP explanations
  - Model Performance: metrics, confusion matrix, ROC/PR curves, global SHAP importance
  - Data Exploration (EDA): histograms, scatter plots, summary statistics
"""

import streamlit as st
import pandas as pd
import numpy as np
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
)

# --- Page config ---
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Import project modules ---
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
    st.error(f"Import error: {e}. Run Streamlit from the project root directory.")
    st.stop()

# --- Optional SHAP ---
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


# --- Cached loaders ---
@st.cache_resource
def load_model_assets():
    """Load the trained pipeline and SHAP explainer."""
    pipeline = load_pipeline(MODEL_SAVE_PATH)
    explainer = None
    if SHAP_AVAILABLE:
        explainer = load_joblib(SHAP_EXPLAINER_SAVE_PATH)
    if pipeline is None:
        st.error(f"Pipeline not found: {MODEL_SAVE_PATH}")
        st.stop()
    return pipeline, explainer


@st.cache_data
def load_other_assets():
    """Load metrics, SHAP values, feature names, and reconstruct data splits."""
    metrics = load_joblib(METRICS_SAVE_PATH)
    shap_values_sample = None
    X_shap_sample_transformed = None

    if SHAP_AVAILABLE:
        shap_values_full_obj = load_joblib(SHAP_VALUES_SAVE_PATH)
        if shap_values_full_obj is not None:
            shap_values_sample = (
                shap_values_full_obj.values
                if hasattr(shap_values_full_obj, "values")
                else shap_values_full_obj
            )
            if hasattr(shap_values_full_obj, "data") and isinstance(
                shap_values_full_obj.data, pd.DataFrame
            ):
                X_shap_sample_transformed = shap_values_full_obj.data
            else:
                X_shap_sample_transformed = load_joblib(TRANSFORMED_DATA_SAVE_PATH)
        if X_shap_sample_transformed is None:
            st.sidebar.warning("SHAP dependence plot data not found.", icon="⚠️")
        if shap_values_sample is None:
            st.sidebar.warning("SHAP values file not found.", icon="⚠️")

    feature_names_transformed = load_joblib(FEATURE_NAMES_SAVE_PATH)

    df = load_data(DATA_PATH)
    if df is None:
        st.error(f"Data not found: {DATA_PATH}")
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

    if metrics is None:
        st.sidebar.warning("Metrics file not found.", icon="⚠️")
    if feature_names_transformed is None:
        st.sidebar.warning("Feature names file not found.", icon="⚠️")

    legit_indices = y_test[y_test == 0].index.tolist()
    fraud_indices = y_test[y_test == 1].index.tolist()

    return (
        metrics,
        shap_values_sample,
        X_shap_sample_transformed,
        feature_names_transformed,
        df,
        list(X.columns),
        X_test,
        y_test,
        X_val,
        y_val,
        legit_indices,
        fraud_indices,
    )


@st.cache_data
def compute_test_probabilities(_pipeline, _X_test):
    """Return predicted fraud probabilities for the test set (cached)."""
    return _pipeline.predict_proba(_X_test)[:, 1]


# --- Session state ---
for key, default in [
    ("current_example_data", None),
    ("current_example_is_fraud", None),
    ("prediction_result", None),
    ("what_if_params", {}),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# --- Load all assets ---
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
except Exception as e:
    st.error(f"Failed to load assets: {e}")
    st.stop()

# --- Title ---
st.title("Fraud Detection Dashboard")

# --- Tabs ---
tab_explain, tab_dashboard, tab_eda = st.tabs(
    ["🔍 Prediction & What-If", "📊 Model Performance", "📈 Data Exploration (EDA)"]
)

# =====================================================================
# TAB 1: PREDICTION & WHAT-IF
# =====================================================================
with tab_explain:
    st.header("Transaction Prediction & SHAP Explanation")
    st.write(
        "Load a random transaction from the test set to see the model prediction "
        "and a per-feature SHAP breakdown."
    )

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("Load Legitimate", key="btn_legit_explain", width='stretch'):
            if legit_indices:
                idx = np.random.choice(legit_indices)
                st.session_state.current_example_data = X_test.loc[[idx]]
                st.session_state.current_example_is_fraud = False
                st.session_state.prediction_result = None
                st.session_state.what_if_params = {}
            else:
                st.warning("No legitimate examples found.")
            st.rerun()

    with col_btn2:
        if st.button("Load Fraudulent", key="btn_fraud_explain", width='stretch'):
            if fraud_indices:
                idx = np.random.choice(fraud_indices)
                st.session_state.current_example_data = X_test.loc[[idx]]
                st.session_state.current_example_is_fraud = True
                st.session_state.prediction_result = None
                st.session_state.what_if_params = {}
            else:
                st.warning("No fraudulent examples found.")
            st.rerun()

    if st.session_state.current_example_data is not None:
        example_data = st.session_state.current_example_data
        is_real_fraud = st.session_state.current_example_is_fraud
        example_idx = example_data.index[0]

        label_html = (
            '<span style="color:red;font-weight:bold;">Fraud</span>'
            if is_real_fraud
            else '<span style="color:green;font-weight:bold;">Legitimate</span>'
        )
        st.markdown(
            f"**Sample #{example_idx} — True label: {label_html}**",
            unsafe_allow_html=True,
        )
        st.write(
            f"Time = `{example_data['Time'].iloc[0]:.1f}` s &nbsp;&nbsp; "
            f"Amount = `${example_data['Amount'].iloc[0]:.2f}`"
        )

        with st.expander("Top-5 V-features by absolute value"):
            st.dataframe(
                example_data.filter(regex="^V")
                .T.iloc[:, 0]
                .abs()
                .nlargest(5)
                .to_frame(name="Value")
            )

        # Prediction
        try:
            proba_fraud = pipeline.predict_proba(example_data)[0, 1]
            is_pred_fraud = proba_fraud >= OPTIMAL_THRESHOLD
            pred_ok = True
        except Exception as e_pred:
            st.error(f"Prediction error: {e_pred}")
            pred_ok = False

        if pred_ok:
            st.subheader("Model Prediction")
            if is_pred_fraud:
                st.error(f"🔴 FRAUD — Probability: {proba_fraud:.2%}  (threshold: {OPTIMAL_THRESHOLD:.2f})")
            else:
                st.success(f"✅ LEGITIMATE — Probability: {proba_fraud:.2%}  (threshold: {OPTIMAL_THRESHOLD:.2f})")

            # SHAP explanation
            if SHAP_AVAILABLE and explainer and feature_names_transformed:
                st.markdown("---")
                st.subheader("SHAP Explanation")
                with st.spinner("Computing SHAP values..."):
                    try:
                        preprocessor = Pipeline(pipeline.steps[:-1])
                        input_tf = preprocessor.transform(example_data)
                        input_tf_df = pd.DataFrame(
                            input_tf,
                            columns=feature_names_transformed,
                            index=example_data.index,
                        )
                        sv_single = explainer(input_tf_df)

                        # Most influential feature caption
                        if shap_values_sample is not None:
                            local_sv = sv_single[0].values
                            top_idx = int(np.argmax(np.abs(local_sv)))
                            top_name = feature_names_transformed[top_idx]
                            top_local = local_sv[top_idx]
                            global_mean = float(
                                np.abs(shap_values_sample).mean(axis=0)[top_idx]
                            )
                            st.caption(
                                f"Most influential feature: **{top_name}** — "
                                f"local SHAP: {top_local:+.3f}, "
                                f"global mean |SHAP|: {global_mean:.3f}"
                            )

                        # Waterfall + Force plots side by side
                        col_wf, col_fp = st.columns(2)

                        with col_wf:
                            st.markdown("**Waterfall Plot**")
                            fig_wf, _ = plt.subplots(figsize=(8, 4))
                            shap.waterfall_plot(sv_single[0], show=False, max_display=15)
                            st.pyplot(fig_wf, width='stretch')
                            plt.close(fig_wf)

                        with col_fp:
                            st.markdown("**Force Plot**")
                            try:
                                base_val = explainer.expected_value
                                if hasattr(base_val, "__len__"):
                                    base_val = base_val[1]
                                shap.force_plot(
                                    base_val,
                                    sv_single[0].values,
                                    sv_single[0].data,
                                    feature_names=feature_names_transformed,
                                    matplotlib=True,
                                    show=False,
                                )
                                fig_fp = plt.gcf()
                                st.pyplot(fig_fp, width='stretch')
                                plt.close(fig_fp)
                            except Exception as e_fp:
                                st.info(f"Force plot unavailable: {e_fp}")

                    except Exception as e_shap:
                        st.warning(f"SHAP computation error: {e_shap}")

            # What-If Analysis
            st.markdown("---")
            st.subheader("What-If Analysis")
            st.caption(
                "Adjust feature values and click **Recalculate** to see how the prediction changes."
            )
            what_if_data = example_data.copy()
            important_features = ["Time", "Amount", "V4", "V12", "V14", "V10", "V17"]

            with st.form("what_if_form"):
                wi_cols = st.columns(4)
                for i, feature in enumerate(important_features):
                    if feature in what_if_data.columns:
                        with wi_cols[i % 4]:
                            orig = example_data[feature].iloc[0]
                            default = float(
                                st.session_state.what_if_params.get(feature, orig)
                            )
                            step = 1000.0 if feature == "Time" else (10.0 if feature == "Amount" else 0.1)
                            fmt = "%.1f" if feature == "Time" else ("%.2f" if feature == "Amount" else "%.4f")
                            val = st.number_input(
                                f"{feature}:",
                                value=default,
                                step=step,
                                format=fmt,
                                min_value=0.0 if feature == "Amount" else None,
                                key=f"wi_{feature}_{example_idx}",
                            )
                            what_if_data[feature] = val
                submitted = st.form_submit_button("Recalculate Prediction")

            if submitted:
                try:
                    wi_proba = pipeline.predict_proba(what_if_data)[0, 1]
                    wi_pred = wi_proba >= OPTIMAL_THRESHOLD
                    if wi_pred:
                        st.error(f"🔴 FRAUD — Probability: {wi_proba:.2%}")
                    else:
                        st.success(f"✅ LEGITIMATE — Probability: {wi_proba:.2%}")
                    for feature in important_features:
                        if feature in what_if_data.columns:
                            st.session_state.what_if_params[feature] = what_if_data[feature].iloc[0]
                except Exception as e_wi:
                    st.error(f"What-If prediction error: {e_wi}")


# =====================================================================
# TAB 2: MODEL PERFORMANCE
# =====================================================================
with tab_dashboard:
    st.header("Model Performance Overview (test set)")

    if metrics:
        st.sidebar.header("Dashboard Settings")
        threshold_mode = st.sidebar.selectbox(
            "Optimize threshold by:",
            [
                "From Config",
                "Maximize F1",
                f"Max Recall (Precision ≥ {MIN_PRECISION_TARGET:.2f})",
                "Max Precision (Recall ≥ 0.75)",
            ],
            key="threshold_select",
        )

        display_threshold = OPTIMAL_THRESHOLD
        shown_metrics = metrics.copy()

        if threshold_mode != "From Config":
            try:
                with st.spinner("Recalculating threshold on validation set..."):
                    y_proba_val = pipeline.predict_proba(X_val)[:, 1]
                    precs, recs, ths = precision_recall_curve(y_val, y_proba_val)
                    ths = np.append(ths, 1.0)
                    f1s = np.divide(
                        2 * precs * recs,
                        precs + recs,
                        out=np.zeros_like(precs),
                        where=(precs + recs) != 0,
                    )
                    new_thresh = OPTIMAL_THRESHOLD
                    if threshold_mode == "Maximize F1":
                        new_thresh = ths[np.argmax(f1s)]
                    elif "Max Recall" in threshold_mode:
                        valid = np.where(precs >= MIN_PRECISION_TARGET)[0]
                        if len(valid):
                            new_thresh = ths[valid[np.argmax(recs[valid])]]
                        else:
                            st.sidebar.warning(f"No threshold satisfies Precision ≥ {MIN_PRECISION_TARGET:.2f}")
                    elif "Max Precision" in threshold_mode:
                        valid = np.where(recs >= 0.75)[0]
                        if len(valid):
                            new_thresh = ths[valid[np.argmax(precs[valid])]]
                        else:
                            st.sidebar.warning("No threshold satisfies Recall ≥ 0.75")

                    display_threshold = new_thresh
                    st.sidebar.info(f"Selected threshold: {display_threshold:.4f}")

                    y_proba_test = pipeline.predict_proba(X_test)[:, 1]
                    y_pred_new = (y_proba_test >= display_threshold).astype(int)
                    shown_metrics["recall_fraud"] = recall_score(y_test, y_pred_new, pos_label=1)
                    shown_metrics["precision_fraud"] = precision_score(y_test, y_pred_new, pos_label=1)
                    shown_metrics["f1_fraud"] = f1_score(y_test, y_pred_new, pos_label=1)
                    shown_metrics["cm"] = confusion_matrix(y_test, y_pred_new).tolist()
            except Exception as e_thresh:
                st.sidebar.error(f"Threshold optimization error: {e_thresh}")
                display_threshold = OPTIMAL_THRESHOLD

        st.markdown(f"*Metrics computed with threshold: **{display_threshold:.4f}***")
        st.write("")

        c1, c2, c3 = st.columns(3)
        c4, c5, _ = st.columns(3)
        c1.metric("Recall (Fraud)", f"{shown_metrics.get('recall_fraud', 'N/A'):.3f}")
        c2.metric("Precision (Fraud)", f"{shown_metrics.get('precision_fraud', 'N/A'):.3f}")
        c3.metric("F1-Score (Fraud)", f"{shown_metrics.get('f1_fraud', 'N/A'):.3f}")
        c4.metric("ROC AUC", f"{shown_metrics.get('roc_auc', 'N/A'):.3f}")
        c5.metric("PR AUC", f"{shown_metrics.get('pr_auc', 'N/A'):.3f}")

        st.divider()

        # Confusion Matrix
        st.subheader("Confusion Matrix")
        cm_data = shown_metrics.get("cm")
        if cm_data:
            fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
            ConfusionMatrixDisplay(
                confusion_matrix=np.array(cm_data),
                display_labels=["Legit (0)", "Fraud (1)"],
            ).plot(ax=ax_cm, cmap="Blues", values_format="d")
            st.pyplot(fig_cm, width='content')
            plt.close(fig_cm)
        else:
            st.warning("Confusion matrix data not available.")

        st.divider()

        # ROC Curve + PR Curve
        st.subheader("ROC Curve & Precision-Recall Curve")
        with st.spinner("Computing curves..."):
            try:
                y_proba_cached = compute_test_probabilities(pipeline, X_test)
                col_roc, col_pr = st.columns(2)

                with col_roc:
                    fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
                    RocCurveDisplay.from_predictions(
                        y_test, y_proba_cached, ax=ax_roc, name="LightGBM"
                    )
                    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
                    ax_roc.set_title("ROC Curve")
                    ax_roc.legend()
                    st.pyplot(fig_roc, width='stretch')
                    plt.close(fig_roc)

                with col_pr:
                    fig_pr, ax_pr = plt.subplots(figsize=(6, 5))
                    PrecisionRecallDisplay.from_predictions(
                        y_test, y_proba_cached, ax=ax_pr, name="LightGBM"
                    )
                    ax_pr.set_title("Precision-Recall Curve")
                    st.pyplot(fig_pr, width='stretch')
                    plt.close(fig_pr)
            except Exception as e_curves:
                st.warning(f"Could not render curves: {e_curves}")

        st.divider()

    else:
        st.warning("Metrics file not loaded.")

    # Global SHAP Feature Importance
    st.subheader("Global Feature Importance (SHAP)")
    if (
        SHAP_AVAILABLE
        and shap_values_sample is not None
        and feature_names_transformed is not None
        and X_shap_sample_transformed is not None
    ):
        with st.spinner("Rendering SHAP summary..."):
            try:
                shap_df = pd.DataFrame(
                    X_shap_sample_transformed, columns=feature_names_transformed
                )
                fig_shap, _ = plt.subplots(figsize=(8, 6))
                shap.summary_plot(
                    shap_values_sample,
                    features=shap_df,
                    feature_names=feature_names_transformed,
                    plot_type="bar",
                    show=False,
                    max_display=20,
                )
                st.pyplot(fig_shap, width='content')
                plt.close(fig_shap)
            except Exception as e_shap:
                st.warning(f"SHAP summary plot error: {e_shap}")
    else:
        st.warning("SHAP values or feature names not loaded.")

    st.divider()

    # SHAP Dependence Plot
    st.subheader("SHAP Dependence Plot")
    if (
        SHAP_AVAILABLE
        and shap_values_sample is not None
        and X_shap_sample_transformed is not None
        and feature_names_transformed is not None
    ):
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            feat_dep = st.selectbox(
                "Feature:", options=feature_names_transformed, key="shap_dep_main"
            )
        with col_d2:
            interact_feat = st.selectbox(
                "Interaction feature:",
                options=["auto"] + feature_names_transformed,
                key="shap_dep_interaction",
            )
        if feat_dep:
            with st.spinner("Rendering dependence plot..."):
                try:
                    fig_dep, ax_dep = plt.subplots(figsize=(8, 5))
                    shap.dependence_plot(
                        feat_dep,
                        shap_values_sample,
                        X_shap_sample_transformed,
                        feature_names=feature_names_transformed,
                        interaction_index=None if interact_feat == "auto" else interact_feat,
                        ax=ax_dep,
                        show=False,
                    )
                    st.pyplot(fig_dep, width='stretch')
                    plt.close(fig_dep)
                except Exception as e_dep:
                    st.warning(f"Dependence plot error: {e_dep}")
    else:
        st.warning("SHAP values or feature names not loaded for dependence plot.")


# =====================================================================
# TAB 3: DATA EXPLORATION (EDA)
# =====================================================================
with tab_eda:
    st.header("Data Exploration")

    if df_full is not None:
        st.sidebar.header("EDA Filters")
        max_amount = int(df_full["Amount"].max())
        amount_range = st.sidebar.slider(
            "Amount range ($):",
            0,
            max_amount,
            (0, max_amount),
            step=100,
            key="eda_filter_amount",
        )
        df_filtered = df_full[
            df_full["Amount"].between(amount_range[0], amount_range[1])
        ]
        st.sidebar.info(f"{len(df_filtered):,} / {len(df_full):,} records after filter.")

        # Histogram
        feat_hist = st.selectbox(
            "Feature for distribution plot:", options=X_feature_columns, key="eda_hist"
        )
        if feat_hist:
            st.subheader(f"Distribution: {feat_hist}")
            if not df_filtered.empty:
                fig_h, axes = plt.subplots(1, 2, figsize=(14, 4))
                sns.histplot(df_filtered[feat_hist], kde=True, ax=axes[0])
                axes[0].set_title(f"{feat_hist} — overall (filtered)")
                sns.histplot(
                    data=df_filtered,
                    x=feat_hist,
                    hue=TARGET_COLUMN,
                    kde=True,
                    ax=axes[1],
                    palette="viridis",
                )
                axes[1].set_title(f"{feat_hist} — by class (filtered)")
                st.pyplot(fig_h, width='stretch')
                plt.close(fig_h)
            else:
                st.warning("No data matches the current filter.")

        st.divider()

        # Scatter Plot
        st.subheader("Feature Relationship")
        sc1, sc2 = st.columns(2)
        with sc1:
            feat_x = st.selectbox(
                "X axis:",
                options=X_feature_columns,
                key="eda_scatter_x",
                index=X_feature_columns.index("V4") if "V4" in X_feature_columns else 0,
            )
        with sc2:
            feat_y = st.selectbox(
                "Y axis:",
                options=X_feature_columns,
                key="eda_scatter_y",
                index=X_feature_columns.index("V14") if "V14" in X_feature_columns else 1,
            )

        if feat_x and feat_y and not df_filtered.empty:
            n_scatter = min(5000, len(df_filtered))
            df_sample = df_filtered.sample(n_scatter, random_state=RANDOM_STATE)
            fig_sc, ax_sc = plt.subplots(figsize=(8, 6))
            sns.scatterplot(
                data=df_sample,
                x=feat_x,
                y=feat_y,
                hue=TARGET_COLUMN,
                alpha=0.5,
                palette="viridis",
                ax=ax_sc,
            )
            ax_sc.set_title(f"{feat_x} vs {feat_y}  (n={n_scatter:,}, filtered)")
            st.pyplot(fig_sc, width='stretch')
            plt.close(fig_sc)
        elif df_filtered.empty:
            st.warning("No data matches the current filter.")

        st.divider()

        # Summary Statistics
        st.subheader("Summary Statistics (filtered data)")
        if not df_filtered.empty:
            st.dataframe(
                df_filtered.groupby(TARGET_COLUMN)[X_feature_columns].describe().T
            )
        else:
            st.warning("No data matches the current filter.")
    else:
        st.error("Could not load dataset for EDA.")
