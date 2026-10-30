"""
Streamlit Web Application: Explainable Random Forest Intrusion Detection Dashboard.
Network flow log analysis, multi-class threat classification, and SHAP explainability.
Project Title: Explainable Random Forest Framework for Multi-Class Intrusion Detection in Cloud Network Flow Logs
"""

import io
import json
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Configure Streamlit page layout and metadata
st.set_page_config(
    page_title="Explainable Random Forest Framework for Multi-Class Intrusion Detection in Cloud Network Flow Logs",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
DEMO_DIR = Path(__file__).parent / "demo_datasets"

# Light theme custom styling
CUSTOM_LIGHT_CSS = """
<style>
    /* Global background and font styling */
    .stApp {
        background-color: #F8FAFC;
        color: #0F172A;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    
    /* Clean sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E2E8F0;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }
    
    /* Card containers */
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.1rem 1.25rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.04), 0 1px 2px 0 rgba(0, 0, 0, 0.02);
        margin-bottom: 0.75rem;
    }
    .metric-card-title {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        margin-bottom: 0.35rem;
    }
    .metric-card-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #0F172A;
        line-height: 1.2;
    }
    .metric-card-subtitle {
        font-size: 0.78rem;
        color: #64748B;
        margin-top: 0.35rem;
    }
    
    /* Status pills */
    .status-pill {
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .status-pill-green {
        background-color: #DEF7EC;
        color: #03543F;
        border: 1px solid #BCF0DA;
    }
    .status-pill-amber {
        background-color: #FEF08A;
        color: #854D0E;
        border: 1px solid #FDE047;
    }
    .status-pill-red {
        background-color: #FDE8E8;
        color: #9B1C1C;
        border: 1px solid #FBD5D5;
    }
    .status-pill-blue {
        background-color: #E1EFFE;
        color: #1E429F;
        border: 1px solid #C3DDFD;
    }

    /* Comprehensive text color enforcement across all standard Streamlit elements */
    p, span, label, div, li, a {
        color: #0F172A;
    }
    
    /* Section headers */
    h1, h2, h3, h4, h5, h6 {
        color: #0F172A !important;
        font-weight: 600 !important;
    }
    
    /* Input widgets: selectboxes, file uploader, radio buttons, buttons */
    div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        border-color: #CBD5E1 !important;
    }
    div[data-baseweb="popover"], div[data-baseweb="menu"], ul[data-baseweb="menu"] {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
    }
    li[data-baseweb="menu-item"] {
        color: #0F172A !important;
        background-color: #FFFFFF !important;
    }
    li[data-baseweb="menu-item"]:hover {
        background-color: #F1F5F9 !important;
        color: #0F172A !important;
    }
    
    /* Radio options & labels */
    div[role="radiogroup"] label p {
        color: #1E293B !important;
        font-weight: 500 !important;
    }
    
    /* File uploader dropzone */
    div[data-testid="stFileUploader"] section {
        background-color: #FFFFFF !important;
        border: 2px dashed #CBD5E1 !important;
        color: #0F172A !important;
    }
    div[data-testid="stFileUploader"] section * {
        color: #334155 !important;
    }

    /* Buttons */
    button[kind="primary"], div.stButton > button[kind="primary"] {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        border: none !important;
        font-weight: 600 !important;
    }
    button[kind="secondary"], div.stButton > button[kind="secondary"], [data-testid="stDownloadButton"] > button {
        background-color: #FFFFFF !important;
        color: #1E293B !important;
        border: 1px solid #CBD5E1 !important;
        font-weight: 500 !important;
    }
    button[kind="secondary"]:hover, [data-testid="stDownloadButton"] > button:hover {
        background-color: #F8FAFC !important;
        border-color: #94A3B8 !important;
        color: #0F172A !important;
    }

    /* Expander styling */
    div[data-testid="stExpander"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 8px !important;
    }
    div[data-testid="stExpander"] details summary span {
        color: #0F172A !important;
        font-weight: 600 !important;
    }
    div[data-testid="stExpander"] details div[data-testid="stExpanderDetails"] * {
        color: #334155 !important;
    }

    /* Alerts and notification callouts */
    div[data-testid="stAlert"] {
        border-radius: 8px !important;
    }
    div[data-testid="stAlert"] * {
        color: #0F172A !important;
    }

    /* Inline code blocks */
    code {
        color: #B91C1C !important;
        background-color: #F1F5F9 !important;
        padding: 0.15rem 0.35rem !important;
        border-radius: 4px !important;
        border: 1px solid #E2E8F0 !important;
        font-size: 0.88em !important;
    }
    
    /* Streamlit tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #E2E8F0;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-bottom: none;
        border-top-left-radius: 8px;
        border-top-right-radius: 8px;
        padding: 0 20px;
        color: #475569;
        font-weight: 500;
        font-size: 0.92rem;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        border-color: #2563EB !important;
    }
    
    /* Dataframe wrapper border */
    [data-testid="stDataFrame"] {
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        background-color: #FFFFFF;
    }
</style>
"""


@st.cache_resource
def load_models_and_artifacts():
    """Load serialized model, preprocessor, SHAP metadata, and metrics from disk."""
    model_path = ARTIFACTS_DIR / "rf_model.joblib"
    prep_path = ARTIFACTS_DIR / "preprocessor.joblib"
    metrics_path = ARTIFACTS_DIR / "metrics.json"
    explainer_path = ARTIFACTS_DIR / "explainer.joblib"
    shap_global_path = ARTIFACTS_DIR / "shap_global.csv"
    gini_path = ARTIFACTS_DIR / "feature_importances.csv"

    model = joblib.load(model_path) if model_path.exists() else None
    preprocessor = joblib.load(prep_path) if prep_path.exists() else None

    metrics = None
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

    explainer_meta = joblib.load(explainer_path) if explainer_path.exists() else None
    shap_global = pd.read_csv(shap_global_path) if shap_global_path.exists() else None
    gini_df = pd.read_csv(gini_path) if gini_path.exists() else None

    return model, preprocessor, metrics, explainer_meta, shap_global, gini_df


def render_light_waterfall(df_contrib, selected_row_idx, target_pred, target_conf):
    """Render a clean, high-contrast light-mode SHAP horizontal bar chart."""
    top_12 = df_contrib.head(12).iloc[::-1]
    bar_colors = ["#DC2626" if x > 0 else "#2563EB" for x in top_12["SHAP Contribution"]]

    fig, ax = plt.subplots(figsize=(10, 5), facecolor="#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    bars = ax.barh(top_12["Feature"], top_12["SHAP Contribution"], color=bar_colors, height=0.65)
    ax.axvline(0, color="#94A3B8", linewidth=1.0, linestyle="--")

    ax.set_title(
        f"Local SHAP Feature Impact for Flow #{selected_row_idx} ({target_pred} - {target_conf:.1%})",
        fontsize=11,
        fontweight="bold",
        color="#0F172A",
        pad=12,
    )
    ax.set_xlabel("SHAP Attribution (Impact on Class Log-Odds)", fontsize=10, color="#475569")
    ax.tick_params(axis="both", labelsize=9, colors="#334155")
    ax.grid(axis="x", linestyle="--", alpha=0.5, color="#E2E8F0")

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#CBD5E1")

    plt.tight_layout()
    return fig


def main():
    st.markdown(CUSTOM_LIGHT_CSS, unsafe_allow_html=True)
    model, preprocessor, metrics, explainer_meta, shap_global, gini_df = load_models_and_artifacts()

    # Model status resolution
    model_loaded = model is not None and preprocessor is not None

    # ---------------- SIDEBAR ----------------
    with st.sidebar:
        st.markdown(
            """
            <div style="display:flex; align-items:center; gap:10px; margin-bottom:1rem;">
                <span style="font-size:1.8rem;">🛡️</span>
                <div>
                    <div style="font-size:1.05rem; font-weight:700; color:#0F172A; line-height:1.2;">FlowGuard NIDS</div>
                    <div style="font-size:0.75rem; color:#64748B;">Explainable AI Framework</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div style="background-color:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:0.75rem; margin-bottom:1rem;">
                <div style="font-size:0.72rem; text-transform:uppercase; font-weight:600; color:#64748B;">Core Architecture</div>
                <div style="font-size:0.85rem; font-weight:600; color:#1E293B; margin-top:2px;">Random Forest Ensemble</div>
                <div style="font-size:0.78rem; color:#475569; margin-top:4px;">Paired with TreeSHAP Game-Theoretic Attributions</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Model Status indicator
        if model_loaded:
            st.markdown(
                """
                <div class="status-pill status-pill-green" style="margin-bottom:1.25rem;">
                    ● Model Ready (15 Classes)
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="status-pill status-pill-red" style="margin-bottom:1.25rem;">
                    ● Artifacts Missing
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Navigation
        st.markdown("##### Navigation")
        app_mode = st.radio(
            "Select View:",
            [
                "📊 Traffic Analysis & Detection",
                "📈 Model Performance & Metrics",
                "🔍 Explainability & Feature Importance",
                "📖 Threat Taxonomy & Dataset Info",
            ],
            label_visibility="collapsed",
        )

        st.markdown("---")

        # Recorded Metrics Sidebar Summary
        if metrics:
            st.markdown(
                """
                <div style="font-size:0.78rem; font-weight:600; text-transform:uppercase; color:#64748B; margin-bottom:0.5rem;">
                    Recorded Evaluation
                </div>
                """,
                unsafe_allow_html=True,
            )
            col_sb1, col_sb2 = st.columns(2)
            with col_sb1:
                st.markdown(
                    f"""
                    <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:6px; padding:0.5rem 0.65rem;">
                        <div style="font-size:0.7rem; color:#64748B;">ACCURACY</div>
                        <div style="font-size:1.15rem; font-weight:700; color:#0F172A;">{metrics.get('accuracy', 0):.2%}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col_sb2:
                st.markdown(
                    f"""
                    <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:6px; padding:0.5rem 0.65rem;">
                        <div style="font-size:0.7rem; color:#64748B;">WEIGHTED F1</div>
                        <div style="font-size:1.15rem; font-weight:700; color:#0F172A;">{metrics.get('f1_weighted', 0):.4f}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown(
                f"""
                <div style="font-size:0.78rem; color:#64748B; margin-top:0.75rem; line-height:1.5;">
                    • <b>Dataset:</b> CSE-CIC-IDS2018<br>
                    • <b>Evaluated Flows:</b> {metrics.get('test_size', 0):,}<br>
                    • <b>Active Features:</b> {metrics.get('num_features', 0)}
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.caption("Machine Learning Project | CSE-CIC-IDS2018 Benchmark")

    # ---------------- HEADER ----------------
    st.markdown(
        """
        <div style="background-color:#FFFFFF; border:1px solid #E2E8F0; border-radius:10px; padding:1.25rem 1.5rem; margin-bottom:1.5rem; box-shadow:0 1px 3px 0 rgba(0,0,0,0.03);">
            <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:10px;">
                <div>
                    <h1 style="font-size:1.55rem; margin:0 0 0.35rem 0; color:#0F172A;">
                        Explainable Random Forest Framework for Multi-Class Intrusion Detection in Cloud Network Flow Logs
                    </h1>
                    <p style="font-size:0.9rem; color:#475569; margin:0;">
                        Offline machine learning framework for multi-class network flow traffic classification and instance-level SHAP explainability.
                    </p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------------- VIEW 1: TRAFFIC ANALYSIS & DETECTION ----------------
    if app_mode == "📊 Traffic Analysis & Detection":
        if not model_loaded:
            st.error("Model artifacts not found in `artifacts/`. Please run `python train.py` first.")
            return

        st.markdown("### 1. Network Flow Ingestion & Schema Specification")
        st.markdown(
            "Upload bidirectional network flow telemetry files (`.csv` or `.parquet`) generated by network flow extractors "
            "(e.g., CICFlowMeter, NetFlow/IPFIX converters). The preprocessor automatically aligns 69 statistical features, "
            "imputes median values for any missing attributes, and prunes constant features."
        )

        c_up1, c_up2 = st.columns([1.6, 1.4])
        with c_up1:
            uploaded_file = st.file_uploader(
                "Upload Flow Telemetry File (.csv or .parquet):",
                type=["csv", "parquet"],
                help="Requires tabular network connection flows containing statistical features (e.g., Flow Duration, Packet Lengths, IAT, Flags).",
            )

        with c_up2:
            st.markdown(
                """
                <div style="background-color:#FFFFFF; border:1px solid #E2E8F0; border-radius:8px; padding:0.85rem 1rem;">
                    <div style="font-size:0.82rem; font-weight:600; color:#0F172A; margin-bottom:0.35rem;">
                        📁 Packaged Demonstration Datasets
                    </div>
                    <div style="font-size:0.78rem; color:#64748B; margin-bottom:0.6rem;">
                        Select from ready-to-evaluate test records covering various attack scenarios:
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Available sample files in demo_datasets
            sample_options = {
                "demo_datasets/1_all_15_attack_scenarios.csv": "⚡ 1. All 15 Attack & Benign Scenarios (Combined Demo)",
                "demo_datasets/2_normal_benign_traffic.csv": "🟢 2. Normal Benign Traffic Only (Safe Baseline)",
                "demo_datasets/3_ddos_flood_incident.csv": "🔴 3. High-Rate DDoS Flood Attack Incident",
                "demo_datasets/4_bruteforce_ssh_ftp.csv": "🟠 4. SSH & FTP Credential Brute-Force",
                "demo_datasets/5_web_attacks_and_sqli.csv": "🟣 5. Web Exploits & SQL Injections",
                "demo_datasets/6_mixed_enterprise_traffic.csv": "🔵 6. Multi-Class Enterprise Threat Mix",
                "demo_datasets/7_raw_unlabeled_network_flows.csv": "⚪ 7. Raw Unlabeled Network Flow Records",
                "demo_datasets/8_sample_network_flows.csv": "📊 8. Representative Network Flow Sample",
            }

            # Filter only existing paths
            existing_sample_options = {}
            for rel_path, desc in sample_options.items():
                p = Path(__file__).parent / rel_path
                if p.exists():
                    existing_sample_options[rel_path] = desc

            selected_sample_rel = st.selectbox(
                "Choose an existing sample scenario:",
                list(existing_sample_options.keys()),
                format_func=lambda k: existing_sample_options[k],
                label_visibility="collapsed",
            )
            sample_target_path = Path(__file__).parent / selected_sample_rel

            c_btn1, c_btn2 = st.columns(2)
            with c_btn1:
                use_selected_sample = st.button("🚀 Load Selected Dataset", type="primary", use_container_width=True)
            with c_btn2:
                if sample_target_path.exists():
                    with open(sample_target_path, "rb") as sf:
                        sample_bytes = sf.read()
                    st.download_button(
                        label="📥 Download CSV",
                        data=sample_bytes,
                        file_name=sample_target_path.name,
                        mime="text/csv",
                        use_container_width=True,
                    )

        # Ingestion logic
        df_uploaded = None
        data_source_name = ""

        if uploaded_file is not None:
            data_source_name = uploaded_file.name
            try:
                if uploaded_file.name.endswith(".parquet"):
                    df_uploaded = pd.read_parquet(uploaded_file)
                else:
                    df_uploaded = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error(f"Error reading uploaded file: {e}")
        elif use_selected_sample or ("active_sample" in st.session_state and st.session_state["active_sample"]):
            if use_selected_sample:
                st.session_state["active_sample"] = selected_sample_rel
            active_file_rel = st.session_state.get("active_sample", selected_sample_rel)
            cur_path = Path(__file__).parent / active_file_rel
            if cur_path.exists():
                data_source_name = cur_path.name
                df_uploaded = pd.read_csv(cur_path) if cur_path.suffix == ".csv" else pd.read_parquet(cur_path)

        if df_uploaded is not None:
            st.markdown("---")
            st.markdown(
                f"""
                <div style="background-color:#F0FDF4; border:1px solid #BBF7D0; border-radius:8px; padding:0.75rem 1rem; margin-bottom:1.25rem;">
                    <span style="color:#166534; font-weight:600;">✓ Ingested Dataset:</span> 
                    <code>{data_source_name}</code> — <b>{len(df_uploaded):,}</b> network flow records, <b>{df_uploaded.shape[1]}</b> telemetry columns.
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Detect ground truth if present
            ground_truth_col = None
            for candidate in ["True_Label", "Label", "label", "target", "Attack"]:
                if candidate in df_uploaded.columns:
                    ground_truth_col = candidate
                    break

            # Separate feature matrix
            ignore_cols = [c for c in ["True_Label", "Label", "label", "target", "Attack"] if c in df_uploaded.columns]
            features_df = df_uploaded.drop(columns=ignore_cols, errors="ignore")

            with st.spinner("Executing Random Forest multi-class inference..."):
                X_trans = preprocessor.transform_features(features_df)
                predictions_idx = model.predict(X_trans)
                probabilities = model.predict_proba(X_trans)

                class_names = preprocessor.label_encoder.classes_
                predicted_labels = [class_names[i] for i in predictions_idx]
                max_confidences = [float(probabilities[i, predictions_idx[i]]) for i in range(len(predictions_idx))]

                results_df = df_uploaded.copy()
                results_df["Predicted_Class"] = predicted_labels
                results_df["Model_Confidence"] = max_confidences
                results_df["Is_Attack"] = [l.lower() != "benign" for l in predicted_labels]

            # ---------------- PREDICTION SUMMARY CARDS ----------------
            st.markdown("### 2. Model Prediction Summary")
            total_flows = len(results_df)
            total_attacks = int(results_df["Is_Attack"].sum())
            total_benign = total_flows - total_attacks
            attack_pct = (total_attacks / total_flows) * 100 if total_flows > 0 else 0

            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            with kpi1:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-card-title">Flows Evaluated</div>
                        <div class="metric-card-value">{total_flows:,}</div>
                        <div class="metric-card-subtitle">Total input flow records</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with kpi2:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-card-title">Benign / Normal</div>
                        <div class="metric-card-value" style="color:#059669;">{total_benign:,}</div>
                        <div class="metric-card-subtitle">{100 - attack_pct:.1f}% classified safe</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with kpi3:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-card-title">Attacks Flagged</div>
                        <div class="metric-card-value" style="color:#DC2626;">{total_attacks:,}</div>
                        <div class="metric-card-subtitle">{attack_pct:.1f}% potential threats</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with kpi4:
                if attack_pct == 0:
                    status_badge = '<span class="status-pill status-pill-green">🟢 NORMAL (0% Attacks)</span>'
                elif attack_pct < 25:
                    status_badge = '<span class="status-pill status-pill-amber">🟡 ELEVATED RISK</span>'
                else:
                    status_badge = '<span class="status-pill status-pill-red">🔴 HIGH ATTACK DENSITY</span>'

                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-card-title">Detection Status</div>
                        <div style="margin-top:0.35rem; margin-bottom:0.25rem;">{status_badge}</div>
                        <div class="metric-card-subtitle">Based on model output</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Charts
            c_ch1, c_ch2 = st.columns([1, 1.3])
            with c_ch1:
                st.markdown("##### 🥧 Traffic Proportion")
                fig_pie, ax_pie = plt.subplots(figsize=(4.5, 3.8), facecolor="#FFFFFF")
                ax_pie.set_facecolor("#FFFFFF")
                ax_pie.pie(
                    [total_benign, total_attacks],
                    labels=["Benign", "Attacks"],
                    autopct="%1.1f%%",
                    colors=["#10B981", "#EF4444"],
                    startangle=90,
                    textprops={"color": "#1E293B", "fontsize": 9, "weight": "bold"},
                    explode=(0, 0.08) if total_attacks > 0 else (0, 0),
                )
                ax_pie.axis("equal")
                plt.tight_layout()
                st.pyplot(fig_pie)
                plt.close(fig_pie)

            with c_ch2:
                st.markdown("##### 🚨 Predicted Attack Category Breakdown")
                attack_df = results_df[results_df["Is_Attack"]]
                if not attack_df.empty:
                    counts = attack_df["Predicted_Class"].value_counts()
                    fig_bar, ax_bar = plt.subplots(figsize=(6.5, 3.8), facecolor="#FFFFFF")
                    ax_bar.set_facecolor("#FFFFFF")
                    counts.plot(kind="barh", color="#EF4444", ax=ax_bar)
                    ax_bar.set_xlabel("Number of Flows", fontsize=9, color="#475569")
                    ax_bar.tick_params(axis="both", labelsize=8.5, colors="#334155")
                    ax_bar.grid(axis="x", linestyle="--", alpha=0.5, color="#E2E8F0")
                    for spine in ["top", "right"]:
                        ax_bar.spines[spine].set_visible(False)
                    for spine in ["left", "bottom"]:
                        ax_bar.spines[spine].set_color("#CBD5E1")
                    plt.tight_layout()
                    st.pyplot(fig_bar)
                    plt.close(fig_bar)
                else:
                    st.markdown(
                        """
                        <div style="background-color:#F0FDF4; border:1px solid #BBF7D0; border-radius:8px; padding:1.5rem; text-align:center; color:#166534; margin-top:2rem;">
                            ✓ All evaluated network flows in this dataset were classified as Benign.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # ---------------- FILTERABLE DATA TABLE ----------------
            st.markdown("---")
            st.markdown("### 3. Flow-Level Inspection Table")
            filter_mode = st.radio(
                "Filter displayed records:",
                ["All Flows", "Flagged Attacks Only 🚨", "Benign Only ✅"],
                horizontal=True,
            )

            if filter_mode == "Flagged Attacks Only 🚨":
                display_df = results_df[results_df["Is_Attack"]]
            elif filter_mode == "Benign Only ✅":
                display_df = results_df[~results_df["Is_Attack"]]
            else:
                display_df = results_df

            # Resolve table preview columns
            identity_cols = [
                c for c in [
                    "Source_IP_User", "User", "Client", "Src IP", "Source_IP",
                    "Destination_Server", "Dst IP", "Destination_IP", "Timestamp"
                ] if c in display_df.columns
            ]
            preview_cols = ["Predicted_Class", "Model_Confidence"]
            if ground_truth_col and ground_truth_col in display_df.columns:
                preview_cols.append(ground_truth_col)

            key_telemetry_cols = [
                c for c in [
                    "Flow Duration", "Flow Bytes/s", "Flow Packets/s",
                    "Init Fwd Win Bytes", "Fwd Packet Length Max", "Avg Packet Size"
                ] if c in display_df.columns
            ]
            table_cols = [c for c in (identity_cols + preview_cols + key_telemetry_cols) if c in display_df.columns]

            format_dict = {}
            if "Model_Confidence" in table_cols:
                format_dict["Model_Confidence"] = "{:.2%}"
            if "Flow Bytes/s" in table_cols:
                format_dict["Flow Bytes/s"] = "{:,.1f}"
            if "Flow Packets/s" in table_cols:
                format_dict["Flow Packets/s"] = "{:,.1f}"

            st.dataframe(
                display_df[table_cols].style.format(format_dict),
                use_container_width=True,
                height=300,
            )

            # Download CSV
            csv_buf = io.StringIO()
            results_df.to_csv(csv_buf, index=False)
            st.download_button(
                label="📥 Download Annotated Predictions (CSV)",
                data=csv_buf.getvalue(),
                file_name=f"predictions_{Path(data_source_name).stem}.csv",
                mime="text/csv",
            )

            # ---------------- LOCAL EXPLAINABILITY (SHAP) ----------------
            st.markdown("---")
            st.markdown("### 4. Instance-Level SHAP Explainability")
            st.markdown(
                "Select any individual flow record from the table above to compute its **local SHAP attribution**. "
                "SHAP computes the marginal contribution of each network feature toward increasing (red) or decreasing (blue) "
                "the model's prediction score for the selected class."
            )

            row_indices = display_df.index.tolist()
            if row_indices:
                selected_row_idx = st.selectbox(
                    "Select Flow Index to Explain:",
                    row_indices,
                    format_func=lambda idx: (
                        f"Row #{idx} | Predicted: {results_df.loc[idx, 'Predicted_Class']} "
                        f"({results_df.loc[idx, 'Model_Confidence']:.1%})"
                        + (f" | Ground Truth: {results_df.loc[idx, ground_truth_col]}" if ground_truth_col else "")
                    ),
                )

                target_pred = results_df.loc[selected_row_idx, "Predicted_Class"]
                target_conf = results_df.loc[selected_row_idx, "Model_Confidence"]
                pred_class_idx = list(class_names).index(target_pred)

                if st.button("🔬 Compute SHAP Attribution for Flow", type="primary"):
                    with st.spinner("Calculating TreeSHAP polynomial attributions..."):
                        import shap
                        background = explainer_meta["sample_background"] if explainer_meta else X_trans.iloc[:50]
                        tree_explainer = shap.TreeExplainer(model, data=background)

                        row_trans = X_trans.loc[[selected_row_idx]]
                        shap_vals = tree_explainer.shap_values(row_trans)

                        if isinstance(shap_vals, list):
                            sample_shap = shap_vals[pred_class_idx][0]
                        elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
                            sample_shap = shap_vals[0, :, pred_class_idx]
                        else:
                            sample_shap = shap_vals[0]

                        features_list = preprocessor.feature_names
                        values_list = row_trans.iloc[0].values

                        contrib_list = []
                        for f_name, val, sv in zip(features_list, values_list, sample_shap):
                            contrib_list.append({
                                "Feature": f_name,
                                "Flow Value": float(val),
                                "SHAP Contribution": float(sv),
                                "Abs Impact": abs(float(sv)),
                            })

                        df_contrib = pd.DataFrame(contrib_list).sort_values(by="Abs Impact", ascending=False)
                        pos_factors = df_contrib[df_contrib["SHAP Contribution"] > 0].head(5)
                        neg_factors = df_contrib[df_contrib["SHAP Contribution"] < 0].head(5)

                        c_exp1, c_exp2 = st.columns(2)
                        with c_exp1:
                            st.markdown("##### 🔺 Top Features Increasing Prediction Score")
                            if not pos_factors.empty:
                                for _, r in pos_factors.iterrows():
                                    st.markdown(
                                        f"- **`{r['Feature']}`** = `{r['Flow Value']:,.2f}` "
                                        f"(+**{r['SHAP Contribution']:.4f}** SHAP)"
                                    )
                            else:
                                st.info("No positive contributing factors detected.")

                        with c_exp2:
                            st.markdown("##### 🔻 Top Features Decreasing Prediction Score")
                            if not neg_factors.empty:
                                for _, r in neg_factors.iterrows():
                                    st.markdown(
                                        f"- **`{r['Feature']}`** = `{r['Flow Value']:,.2f}` "
                                        f"(**{r['SHAP Contribution']:.4f}** SHAP)"
                                    )
                            else:
                                st.info("No negative contributing factors detected.")

                        # Waterfall plot
                        st.markdown("##### 📊 Local SHAP Waterfall Plot")
                        fig_waterfall = render_light_waterfall(df_contrib, selected_row_idx, target_pred, target_conf)
                        st.pyplot(fig_waterfall)
                        plt.close(fig_waterfall)

                        # Narrative Summary
                        top_pos_feats = [f"`{r['Feature']}` ({r['Flow Value']:,.1f})" for _, r in pos_factors.head(3).iterrows()]
                        narrative = (
                            f"**Explanation Summary**: Evaluated flow record #{selected_row_idx}. "
                            f"The Random Forest model classified this connection as **{target_pred}** with a confidence score of **{target_conf:.1%}**. "
                        )
                        if top_pos_feats:
                            narrative += f"The primary flow metrics pushing the model toward this classification were {', '.join(top_pos_feats)}. "
                        if target_pred.lower() != "benign":
                            narrative += "Recommended defensive action: Validate source IP, review destination service exposure, and apply network ACL rules if confirmed anomalous."
                        else:
                            narrative += "The telemetry metrics fall within baseline distribution parameters for benign network traffic."

                        st.info(narrative)
            else:
                st.info("No rows match the current filter selection.")
        else:
            st.info("👆 Please upload a network flow file or select a demonstration dataset above to begin analysis.")

    # ---------------- VIEW 2: MODEL PERFORMANCE & METRICS ----------------
    elif app_mode == "📈 Model Performance & Metrics":
        st.markdown("### Model Benchmark & Multi-Class Evaluation Results")
        st.markdown(
            "Below are the **recorded evaluation results** obtained by testing the trained Random Forest model "
            "on the held-out test split of **24,801 flow records** across 15 distinct multi-class categories."
        )

        if not metrics:
            st.warning("Evaluation metrics file (`artifacts/metrics.json`) not found.")
            return

        # Recorded KPI Cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-card-title">Recorded Accuracy</div>
                    <div class="metric-card-value">{metrics['accuracy']:.2%}</div>
                    <div class="metric-card-subtitle">Overall test accuracy</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-card-title">Macro Precision</div>
                    <div class="metric-card-value">{metrics['precision_macro']:.4f}</div>
                    <div class="metric-card-subtitle">Unweighted mean across classes</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-card-title">Macro Recall</div>
                    <div class="metric-card-value">{metrics['recall_macro']:.4f}</div>
                    <div class="metric-card-subtitle">Unweighted mean detection rate</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-card-title">Macro F1-Score</div>
                    <div class="metric-card-value" style="color:#2563EB;">{metrics['f1_macro']:.4f}</div>
                    <div class="metric-card-subtitle">Weighted F1: {metrics['f1_weighted']:.4f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")
        c_perf1, c_perf2 = st.columns([1.2, 1])

        with c_perf1:
            st.markdown("#### 🧩 Confusion Matrix Heatmap")
            cm_img_path = FIGURES_DIR / "confusion_matrix.png"
            if cm_img_path.exists():
                st.image(str(cm_img_path), caption="Normalized Multi-Class Confusion Matrix (24,801 Test Records)", use_container_width=True)
            else:
                st.warning("Confusion matrix plot not found in `artifacts/figures/`.")

        with c_perf2:
            st.markdown("#### 📊 Per-Class Metric Overview")
            class_img_path = FIGURES_DIR / "per_class_performance.png"
            if class_img_path.exists():
                st.image(str(class_img_path), caption="Precision, Recall, and F1 across Attack Categories", use_container_width=True)
            else:
                st.warning("Per-class performance plot not found.")

        # Detailed Report Table
        st.markdown("#### 📋 Comprehensive Classification Breakdown")
        report_data = []
        for cls, vals in metrics["classification_report"].items():
            if isinstance(vals, dict):
                report_data.append({
                    "Traffic / Attack Class": cls,
                    "Precision": f"{vals['precision']:.4f}",
                    "Recall": f"{vals['recall']:.4f}",
                    "F1-Score": f"{vals['f1-score']:.4f}",
                    "Support (Flow Records)": f"{int(vals['support']):,}",
                })
        st.dataframe(pd.DataFrame(report_data), use_container_width=True)

        st.markdown(
            """
            <div style="background-color:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:1rem; margin-top:1rem; font-size:0.85rem; color:#475569;">
                <b>Performance Notes:</b> High-volume attack vectors (DDoS, DoS, Botnets, and SSH-Bruteforce) achieve near-perfect 
                F1-scores (&ge;0.99) due to distinct packet size and rate anomalies. Conversely, classes with extreme minority representation 
                in the raw dataset (e.g., Infiltration, FTP-BruteForce, SQL Injection) exhibit lower precision due to flow similarities with 
                benign administrative traffic.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---------------- VIEW 3: EXPLAINABILITY & FEATURE IMPORTANCE ----------------
    elif app_mode == "🔍 Explainability & Feature Importance":
        st.markdown("### Global Feature Explainability & Attributions")
        st.markdown(
            "Global explainability reveals which network flow features have the greatest overall influence on the model's decisions "
            "across the entire dataset. This differs from local SHAP explanations, which explain individual predictions."
        )

        col_g1, col_g2 = st.columns([1.1, 1.1])
        with col_g1:
            st.markdown("#### 🏆 Global SHAP Feature Ranking")
            shap_img = FIGURES_DIR / "shap_importance.png"
            if shap_img.exists():
                st.image(str(shap_img), caption="Mean Absolute SHAP Value across Evaluated Samples", use_container_width=True)
            else:
                st.warning("SHAP importance image not found.")

        with col_g2:
            st.markdown("#### 🌲 Random Forest Gini Impurity (MDI)")
            gini_img = FIGURES_DIR / "feature_importances.png"
            if gini_img.exists():
                st.image(str(gini_img), caption="Top 20 Features via Mean Decrease in Impurity (Gini)", use_container_width=True)
            else:
                st.warning("Gini feature importance image not found.")

        # Tables
        c_tab1, c_tab2 = st.columns(2)
        with c_tab1:
            if shap_global is not None:
                st.markdown("##### SHAP Global Importance Ranking (Top 15)")
                st.dataframe(
                    shap_global.head(15).style.format({"Mean_Abs_SHAP": "{:.5f}"}),
                    use_container_width=True,
                    height=350,
                )
        with c_tab2:
            if gini_df is not None:
                st.markdown("##### Gini Impurity Importance (Top 15)")
                st.dataframe(
                    gini_df.head(15).style.format({gini_df.columns[1]: "{:.5f}"}),
                    use_container_width=True,
                    height=350,
                )

    # ---------------- VIEW 4: THREAT TAXONOMY & DATASET INFO ----------------
    elif app_mode == "📖 Threat Taxonomy & Dataset Info":
        st.markdown("### Network Cyber-Threat Taxonomy & Remediation Guidelines")
        st.markdown(
            "The **CSE-CIC-IDS2018** benchmark includes 15 distinct classes covering common enterprise network intrusions. "
            "Below is the technical profile, primary flow indicators, and defensive remediation strategies for each category:"
        )

        threats = [
            {
                "Family": "DDoS (Distributed Denial of Service)",
                "Specific Variants": "DDOS attack-HOIC, LOIC-UDP, LOIC-HTTP",
                "Description": "Volumetric and application-layer floods designed to exhaust network bandwidth and server compute resources.",
                "Key Flow Signatures": "High `Flow Packets/s`, small `Flow IAT Mean`, repetitive `Init Fwd Win Bytes`, abnormal `Fwd Packets Length Total`.",
                "Defense": "Upstream DDoS mitigation, edge rate limiting, DNS filtering, auto-scaling thresholds.",
            },
            {
                "Family": "DoS (Denial of Service)",
                "Specific Variants": "GoldenEye, Hulk, Slowloris, SlowHTTPTest",
                "Description": "Targeted resource starvation attacks, such as slow HTTP headers (Slowloris) or continuous TCP session holding.",
                "Key Flow Signatures": "Extremely elevated `Flow Duration`, low `Flow Bytes/s`, elevated `Idle Mean`, persistent TCP SYN/ACK flags.",
                "Defense": "Reverse-proxy timeout clamping, Web Application Firewall (WAF) rate rules, connection limits.",
            },
            {
                "Family": "Brute Force",
                "Specific Variants": "SSH-Bruteforce, FTP-BruteForce",
                "Description": "Automated credential-stuffing dictionary attacks against administrative service ports (22, 21).",
                "Key Flow Signatures": "Burst of short flows with identical `Fwd Header Length`, high TCP RST/FIN flags, rapid inter-arrival time.",
                "Defense": "Firewall rules restricting ports 22/21 to VPN bastions, key-based authentication, fail2ban.",
            },
            {
                "Family": "Web Attacks",
                "Specific Variants": "Brute Force -Web, Brute Force -XSS, SQL Injection",
                "Description": "Attacks targeting web applications hosted on servers, attempting data exfiltration or script injection.",
                "Key Flow Signatures": "Asymmetric `Fwd Packets Length Total` vs `Bwd Packets Length Total`, unusual `Avg Packet Size`.",
                "Defense": "Web Application Firewall (WAF) managed rulesets (SQLi/XSS), strict input sanitization, parameterized queries.",
            },
            {
                "Family": "Botnet & Infiltration",
                "Specific Variants": "Bot, Infilteration",
                "Description": "Command & Control (C2) beaconing, post-exploitation lateral movement, and internal subnet staging.",
                "Key Flow Signatures": "Periodic `Active Mean` / `Idle Mean` cycles, consistent beacon packet sizes, low-bandwidth persistence.",
                "Defense": "Network threat intelligence feeds, DNS anomaly monitoring, egress proxy filtering, endpoint isolation.",
            },
        ]

        for t in threats:
            with st.expander(f"🛡️ {t['Family']} — {t['Specific Variants']}"):
                st.markdown(f"**Threat Profile**: {t['Description']}")
                st.markdown(f"**Flow Indicators**: `{t['Key Flow Signatures']}`")
                st.markdown(f"**Mitigation Strategy**: {t['Defense']}")

        st.markdown("---")
        st.markdown("### Technical Framework Information")
        st.markdown(
            """
            - **Model Algorithm:** Scikit-Learn `RandomForestClassifier` (100 estimators, max depth 20, balanced class weights).
            - **Explainability:** SHAP `TreeExplainer` providing local polynomial attributions and global feature rankings.
            - **Feature Count:** 69 numerical features retained after zero-variance pruning and median imputation.
            - **Offline Evaluation Context:** All metrics reflect evaluation on a static offline test split of the CSE-CIC-IDS2018 benchmark.
            """
        )


if __name__ == "__main__":
    main()
