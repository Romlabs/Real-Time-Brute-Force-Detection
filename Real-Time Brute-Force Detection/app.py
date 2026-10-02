"""
app.py - Streamlit GUI for Real-Time Brute-Force Detection.

Run locally:   streamlit run app.py
All ML logic lives in brute_force_detector.py.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

import brute_force_detector as bfd

st.set_page_config(page_title="Brute-Force Detection", page_icon="🛡️", layout="wide")


# --------------------------------------------------------------------------
# Cached helpers
# --------------------------------------------------------------------------
@st.cache_data(show_spinner="Downloading dataset from Kaggle...")
def cached_kaggle() -> pd.DataFrame:
    return bfd.load_from_kaggle()


@st.cache_data
def cached_clean(df: pd.DataFrame):
    return bfd.clean_data(df)


# --------------------------------------------------------------------------
# Sidebar: data source + training settings
# --------------------------------------------------------------------------
st.sidebar.title("🛡️ Settings")

st.sidebar.subheader("1. Data")
source = st.sidebar.radio("Data source", ["Upload CSV", "Download from Kaggle"])

raw_df = None
if source == "Upload CSV":
    up = st.sidebar.file_uploader("Intrusion-detection CSV", type="csv")
    if up is not None:
        raw_df = bfd.load_from_csv(up)
else:
    if st.sidebar.button("Download dataset"):
        try:
            st.session_state["kaggle_df"] = cached_kaggle()
        except Exception as e:  # network / kaggle issues
            st.sidebar.error(f"Could not download: {e}")
    raw_df = st.session_state.get("kaggle_df")

st.sidebar.subheader("2. Model")
test_size = st.sidebar.slider("Test split", 0.10, 0.40, 0.20, 0.05)
use_smote = st.sidebar.checkbox("Balance classes with SMOTE", value=True)
tune = st.sidebar.checkbox("Hyperparameter tuning (Randomized Search)", value=False)

if tune:
    n_estimators = st.sidebar.select_slider("Trees (n_estimators)", [100, 200, 300, 400], value=200)
    search_iter = st.sidebar.slider("Search iterations", 5, 40, 15)
    cv_folds = st.sidebar.slider("CV folds", 3, 5, 3)
    max_depth, mss, msl = None, 2, 1
else:
    n_estimators = st.sidebar.select_slider("Trees (n_estimators)", [50, 100, 200, 300, 400], value=200)
    depth_choice = st.sidebar.select_slider("Max depth", [5, 10, 20, 30, "None"], value="None")
    max_depth = None if depth_choice == "None" else int(depth_choice)
    mss = st.sidebar.slider("Min samples split", 2, 15, 2)
    msl = st.sidebar.slider("Min samples leaf", 1, 8, 1)
    search_iter, cv_folds = 15, 3

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
st.title("Real-Time Brute-Force Detection")
st.caption("Predictive machine learning (Random Forest) for spotting attack sessions.")

if raw_df is None:
    st.info("👈 Upload the dataset CSV or download it from Kaggle in the sidebar to begin.")
    st.stop()

missing = bfd.validate_columns(raw_df)
if missing:
    st.error(f"The dataset is missing required columns: {missing}")
    st.stop()

clean_df, report = cached_clean(raw_df)

tab_data, tab_train, tab_live, tab_batch = st.tabs(
    ["📊 Data Explorer", "🧠 Train & Evaluate", "⚡ Live Check", "📁 Batch Scoring"]
)

# ---------------------------- Data Explorer -------------------------------
with tab_data:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows (raw)", f"{report['rows_before']:,}")
    c2.metric("Duplicates removed", f"{report['duplicates_removed']:,}")
    c3.metric("Rows (clean)", f"{report['rows_after']:,}")
    c4.metric("Missing cells (raw)", f"{int(raw_df.isnull().sum().sum()):,}")

    st.subheader("Preview")
    st.dataframe(raw_df.head(10), use_container_width=True)

    with st.expander("Summary statistics"):
        st.dataframe(clean_df.describe(), use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Class balance")
        counts = clean_df[bfd.TARGET].value_counts().sort_index()
        counts.index = ["Normal (0)" if i == 0 else "Attack (1)" for i in counts.index]
        st.bar_chart(counts)
        st.write((counts / counts.sum() * 100).round(2).astype(str) + " %")
    with right:
        st.subheader("session_duration outliers (IQR)")
        out = bfd.iqr_outliers(clean_df["session_duration"])
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.boxplot(clean_df["session_duration"].dropna(), vert=False)
        ax.set_xlabel("session_duration")
        st.pyplot(fig)
        st.write(f"Bounds: **{out['lower']:.1f}** to **{out['upper']:.1f}** | "
                 f"Possible outliers: **{out['count']}**")

    st.subheader("Correlation heatmap")
    encoded_preview, _ = bfd.encode_categoricals(clean_df)
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(encoded_preview.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# --------------------------- Train & Evaluate -----------------------------
with tab_train:
    st.write("Configure the model in the sidebar, then train.")
    if st.button("🚀 Train model", type="primary"):
        with st.spinner("Training... tuning can take a few minutes."):
            st.session_state["trained"] = bfd.train_model(
                clean_df,
                test_size=test_size,
                use_smote=use_smote,
                tune=tune,
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=mss,
                min_samples_leaf=msl,
                search_iterations=search_iter,
                cv_folds=cv_folds,
            )

    trained = st.session_state.get("trained")
    if trained is None:
        st.info("No model trained yet.")
    else:
        st.subheader("Test-set metrics")
        cols = st.columns(len(trained.metrics))
        for col, (name, val) in zip(cols, trained.metrics.items()):
            col.metric(name, f"{val * 100:.2f}%")

        if trained.best_params:
            st.write("**Best parameters:**", trained.best_params)
            st.write(f"**Best CV F1 (weighted):** {trained.cv_score * 100:.2f}%")
        if use_smote:
            st.caption(f"Training class counts after SMOTE: {trained.train_class_counts}")

        l, r = st.columns(2)
        with l:
            st.subheader("Confusion matrix")
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(trained.confusion, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"], ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            st.pyplot(fig)
        with r:
            st.subheader("Feature importance")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(data=trained.importances, x="Importance", y="Feature", ax=ax)
            ax.grid(axis="x", linestyle="--", alpha=0.6)
            st.pyplot(fig)

        st.download_button(
            "💾 Download trained model (.joblib)",
            data=bfd.export_model_bytes(trained),
            file_name="brute_force_detector.joblib",
        )

# ------------------------------ Live Check --------------------------------
with tab_live:
    trained = st.session_state.get("trained")
    if trained is None:
        st.warning("Train a model first (🧠 Train & Evaluate tab).")
    else:
        st.write("Enter the details of a session to score it.")
        a, b, c = st.columns(3)
        with a:
            packet = st.number_input("Network packet size", 0, 100000,
                                     int(clean_df["network_packet_size"].median()))
            logins = st.number_input("Login attempts", 0, 1000,
                                     int(clean_df["login_attempts"].median()))
            failed = st.number_input("Failed logins", 0, 1000,
                                     int(clean_df["failed_logins"].median()))
        with b:
            duration = st.number_input("Session duration (s)", 0.0, 100000.0,
                                       float(clean_df["session_duration"].median()))
            rep = st.slider("IP reputation score", 0.0, 1.0,
                            float(np.clip(clean_df["ip_reputation_score"].median(), 0, 1)), 0.01)
            odd = st.selectbox("Unusual time access", [0, 1])
        with c:
            proto = st.selectbox("Protocol", sorted(trained.encoders["protocol_type"]))
            enc = st.selectbox("Encryption", sorted(trained.encoders["encryption_used"]))
            browser = st.selectbox("Browser", sorted(trained.encoders["browser_type"]))

        if st.button("🔍 Check session", type="primary"):
            row = pd.DataFrame([{
                "network_packet_size": packet, "protocol_type": proto,
                "login_attempts": logins, "session_duration": duration,
                "encryption_used": enc, "ip_reputation_score": rep,
                "failed_logins": failed, "browser_type": browser,
                "unusual_time_access": odd,
            }])
            res = bfd.predict(trained, row).iloc[0]
            prob = float(res["attack_probability"])
            if res["prediction"] == "ATTACK":
                st.error(f"🚨 Likely ATTACK - probability {prob * 100:.1f}%")
            else:
                st.success(f"✅ Looks normal - attack probability {prob * 100:.1f}%")
            st.progress(min(max(prob, 0.0), 1.0))

# ----------------------------- Batch Scoring ------------------------------
with tab_batch:
    trained = st.session_state.get("trained")
    if trained is None:
        st.warning("Train a model first (🧠 Train & Evaluate tab).")
    else:
        st.write("Upload a CSV of sessions (same feature columns; `attack_detected` is optional).")
        batch = st.file_uploader("Sessions CSV", type="csv", key="batch")
        if batch is not None:
            sessions = bfd.load_from_csv(batch)
            try:
                scored = bfd.predict(trained, sessions)
            except ValueError as e:
                st.error(str(e))
            else:
                n_attack = int((scored["prediction"] == "ATTACK").sum())
                st.metric("Flagged as attack", f"{n_attack} / {len(scored)}")
                st.dataframe(scored.sort_values("attack_probability", ascending=False),
                             use_container_width=True)
                st.download_button("Download results", scored.to_csv(index=False).encode(),
                                   "scored_sessions.csv", "text/csv")
