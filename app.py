"""
app.py - Streamlit GUI for Real-Time Brute-Force Detection.

Run locally:   streamlit run app.py
ML logic lives in brute_force_detector.py. A pre-trained model is loaded from
model/brute_force_model.joblib (create it once with: python train_model.py).
"""

import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

import brute_force_detector as bfd

st.set_page_config(page_title="Brute-Force Detection", page_icon="🛡️", layout="wide")

MODEL_PATH = Path(__file__).parent / "model" / "brute_force_model.joblib"


# --------------------------------------------------------------------------
# Cached helpers
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading pre-trained model...")
def load_pretrained(path: str):
    if not os.path.exists(path):
        return None, None
    try:
        return bfd.load_model(path), None
    except Exception as e:  # e.g. scikit-learn version mismatch
        return None, str(e)


@st.cache_data(show_spinner="Downloading dataset from Kaggle...")
def cached_kaggle() -> pd.DataFrame:
    return bfd.load_from_kaggle()


@st.cache_data
def cached_clean(df: pd.DataFrame):
    return bfd.clean_data(df)


def show_fig(fig):
    st.pyplot(fig)
    plt.close(fig)


# --------------------------------------------------------------------------
# Startup: load the pre-trained model into session state
# --------------------------------------------------------------------------
pretrained, load_error = load_pretrained(str(MODEL_PATH))
if "trained" not in st.session_state and pretrained is not None:
    st.session_state["trained"] = pretrained
    st.session_state["model_source"] = "Pre-trained model (loaded from the repository)"

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
st.sidebar.title("🛡️ Settings")

st.sidebar.subheader("Detection threshold")
threshold = st.sidebar.slider(
    "Alert threshold", 0.05, 0.95, 0.50, 0.05,
    help="Lower = catches more attacks but raises more false alarms. "
         "Higher = fewer false alarms but more missed attacks.",
)

st.sidebar.subheader("Data (only for exploring / retraining)")
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
        except Exception as e:
            st.sidebar.error(f"Could not download: {e}")
    raw_df = st.session_state.get("kaggle_df")

clean_df, report = (None, None)
if raw_df is not None:
    missing = bfd.validate_columns(raw_df)
    if missing:
        st.sidebar.error(f"Dataset is missing columns: {missing}")
        raw_df = None
    else:
        clean_df, report = cached_clean(raw_df)

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("Real-Time Brute-Force Detection")
st.caption("Random Forest classifier that flags attack sessions from login and network behaviour.")

trained = st.session_state.get("trained")
if trained is None:
    if load_error:
        st.warning(f"A saved model was found but could not be loaded ({load_error}). "
                   "Re-run `python train_model.py` with the same package versions as the app, "
                   "or use the Retrain tab.")
    else:
        st.info("No model yet. Run `python train_model.py` and commit `model/brute_force_model.joblib`, "
                "or train one in the **Retrain** tab (needs the dataset).")
else:
    st.caption(f"Model: {st.session_state.get('model_source', 'Trained in this session')}")

tab_perf, tab_live, tab_check, tab_batch, tab_data, tab_train = st.tabs([
    "📈 Performance", "⚡ Live Stream", "🔍 Check a Session",
    "📁 Batch Scoring", "📊 Data Explorer", "🧠 Retrain",
])

NEEDS_MODEL = "No model available yet - see the message at the top of the page."

# ------------------------------ Performance -------------------------------
with tab_perf:
    if trained is None or trained.y_test is None:
        st.warning(NEEDS_MODEL)
    else:
        m = bfd.metrics_at_threshold(trained.y_test, trained.y_proba, threshold)
        st.subheader(f"Results on held-out test data at threshold {threshold:.2f}")
        cols = st.columns(6)
        cols[0].metric("Accuracy", f"{m['Accuracy'] * 100:.2f}%")
        cols[1].metric("Precision", f"{m['Precision'] * 100:.2f}%")
        cols[2].metric("Recall", f"{m['Recall'] * 100:.2f}%")
        cols[3].metric("F1 Score", f"{m['F1 Score'] * 100:.2f}%")
        cols[4].metric("Missed attacks", m["FN"])
        cols[5].metric("False alarms", m["FP"])
        st.caption(f"Threshold-independent: ROC-AUC {trained.metrics['ROC-AUC'] * 100:.2f}% | "
                   f"PR-AUC {trained.metrics['PR-AUC'] * 100:.2f}%. Move the **Alert threshold** "
                   "slider in the sidebar to trade missed attacks against false alarms.")

        curves = bfd.roc_pr_curves(trained.y_test, trained.y_proba)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        ax1.plot(curves["fpr"], curves["tpr"], label="Model")
        ax1.plot([0, 1], [0, 1], "--", color="grey", label="Random guess")
        ax1.scatter([m["FPR"]], [m["Recall"]], color="red", zorder=5, label="Current threshold")
        ax1.set(xlabel="False positive rate", ylabel="True positive rate (recall)",
                title=f"ROC curve (AUC {curves['roc_auc']:.3f})")
        ax1.legend(loc="lower right")
        ax2.plot(curves["recall"], curves["precision"], label="Model")
        ax2.scatter([m["Recall"]], [m["Precision"]], color="red", zorder=5, label="Current threshold")
        ax2.set(xlabel="Recall", ylabel="Precision",
                title=f"Precision-Recall curve (AP {curves['pr_auc']:.3f})")
        ax2.legend(loc="lower left")
        fig.tight_layout()
        show_fig(fig)

        l, r = st.columns(2)
        with l:
            st.subheader("Confusion matrix")
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(m["confusion"], annot=True, fmt="d", cmap="Blues",
                        xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"], ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            show_fig(fig)
        with r:
            st.subheader("Feature importance")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(data=trained.importances, x="Importance", y="Feature", ax=ax)
            ax.grid(axis="x", linestyle="--", alpha=0.6)
            show_fig(fig)

        st.subheader("Trade-off by threshold")
        sweep = pd.DataFrame([
            {"Threshold": t, **{k: v for k, v in bfd.metrics_at_threshold(
                trained.y_test, trained.y_proba, t).items() if k in ("Precision", "Recall", "F1 Score")}}
            for t in np.arange(0.05, 1.0, 0.05)
        ]).set_index("Threshold")
        st.line_chart(sweep)
        st.caption("Scores come from a held-out split of this dataset. Very high numbers on a clean "
                   "benchmark dataset do not guarantee the same results on real network traffic.")

# ------------------------------ Live Stream -------------------------------
with tab_live:
    if trained is None or trained.test_sessions is None:
        st.warning(NEEDS_MODEL)
    else:
        st.write("Replays sessions the model never saw during training, one at a time, "
                 "and raises an alert for each session scored above the threshold.")
        c1, c2, c3 = st.columns([2, 2, 1])
        n_sessions = c1.slider("Sessions to replay", 20, 200, 60, 10)
        speed = c2.slider("Speed (sessions per second)", 1, 10, 4)
        start = c3.button("▶ Start", type="primary", width="stretch")
        st.caption("To stop early, change any control.")

        counters_ph, chart_ph, table_ph = st.empty(), st.empty(), st.empty()

        if start:
            stream = bfd.simulate_stream(trained, n=n_sessions, threshold=threshold)
            alert = (stream["prediction"] == "ATTACK").to_numpy()
            actual = stream["actual"].to_numpy() == 1

            def outcome(a, p):
                if a and p:
                    return "✔ attack caught"
                if a and not p:
                    return "✖ attack missed"
                if p:
                    return "✖ false alarm"
                return "✔ normal"

            for i in range(len(stream)):
                a, p = alert[: i + 1], actual[: i + 1]
                with counters_ph.container():
                    k = st.columns(5)
                    k[0].metric("Sessions seen", i + 1)
                    k[1].metric("Alerts raised", int(a.sum()))
                    k[2].metric("Attacks caught", int((a & p).sum()))
                    k[3].metric("Attacks missed", int((~a & p).sum()))
                    k[4].metric("False alarms", int((a & ~p).sum()))

                probs = stream["attack_probability"].iloc[: i + 1].reset_index(drop=True)
                chart_ph.line_chart(pd.DataFrame({"Attack probability": probs,
                                                  "Alert threshold": threshold}))

                recent = stream.iloc[max(0, i - 7): i + 1].iloc[::-1]
                table_ph.dataframe(pd.DataFrame({
                    "Status": np.where(recent["prediction"] == "ATTACK", "🚨 ALERT", "✅ ok"),
                    "Attack probability": recent["attack_probability"].round(3),
                    "Actual": np.where(recent["actual"] == 1, "Attack", "Normal"),
                    "Outcome": [outcome(x == 1, y == "ATTACK")
                                for x, y in zip(recent["actual"], recent["prediction"])],
                    "Login attempts": recent["login_attempts"],
                    "Failed logins": recent["failed_logins"],
                    "IP reputation": recent["ip_reputation_score"].round(3),
                }), width="stretch", hide_index=True)
                time.sleep(1 / speed)

# ---------------------------- Check a Session -----------------------------
with tab_check:
    if trained is None or trained.test_sessions is None:
        st.warning(NEEDS_MODEL)
    else:
        ts = trained.test_sessions
        st.write("Enter the details of a session to score it.")
        a, b, c = st.columns(3)
        with a:
            packet = st.number_input("Network packet size", 0, 100000, int(ts["network_packet_size"].median()))
            logins = st.number_input("Login attempts", 0, 1000, int(ts["login_attempts"].median()))
            failed = st.number_input("Failed logins", 0, 1000, int(ts["failed_logins"].median()))
        with b:
            duration = st.number_input("Session duration (s)", 0.0, 100000.0, float(ts["session_duration"].median()))
            rep = st.slider("IP reputation score", 0.0, 1.0,
                            float(np.clip(ts["ip_reputation_score"].median(), 0, 1)), 0.01)
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
            res = bfd.predict(trained, row, threshold=threshold).iloc[0]
            prob = float(res["attack_probability"])
            if res["prediction"] == "ATTACK":
                st.error(f"🚨 Likely ATTACK - probability {prob * 100:.1f}% (threshold {threshold:.2f})")
            else:
                st.success(f"✅ Looks normal - attack probability {prob * 100:.1f}% (threshold {threshold:.2f})")
            st.progress(min(max(prob, 0.0), 1.0))

# ----------------------------- Batch Scoring ------------------------------
with tab_batch:
    if trained is None:
        st.warning(NEEDS_MODEL)
    else:
        st.write("Upload a CSV of sessions (same feature columns; `attack_detected` is optional).")
        batch = st.file_uploader("Sessions CSV", type="csv", key="batch")
        if batch is not None:
            try:
                scored = bfd.predict(trained, bfd.load_from_csv(batch), threshold=threshold)
            except ValueError as e:
                st.error(str(e))
            else:
                n_attack = int((scored["prediction"] == "ATTACK").sum())
                st.metric("Flagged as attack", f"{n_attack} / {len(scored)}")
                st.dataframe(scored.sort_values("attack_probability", ascending=False),
                             width="stretch")
                st.download_button("Download results", scored.to_csv(index=False).encode(),
                                   "scored_sessions.csv", "text/csv")

# ----------------------------- Data Explorer ------------------------------
with tab_data:
    if clean_df is None:
        st.info("Upload the dataset CSV or download it from Kaggle (sidebar) to explore the data.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows (raw)", f"{report['rows_before']:,}")
        c2.metric("Duplicates removed", f"{report['duplicates_removed']:,}")
        c3.metric("Rows (clean)", f"{report['rows_after']:,}")
        c4.metric("Missing cells (raw)", f"{int(raw_df.isnull().sum().sum()):,}")

        st.subheader("Preview")
        st.dataframe(raw_df.head(10), width="stretch")
        with st.expander("Summary statistics"):
            st.dataframe(clean_df.describe(), width="stretch")

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
            sns.boxplot(x=clean_df["session_duration"].dropna(), ax=ax)
            ax.set_xlabel("session_duration")
            show_fig(fig)
            st.write(f"Bounds: **{out['lower']:.1f}** to **{out['upper']:.1f}** | "
                     f"Possible outliers: **{out['count']}**")

        st.subheader("Correlation heatmap")
        encoded_preview, _ = bfd.encode_categoricals(clean_df)
        fig, ax = plt.subplots(figsize=(9, 6))
        sns.heatmap(encoded_preview.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        show_fig(fig)

# -------------------------------- Retrain ---------------------------------
with tab_train:
    if clean_df is None:
        st.info("Load the dataset in the sidebar to retrain. Retraining here only affects "
                "your current session; to change the deployed model, run `train_model.py` "
                "and commit the new file.")
    else:
        t1, t2, t3 = st.columns(3)
        test_size = t1.slider("Test split", 0.10, 0.40, 0.20, 0.05)
        use_smote = t1.checkbox("Balance classes with SMOTE", value=True)
        tune = t2.checkbox("Randomized Search tuning (slower)", value=False)
        n_estimators = t2.select_slider("Trees", [50, 100, 200, 300, 400], value=200)
        if tune:
            search_iter = t3.slider("Search iterations", 5, 40, 15)
            cv_folds = t3.slider("CV folds", 3, 5, 3)
            max_depth, mss, msl = None, 2, 1
        else:
            depth_choice = t3.select_slider("Max depth", [5, 10, 20, 30, "None"], value="None")
            max_depth = None if depth_choice == "None" else int(depth_choice)
            mss = t3.slider("Min samples split", 2, 15, 2)
            msl = t3.slider("Min samples leaf", 1, 8, 1)
            search_iter, cv_folds = 15, 3

        if st.button("🚀 Train model", type="primary"):
            with st.spinner("Training... tuning can take a few minutes."):
                st.session_state["trained"] = bfd.train_model(
                    clean_df, test_size=test_size, use_smote=use_smote, tune=tune,
                    n_estimators=n_estimators, max_depth=max_depth,
                    min_samples_split=mss, min_samples_leaf=msl,
                    search_iterations=search_iter, cv_folds=cv_folds,
                )
                st.session_state["model_source"] = "Trained in this session"
                st.session_state.pop("model_bytes", None)
            st.success("Done - open the Performance tab to see the results.")
            st.rerun()

        if trained is not None:
            # Serializing the model is heavy, so only do it when asked (not on every rerun).
            if st.button("Prepare model file for download"):
                st.session_state["model_bytes"] = bfd.export_model_bytes(trained)
            if "model_bytes" in st.session_state:
                st.download_button("💾 Download model (.joblib)",
                                   data=st.session_state["model_bytes"],
                                   file_name="brute_force_model.joblib")
