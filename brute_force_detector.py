"""
brute_force_detector.py
=======================
Core logic for "Real-Time Brute-Force Detection via Predictive Machine Learning".

This is the notebook's pipeline refactored into importable functions so the
Streamlit GUI (app.py) can call it:

    load -> clean -> encode -> split -> (SMOTE) -> scale -> train/tune -> evaluate -> predict
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
KAGGLE_DATASET = "dnkumars/cybersecurity-intrusion-detection-dataset"
TARGET = "attack_detected"
ID_COLUMN = "session_id"
CATEGORICAL_COLUMNS = ["protocol_type", "browser_type", "encryption_used"]
NUMERIC_COLUMNS = [
    "network_packet_size",
    "login_attempts",
    "session_duration",
    "ip_reputation_score",
    "failed_logins",
    "unusual_time_access",
]
FEATURE_COLUMNS = [
    "network_packet_size",
    "protocol_type",
    "login_attempts",
    "session_duration",
    "encryption_used",
    "ip_reputation_score",
    "failed_logins",
    "browser_type",
    "unusual_time_access",
]


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def load_from_kaggle() -> pd.DataFrame:
    """Download the public dataset through kagglehub and return it as a DataFrame."""
    import kagglehub  # imported lazily so the app starts even if it is missing

    path = kagglehub.dataset_download(KAGGLE_DATASET)
    for root, _, files in os.walk(path):
        for f in files:
            if f.lower().endswith(".csv"):
                return pd.read_csv(os.path.join(root, f))
    raise FileNotFoundError(f"No CSV file found in {path}")


def load_from_csv(file_or_path) -> pd.DataFrame:
    """Load a CSV from an uploaded file object or a path."""
    return pd.read_csv(file_or_path)


def validate_columns(df: pd.DataFrame, need_target: bool = True) -> list[str]:
    """Return the list of required columns that are missing from df."""
    required = FEATURE_COLUMNS + ([TARGET] if need_target else [])
    return [c for c in required if c not in df.columns]


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop the ID column, fill gaps, remove duplicates. Returns (clean_df, report)."""
    out = df.copy()
    report = {"rows_before": len(out)}

    if ID_COLUMN in out.columns:
        out = out.drop(columns=ID_COLUMN)

    # Categorical gaps (e.g. encryption_used can be empty) become their own category.
    for col in CATEGORICAL_COLUMNS:
        if col in out.columns:
            out[col] = out[col].fillna("Unknown").astype(str)

    # Numeric gaps are filled with the median.
    for col in NUMERIC_COLUMNS:
        if col in out.columns and out[col].isnull().any():
            out[col] = out[col].fillna(out[col].median())

    report["duplicates_removed"] = int(out.duplicated().sum())
    out = out.drop_duplicates().reset_index(drop=True)
    report["rows_after"] = len(out)
    return out, report


def iqr_outliers(series: pd.Series) -> dict:
    """IQR outlier summary (same rule the notebook uses for session_duration)."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = (series < lower) | (series > upper)
    return {"q1": q1, "q3": q3, "iqr": iqr, "lower": lower, "upper": upper,
            "count": int(mask.sum()), "mask": mask}


# --------------------------------------------------------------------------
# Encoding
# --------------------------------------------------------------------------
def encode_categoricals(df: pd.DataFrame, encoders: dict | None = None):
    """
    Label-encode categorical columns.

    Fit mode (encoders=None): learn one mapping per column.
    Transform mode: reuse the mappings; unseen categories become -1.
    Returns (encoded_df, encoders).
    """
    out = df.copy()
    fit = encoders is None
    encoders = {} if fit else encoders

    for col in CATEGORICAL_COLUMNS:
        values = out[col].fillna("Unknown").astype(str)
        if fit:
            classes = sorted(values.unique())
            encoders[col] = {c: i for i, c in enumerate(classes)}
        out[col] = values.map(encoders[col]).fillna(-1).astype(int)
    return out, encoders


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
@dataclass
class TrainedModel:
    model: RandomForestClassifier
    scaler: StandardScaler
    encoders: dict
    feature_names: list
    metrics: dict
    confusion: np.ndarray
    importances: pd.DataFrame
    best_params: dict = field(default_factory=dict)
    cv_score: float | None = None
    train_class_counts: dict = field(default_factory=dict)


def train_model(
    clean_df: pd.DataFrame,
    test_size: float = 0.20,
    use_smote: bool = True,
    tune: bool = False,
    n_estimators: int = 200,
    max_depth: int | None = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    search_iterations: int = 20,
    cv_folds: int = 3,
    random_state: int = 42,
) -> TrainedModel:
    """
    Train a Random Forest brute-force detector.

    Improvements over the notebook:
      * SMOTE is applied to the TRAINING split only (never to test data).
      * PR-AUC is computed from predicted probabilities, not hard labels.
      * Encoders/scaler are kept so new sessions can be scored later.
    """
    encoded, encoders = encode_categoricals(clean_df)
    X = encoded[FEATURE_COLUMNS]
    y = encoded[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    if use_smote:
        X_train, y_train = SMOTE(random_state=random_state).fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    best_params, cv_score = {}, None
    if tune:
        search = RandomizedSearchCV(
            RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1),
            param_distributions={
                "max_features": ["sqrt", "log2", None],
                "max_depth": [10, 20, 30, None],
                "min_samples_split": randint(2, 15),
                "min_samples_leaf": randint(1, 8),
                "bootstrap": [True, False],
                "class_weight": [None, "balanced"],
            },
            n_iter=search_iterations,
            cv=cv_folds,
            scoring="f1_weighted",
            n_jobs=-1,
            random_state=random_state,
        )
        search.fit(X_train_s, y_train)
        model, best_params, cv_score = search.best_estimator_, search.best_params_, search.best_score_
    else:
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)[:, 1]
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0),
        "PR-AUC": average_precision_score(y_test, y_proba),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
    }

    importances = (
        pd.DataFrame({"Feature": FEATURE_COLUMNS, "Importance": model.feature_importances_})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )

    return TrainedModel(
        model=model,
        scaler=scaler,
        encoders=encoders,
        feature_names=FEATURE_COLUMNS,
        metrics=metrics,
        confusion=confusion_matrix(y_test, y_pred),
        importances=importances,
        best_params=best_params,
        cv_score=cv_score,
        train_class_counts={int(k): int(v) for k, v in y_train.value_counts().items()},
    )


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------
def predict(trained: TrainedModel, sessions: pd.DataFrame) -> pd.DataFrame:
    """Score one or more sessions. Returns a copy with attack probability + label."""
    missing = validate_columns(sessions, need_target=False)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data = sessions.copy()
    for col in NUMERIC_COLUMNS:
        data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)

    encoded, _ = encode_categoricals(data, trained.encoders)
    X = trained.scaler.transform(encoded[trained.feature_names])
    proba = trained.model.predict_proba(X)[:, 1]

    result = sessions.copy()
    result["attack_probability"] = proba
    result["prediction"] = np.where(proba >= 0.5, "ATTACK", "Normal")
    return result


def export_model_bytes(trained: TrainedModel) -> bytes:
    """Serialize the trained pipeline (model + scaler + encoders) for download."""
    buf = io.BytesIO()
    joblib.dump(trained, buf)
    return buf.getvalue()
