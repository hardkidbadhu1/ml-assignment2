"""
Streamlit front-end for ML Assignment 2.

Loads the Pipelines persisted by `model/train.py`, scores an uploaded test CSV,
and reports the six required metrics plus a confusion matrix and classification
report for the selected model, alongside a cross-model comparison.

Run locally:  streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

from ml_pipeline import METRIC_ORDER, compute_metrics, positive_index, score

ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT / "model" / "artifacts"
METADATA_PATH = ARTIFACT_DIR / "metadata.json"
BUNDLED_TEST = ROOT / "test_data.csv"

METRIC_HELP = {
    "Accuracy": "Fraction of correct predictions. Flattering under class imbalance.",
    "AUC": "Ranking quality across all thresholds; 0.5 is a coin flip.",
    "Precision": "Of the instances predicted positive, how many truly are.",
    "Recall": "Of the truly positive instances, how many were caught.",
    "F1": "Harmonic mean of precision and recall.",
    "MCC": "Correlation over the whole confusion matrix; robust to imbalance. Range -1 to +1.",
}

st.set_page_config(page_title="ML Assignment 2 — Classifier Comparison", layout="wide")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading trained models…")
def load_metadata() -> dict:
    if not METADATA_PATH.exists():
        st.error(
            "`model/artifacts/metadata.json` not found. Run "
            "`python model/train.py --data <csv> --target <col>` and commit the artifacts."
        )
        st.stop()
    return json.loads(METADATA_PATH.read_text())


@st.cache_resource(show_spinner=False)
def load_models(artifacts: dict[str, str]) -> dict:
    models = {}
    for name, filename in artifacts.items():
        path = ARTIFACT_DIR / filename
        if path.exists():
            models[name] = joblib.load(path)
    if not models:
        st.error(f"No `.joblib` artifacts found under `{ARTIFACT_DIR}`.")
        st.stop()
    return models


@st.cache_data(show_spinner=False)
def read_csv(source) -> pd.DataFrame:
    return pd.read_csv(source)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_confusion(cm: np.ndarray, labels: list[str], normalise: bool) -> plt.Figure:
    data = cm.astype(float)
    if normalise:
        totals = data.sum(axis=1, keepdims=True)
        data = np.divide(data, totals, out=np.zeros_like(data), where=totals != 0)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        data,
        annot=True,
        fmt=".2f" if normalise else ".0f",
        cmap="Blues",
        cbar=False,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix" + (" (row-normalised)" if normalise else ""))
    fig.tight_layout()
    return fig


def plot_roc(y_true: np.ndarray, y_proba: np.ndarray, classes, positive_label: str) -> plt.Figure:
    idx = positive_index(classes, positive_label)
    y_bin = (np.asarray(y_true).astype(str) == str(positive_label)).astype(int)
    fpr, tpr, _ = roc_curve(y_bin, y_proba[:, idx])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc_score(y_bin, y_proba[:, idx]):.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="Random (AUC = 0.5)")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC curve — positive class `{positive_label}`")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
metadata = load_metadata()
schema = metadata["schema"]
models = load_models(metadata["artifacts"])

target = schema["target"]
feature_cols = schema["numeric_features"] + schema["categorical_features"]
classes = np.array(schema["classes"])
task = schema["task"]
positive_label = schema["positive_label"]

st.title("Classifier Comparison Dashboard")
st.caption(
    f"Dataset: **{metadata['dataset_file']}** · {len(feature_cols)} features · "
    f"{task} classification over {len(classes)} classes · "
    f"trained on {schema['n_train']:,} instances · {len(models)} models"
)

with st.sidebar:
    st.header("1 · Test data")
    uploaded = st.file_uploader(
        "Upload a test CSV",
        type="csv",
        help=f"Must contain the {len(feature_cols)} feature columns; include `{target}` for metrics.",
    )
    use_bundled = st.checkbox(
        "Use bundled test_data.csv", value=True, disabled=uploaded is not None
    )

    st.header("2 · Model")
    model_name = st.selectbox("Classifier", list(models.keys()))

    st.header("3 · Display")
    normalise_cm = st.checkbox("Normalise confusion matrix", value=False)

    trained_sk = schema["versions"]["scikit-learn"]
    if trained_sk != sklearn.__version__:
        st.warning(
            f"Artifacts were trained on scikit-learn {trained_sk}; this runtime has "
            f"{sklearn.__version__}. Pin the exact version in requirements.txt.",
            icon="⚠️",
        )

    with st.expander("Expected schema"):
        st.write(f"**Target:** `{target}`")
        st.write(f"**Numeric ({len(schema['numeric_features'])}):**")
        st.code(", ".join(schema["numeric_features"]) or "—")
        st.write(f"**Categorical ({len(schema['categorical_features'])}):**")
        st.code(", ".join(schema["categorical_features"]) or "—")

if uploaded is not None:
    df = read_csv(uploaded)
    source_label = uploaded.name
elif use_bundled and BUNDLED_TEST.exists():
    df = read_csv(BUNDLED_TEST)
    source_label = "test_data.csv (bundled)"
else:
    st.info("Upload a test CSV in the sidebar, or tick **Use bundled test_data.csv**.")
    st.stop()

missing = [c for c in feature_cols if c not in df.columns]
if missing:
    st.error(f"Uploaded CSV is missing {len(missing)} required feature column(s): {missing}")
    st.stop()

has_labels = target in df.columns
X = df[feature_cols]
y_true = df[target].astype(str).to_numpy() if has_labels else None

st.success(f"Loaded **{source_label}** — {len(df):,} rows × {df.shape[1]} columns.")
if not has_labels:
    st.warning(
        f"No `{target}` column found, so metrics cannot be computed. "
        "Predictions are still available in the *Predictions* tab.",
        icon="ℹ️",
    )

tab_metrics, tab_matrix, tab_compare, tab_preds, tab_data = st.tabs(
    ["Metrics", "Confusion matrix & report", "Compare all models", "Predictions", "Data preview"]
)

model = models[model_name]
y_pred, y_proba = score(model, X)

with tab_metrics:
    st.subheader(f"{model_name} — evaluation metrics")
    if has_labels:
        m = compute_metrics(y_true, y_pred, y_proba, model.classes_, task, positive_label)
        for col, key in zip(st.columns(len(METRIC_ORDER)), METRIC_ORDER):
            value = m[key]
            col.metric(key, "n/a" if np.isnan(value) else f"{value:.4f}", help=METRIC_HELP[key])
        if task == "binary":
            st.caption(
                f"Precision / Recall / F1 are computed for the positive class "
                f"`{positive_label}`; Accuracy and MCC span the whole confusion matrix."
            )
            if y_proba is not None:
                st.pyplot(plot_roc(y_true, y_proba, model.classes_, positive_label))
        else:
            st.caption(
                "Precision / Recall / F1 use macro averaging; AUC is one-vs-rest macro."
            )
    else:
        st.info("Metrics require ground-truth labels.")

with tab_matrix:
    if has_labels:
        labels = [str(c) for c in model.classes_]
        cm = confusion_matrix(y_true, np.asarray(y_pred).astype(str), labels=labels)
        left, right = st.columns(2)
        with left:
            st.pyplot(plot_confusion(cm, labels, normalise_cm))
        with right:
            st.subheader("Classification report")
            report = classification_report(
                y_true,
                np.asarray(y_pred).astype(str),
                labels=labels,
                output_dict=True,
                zero_division=0,
            )
            st.dataframe(pd.DataFrame(report).T.round(4), width="stretch")
    else:
        st.info("Confusion matrix requires ground-truth labels.")

with tab_compare:
    st.subheader("All models on this test set")
    if has_labels:
        rows = []
        for name, mdl in models.items():
            p, pr = score(mdl, X)
            row = {"Model": name}
            row.update(compute_metrics(y_true, p, pr, mdl.classes_, task, positive_label))
            rows.append(row)
        comparison = pd.DataFrame(rows).set_index("Model")[METRIC_ORDER]
        try:
            # pandas' .style accessor is an optional feature gated on jinja2 being
            # installed. It is pinned in requirements.txt, but a missing transitive
            # dependency should degrade to an unstyled table rather than take down
            # the whole tab on Streamlit Cloud.
            styled = comparison.style.format("{:.4f}").highlight_max(axis=0, color="#c8e6c9")
        except (ImportError, AttributeError):
            styled = comparison.round(4)
        st.dataframe(styled, width="stretch")
        metric_choice = st.selectbox("Chart metric", METRIC_ORDER, index=METRIC_ORDER.index("MCC"))
        st.bar_chart(comparison[metric_choice])
        st.download_button(
            "Download comparison as CSV",
            comparison.to_csv().encode(),
            file_name="model_comparison.csv",
            mime="text/csv",
        )
    else:
        st.info("Comparison requires ground-truth labels.")

with tab_preds:
    st.subheader(f"Predictions — {model_name}")
    out = df.copy()
    out["prediction"] = y_pred
    if y_proba is not None:
        for i, cls in enumerate(model.classes_):
            out[f"proba_{cls}"] = y_proba[:, i].round(4)
    if has_labels:
        out["correct"] = out["prediction"].astype(str) == out[target].astype(str)
        st.caption(f"{int(out['correct'].sum()):,} of {len(out):,} correct.")
    st.dataframe(out.head(200), width="stretch")
    st.download_button(
        "Download predictions as CSV",
        out.to_csv(index=False).encode(),
        file_name=f"predictions_{model_name.replace(' ', '_').lower()}.csv",
        mime="text/csv",
    )

with tab_data:
    st.subheader("Uploaded data")
    st.dataframe(df.head(100), width="stretch")
    st.write("Numeric summary")
    st.dataframe(df.describe().T.round(3), width="stretch")
    if has_labels:
        st.write("Class distribution")
        st.bar_chart(df[target].astype(str).value_counts())
