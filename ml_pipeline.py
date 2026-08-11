"""
Shared feature-engineering and metric code.

Both `model/train.py` and `app.py` import from here. That is deliberate: joblib
pickles transformers *by reference*, so the classes/functions embedded in a
saved Pipeline must be importable at load time from the same module path. Keeping
them in one place also guarantees training and serving cannot drift apart.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

RANDOM_STATE = 42
METRIC_ORDER = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


# --------------------------------------------------------------------------- #
# Feature typing
# --------------------------------------------------------------------------- #
def as_string_frame(X: Any) -> pd.DataFrame:
    """Coerce every categorical column to str while preserving NaN.

    Needed because a categorical block can legitimately mix dtypes (a bool
    `Weekend`, an int `OperatingSystems`, a str `Month`). sklearn's
    SimpleImputer(most_frequent) receives that as a single object array and
    tries to cast it to float, which explodes on the first string. Must be a
    module-level function, not a lambda, so joblib can pickle it.
    """
    df = pd.DataFrame(X).copy()
    for col in df.columns:
        s = df[col]
        df[col] = s.astype("object").where(s.isna(), s.astype(str))
    return df


def infer_feature_types(
    X: pd.DataFrame,
    max_cardinality: int = 50,
    force_numeric: tuple[str, ...] | list[str] = (),
) -> tuple[list[str], list[str], list[str]]:
    """Split columns into (numeric, categorical, dropped).

    Low-cardinality integer columns default to categorical: an integer-coded
    `Month` or `Region` is a label, not a quantity, and standardising it would
    invent an ordering that does not exist.

    That default is wrong for *count* columns, where the ordering is real —
    3 tackles is genuinely more than 1. One-hot encoding a count throws away
    that ordering, and inflates dimensionality in a way that specifically hurts
    kNN (Euclidean distance degrades) and GaussianNB (a Gaussian fitted to a
    0/1 dummy is a poor density model). Pass such columns in `force_numeric`.
    """
    numeric: list[str] = []
    categorical: list[str] = []
    dropped: list[str] = []
    forced = set(force_numeric)

    for col in X.columns:
        s = X[col]
        if col in forced and pd.api.types.is_numeric_dtype(s):
            numeric.append(col)
        elif pd.api.types.is_bool_dtype(s):
            categorical.append(col)
        elif pd.api.types.is_numeric_dtype(s):
            if pd.api.types.is_integer_dtype(s) and s.nunique(dropna=True) <= 10:
                categorical.append(col)
            else:
                numeric.append(col)
        else:
            if s.nunique(dropna=True) > max_cardinality:
                dropped.append(col)  # near-unique string column, almost certainly an ID
            else:
                categorical.append(col)
    return numeric, categorical, dropped


def build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    """Median-impute + standardise numerics; stringify + mode-impute + one-hot categoricals.

    Two choices that matter at serving time:
      * `handle_unknown="ignore"` — a category present in the uploaded test CSV
        but unseen during training encodes to all-zeros rather than raising.
      * `sparse_output=False` — GaussianNB cannot consume a sparse matrix.
    """
    numeric_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("stringify", FunctionTransformer(as_string_frame, feature_names_out="one-to-one")),
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipe, numeric),
            ("cat", categorical_pipe, categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def positive_index(classes: np.ndarray, positive_label: Any) -> int:
    matches = np.where(np.asarray(classes).astype(str) == str(positive_label))[0]
    if len(matches) == 0:
        raise ValueError(f"positive label {positive_label!r} not in classes {list(classes)}")
    return int(matches[0])


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
    classes: np.ndarray,
    task: str,
    positive_label: str | None,
) -> dict[str, float]:
    """The six metrics the assignment asks for.

    Averaging switches on task type. Binary reports precision/recall/F1 for the
    positive class only — macro-averaging a 90/10 split would report a flattering
    number driven by the majority class. Multiclass uses macro so every class
    counts equally regardless of support.
    """
    y_true = np.asarray(y_true).astype(str)
    y_pred = np.asarray(y_pred).astype(str)
    average = "binary" if task == "binary" else "macro"
    kwargs: dict[str, Any] = {"zero_division": 0}
    if task == "binary":
        kwargs["pos_label"] = str(positive_label)

    metrics: dict[str, float] = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "AUC": float("nan"),
        "Precision": float(precision_score(y_true, y_pred, average=average, **kwargs)),
        "Recall": float(recall_score(y_true, y_pred, average=average, **kwargs)),
        "F1": float(f1_score(y_true, y_pred, average=average, **kwargs)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
    }

    if y_proba is not None:
        classes_str = np.asarray(classes).astype(str)
        try:
            if task == "binary":
                idx = positive_index(classes_str, positive_label)
                y_bin = (y_true == str(positive_label)).astype(int)
                metrics["AUC"] = float(roc_auc_score(y_bin, y_proba[:, idx]))
            else:
                metrics["AUC"] = float(
                    roc_auc_score(
                        y_true,
                        y_proba,
                        multi_class="ovr",
                        average="macro",
                        labels=list(classes_str),
                    )
                )
        except ValueError as exc:
            # Typically: the evaluation slice is missing one of the classes.
            warnings.warn(f"AUC undefined for this split: {exc}", stacklevel=2)
    return metrics


def score(model, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray | None]:
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X) if hasattr(model, "predict_proba") else None
    return y_pred, y_proba
