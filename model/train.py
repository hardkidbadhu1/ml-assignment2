"""
Train and evaluate classification models for ML Assignment 2.

Dataset-agnostic: point it at any CSV, name the target column, and it infers
numeric/categorical features, builds a preprocessing + estimator Pipeline per
algorithm, evaluates on a held-out stratified test split, and persists
everything the Streamlit app needs.

Usage
-----
    python model/train.py --data data/online_shoppers_intention.csv --target Revenue
    python model/train.py --data data/x.csv --target y --positive-label 1 --drop id

Outputs
-------
    model/artifacts/<slug>.joblib   one fitted Pipeline per model
    model/artifacts/metadata.json   schema, class labels, library versions
    reports/metrics.csv             metric table (machine readable)
    reports/metrics.md              metric table (paste straight into README.md)
    test_data.csv                   held-out split, raw feature values, for the app
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so `ml_pipeline` resolves regardless of cwd

from ml_pipeline import (  # noqa: E402
    METRIC_ORDER,
    RANDOM_STATE,
    build_preprocessor,
    compute_metrics,
    infer_feature_types,
    score,
)
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score  # noqa: E402
from sklearn.model_selection import (  # noqa: E402
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.svm import SVC  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

ARTIFACT_DIR = ROOT / "model" / "artifacts"
REPORT_DIR = ROOT / "reports"

# Display name -> estimator factory. These names are what land in the README table.
REQUIRED_MODELS: dict[str, Any] = {
    "Logistic Regression": lambda: LogisticRegression(
        max_iter=2000, random_state=RANDOM_STATE
    ),
    "Decision Tree": lambda: DecisionTreeClassifier(
        random_state=RANDOM_STATE, min_samples_leaf=5
    ),
    "kNN": lambda: KNeighborsClassifier(n_neighbors=15, weights="distance"),
    "Naive Bayes (Gaussian)": lambda: GaussianNB(),
    "Random Forest (Ensemble)": lambda: RandomForestClassifier(
        n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1, min_samples_leaf=2
    ),
}

# The brief lists five models but repeatedly refers to "all the 6 ML models".
# Adding a sixth from the syllabus closes that ambiguity for free.
EXTRA_MODELS: dict[str, tuple[str, Any]] = {
    "svm": (
        "SVM (RBF)",
        lambda: SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
    ),
    "gbm": (
        "Gradient Boosting (Ensemble)",
        lambda: GradientBoostingClassifier(random_state=RANDOM_STATE),
    ),
}


def slugify(name: str) -> str:
    out = "".join(c.lower() if c.isalnum() else "_" for c in name)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


def load_dataset(path: Path, target: str, drop: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    if target not in df.columns:
        raise SystemExit(f"Target column {target!r} not found. Available: {list(df.columns)}")
    if drop:
        df = df.drop(columns=[c for c in drop if c in df.columns])
    df = df.dropna(subset=[target])

    if df.shape[1] - 1 < 12:
        warnings.warn(
            f"Only {df.shape[1] - 1} features — the assignment requires >= 12.", stacklevel=2
        )
    if len(df) < 500:
        warnings.warn(f"Only {len(df)} instances — the assignment requires >= 500.", stacklevel=2)
    return df


def cross_val_summary(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, task: str) -> dict:
    """5-fold stratified CV on the training split only.

    The single test-split number is one sample of a random variable; the CV
    standard deviation tells you whether a 0.01 gap between two models is signal
    or noise. This is the evidence behind the observations table.
    """
    scoring = ["accuracy", "f1" if task == "binary" else "f1_macro"]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    return {
        "CV Accuracy (mean)": float(scores["test_accuracy"].mean()),
        "CV Accuracy (std)": float(scores["test_accuracy"].std()),
        "CV F1 (mean)": float(scores[f"test_{scoring[1]}"].mean()),
        "Fit time (s)": float(scores["fit_time"].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="path to the raw CSV")
    parser.add_argument("--target", required=True, help="name of the label column")
    parser.add_argument(
        "--positive-label", default=None, help="binary only; defaults to the minority class"
    )
    parser.add_argument("--drop", nargs="*", default=[], help="columns to drop (IDs etc.)")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument(
        "--extra-models", default="svm", help="extras beyond the required 5: svm,gbm or 'none'"
    )
    parser.add_argument(
        "--max-test-rows",
        type=int,
        default=2000,
        help="cap rows in test_data.csv (Streamlit Cloud has a small memory quota)",
    )
    parser.add_argument("--max-cardinality", type=int, default=50)
    parser.add_argument("--skip-cv", action="store_true")
    args = parser.parse_args()

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.data, args.target, args.drop)
    y = df[args.target].astype(str)
    X = df.drop(columns=[args.target])

    numeric, categorical, dropped = infer_feature_types(X, args.max_cardinality)
    if dropped:
        print(f"Dropped high-cardinality columns (likely IDs): {dropped}")
    X = X[numeric + categorical]

    classes = np.array(sorted(y.unique()))
    task = "binary" if len(classes) == 2 else "multiclass"
    positive_label = None
    if task == "binary":
        positive_label = (
            str(args.positive_label)
            if args.positive_label is not None
            else str(y.value_counts().idxmin())  # minority class = the event of interest
        )

    print(f"\nRows={len(df)}  Features={X.shape[1]}  Task={task}  Classes={list(classes)}")
    print(f"  numeric ({len(numeric)}): {numeric}")
    print(f"  categorical ({len(categorical)}): {categorical}")
    if positive_label:
        print(f"  positive label: {positive_label!r}")
    print(f"  class balance:\n{y.value_counts(normalize=True).round(4).to_string()}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=RANDOM_STATE
    )

    registry = dict(REQUIRED_MODELS)
    if args.extra_models and args.extra_models.lower() != "none":
        for key in (k.strip() for k in args.extra_models.split(",") if k.strip()):
            if key not in EXTRA_MODELS:
                raise SystemExit(f"Unknown extra model {key!r}. Options: {list(EXTRA_MODELS)}")
            name, factory = EXTRA_MODELS[key]
            registry[name] = factory

    rows: list[dict] = []
    artifacts: dict[str, str] = {}

    for name, factory in registry.items():
        pipe = Pipeline([("prep", build_preprocessor(numeric, categorical)), ("clf", factory())])
        pipe.fit(X_train, y_train)

        y_pred, y_proba = score(pipe, X_test)
        row: dict[str, Any] = {"Model": name}
        row.update(
            compute_metrics(
                y_test.to_numpy(), y_pred, y_proba, pipe.classes_, task, positive_label
            )
        )
        if not args.skip_cv:
            row.update(cross_val_summary(pipe, X_train, y_train, task))
        # Train-vs-test accuracy gap is the overfitting evidence for the write-up.
        row["Train Accuracy"] = float(accuracy_score(y_train, pipe.predict(X_train)))
        rows.append(row)

        slug = slugify(name)
        joblib.dump(pipe, ARTIFACT_DIR / f"{slug}.joblib")
        artifacts[name] = f"{slug}.joblib"
        print(
            f"{name:<32} acc={row['Accuracy']:.4f}  auc={row['AUC']:.4f}  "
            f"f1={row['F1']:.4f}  mcc={row['MCC']:.4f}"
        )

    results = pd.DataFrame(rows).set_index("Model")
    results.to_csv(REPORT_DIR / "metrics.csv")

    diagnostics = [c for c in results.columns if c not in METRIC_ORDER]
    (REPORT_DIR / "metrics.md").write_text(
        "## Model comparison (held-out test set)\n\n"
        + results[METRIC_ORDER].round(4).to_markdown()
        + "\n\n## Diagnostics (5-fold CV on the training split)\n\n"
        + results[diagnostics].round(4).to_markdown()
        + "\n"
    )

    metadata = {
        "dataset_file": args.data.name,
        "artifacts": artifacts,
        "schema": {
            "target": args.target,
            "numeric_features": numeric,
            "categorical_features": categorical,
            "classes": [str(c) for c in classes],
            "positive_label": positive_label,
            "task": task,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "versions": {
                "python": platform.python_version(),
                "scikit-learn": sklearn.__version__,
                "pandas": pd.__version__,
                "numpy": np.__version__,
                "joblib": joblib.__version__,
            },
        },
    }
    (ARTIFACT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))

    test_df = X_test.copy()
    test_df[args.target] = y_test
    if len(test_df) > args.max_test_rows:
        test_df = test_df.sample(args.max_test_rows, random_state=RANDOM_STATE).sort_index()
        print(f"\ntest_data.csv downsampled to {args.max_test_rows} rows for the app.")
    test_df.to_csv(ROOT / "test_data.csv", index=False)

    print(f"\nWrote {len(artifacts)} models -> {ARTIFACT_DIR}")
    print(f"Wrote metrics -> {REPORT_DIR / 'metrics.md'}")
    print(f"Wrote test_data.csv ({len(test_df)} rows)")

    # Unpickling a Pipeline under a different scikit-learn than it was fitted on
    # is the most common Streamlit Cloud failure. Emit the exact pins.
    print(
        "\nPin these in requirements.txt so the deployed runtime matches this one:\n"
        f"  scikit-learn=={sklearn.__version__}\n"
        f"  pandas=={pd.__version__}\n"
        f"  numpy=={np.__version__}\n"
        f"  joblib=={joblib.__version__}\n"
        f"  (Python {platform.python_version()} — set the same major.minor in the "
        "Streamlit Cloud app's Advanced settings)"
    )


if __name__ == "__main__":
    main()
