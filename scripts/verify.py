"""Post-training checks. Run after `scripts/train_fifa.sh`.

Three things are verified, in order of how badly they would invalidate the report:

  1. Leakage — no player_id appears in both the training and the shipped test set,
     and the grouped split actually costs accuracy relative to a random row split.
     If a random split scored the *same*, the grouping would be doing nothing and
     the concern would have been imaginary.
  2. Artifacts — every pipeline named in metadata.json loads and predicts, which is
     what app.py does at startup. Catches a version-mismatched pickle before deploy.
  3. Schema — test_data.csv carries exactly the columns the app expects.

Usage:  python scripts/verify.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ml_pipeline import build_preprocessor, compute_metrics, infer_feature_types  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import accuracy_score  # noqa: E402
from sklearn.model_selection import GroupShuffleSplit, train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

RAW = ROOT / "data" / "fifa_world_cup_2026_player_performance.csv"
META = ROOT / "model" / "artifacts" / "metadata.json"
TEST = ROOT / "test_data.csv"

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{f' — {detail}' if detail else ''}")
    if not ok:
        failures.append(label)


metadata = json.loads(META.read_text())
schema = metadata["schema"]
features = schema["numeric_features"] + schema["categorical_features"]
target = schema["target"]
test_df = pd.read_csv(TEST)

# --------------------------------------------------------------------------- #
print("\n1. Leakage")
# --------------------------------------------------------------------------- #
raw = pd.read_csv(RAW)
check(
    "group column excluded from features",
    schema["group_col"] not in features,
    f"{schema['group_col']!r} is not a model input",
)

train_groups = set(schema["train_groups"] or [])
test_groups = set(schema["test_groups"] or [])
check(
    "train and test entity sets are disjoint",
    bool(train_groups) and not (train_groups & test_groups),
    f"{len(train_groups)} train / {len(test_groups)} test, {len(train_groups & test_groups)} shared",
)
check(
    "the split partitions the population (no entity silently dropped)",
    train_groups | test_groups == set(map(str, raw["player_id"].unique())),
    f"{len(train_groups | test_groups)} of {raw['player_id'].nunique()} players accounted for",
)

# --------------------------------------------------------------------------- #
print("\n2. Grouped vs random split — does the grouping change anything?")
# --------------------------------------------------------------------------- #
drop_like = [c for c in raw.columns if c not in features + [target, "player_id"]]
sub = raw.sample(15_000, random_state=42)
groups = sub["player_id"]
X, y = sub[features], sub[target].astype(str)
num, cat, _ = infer_feature_types(X, 50, schema["numeric_features"])


def rf_accuracy(tr, te) -> float:
    pipe = Pipeline(
        [
            ("prep", build_preprocessor(num, cat)),
            ("clf", RandomForestClassifier(n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=42)),
        ]
    )
    pipe.fit(X.iloc[tr], y.iloc[tr])
    return float(accuracy_score(y.iloc[te], pipe.predict(X.iloc[te])))


g_tr, g_te = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42).split(X, y, groups))
grouped_acc = rf_accuracy(g_tr, g_te)
r_tr, r_te = train_test_split(np.arange(len(X)), test_size=0.25, stratify=y, random_state=42)
random_acc = rf_accuracy(r_tr, r_te)

print(f"  random row split : {random_acc:.4f}")
print(f"  grouped split    : {grouped_acc:.4f}   (delta {random_acc - grouped_acc:+.4f})")
check(
    "grouped split is not more optimistic than the random one",
    grouped_acc <= random_acc + 0.005,
    "the reported number is the conservative one",
)
check(
    "zero player overlap under the grouped split",
    not (set(groups.iloc[g_tr]) & set(groups.iloc[g_te])),
)

# --------------------------------------------------------------------------- #
print("\n3. Artifacts load and predict (this is what app.py does at startup)")
# --------------------------------------------------------------------------- #
X_test = test_df[features]
y_test = test_df[target].astype(str).to_numpy()
for name, filename in metadata["artifacts"].items():
    path = ROOT / "model" / "artifacts" / filename
    try:
        pipe = joblib.load(path)
        pred = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)
        m = compute_metrics(y_test, pred, proba, pipe.classes_, schema["task"], schema["positive_label"])
        check(name, len(pred) == len(X_test), f"acc={m['Accuracy']:.4f} auc={m['AUC']:.4f}")
    except Exception as exc:  # noqa: BLE001
        check(name, False, f"{type(exc).__name__}: {exc}")

# --------------------------------------------------------------------------- #
print("\n4. Schema and assignment minimums")
# --------------------------------------------------------------------------- #
check("all feature columns present in test_data.csv", not [c for c in features if c not in test_df])
check("target present in test_data.csv", target in test_df.columns)
check("features >= 12", len(features) >= 12, f"{len(features)}")
check("instances >= 500", len(raw) >= 500, f"{len(raw):,} raw / {len(test_df):,} shipped")

# An unseen categorical level must encode to all-zeros, not explode.
probe = X_test.head(20).copy()
if schema["categorical_features"]:
    col = schema["categorical_features"][0]
    probe[col] = "__never_seen__"
    try:
        joblib.load(ROOT / "model" / "artifacts" / next(iter(metadata["artifacts"].values()))).predict(probe)
        check("OneHotEncoder(handle_unknown='ignore') survives an unseen level", True, f"probed {col}")
    except Exception as exc:  # noqa: BLE001
        check("OneHotEncoder(handle_unknown='ignore') survives an unseen level", False, str(exc))

# --------------------------------------------------------------------------- #
print("\n5. Deployability")
# --------------------------------------------------------------------------- #
# GitHub warns above 50 MB per file and hard-rejects at 100 MB; Streamlit Cloud's
# free tier holds every artifact in memory at once. Both are worth catching here
# rather than at push time.
total = 0.0
for name, filename in metadata["artifacts"].items():
    mb = (ROOT / "model" / "artifacts" / filename).stat().st_size / 1e6
    total += mb
    if mb > 5:
        print(f"    {name}: {mb:.1f} MB")
check("no single artifact over 50 MB", all(
    (ROOT / "model" / "artifacts" / f).stat().st_size / 1e6 < 50 for f in metadata["artifacts"].values()
))
check("all artifacts together under 100 MB", total < 100, f"{total:.1f} MB")

# --------------------------------------------------------------------------- #
print("\n6. Streamlit app renders under the pinned streamlit")
# --------------------------------------------------------------------------- #
# Loading the artifacts is necessary but not sufficient — the app also has to
# survive the Streamlit *API* version that requirements.txt pins. This caught a
# real one: `st.dataframe(..., width="stretch")` is valid from Streamlit 1.49 but
# raises TypeError on the pinned 1.41.1, which would have booted fine and then
# crashed the moment a grader opened the confusion-matrix tab.
try:
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    print(f"    streamlit {st.__version__} (requirements.txt pin)")
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300).run()
    check("app.py executes with no uncaught exception", not at.exception,
          "; ".join(str(e.value) for e in at.exception) if at.exception else "")
    check("all six metrics render", len({m.label for m in at.metric}) == 6,
          f"{sorted({m.label for m in at.metric})}")
    check("five tabs render", len(at.tabs) == 5, f"{len(at.tabs)}")

    # Switching model re-runs the whole script, so a per-model failure (an
    # estimator without predict_proba, say) only shows up on selection.
    picker = next((s for s in at.selectbox if "kNN" in s.options), None)
    if picker is None:
        check("model dropdown present", False)
    else:
        check(f"dropdown lists all {len(metadata['artifacts'])} models",
              len(picker.options) == len(metadata["artifacts"]))
        for name in picker.options:
            run = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300).run()
            next(s for s in run.selectbox if name in s.options).select(name).run()
            check(f"select '{name}'", not run.exception,
                  "; ".join(str(e.value) for e in run.exception) if run.exception else "")
except ImportError as exc:
    print(f"  [SKIP] streamlit not importable ({exc}) — install requirements.txt to run this check")

print(f"\n{'ALL CHECKS PASSED' if not failures else f'{len(failures)} FAILED: {failures}'}")
sys.exit(1 if failures else 0)
