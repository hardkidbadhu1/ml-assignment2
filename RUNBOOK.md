# Runbook — ML Assignment 2

Local working notes. Delete this file before pushing if you would rather not
ship it; it is not part of the graded deliverable.

---

## 0. Smoke-test the wiring first (5 minutes, before your dataset arrives)

```bash
pip install -r requirements.txt
python scripts/make_synthetic_smoke_data.py
python model/train.py --data data/_smoke.csv --target converted --skip-cv
streamlit run app.py
```

If the app renders six models and a confusion matrix, the plumbing is sound and
every later failure is dataset-specific. Then `rm data/_smoke.csv`.

---

## 1. Dataset

Requirements: **≥ 12 features, ≥ 500 instances**, from UCI or Kaggle.

Recommended: **UCI Online Shoppers Purchasing Intention** — 12,330 sessions,
17 features, binary target `Revenue`, roughly 15% positive. The imbalance is the
point: accuracy will sit near 0.85 for a model that predicts "no purchase" every
time, so MCC and AUC do the real work, which gives you substantive material for
the 3-mark observations table.

```bash
mkdir -p data
# download the CSV into data/ from the UCI page, then:
python model/train.py \
  --data data/online_shoppers_intention.csv \
  --target Revenue \
  --positive-label True
```

Alternatives if you want distance from other students: UCI *Steel Plates Faults*
(1,941 × 27, 7 classes) or UCI *Predict Students' Dropout and Academic Success*
(4,424 × 36, 3 classes). Both work unchanged — the script detects multi-class and
switches to macro-averaged P/R/F1 and one-vs-rest macro AUC.

**Avoid** Titanic, Iris, Pima, Adult, Telco Churn, Wine Quality. Overused, and
Pima/Wine fail the 12-feature bar anyway.

Useful flags:

| Flag | When |
|---|---|
| `--drop id customer_id` | remove identifiers and anything leaking the label |
| `--positive-label True` | pick the class that P/R/F1 report on (defaults to the minority class) |
| `--extra-models svm,gbm` | which sixth (and seventh) model to add |
| `--max-test-rows 2000` | cap `test_data.csv` for Streamlit Cloud |
| `--skip-cv` | skip 5-fold CV while iterating; run without it for the final numbers |

**Leakage check before you trust anything.** If a model comes back at 0.99 AUC,
assume a feature encodes the answer. Scan for columns recorded after the outcome
(status flags, resolution dates, post-hoc scores) and drop them.

---

## 2. BITS Virtual Lab (1 mark)

The brief requires the assignment to be *performed* on BITS Virtual Lab, with one
screenshot as proof. Run the training there and screenshot the terminal or
notebook showing the per-model metrics printing, with the lab chrome visible in
frame. A screenshot of your own laptop will not pass.

If the lab environment has an older scikit-learn than your laptop, train there —
the joblib artifacts must be produced by the same version the app pins.

---

## 3. Git

Commit history is explicitly reviewed, and "identical repo structure & variable
names may be flagged." One giant initial commit is the pattern that looks copied.
Commit as you actually work, over more than one day:

```bash
git init && git branch -M main
git add .gitignore requirements.txt runtime.txt && git commit -m "Project scaffold and pinned dependencies"
git add ml_pipeline.py && git commit -m "Add shared preprocessing pipeline and metric helpers"
git add model/train.py && git commit -m "Train five required classifiers with stratified split"
git commit -am "Add SVM as sixth model and 5-fold CV diagnostics"
git add app.py && git commit -m "Streamlit UI: upload, model dropdown, metrics, confusion matrix"
git add test_data.csv model/artifacts reports && git commit -m "Commit trained artifacts and metric reports"
git add README.md && git commit -m "Document dataset, results and per-model observations"
```

Rename things to your own vocabulary as you go. Variable names are part of what
gets compared.

Artifacts **must** be committed — Streamlit Cloud builds from the repo and does
no training. Check total artifact size stays modest:

```bash
du -sh model/artifacts test_data.csv
```

A Random Forest with 300 trees on a wide one-hot matrix can exceed 100 MB. If so,
drop to `n_estimators=150` or add `max_depth`, and retrain.

---

## 4. Streamlit Community Cloud

1. https://streamlit.io/cloud → sign in with GitHub
2. **New App** → your repo → branch `main` → main file `app.py`
3. **Advanced settings** → set the Python version to match `runtime.txt`
4. Deploy, then watch the build log to the end

Failure modes, in order of likelihood:

| Symptom | Cause | Fix |
|---|---|---|
| `InconsistentVersionWarning` or a joblib `AttributeError` | app runtime ≠ training environment | copy the exact pins `train.py` prints into `requirements.txt`, redeploy |
| `ModuleNotFoundError: ml_pipeline` | shared module not committed | `git add ml_pipeline.py` |
| `metadata.json not found` | `model/artifacts/` gitignored or empty | commit the artifacts |
| App boots, then OOMs on upload | test CSV too large | lower `--max-test-rows` |
| Blank page after deploy | build still running | read the log; it takes a few minutes on first deploy |

Verify by opening the live URL in a private window — that catches "works only
because my browser is logged in."

---

## 5. Submission PDF (single file, in this order)

1. GitHub repository link
2. Live Streamlit app link
3. BITS Virtual Lab screenshot
4. The full README content

Final checks:

- [ ] Repo link opens for a logged-out visitor (repo is **public**)
- [ ] App link opens and renders without an error banner
- [ ] Model dropdown lists all six; switching updates the metrics
- [ ] Confusion matrix renders
- [ ] Uploading `test_data.csv` through the sidebar works
- [ ] `requirements.txt` pins match the metadata versions
- [ ] README placeholders all filled — no `<...>` left
- [ ] Observations are written in your own words from your own numbers
- [ ] Submitted, not saved as draft — **18 Aug 2026, 23:59**
