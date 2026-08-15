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
screenshot as proof. The lab is a provisioned **AWS VM with shell access and full
internet**, so the repo runs there unchanged. A screenshot of your own laptop will
not pass.

The lab image is **Rocky Linux 9.5**, which matters for one reason (below).

`scripts/lab_run.sh` does the whole thing in one command and prints the evidence
and the result in a single frame:

```bash
ssh <user>@<lab-host>
sudo dnf install -y git python3.12 python3.12-pip   # see "Python version" below
git clone https://github.com/<user>/<repo>.git && cd <repo>
bash scripts/lab_run.sh
```

### Python version — the one thing that will actually break

Rocky 9 ships **Python 3.9** as the system `python3`. Both pinned libraries need
newer:

| Package | Pinned | `requires_python` |
|---|---|---|
| scikit-learn | 1.7.2 | `>= 3.10` |
| numpy | 2.2.6 | `>= 3.10` |

On 3.9, `pip install -r requirements.txt` does not fail loudly — it resolves
*backwards* to whatever old scikit-learn still supports 3.9. You then get
artifacts that do not match the pins, and Streamlit Cloud unpickles a mismatched
pipeline. Install 3.12 from AppStream instead — it also matches `runtime.txt`, so
lab and Streamlit Cloud agree:

```bash
sudo dnf install -y python3.12 python3.12-pip
```

The script prefers `python3.12 > python3.11 > python3.10`, refuses to run on
anything older, and installs into a `.venv` so nothing touches the dnf-managed
system site-packages. If `python -m venv` is unavailable it falls back to
`pip install --user` rather than stopping.

It prints hostname, kernel, distro, CPU/RAM, the **EC2 instance id** and the
library versions; then installs dependencies, trains all six models, runs the 18
verification checks, and re-prints the metric table under a
`RESULTS — trained on <hostname>` banner. Screenshot that window. The instance id
is stronger evidence than lab chrome in the frame — it is unforgeable from a laptop.

**Getting the dataset there.** `data/*.csv` is gitignored, so a fresh clone will
not have the 17 MB file and the script will stop with instructions. Easiest is to
push it from your laptop:

```bash
scp data/fifa_world_cup_2026_player_performance.csv <user>@<lab-host>:~/<repo>/data/
```

Or configure the Kaggle CLI on the VM (`pip install kaggle`, drop your API token
at `~/.kaggle/kaggle.json`, `export KAGGLE_DATASET=<owner>/<slug>`) and re-run.

**If the VM is small.** SVC is superlinear in *n* and is the only thing that
meaningfully moves runtime — the other five models finish in under two seconds
combined. On a `t3.micro`-class box:

```bash
LAB_MAX_ROWS=6000 bash scripts/lab_run.sh
```

That still clears the 500-instance requirement by 12×. Metrics shift by a few
thousandths; the model ranking holds.

**Version pinning — decide this deliberately.** The joblib artifacts must be
produced by the same scikit-learn that `requirements.txt` pins, or Streamlit Cloud
unpickles a mismatched pipeline and can mis-predict *silently* rather than fail.
So pick one environment and let it own the artifacts:

- If the lab's scikit-learn differs from your laptop's, **train in the lab**, then
  copy the version block `train.py` prints into `requirements.txt` and commit the
  artifacts produced there.
- Do not train locally, screenshot the lab, and ship the local artifacts — the
  screenshot would then show numbers that no committed artifact reproduces.

`scripts/verify.py` re-loads every artifact and re-scores it, so run it wherever
you trained; it will catch the mismatch before Streamlit Cloud does.

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
3. **Advanced settings** → set **Python 3.12** from the dropdown. Do this every
   time; see the warning below.
4. **Custom subdomain** → set one (e.g. `badhu-ml-assignment2`). The auto-generated
   URL contains a random hash that changes if you ever redeploy, which would
   invalidate the link already printed in the README and the submission PDF.
5. Deploy, then watch the build log to the end

### ⚠️ `runtime.txt` is currently ignored by Community Cloud

Community Cloud now installs with `uv`, and there is an open bug where the
platform forces a recent Python (3.13/3.14) regardless of `runtime.txt`. The log
line to look for is:

```
Using uv pip install.
Using Python 3.14.7 environment at /home/adminuser/venv
Resolved 53 packages in 531ms          <-- then it appears to hang
```

It is not hanging. numpy 2.2.6, pandas 2.3.3 and scikit-learn 1.7.2 have no
prebuilt wheels for 3.14, so uv is compiling them from source — twenty-plus
minutes, usually ending in a compiler error.

**The Python version cannot be changed after an app is created.** You must delete
the app and redeploy, selecting 3.12 in *Advanced settings*. The dropdown is
honoured; only `runtime.txt` is ignored. A `.python-version` file (read by `uv`)
is committed as a second line of defence, but do not rely on it alone.

If 3.12 is ever unavailable, the fallback is to retrain so the artifacts match a
Python the platform will give you — never to loosen the pins while keeping old
artifacts, which reintroduces the version-mismatch failure.

Refs: [streamlit#15326](https://github.com/streamlit/streamlit/issues/15326),
[upgrade-python docs](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app/upgrade-python)

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

`scripts/make_pdf.py` builds all four in one render — no manual merging, one set
of page numbers, and the order cannot come out wrong:

Screenshots live in two folders, one per section of the report:

```bash
screenshots/lab/    one frame from the end of scripts/lab_run.sh
screenshots/app/    one capture per tab of the deployed Streamlit app
```

Then:

```bash
cp ~/Desktop/lab-run.png  screenshots/lab/01-lab-run.png
cp ~/Desktop/metrics.png  screenshots/app/01-metrics.png
pip install -r requirements-dev.txt      # weasyprint and markdown, build-time only
python scripts/make_pdf.py
```

It prints exactly what went into each section, so you can confirm before
submitting:

```
  1. GitHub repository    https://github.com/hardkidbadhu1/ml-assignment2
  2. Live Streamlit app   https://badhu-ml-assignment2.streamlit.app/
  3. Lab evidence         1 image(s): 01-lab-run.png
  4. App screenshots      5 image(s): 01-metrics.png, ...
  5. Report body from README.md
```

Notes:

- The two links are **parsed out of README section c** rather than restated in
  the script, so the cover page cannot disagree with the body. This matters:
  the app URL already changed once.
- Images are embedded in **filename order**, so prefix them `01-`, `02-` to
  control the sequence. Captions are derived from the filename, with the numeric
  prefix stripped, so `02-confusion-matrix.png` prints as
  *Figure 2. Confusion matrix*.
- It **refuses to build** if either folder is empty, because a PDF missing the
  evidence looks complete while quietly dropping marks. Use
  `--allow-missing-screenshots` to override, or `--readme-only` for just the
  body with no cover or figures.

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
