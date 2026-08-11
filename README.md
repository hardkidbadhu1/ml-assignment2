# <Dataset name> — Classifier Comparison

> **Fill every `<...>` placeholder before submitting.** The numbers come from
> `reports/metrics.md`, which `model/train.py` writes on every run. This whole
> file also has to be pasted into the submission PDF (Section 2, item 4).

**Course:** M.Tech (AIML/DSE) — Machine Learning · **Assignment 2**
**Name / BITS ID:** `Badhmanaban M` / `2025AC05386`

---

## a. Problem statement

`<2–4 sentences. State the prediction task, why it matters, and the decision the
model would inform. Name the target variable and what a positive prediction
means operationally — e.g. "flag a browsing session that will end in a purchase
so the site can trigger a retention offer." Say whether it is binary or
multi-class.>`

---

## b. Dataset description

| | |
|---|---|
| Source | `<UCI / Kaggle URL>` |
| Instances | `<n>` (requirement: ≥ 500) |
| Features | `<n>` (requirement: ≥ 12) — `<a>` numeric, `<b>` categorical |
| Target | `<column>` — classes `<list>` |
| Class balance | `<e.g. 84.5% False / 15.5% True>` |
| Missing values | `<count, and how handled>` |
| Train / test split | `<75 / 25>`, stratified, `random_state=42` |

**Feature summary**

| # | Feature | Type | Meaning |
|---|---------|------|---------|
| 1 | `<name>` | numeric | `<one line>` |
| … | | | |

**Preprocessing.** Every model is wrapped in a single `sklearn.Pipeline` so that
imputation and encoding are fitted on the training fold only and travel with the
serialised model — the Streamlit app therefore cannot apply a different transform
at inference than was used at training.

- Numeric: median imputation → `StandardScaler`. Scaling matters for kNN
  (distance-based), logistic regression (convergence) and SVM (kernel width); it
  is a no-op for the tree-based models.
- Categorical: mode imputation → `OneHotEncoder(handle_unknown="ignore")`, so a
  category present in the uploaded test CSV but unseen in training encodes to
  all-zeros rather than raising at inference time.
- Low-cardinality integer columns (`<list them>`) are treated as categorical, not
  numeric — an integer-coded month is a label, not a quantity.
- Columns dropped: `<list, with reason — IDs, leakage, near-constant>`.

---

## c. GitHub repository link

- **Repository:** `<https://github.com/<user>/<repo>>`
- **Live Streamlit app:** `<https://<app>.streamlit.app>`

```
project-folder/
├── app.py                      Streamlit UI
├── ml_pipeline.py              shared preprocessing + metric code (train and serve)
├── requirements.txt            pinned dependencies
├── runtime.txt                 Python version for Streamlit Cloud
├── README.md
├── test_data.csv               held-out split used by the app
├── data/                       raw dataset (gitignored)
├── model/
│   ├── train.py                training + evaluation entry point
│   └── artifacts/              *.joblib pipelines + metadata.json
├── reports/                    metrics.csv, metrics.md
└── scripts/                    local smoke-test data generator
```

**Reproduce:**

```bash
pip install -r requirements.txt
python model/train.py --data data/<file>.csv --target <target> --positive-label <label>
streamlit run app.py
```

---

## d. Models used

All six models are trained on the same dataset, the same stratified split, and
the same preprocessing pipeline, so the comparison isolates the effect of the
learning algorithm.

| Model | Key hyperparameters | Why included |
|---|---|---|
| Logistic Regression | `max_iter=2000` | Linear baseline; calibrated probabilities |
| Decision Tree | `min_samples_leaf=5` | Non-linear, interpretable, high variance |
| kNN | `k=15`, distance-weighted | Instance-based; needs the scaler |
| Naive Bayes (Gaussian) | defaults | Generative; strong independence assumption |
| Random Forest | `n_estimators=300`, `min_samples_leaf=2` | Bagging ensemble — variance reduction |
| SVM (RBF) | `probability=True` | Max-margin with kernel; sixth model |

### Comparison table

> Paste the table from `reports/metrics.md`. Binary tasks report Precision /
> Recall / F1 for the positive class `<label>`; multi-class uses macro averaging
> and one-vs-rest macro AUC.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | | | | | | |
| Decision Tree | | | | | | |
| kNN | | | | | | |
| Naive Bayes | | | | | | |
| Random Forest (Ensemble) | | | | | | |
| SVM (RBF) | | | | | | |

### Diagnostics

> From the second table in `reports/metrics.md`. Train-vs-test accuracy gap is
> the overfitting evidence; CV standard deviation tells you whether a small gap
> between two models is signal or noise.

| Model | Train Acc | Test Acc | CV Acc (mean ± sd) | Fit time (s) |
|---|---|---|---|---|
| | | | | |

### Observations

> Three marks live here. Write from *your* numbers, and make each row explain a
> mechanism rather than restate the metric. Prompts, not answers:
>
> - **Logistic Regression** — how close is it to the best model? If it is within
>   noise of the ensembles, the decision boundary is close to linear. Compare its
>   AUC to its F1: a good AUC with poor F1 means the ranking is fine and the 0.5
>   threshold is wrong for this class balance.
> - **Decision Tree** — look at train vs test accuracy. A large gap is variance;
>   tie it to the depth the tree reached and to `min_samples_leaf`.
> - **kNN** — did scaling matter, and what does `k=15` do to the bias/variance
>   balance? One-hot encoding inflates dimensionality, which weakens Euclidean
>   distance; say whether you see that.
> - **Naive Bayes** — the conditional-independence assumption is violated by any
>   correlated features (say which). Note that it is often the fastest to fit and
>   whether the accuracy cost was worth it.
> - **Random Forest** — bagging reduces variance relative to the single tree.
>   Quantify it: compare the train–test gap of the two.
> - **SVM (RBF)** — compare against logistic regression to judge whether the
>   non-linear boundary bought anything, and note the fit-time cost.
> - **Overall winner** — pick on the metric that matches the decision cost, not
>   on accuracy. Under class imbalance, accuracy is dominated by the majority
>   class; MCC and AUC are the honest summaries. Justify the choice explicitly.

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | `<...>` |
| Decision Tree | `<...>` |
| kNN | `<...>` |
| Naive Bayes | `<...>` |
| Random Forest (Ensemble) | `<...>` |
| SVM (RBF) | `<...>` |
| **Overall winner** | `<model>` — `<which metric you selected on, and why that metric fits this problem>` |

---

## Streamlit app features

| Requirement | Where |
|---|---|
| Dataset upload option (CSV) | Sidebar → *1 · Test data* |
| Model selection dropdown | Sidebar → *2 · Model* |
| Display of evaluation metrics | *Metrics* tab — all six, plus ROC curve |
| Confusion matrix / classification report | *Confusion matrix & report* tab |
| All models on the test data | *Compare all models* tab |

---

## Limitations

`<Be honest — it reads better than overclaiming. Candidates: no hyperparameter
search, so these are default-ish configurations; no threshold tuning despite the
class imbalance; single split plus 5-fold CV rather than repeated CV; test_data.csv
is downsampled to keep within Streamlit Cloud's memory quota.>`
