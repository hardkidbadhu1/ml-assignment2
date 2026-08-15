# FIFA World Cup 2026 Player Performance: a comparison of six classifiers

**Course:** M.Tech (AIML/DSE), Machine Learning, Assignment 2<br>
**Name / BITS ID:** `Badhmanaban M` / `2025AC05386`

---

## a. Problem statement

Given one player's record from a single match, that is, what he did on the pitch
over those ninety minutes plus his physical attributes, predict which of four
**positions** he plays: Goalkeeper, Defender, Midfielder or Forward. This is a
**multi-class** classification problem over four classes.

The practical motivation is role inference from telemetry. Match event feeds
(passes, tackles, distance covered, sprint distance) arrive reliably, but the
position label is squad metadata that is often missing, stale, or simply wrong
for a player deployed out of his nominal role. A model that recovers position
from behaviour is useful in two ways:

* It fills gaps where a third-party feed has no position recorded.
* It flags the disagreements, which are the interesting cases. A player listed
  as a Midfielder whose match profile scores as a Defender is exactly the
  tactical role change an analyst wants brought to their attention.

---

## b. Dataset description

| Property | Value |
|---|---|
| Source | [FIFA World Cup 2026, Player Performance (Kaggle)](https://www.kaggle.com/datasets) |
| Instances | **54,600** raw player-match rows (requirement: at least 500) |
| Features | **53** after cleaning (requirement: at least 12), being 48 numeric and 5 categorical |
| Target | `position`, one of Goalkeeper, Defender, Midfielder, Forward |
| Class balance | Defender 34.62%, Midfielder 30.77%, Forward 23.08%, Goalkeeper 11.54% |
| Missing values | **0**, so the imputation step in the pipeline is never actually exercised |
| Rows used | 15,000, drawn as a stratified subsample (reasoning below) |
| Train / test split | 11,187 / 3,813 rows, `GroupShuffleSplit` on `player_id`, `random_state=42` |

### Why the split is grouped rather than stratified

This is the single most important decision in the project, so it is worth
setting out carefully.

The grain of this table is the player-match, not the player. There are 1,248
players contributing roughly 44 rows each. Several columns never vary within a
player: `age`, `height_cm`, `weight_kg`, `market_value_eur` and
`preferred_foot`. Under an ordinary `train_test_split`, the consequences are:

* The same player appears on both sides of the split.
* Those constant columns then act as an identity fingerprint.
* The model can score well by recognising players it has already seen.
* The reported test number measures recall of the training set rather than
  genuine generalisation.

This is not a theoretical worry. Measured on this data:

| Target | Random row split | Grouped split on `player_id` | Majority baseline |
|---|---|---|---|
| `preferred_foot` | 0.846 | **0.712** | 0.745 |
| `position` | 0.950 | **0.944** | 0.346 |

Reading those two rows:

* `preferred_foot` looks like a respectable 0.85 accuracy under a random split.
  Under a grouped split it drops to 0.712, which is **below the majority-class
  baseline of 0.745**. All of the apparent skill was memorisation of players.
  A model predicting "Right" for everybody would have done better.
* `position` barely moves, falling only from 0.950 to 0.944. Its signal is
  genuinely per-match rather than per-player, which is what makes it a sound
  choice of target here.

Cross-validation uses `StratifiedGroupKFold` for the same reason. Ordinary
`StratifiedKFold` would report the same inflated figure and would be useless as
a check. Both numbers in the table above are reproduced by `scripts/verify.py`.

### Columns dropped, and the reasoning in each case

The executable version of this list is in `scripts/train_fifa.sh`. There are
three distinct groups.

**1. Identity and fixture metadata (11 columns).** `player_name`, `match_id`,
`match_date`, `jersey_number`, `club_name`, `team`, `nationality`,
`opponent_team`, `stadium`, `city`, `market_value_eur`.

* None of them says anything about *position*.
* Several uniquely fingerprint a player, which is the leakage described above.
* `player_id` is held out separately as the grouping key and is never offered to
  the model as a feature.

**2. Tournament-level rollups (5 columns).** `total_goals_tournament`,
`total_assists_tournament`, `total_minutes_tournament`,
`player_of_match_awards`, `tournament_rating`.

* Each aggregates the whole competition, including matches that fall in the
  test split.
* Using them to predict a single match is therefore target leakage across time.

**3. Team-match outcome (4 columns).** `match_result`, `goals_team`,
`goals_opponent`, `tournament_stage`.

* These describe the fixture, not the player.
* They also carry no signal whatsoever. A Random Forest predicting
  `match_result` from every other column scores **0.380** against a majority
  baseline of **0.369**. The generator sampled these independently of everything
  else, which is a useful reminder that the dataset is synthetic.

### Preprocessing

Every model sits inside a single `sklearn.Pipeline`. Imputation and encoding are
therefore fitted on the training fold alone, and they travel with the serialised
model. The Streamlit app cannot apply a different transformation at inference
than was used at training, which is the class of bug that produces an excellent
notebook and a broken deployment.

* **Numeric columns (48):** median imputation, then `StandardScaler`. Scaling is
  what makes kNN (Euclidean distance), logistic regression (convergence) and the
  RBF SVM (kernel width) work at all. It makes no difference to the tree-based
  models.
* **Categorical columns (5):** `preferred_foot`, `yellow_cards`, `red_cards`,
  `clean_sheet`, `penalty_saves`. Mode imputation, then
  `OneHotEncoder(handle_unknown="ignore")`, so a category present in an uploaded
  CSV but absent from training encodes as all zeros instead of raising an error
  at inference time.
* **Counts are kept numeric.** The default heuristic sends any integer column
  with ten or fewer distinct values down the one-hot branch. That is right for
  an integer-coded month and wrong for `tackles` or `goals`, where the ordering
  is real and three genuinely is more than one. Forcing the 18 count columns
  back to numeric moved Naive Bayes from **0.489 to 0.706** accuracy, with AUC
  going from 0.796 to 0.947. This was the single largest modelling change in the
  project. A Gaussian fitted to a 0/1 dummy variable is a poor density model.

### Why the data is subsampled to 15,000 rows

* `SVC` scales somewhere between quadratically and cubically in the number of
  training samples, and `probability=True` wraps it in an internal five-fold
  Platt calibration on top of that.
* The RBF fit, rather than the size of the data, is what actually constrains
  end-to-end runtime. On the lab machine it takes 15.72 seconds against 0.19
  seconds for the fastest model.
* Sampling is stratified on `position`, so the models still see the population
  class prior.
* Even after the cut, 15,000 rows is thirty times the assignment minimum.

---

## c. GitHub repository link

- **Repository:** <https://github.com/hardkidbadhu1/ml-assignment2>
- **Live Streamlit app:** <https://badhu-ml-assignment2.streamlit.app/>

```
project-folder/
├── app.py                      Streamlit user interface
├── ml_pipeline.py              preprocessing and metric code shared by training and serving
├── requirements.txt            pinned runtime dependencies, installed by Streamlit Cloud
├── requirements-dev.txt        build-time only, for PDF rendering
├── runtime.txt                 Python version request for Streamlit Cloud
├── .python-version             Python version for uv, which Streamlit Cloud now uses
├── README.md
├── test_data.csv               held-out split used by the app, 2,000 rows
├── data/                       raw dataset, gitignored at 17 MB
├── model/
│   ├── train.py                training and evaluation entry point
│   └── artifacts/              one .joblib pipeline per model, plus metadata.json
├── reports/
│   ├── metrics.csv, metrics.md          generated metric tables
│   └── ML_Assignment2_...pdf            submission PDF, rendered from this README
├── screenshots/
│   ├── lab/                    BITS Virtual Lab evidence
│   └── app/                    deployed Streamlit application
└── scripts/
    ├── train_fifa.sh           the exact training invocation, with its reasoning
    ├── lab_run.sh              one-shot run for the BITS Virtual Lab VM
    ├── verify.py               leakage, artifact and deployment checks
    └── make_pdf.py             builds the submission PDF from this README
```

**To reproduce:**

```bash
pip install -r requirements.txt
bash scripts/train_fifa.sh     # about 25 seconds
python scripts/verify.py       # 28 checks, exits non-zero on any failure
streamlit run app.py

# Submission PDF, using build-time dependencies only
pip install -r requirements-dev.txt
python scripts/make_pdf.py
```

A note on versions, because this caused real trouble during deployment. A
`Pipeline` unpickled under a different scikit-learn version than it was fitted
on is the most common Streamlit Cloud failure, and the bad case is silent
mis-prediction rather than a loud error. The artifacts committed here were
fitted on the lab VM under Python 3.12.7 with scikit-learn 1.7.2, and
`requirements.txt` pins exactly that. If you retrain elsewhere, copy the version
block that `train.py` prints at the end of its run into `requirements.txt`. The
app also shows a warning in its sidebar whenever the running scikit-learn
differs from the one recorded in `metadata.json`.

---

## d. Models used

All six models see the same dataset, the same grouped split and the same
preprocessing pipeline, so the comparison isolates the effect of the learning
algorithm and nothing else.

| Model | Key hyperparameters | Reason for inclusion |
|---|---|---|
| Logistic Regression | `max_iter=2000` | Linear baseline with calibrated probabilities |
| Decision Tree | `min_samples_leaf=5` | Non-linear, interpretable, high variance |
| kNN | `k=15`, distance-weighted | Instance-based, depends entirely on the scaler |
| Naive Bayes (Gaussian) | defaults | Generative, with a strong independence assumption |
| Random Forest | `n_estimators=200`, `min_samples_leaf=5` | Bagging ensemble, for variance reduction |
| SVM (RBF) | `probability=True` | Max-margin with a kernel, the sixth model |

The forest uses `min_samples_leaf=5` rather than the more usual 2. A fully grown
300-tree forest pickles to 46 MB, which is awkward to commit and wasteful to
hold in Streamlit Cloud's memory. Reducing it costs 0.003 accuracy and produces
an artifact 2.4 times smaller, and the bagging-versus-single-tree comparison
that the write-up depends on survives intact.

### Comparison table

Measured on the held-out test set of 3,813 rows, covering 312 players never seen
during training. Precision, recall and F1 are **macro**-averaged; AUC is
**one-vs-rest macro**.

These figures were produced on the BITS Virtual Lab VM (Rocky Linux 9.5, Python
3.12.7, scikit-learn 1.7.2). They come from the same run as the screenshot in
the report, and from the same artifacts committed under `model/artifacts/`. Fit
times are specific to that eight-vCPU instance, but the metrics themselves
reproduce exactly on other hardware, since every model is seeded with
`random_state=42`.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | **0.9580** | **0.9972** | **0.9608** | **0.9573** | **0.9589** | **0.9416** |
| Decision Tree | 0.8985 | 0.9602 | 0.9041 | 0.8973 | 0.9002 | 0.8587 |
| kNN | 0.8893 | 0.9829 | 0.9001 | 0.8839 | 0.8911 | 0.8457 |
| Naive Bayes (Gaussian) | 0.7060 | 0.9465 | 0.7197 | 0.7713 | 0.7011 | 0.6310 |
| Random Forest (Ensemble) | 0.9441 | 0.9948 | 0.9486 | 0.9439 | 0.9461 | 0.9223 |
| SVM (RBF) | 0.9533 | 0.9961 | 0.9567 | 0.9502 | 0.9533 | 0.9349 |

### Diagnostics

Five-fold `StratifiedGroupKFold`, run on the training split only.

| Model | Train Acc | Test Acc | Gap | CV Acc (mean ± sd) | Fit time (s) |
|---|---|---|---|---|---|
| Logistic Regression | 0.9621 | 0.9580 | 0.004 | 0.9551 ± 0.0066 | 1.44 |
| Decision Tree | 0.9634 | 0.8985 | **0.065** | 0.8999 ± 0.0021 | 0.46 |
| kNN | 1.0000 | 0.8893 | **0.111** | 0.8866 ± 0.0101 | 0.20 |
| Naive Bayes (Gaussian) | 0.7293 | 0.7060 | 0.023 | 0.7274 ± 0.0159 | 0.19 |
| Random Forest (Ensemble) | 0.9790 | 0.9441 | **0.035** | 0.9445 ± 0.0064 | 3.69 |
| SVM (RBF) | 0.9631 | 0.9533 | 0.010 | 0.9518 ± 0.0075 | 15.72 |

Two things this table is for. The train-versus-test gap is the evidence of
overfitting, and the cross-validation standard deviation tells you whether a
small difference between two models is signal or noise. A gap of 0.005 between
two models whose CV standard deviation is 0.007 is not a real difference.

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best model on all six metrics, and also the simplest one, which is the finding rather than an accident. The engineered composite columns (`offensive_contribution`, `defensive_contribution`, `creativity_score`) act almost as linear discriminants for role, so the four classes are very nearly linearly separable in this feature space and a hyperplane is the right hypothesis class. Its train-test gap of 0.004, against a CV standard deviation of 0.0066, says it is not overfitting and has capacity to spare. The 0.5-point lead over the RBF SVM is only about 0.7 of a standard deviation, so the honest reading is that logistic regression and the SVM are tied at the top, ahead of the forest. |
| Decision Tree | Trains to 0.9634 and tests at 0.8985, a gap of 0.065 that is pure variance. A single tree partitions with axis-parallel cuts, so it approximates a diagonal boundary with a staircase and spends depth doing it; `min_samples_leaf=5` bounds this without removing it. Worth noting that its CV standard deviation of 0.0021 is the lowest in the table. It is consistently mediocre rather than erratic, so the gap reflects bias towards the training set rather than sensitivity to which fold it sees. |
| kNN | Train accuracy is exactly 1.0000, which gives away `weights="distance"`: a training point sits at distance zero from itself and takes infinite weight, so the model reproduces the training set by construction. Real generalisation is 0.8893, and the gap of 0.111 is the largest here. Across 48 scaled numeric dimensions, Euclidean distances are already concentrating, so a fixed k averages over neighbours that are not especially near. This is also the model most dependent on the `StandardScaler`. Without it, columns on the scale of market value would dominate the distance metric outright. |
| Naive Bayes (Gaussian) | Clearly the weakest on accuracy at 0.7060, yet it holds an AUC of 0.9465. That combination is the signature of violated conditional independence: the class ranking is good while the calibration is bad. `offensive_contribution`, `creativity_score`, `possession_impact` and the underlying pass and shot counts are strongly correlated, so the model multiplies what is effectively the same evidence several times over, pushing posteriors to overconfident extremes. It mislabels cases near the boundaries while keeping the overall ordering roughly right. Recall of 0.7713 exceeding precision of 0.7197 says it over-predicts the broad classes. It is also the fastest to fit at 0.19 seconds. The accuracy cost is not worth paying here, but the AUC shows the features themselves are informative. |
| Random Forest (Ensemble) | Bagging does exactly what it promises. Set against the single tree, train accuracy rises from 0.9634 to 0.9790 while the train-test gap halves, from 0.065 to 0.035, and test accuracy climbs from 0.8985 to 0.9441. Averaging 200 decorrelated trees cancels the variance of any individual one. It still trails logistic regression, which is the useful negative result: the ensemble's extra capacity goes into modelling a boundary that was close to linear to begin with, so there was nothing there to buy. |
| SVM (RBF) | Statistically tied with logistic regression, 0.9533 against 0.9580, less than one CV standard deviation apart, at roughly eleven times the fit cost of 15.72 seconds against 1.44. That near-equality is itself the evidence. If the RBF kernel's non-linear boundary were buying real separation, it would show up as a clear win, and it does not. This independently confirms the linear-separability reading. It is the slowest model in the table, because `probability=True` adds an internal five-fold Platt calibration on top of a fit that is already superlinear. |
| **Overall Winner for your dataset?** | **Logistic Regression**, selected on **MCC (0.9416)**. With four classes running from 34.6% down to 11.5%, plain accuracy is dominated by Defender and Midfielder, and would let a model that ignores Goalkeepers entirely still look respectable. MCC accounts for every cell of the 4x4 confusion matrix and does not reward that. Logistic regression happens to top all six metrics, so the choice is not contested on this data, but it also wins on the criteria that matter in practice: it fits roughly eleven times faster than the SVM it is tied with, it produces by far the smallest artifact at 11 KB against the forest's 19 MB, and it is the only model whose coefficients can be read directly as a statement about which behaviours mark out a Defender. |

---

## Streamlit app features

| Requirement | Where it lives in the app |
|---|---|
| Dataset upload option (CSV) | Sidebar, *1 · Test data* |
| Model selection dropdown | Sidebar, *2 · Model* |
| Display of evaluation metrics | *Metrics* tab, all six |
| Confusion matrix and classification report | *Confusion matrix & report* tab |
| All models on the test data | *Compare all models* tab |

---

## Verification

`scripts/verify.py` runs 28 checks and exits non-zero on any failure. They fall
into five groups.

* **Leakage.** The training and test player sets are disjoint, 936 against 312
  with none shared, and together they account for all 1,248 players. The
  grouping column is confirmed absent from the feature list. The grouped split
  is confirmed to be no more optimistic than a random one.
* **Artifacts.** Every pipeline named in `metadata.json` loads and predicts,
  which is precisely what `app.py` does at startup. This catches a
  version-mismatched pickle before deployment rather than after.
* **Robustness.** An unseen categorical level at inference encodes to zeros
  rather than raising.
* **Deployability.** No artifact exceeds 50 MB, which is GitHub's warning
  threshold, and the total is 25.9 MB.
* **App rendering.** `streamlit.testing.v1.AppTest` executes `app.py` against
  the *pinned* Streamlit version and selects each of the six models in turn.

That last group caught two bugs that artifact-loading tests alone would have
missed, because both are Streamlit API problems rather than model problems.

1. `pandas.DataFrame.style` is an optional accessor gated on `jinja2`, which
   Streamlit does not install transitively. It is now pinned, and the call site
   falls back to an unstyled table rather than taking down the tab.
2. `st.dataframe(..., width="stretch")` is valid from Streamlit 1.49 onwards but
   raises `TypeError` on the pinned 1.41.1. The app would have started cleanly
   and then crashed the moment anyone opened the *Confusion matrix* tab. It now
   uses `use_container_width=True`, the correct API for the pinned version.

The general lesson is worth stating plainly: pinning a dependency and then
writing code against a newer version of its API fails at runtime, not at install
time. The only way to catch it is to actually execute the app under the pin.

---

## Limitations

Being straightforward about these reads better than overclaiming.

* **The dataset is synthetic, and it shows.** Zero missing values across 54,600
  rows and 75 columns is not a property real match data has, and the
  `match_result` columns are provably independent of everything else, at 0.380
  against a 0.369 baseline. The conclusions here are about model behaviour on
  this feature geometry, not about football.
* **No hyperparameter search was performed.** These are close to default
  configurations. A tuned tree or a tuned k would narrow some of the gaps
  reported above, and `min_samples_leaf=5` on the forest was chosen for artifact
  size rather than accuracy.
* **A single grouped split plus five-fold CV**, rather than repeated CV. Any
  difference below roughly 0.01 accuracy should be treated as noise, which is
  exactly why the logistic regression versus SVM result is reported as a tie
  rather than a win.
* **Position is an easy target.** Goalkeepers are nearly separable on `saves`
  alone. Almost all the remaining difficulty sits in a single pair of classes:
  of the winning model's 69 errors on `test_data.csv`, **46 are Midfielder
  confused with Forward**, in both directions. That boundary is genuinely fuzzy,
  since an attacking midfielder and a withdrawn forward produce similar match
  profiles. The per-class rows of the classification report in the app carry
  more information than the macro averages in the tables above.
* **`test_data.csv` is downsampled to 2,000 rows** to stay within Streamlit
  Cloud's memory quota, so the metrics the app reports differ slightly from the
  tables here, which were computed on the full 3,813-row test split.
