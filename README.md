# Wine Quality: Does Splitting by Wine Type Matter, and Which Chemical Features Matter Most?

A small, reproducible empirical study on the UCI Wine Quality dataset (red and white *vinho verde*).

---

## 1. Goal

**Research question.** How does having or not having a split of the data by wine type (red / white) affect cross-validation results, and which chemical measurements turn out to be the most important when ranked with permutation importance?

**Task.** Predict the sensory quality score (`quality`, integers 3–9) of a wine from its 11 physicochemical measurements. Treated as **regression**; predictions are also rounded to the nearest integer to report classification-style metrics.

**Why it matters.** Red and white wines differ chemically, and a pooled model may either blur those differences or simply learn the wine type from other features. Comparing pooled and per-type setups shows whether splitting the data actually changes the validation picture, and whether "what drives quality" depends on the wine type.

**Scenarios compared** (Random Forest, identical CV folds for all):

| # | Scenario | Description |
|---|---|---|
| 1 | `red only` | Model trained and tested on red wine only |
| 2 | `white only` | Model trained and tested on white wine only |
| 3 | `combined + wine_type` | One pooled model, `wine_type` is a feature |
| 4 | `combined, no wine_type` | One pooled model, no type information |

Scenarios 1 + 2 together form the "separate models" approach (each wine is scored by the model of its own type).

**Scope.** This repository covers the Random Forest comparison. An SVM comparison is planned as a separate notebook and is **not part of the current results**.

---

## 2. Data

- **Source:** UCI Machine Learning Repository, *Wine Quality* (Cortez et al., 2009). Files in `dataset/`: `winequality-red.csv`, `winequality-white.csv` (`;`-separated), `winequality.names` (UCI description).
- **Size:** 1,599 red + 4,898 white = 6,497 rows, 11 features + `quality`.
- **Features (all numeric):** fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol.
- **Missing values:** none.
- **Duplicates:** many exact duplicate rows (240 red = 15.0%, 937 white = 19.1%). They were removed in the EDA notebook **before** any splitting, because identical rows in train and test folds would leak information. After cleaning: **1,359 red + 3,961 white = 5,320 rows**, saved as `dataset/wine_clean.csv`, which is the input of the Random Forest notebook.
- **Target distribution after cleaning:**

| quality | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|
| red | 10 | 53 | 577 | 535 | 167 | 17 | 0 |
| white | 20 | 153 | 1175 | 1788 | 689 | 131 | 5 |

  Scores 5–6 dominate and the extremes are rare, so the target is imbalanced.
- **Other observations from the EDA:**
  - White wine has about three times more samples than red.
  - Red and white are chemically distinguishable (e.g. white has about 3× the total SO₂ and 2.5× the residual sugar; red has about 2× the volatile acidity and chlorides).
  - Strongly correlated feature pairs (|r| > 0.6): free vs total SO₂ (0.72) and density vs alcohol (0.69). They were kept on purpose.
  - Features are on very different scales and some are strongly skewed.

---

## 3. Method

**Baseline.** `DummyRegressor(strategy="mean")` (constant prediction) for the final comparison. Earlier exploratory work in `main.ipynb` also used linear regression and polynomial regression as baselines (see section 4.4).

**Preprocessing and leakage control.**
- Duplicates are removed before any split, in the EDA notebook. The Random Forest notebook loads the cleaned `dataset/wine_clean.csv` and checks that it has no missing values and no duplicate rows.
- The Random Forest needs no scaling, so the modelling pipeline contains only the model and there is no preprocessing that could leak.
- The `StandardScaler` used in the EDA notebook is fitted on all data for plotting only. The scaled copy is deliberately **not** saved, and any later model that needs scaling (e.g. SVM) must put it inside a `Pipeline` fitted within each fold.

**Model.** `RandomForestRegressor(n_estimators=300, random_state=42)` with default hyperparameters. No tuning, because the aim is to compare the scenarios, not to maximise the score.

**Validation.**
- **5-fold `StratifiedKFold`** (shuffled, seed 42), stratified on the combination of `wine_type` and `quality`. The *same folds* are used in all scenarios, so every row has an out-of-fold prediction from every scenario and they can be compared row by row.
- Pooled models are evaluated separately on the red subset and on the white subset, because metrics on different test sets are not directly comparable.
- Fold-to-fold spread of RMSE is reported to judge whether differences are larger than noise.
- A single stratified **80/20 hold-out** (4,256 train / 1,064 test) is used as a secondary check.

**Metrics.** RMSE (main), MAE, R²; plus accuracy of the rounded prediction and the share of predictions within ±1 point. Macro precision/recall/F1 and a confusion matrix are printed by the provided `evaluate_classification_metrics` helper on the hold-out set, but they are unstable because of class imbalance.

**Feature importance.** Permutation importance on the **hold-out test set** (not on training data), 20 repeats, scored by RMSE increase, for the red model, the white model, the pooled model with `wine_type` and the pooled model without it.

**Error analysis.** CV MAE broken down by true quality score.

---

## 4. Results

### 4.1 Cross-validation (5-fold, RMSE, lower is better)

| Subset | Baseline (mean) | Separate models | Combined + `wine_type` | Combined, no `wine_type` |
|---|---|---|---|---|
| All | 0.880 | 0.6867 | 0.6863 | 0.6861 |
| Red | 0.841 | 0.6545 | 0.6475 | 0.6465 |
| White | 0.893 | 0.6971 | 0.6988 | 0.6988 |

All three Random Forest setups reach R² ≈ 0.39 on the full data (baseline: 0), with about 57% exact matches after rounding and about 96% within ±1 point.

Fold-to-fold RMSE spread (std): **±0.034–0.039 on red** and **±0.005 on white**. The differences between scenarios (≤ 0.007) are well inside this noise.

### 4.2 Effect of removing `wine_type` (pooled model, CV)

| Subset | RMSE with type | RMSE without type | ΔRMSE |
|---|---|---|---|
| All | 0.6863 | 0.6861 | −0.0002 |
| Red | 0.6487 | 0.6477 | −0.0010 |
| White | 0.6988 | 0.6988 | 0.0000 |

### 4.3 Permutation importance (RMSE increase, hold-out)

| Feature | Red model | White model | Combined + type | Combined, no type |
|---|---|---|---|---|
| alcohol | 0.136 | 0.174 | 0.169 | 0.170 |
| volatile acidity | 0.047 | 0.052 | 0.063 | 0.062 |
| free sulfur dioxide | 0.007 | 0.051 | 0.045 | 0.045 |
| sulphates | 0.078 | 0.002 | 0.025 | 0.025 |
| total sulfur dioxide | 0.018 | 0.005 | 0.015 | 0.016 |
| pH | 0.007 | 0.018 | 0.013 | 0.013 |
| chlorides | 0.002 | 0.007 | 0.009 | 0.009 |
| residual sugar | 0.005 | 0.010 | 0.007 | 0.007 |
| citric acid | 0.004 | 0.005 | 0.007 | 0.006 |
| fixed acidity | 0.001 | 0.003 | 0.006 | 0.006 |
| density | 0.003 | 0.003 | 0.003 | 0.003 |
| wine_type | – | – | 0.0004 | – |

### 4.4 Hold-out (single 80/20 split) and exploratory baselines

- On the hold-out split, the separate red model looks better than the pooled one (RMSE 0.625 vs 0.637 on red). On white the models are practically equal (0.722 vs 0.723). With one split this is within noise, which is why the CV results are preferred.
- `main.ipynb` (early exploration, **before** duplicate removal, 60/20/20 train/validation/test split) gave linear-regression validation RMSE of 0.674 (red), 0.741 (white), 0.721 (combined). Polynomial regression was best at degree 2 (0.666 / 0.700 / 0.704); degrees 3–4 overfit badly. These numbers are not comparable with the CV table because the data and the split differ.

### 4.5 Error analysis

CV MAE of the pooled model by true quality score:

| quality | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|
| MAE | 2.39 | 1.26 | 0.46 | 0.38 | 0.67 | 1.48 | 2.32 |
| n | 30 | 206 | 1752 | 2323 | 856 | 148 | 5 |

The model is accurate for the common scores 5–6 and fails on the rare extremes: predictions are pulled towards the mean (regression to the mean), so low and high scores are rarely recovered.

---

## 5. Conclusions

1. **Separate vs pooled models.** All Random Forest setups clearly beat the constant baseline (RMSE ≈ 0.69 vs 0.88). The differences between separate and pooled models (≤ 0.007 RMSE) are smaller than the fold-to-fold spread, so these data do **not** support a claim that splitting by wine type helps or hurts.
2. **`wine_type` is nearly redundant.** Removing it changes RMSE by at most 0.001, and its permutation importance is about 0.0004. The type appears to be already encoded in the other measurements (total SO₂, volatile acidity, chlorides, etc.).
3. **Most important features.** `alcohol` ranks first in every model, followed by `volatile acidity`. The type-specific differences: `sulphates` matter for red (0.078) but almost not for white (0.002), while `free sulfur dioxide` matters for white (0.051) but not for red (0.007). The pooled model averages these differences (sulphates 0.025).
4. **Single splits mislead.** The hold-out split suggested an advantage for the separate red model that cross-validation does not support.

---

## 6. Limitations

- **Importance is not causation.** Permutation importance shows what the model relies on, not what makes a wine better. With correlated features (density / alcohol / residual sugar, free / total SO₂) it splits or understates importance.
- **Hyperparameters were not tuned**, and only one model family (Random Forest) is covered so far. The SVM comparison is future work.
- **Noisy, imbalanced target.** `quality` is a subjective tasters' score; macro precision/recall/F1 are unstable and extreme scores are essentially unpredictable.
- **Deduplication is by exact row match.** UCI has no wine identifier, so some "duplicates" may be different wines with identical measurements, and some near-duplicates may remain.
- **Uneven sample sizes.** Red has only 1,359 rows, so its estimates are noisier (larger fold spread) than white.
- **Scope.** One dataset (Portuguese *vinho verde*) and default Random Forest parameters; results may not transfer to other wines or settings.

---

## 7. Repository structure

```
.
├── README.md
├── requirements.txt               # pinned library versions
├── dataset/
│   ├── winequality-red.csv
│   ├── winequality-white.csv
│   ├── winequality.names          # UCI description
│   └── wine_clean.csv             # de-duplicated, merged data (output of the EDA notebook, input of the RF notebook)
├── wine_quality_eda.ipynb         # loading, inspection, cleaning, visualisations
├── wine_rf_comparison.ipynb       # reads wine_clean.csv; baseline, 5-fold CV, hold-out, permutation importance, error analysis
├── main.ipynb                     # early exploration: linear / polynomial regression baselines
└── figures/                       # 4 EDA figures
```

---

## 8. Reproducibility

**Environment.** Python 3.13, ordinary laptop CPU (no GPU). All three notebooks were executed top to bottom in one environment with the library versions pinned in `requirements.txt`:

| Library | Version |
|---|---|
| numpy | 2.5.3 |
| pandas | 3.0.5 |
| scikit-learn | 1.9.1 |
| matplotlib | 3.10.8 |
| seaborn | 0.13.2 |
| ipykernel / nbconvert | 7.4.0 / 7.17.2 (to run the notebooks from the command line) |

**Commands**

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

jupyter nbconvert --to notebook --execute wine_quality_eda.ipynb --inplace
jupyter nbconvert --to notebook --execute main.ipynb --inplace
jupyter nbconvert --to notebook --execute wine_rf_comparison.ipynb --inplace
```

Run the commands from the repository root. `wine_rf_comparison.ipynb` reads `dataset/wine_clean.csv`, which is committed in the repository; if you change the raw data, run the EDA notebook first to regenerate it. The Random Forest notebook takes about 3 minutes on a 2-core machine. Nearly all of that is Random Forest fitting (5-fold CV in four scenarios plus 20-repeat permutation importance), not data loading.

**Seeds.** `SEED = 42` is used for NumPy, the CV splitter, the hold-out split, the Random Forest and permutation importance (`main.ipynb` uses `random_state=42` in its splits). Cells must be run from top to bottom.

---

## 9. Disclosure of AI tool use

**Tools used and for what.** Claude (Anthropic). In this project it was used for: drafting this README (goal, data, method, results, conclusions, limitations, reproducibility, this disclosure); reading the repository through a GitHub connector so that the README matches the actual notebooks; translating `main.ipynb` and `wine_rf_comparison.ipynb` from Ukrainian into English; changing `wine_rf_comparison.ipynb` to load the cleaned `dataset/wine_clean.csv` instead of the raw files (the output tables are identical to the previous version); fixing two typos in `main.ipynb` (the function name `polynomia_Regression` and a confusion-matrix title that said "White Wine" while the red-wine model was evaluated); adding a library-version cell to `main.ipynb`; creating `requirements.txt`; and re-executing all three notebooks in one environment with pinned versions and committing the result.

**Influence on methodological decisions.** In this session the AI did not change any methodological decision: it described the design already present in the notebooks (scenarios, identical stratified folds, duplicate removal before splitting, hold-out permutation importance, error analysis). Its edits to the notebooks were limited to translation, the two typo fixes, the version cell and the switch to `wine_clean.csv`. After re-execution, all numeric results were identical to the earlier outputs.

**Parts written or substantially changed by me.** The research question and scenarios, the baseline and polynomial regression in `main.ipynb`, the choice of metrics, the interpretation of results.

**What I verified and understood independently.** Re-running the notebooks, checking that duplicates are removed before splitting and that no preprocessing is fitted on test folds, comparing README numbers with notebook outputs, and being able to explain permutation importance and stratified K-fold.

**Representative prompts.**
1. "write a README file for the project, covering the goal, the data, the method, the results, the conclusions and the limitations. Instructions for reproducing the work: the environment, the dependencies, and the commands to run."
2. "paste model comparisons into readme"
3. "change notebooks so that all libraries are the same, create a requirements.txt. let all notebooks be in english entirely. commit all changes"
4. "rewrite wine rf comparison so that it uses wine_clean and is therefore quicker. update readme accordingly"

---

## References

Cortez, P., Cerdeira, A., Almeida, F., Matos, T., Reis, J. (2009). *Modeling wine preferences by data mining from physicochemical properties.* Decision Support Systems, 47(4), 547–553. Dataset: UCI Machine Learning Repository, Wine Quality.
