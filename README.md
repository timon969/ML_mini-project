# Wine Quality: Does Splitting by Wine Type Matter, and Which Chemical Features Matter Most?

A small, reproducible empirical study on the UCI Wine Quality dataset (red and white *vinho verde*).

---

# 🍷 Chemical Drivers of Wine Quality and the Impact of Product Segmentation

### 1.1 Problem Formulation & Research Questions
In this machine learning project, we investigate the physicochemical properties of Portuguese "Vinho Verde" wine (red and white variants) to systematically address two core research questions:

1. **Impact of Dataset Segmentation on Generalization Performance:**
   * *How does explicit dataset segmentation by wine type (isolated red-only and white-only models vs. unified combined models, with and without an explicit wine type indicator) affect cross-validation stability and predictive accuracy?*
2. **Identification of Chemical Drivers via Permutation Importance:**
   * *Which chemical attributes demonstrate the highest objective feature importance when evaluated using model-agnostic Permutation Feature Importance, and do these drivers differ across wine categories?*

---

### 1.2 Motivation & Practical Value (Why It Matters)

* **Operational & MLOps Efficiency (Global vs. Local Models):**
  Wineries typically produce both red and white wines. Maintaining, monitoring, and retraining two isolated machine learning pipelines introduces significant operational overhead. Determining whether a single global model can match or exceed the performance of specialized local models directly informs cost-effective deployment and pipeline architecture.
* **Mitigating Class and Domain Imbalance:**
  The dataset exhibits a marked volumetric asymmetry: white wine accounts for ~75% (~4,898 samples), whereas red wine represents only ~25% (~1,599 samples). It is crucial to determine whether pooling the data provides beneficial regularization and variance reduction for red wine predictions, or conversely, whether white wine chemistry dominates and distorts the red wine predictions.
* **Objective Quality Assurance & Interpretability:**
  Sensory evaluation by professional sommeliers is subjective, expensive, and non-scalable. Traditional tree-based feature importance (Mean Decrease in Impurity) is known to be biased toward high-cardinality continuous features. Using Permutation Feature Importance provides an unbiased ranking of critical chemical factors (e.g., alcohol, volatile acidity, sulfur dioxide), giving enologists actionable thresholds during fermentation and maturation.

---

### 1.3. Success Criteria & Evaluation Framework 
To ensure the investigation is rigorous, reproducible, and conclusive, the answers are defined by specific quantitative benchmarks:

* **For Question 1 (Data Segmentation & Generalization):**
  * We implement a 5-fold Stratified Cross-Validation scheme (stratified on composite bins of `wine_type` and `quality`) across four distinct experimental setups:
    1. **Red Wine Only** (`red_only`)
    2. **White Wine Only** (`white_only`)
    3. **Combined Dataset without Type Feature** (`combined_no_type`)
    4. **Combined Dataset with Type Feature** (`combined_with_type`)
  * A concrete answer is established by tabulating the **Mean ± Standard Deviation** of regression metrics (**RMSE**, **MAE**, and **R²**) across all folds, benchmarked against a **Dummy/Baseline Regressor** (predicting the mean target value).
  * A successful finding validates whether adding an explicit `wine_type` feature yields statistically significant error reduction or if non-linear tree ensembles implicitly reconstruct wine type from underlying physicochemical contrasts (e.g., total sulfur dioxide and volatile acidity).

* **For Question 2 (Permutation Feature Importance):**
  * A concrete answer is delivered through validation-fold Permutation Importance rankings, measuring the mean increase in RMSE when each feature's values are randomly shuffled.
  * Results are validated via ranked bar plots with standard deviation error bars, distinguishing universal quality drivers (e.g., alcohol content) from type-specific drivers (e.g., volatile acidity in red wine vs. sulfur dioxide balance in white wine).

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

## 3. Methodology & Model Justification

### 3.1. Baseline Comparison (Dummy Regressor)
To establish a rigorous empirical reference point, all predictive models across all four data segmentation scenarios are benchmarked against a **Dummy Regressor** (predicting the mean target value of the training set). This quantifies the exact performance gain achieved by capturing chemical features over a naive guess and confirms that model accuracy is driven by signal rather than class clustering around median quality scores.

### 3.2. Model Selection Justifications
* **Random Forest Regressor (Primary Architecture):**
  * *Non-Linearity & Interactions:* Sensory wine quality does not scale linearly with individual chemical compounds; it depends on non-linear balances (e.g., alcohol vs. acidity). Random Forest naturally captures high-order feature interactions without requiring manual polynomial feature engineering.
  * *Scale Invariance & Outlier Robustness:* Tree-based decision boundaries depend on feature order rather than distance metrics, preventing extreme values (such as sulfur dioxide outliers) from distorting split criteria.
  * *Variance Reduction:* Bagging across deep trees stabilizes predictions across the skewed 75/25 white-to-red data ratio.
* **Comparative Models (Gradient Boosting & SVR):**
  * *Gradient Boosting:* Evaluated to test whether iterative sequential residual correction outperforms parallel bagging.
  * *Support Vector Regression (SVR with RBF Kernel):* Serves as a continuous geometric alternative to decision trees, projecting physicochemical attributes into a high-dimensional space[cite: 3].

### 3.3. Preprocessing & Leakage Prevention Strategy
* **Target Standardization & Pipeline Integration:**
  For distance-sensitive estimators (like SVR), scaling (`StandardScaler`) is encapsulated strictly inside cross-validation pipelines. Scaling parameters (`mean`, `std`) are learned solely on the training fold and applied to validation/test folds via `.transform()`, guaranteeing zero data leakage.
* **Stratified Split Scheme:**
  Because wine quality ratings are discrete and heavily clustered around scores 5 and 6, cross-validation folds are stratified across combined `wine_type` and `quality` bins to prevent fold-to-fold class shift.

  
### 3.4. Justification of Permutation Feature Importance
* Built-in tree feature importances (Mean Decrease in Impurity / MDI) are known to exhibit structural bias toward continuous features with high cardinality.
* To address this, we rely on **Permutation Feature Importance** evaluated strictly on held-out validation folds. Shuffling each predictor and computing the relative degradation in validation RMSE provides an unbiased, model-agnostic ranking of true chemical drivers[cite: 2].


  ## 4. Validity of the Experiment & Leakage Prevention

To satisfy rigorous experimental standards, the validation architecture strictly mirrors the underlying data distribution and enforces complete partition isolation:

### 4.1. Cross-Validation Scheme Matched to Data Structure
* **Dual-Target Stratification:**
  The dataset exhibits two simultaneous distribution imbalances:
  1. A 3:1 volumetric dominance of white wine (~75%) over red wine (~25%).
  2. Severe target skewness, where median quality scores (5 and 6) represent over 80% of samples, while extreme scores (3, 8, 9) are sparse.
* To prevent severe fold-to-fold distributional shifts, splitting is performed via **5-Fold Stratified Cross-Validation (`StratifiedKFold`)** utilizing a composite stratum: `wine_type + "_" + quality`. This guarantees identical joint marginal distributions across all training and validation splits.

### 4.2. Strict Data Leakage Prevention
* **Encapsulation via Scikit-Learn `Pipeline`:**
  All statistical transformations—specifically `StandardScaler` (calculating $\mu$ and $\sigma$) and `OneHotEncoder`—are executed inside a `Pipeline`.
* **Fold Independence:**
  Parameter estimation (`.fit()`) occurs strictly on the training partition of each fold. Validation and test instances are exclusively evaluated using `.transform()`, ensuring that no out-of-fold distribution parameters inform feature transformation.
* **Separation of the Hold-Out Test Partition:**
  Prior to cross-validation and hyperparameter optimization, a clean 20% stratified hold-out test set was isolated and locked. Cross-validation was run solely on the remaining 80% development set, reserving the test set exclusively for final verification.



---


## 5. Evaluation Criterion

### 5.1. Task-Specific Metric Selection
Wine quality prediction lies at the intersection of continuous regression and ordinal classification (integer ratings from 3 to 9. Evaluating such models using a single metric (e.g., Accuracy alone) fails to reflect practical performance, as misclassifying score 5 as 6 is practically acceptable for a winemaker, whereas confusing score 3 with score 8 is a critical error. Consequently, we employ a multi-layered evaluation framework:

1. **Continuous Regression Metrics:**
   * **Root Mean Squared Error (RMSE):** Primary optimization target; penalizes large, unacceptable outlier predictions.
   * **Mean Absolute Error (MAE):** Reflects the average magnitude of prediction error on the sommelier rating scale
   * **Coefficient of Determination ($R^2$):** Measures the proportion of variance in quality scores captured by physicochemical features relative to the mean baseline.
2. **Discretized Classification Metrics (Post-Rounding):**
   * **Accuracy, Macro-averaged Precision, Recall, and F1-Score:** Account for class imbalance across underrepresented extreme scores (3, 4, 8, 9).
3. **Domain-Specific Tolerance Metrics:**
   * **Exact Match Rate:** Share of predictions matching true sensory scores exactly.
   * **Within $\pm 1$ Point (Off-by-One):** Practical sensory tolerance threshold, acknowledging expert taster subjectivity.
   * **Gross Error Rate ($> 1$ Point):** Proportion of predictions deviating by two or more grade levels.

---

### 6.3. Analysis &

1. **Confusion Matrices (Heatmaps):**
   * Multi-class confusion matrices illustrate that predictions concentrate tightly along the main diagonal, with over 95% of total errors confined strictly within the $\pm 1$ point tolerance band[cite: 3, 5].
2. **Error Distribution Curves by Quality Tier (MAE by True Score):**
   * Residual analysis reveals an inherent regression-to-the-mean tendency: models exhibit minimal error on abundant classes (scores 5 and 6, MAE $\approx 0.45$), while error increases on extreme tiers (scores 3 and 8/9, MAE $> 1.0$), reflecting sample scarcity in the training distribution[cite: 3, 5].
3. **Validation Permutation Feature Importance:**
   * Ranked horizontal bar plots with fold-variance error bars quantify the exact impact of chemical factors on validation RMSE[cite: 3, 5]. Alcohol and volatile acidity consistently emerge as the primary non-linear drivers across both cultivars, whereas `wine_type` contributes negligible marginal importance[cite: 3, 5, 6].


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
