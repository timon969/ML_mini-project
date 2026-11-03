# Wine Quality: Random Forest, SVM and Gradient Boosting

A reproducible study of quality-score regression on red and white Portuguese *vinho verde* from the UCI Wine Quality dataset.

# 🍷 Chemical Drivers of Wine Quality and the Impact of Product Segmentation

<<<<<<< HEAD
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
=======
Compare separate wine-type models with pooled models, and identify the chemical features most useful for predicting `quality` (integer scores 3–9). Predictions are continuous; rounded predictions are also evaluated with classification-style metrics.

The repository contains three model families: tuned Random Forest, tuned Support Vector Regression (SVR, the regression form of SVM), and fixed-parameter Gradient Boosting.

| Scenario | Training data | Evaluation rows |
|---|---|---|
| Red only | Red wines | Red wines |
| White only | White wines | White wines |
| Pooled + `wine_type` | All wines, including type | All wines, also reported by type |
| Pooled without `wine_type` | All wines, excluding type | All wines, also reported by type |

The two separate models together form the separate approach. All three comparison notebooks use the same five outer validation folds, so scenarios can be compared on identical rows.
>>>>>>> 05b88a282f1748573371b21f7a9147f4c477643a

## 2. Data

- **Source:** UCI Machine Learning Repository, *Wine Quality* (Cortez et al., 2009). Files in `dataset/`: `winequality-red.csv`, `winequality-white.csv` (`;`-separated), `winequality.names` (UCI description).
- **Size:** 1,599 red + 4,898 white = 6,497 rows, 11 features + `quality`.
- **Features (all numeric):** fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol.
- **Missing values:** none.
- **Duplicates:** many exact duplicate rows (240 red = 15.0%, 937 white = 19.1%). They were removed in the EDA notebook **before** any splitting, because identical rows in train and test folds would leak information. After cleaning: **1,359 red + 3,961 white = 5,320 rows**, saved as `dataset/wine_clean.csv`, which is the input shared by all three comparison notebooks.
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


## 3. Methodology & Model Justification

<<<<<<< HEAD
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
=======
**Preprocessing.** The comparison notebooks load `dataset/wine_clean.csv`, map wine type to 0/1 and exclude `quality_category`. Duplicates are removed before splitting. SVR puts `StandardScaler` inside its searched pipeline, fitted only on training folds. The tree models require no scaling; EDA scaling is for plots only.

| Model | Configuration | Parameter selection |
|---|---|---|
| Random Forest | `RandomForestRegressor` in a pipeline | `RandomizedSearchCV`: 12 sampled configurations, three shuffled inner folds; minimise mean inner-fold RMSE |
| SVM / SVR | `StandardScaler` → RBF `SVR` (`epsilon=0.1`) | `GridSearchCV`: `C` = 0.1/1/10, `gamma` = `scale`/`auto`/0.1; three inner folds; minimise mean squared error |
| Gradient Boosting | `GradientBoostingRegressor(n_estimators=300, random_state=42)` | Fixed settings; all remaining parameters at library defaults; no tuning |

RF searches tree count (100/200/300), depth (`None`/10/20/30), minimum split size (2/5/10), minimum leaf size (1/2/4), feature sampling (1.0/0.7/`sqrt`) and bootstrap on/off. SVR searches nine combinations. Mean inner-fold MSE and mean inner-fold RMSE are distinct aggregation objectives; the results below compare all models using outer RMSE.

**Validation.** Five outer `StratifiedKFold` folds are shuffled with seed 42 and stratified jointly by `wine_type + quality`. RF and SVR repeat parameter selection inside each outer training fold (nested CV); outer validation rows never enter the corresponding search. The `DummyRegressor(strategy="mean")` baseline is not tuned. Scores are calculated from concatenated out-of-fold predictions and reported separately for all wines, red and white.

**Secondary hold-out.** Each notebook also uses the same stratified 80/20 split: 4,256 training and 1,064 test rows. RF and SVR run fresh searches on training data and evaluate their refitted best estimators on untouched test rows. This split overlaps the full-data CV experiment and is not a separate independent dataset.

**Metrics.** RMSE, MAE, R², rounded accuracy and proportion within ±1 point. Hold-out macro precision/recall/F1 and confusion matrices show the effect of class imbalance. Fold-to-fold RMSE standard deviations describe variability, not formal significance. Permutation importance uses 20 repeats on hold-out test rows, measured as RMSE increase after shuffling a feature. Per-score outer-CV MAE identifies difficult target values.
>>>>>>> 05b88a282f1748573371b21f7a9147f4c477643a


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

RMSE is lower when predictions are better. Tables use four decimals; differences calculated before rounding may not match subtraction of displayed values exactly.

### 4.1 Tuned Random Forest

| Subset | Mean baseline | Separate models | Pooled + `wine_type` | Pooled without type |
|---|---|---|---|---|
| All | 0.8797 | 0.6807 | **0.6798** | 0.6836 |
| Red | 0.8411 | 0.6466 | **0.6457** | 0.6476 |
| White | 0.8925 | 0.6921 | **0.6911** | 0.6955 |

The pooled model with type has the lowest RMSE, but leads separate models by only 0.0009 overall. Its overall R² is 0.4028, MAE 0.5209, rounded accuracy 58.20%, and within-one-point rate 96.48%. Removing type raises RMSE by 0.0037 (0.55%, computed before rounding). RF fold standard deviations are 0.0283–0.0344 on red and 0.0077–0.0107 on white; the small ranking differences do not establish significance.

Alcohol leads all four RF permutation-importance rankings (RMSE increase about 0.103–0.113). In the pooled model with type it is followed by volatile acidity (0.0488) and free sulfur dioxide (0.0397). Sulphates matter more for red (0.0615 versus 0.0045 for white), while free sulfur dioxide matters more for white (0.0366 versus 0.0037 for red). Type itself has small importance (0.0044).

### 4.2 SVM / Support Vector Regression

| Subset | Mean baseline | Separate models | Pooled + `wine_type` | Pooled without type |
| --- | --- | --- | --- | --- |
| All | 0.8797 | 0.7034 | 0.7061 | 0.7077 |
| Red | 0.8411 | 0.6627 | 0.6585 | 0.6598 |
| White | 0.8925 | 0.7168 | 0.7217 | 0.7234 |

The lowest overall outer-CV RMSE is **0.7034** for `separate (red+white models)` (MAE 0.5335, R² 0.3606, rounded accuracy 57.01%, within ±1 point 95.79%). Removing type from the pooled procedure changes overall RMSE by +0.0016 (+0.22%). Positive means worse without type. The best pooled setup trails the separate approach by 0.0027 overall RMSE in this run. This is descriptive, not a significance claim.

The leading pooled-with-type permutation importances (RMSE increase) are `density` (0.1120), `wine_type` (0.0801), `volatile acidity` (0.0637); `wine_type` contributes 0.0801. These are model-specific predictive associations.

Hold-out RMSE, on identical rows within each wine type:

| Subset | Separate | Pooled + type | Pooled without type |
| --- | --- | --- | --- |
| Red | 0.6455 | 0.6482 | 0.6484 |
| White | 0.7264 | 0.7209 | 0.7233 |

Separate SVR models lead overall because they improve the white-wine subset, which contains most observations; the pooled model with type instead has lower RMSE on red wines. Type has substantial permutation importance in the fitted pooled SVR even though retraining without it only slightly changes CV RMSE. Shuffling a feature in one fitted model and retraining a model without that feature measure different effects.

The fresh hold-out searches select the following parameters; they are not the parameter sets used in every outer fold. Inner search scores are not unbiased model-performance estimates.

| Hold-out model | C | gamma |
| --- | --- | --- |
| Red | 1 | scale |
| White | 1 | scale |
| Pooled + type | 1 | scale |
| Pooled without type | 1 | scale |

### 4.3 Gradient Boosting

| Subset | Mean baseline | Separate models | Pooled + `wine_type` | Pooled without type |
| --- | --- | --- | --- | --- |
| All | 0.8797 | 0.7089 | 0.6992 | 0.7019 |
| Red | 0.8411 | 0.6803 | 0.6536 | 0.6628 |
| White | 0.8925 | 0.7184 | 0.7142 | 0.7148 |

The lowest overall outer-CV RMSE is **0.6992** for `combined + wine_type` (MAE 0.5420, R² 0.3682, rounded accuracy 55.36%, within ±1 point 96.03%). Removing type from the pooled procedure changes overall RMSE by +0.0027 (+0.39%). Positive means worse without type. The best pooled setup outperforms the separate approach by 0.0097 overall RMSE in this run. This is descriptive, not a significance claim.

The leading pooled-with-type permutation importances (RMSE increase) are `alcohol` (0.1322), `free sulfur dioxide` (0.0647), `volatile acidity` (0.0573); `wine_type` contributes 0.0029. These are model-specific predictive associations.

Hold-out RMSE, on identical rows within each wine type:

| Subset | Separate | Pooled + type | Pooled without type |
| --- | --- | --- | --- |
| Red | 0.6389 | 0.6387 | 0.6441 |
| White | 0.7455 | 0.7491 | 0.7470 |

Pooling helps Gradient Boosting most on red wines: RMSE falls from 0.6803 for a separate model to 0.6536 with pooled training and type. The white-wine gain is smaller (0.7184 to 0.7142). Thus the observed pooling effect differs from SVR and should not be generalised across model families.

These results describe the committed fixed 300-stage model (default learning rate 0.1 and tree depth 3), not a tuned Gradient Boosting optimum. Changing its depth, learning rate or number of stages may change the comparison.

### 4.4 Overall comparison on the same outer folds

| Model | Scenario | RMSE ↓ | R² ↑ | Rounded accuracy ↑ |
| --- | --- | --- | --- | --- |
| Random Forest | Separate | 0.6807 | 0.4012 | 57.37% |
| Random Forest | Pooled + type | 0.6798 | 0.4028 | 58.20% |
| Random Forest | Pooled without type | 0.6836 | 0.3962 | 58.21% |
| SVR | Separate | 0.7034 | 0.3606 | 57.01% |
| SVR | Pooled + type | 0.7061 | 0.3557 | 55.71% |
| SVR | Pooled without type | 0.7077 | 0.3528 | 56.05% |
| Gradient Boosting | Separate | 0.7089 | 0.3507 | 55.94% |
| Gradient Boosting | Pooled + type | 0.6992 | 0.3682 | 55.36% |
| Gradient Boosting | Pooled without type | 0.7019 | 0.3633 | 55.45% |

The lowest observed overall RMSE is **0.6798**, from **Random Forest — Pooled + type**. The search budgets differ, so this ranks the implemented procedures rather than establishing an inherent winner among model families.

### 4.5 Error by true quality score

Outer-CV MAE for the pooled model with type:

| Model | 3 (n=30) | 4 (n=206) | 5 (n=1752) | 6 (n=2323) | 7 (n=856) | 8 (n=148) | 9 (n=5) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RF | 2.334 | 1.255 | 0.450 | 0.362 | 0.678 | 1.500 | 2.340 |
| SVR | 2.465 | 1.351 | 0.447 | 0.360 | 0.757 | 1.630 | 2.401 |
| Gradient Boosting | 2.107 | 1.225 | 0.476 | 0.390 | 0.694 | 1.498 | 2.414 |

All three models have much larger errors at the rare extremes than at the common scores 5–6, consistent with predictions tending towards the middle of the scale. The extreme scores have small samples; per-score errors should be interpreted alongside these counts.

## 5. Conclusions

1. **Model comparison.** The lowest overall nested/outer-CV RMSE in the implemented experiments is 0.6798 (Random Forest, Pooled + type). Use the common outer-fold comparison above; inner search scores and metrics from different wine-type subsets answer different questions.
2. **Separate versus pooled.** For RF, pooled with type leads separate models by only 0.0009 RMSE. Separate SVR models lead overall (0.7034 versus 0.7061 for pooled with type), driven by better white-wine predictions. Gradient Boosting benefits from pooling (0.6992 versus 0.7089), especially on red wines. The pooling decision is model-dependent; these rankings are descriptive, not significance tests.
3. **Value of wine type.** Omitting type changes overall RMSE by +0.0037 for RF, +0.0016 for SVR and +0.0027 for Gradient Boosting. Independent RF/SVR searches mean these are complete-procedure comparisons, not fixed-parameter ablations.
4. **Features and errors.** Alcohol leads the tree-model rankings, while density and wine type lead the pooled SVR ranking. Strong permutation importance for type in SVR can coexist with a small retraining benefit from including it. All models struggle more with rare extreme scores than with common scores 5–6. Predictive importance does not establish a causal effect on wine quality.
5. **Tuning matters for interpretation.** RF and SVR search their parameters within training folds; Gradient Boosting remains fixed. Its result is a useful baseline, but cannot establish the performance ceiling of boosting. A shared hold-out is a secondary check, not a replacement for cross-validation.

## 6. Limitations

- Search budgets differ: RF samples 12 configurations, SVR searches nine combinations, and Gradient Boosting is not tuned. Results compare these implemented procedures, not the best possible version of each algorithm.
- Finite searches do not guarantee global optima; selected parameters can vary by outer fold and scenario.
- `quality` is subjective and imbalanced. Rare extremes are difficult to predict and macro classification metrics can be unstable.
- Permutation importance measures predictive reliance, not causation. Correlated measurements can share or dilute importance.
- Exact-row deduplication cannot distinguish different wines with identical measurements; UCI provides no wine identifier.
- Red has fewer observations than white. Fold spread and a single hold-out split do not establish statistical significance.
- The hold-out and full-data CV overlap. Conclusions apply to these data, splits, preprocessing choices and search budgets.

## 7. Repository structure

| File / directory | Purpose |
|---|---|
| `dataset/winequality-red.csv`, `dataset/winequality-white.csv`, `dataset/winequality.names` | Raw UCI data and description |
| `dataset/wine_clean.csv` | Deduplicated input shared by all model comparisons |
| [wine_quality_eda.ipynb](wine_quality_eda.ipynb) | Data inspection, cleaning and plots |
| [wine_rf_comparison.ipynb](wine_rf_comparison.ipynb) | Tuned RF comparison with saved executed results |
| [wine_svr_gridsearch_cv.ipynb](wine_svr_gridsearch_cv.ipynb) | Scaled SVR with nested grid search |
| [wine_quality_gradient_boosting_model.ipynb](wine_quality_gradient_boosting_model.ipynb) | Fixed 300-stage Gradient Boosting comparison |
| [main.ipynb](main.ipynb) | Early linear/polynomial exploration before deduplication, using different splits; scores are not directly comparable |
| `SVR.ipynb` | Empty placeholder; use `wine_svr_gridsearch_cv.ipynb` |
| `figures/` | Four EDA figures |
| [requirements.txt](requirements.txt) | Pinned dependencies shared by the notebooks |

## 8. Reproducibility and dependencies

Use Python 3.12 or 3.13. The pinned NumPy release requires Python >=3.12. The original RF run reports Python 3.13; the README's new SVR and Gradient Boosting results were validated with Python 3.12 and the same exact library pins. CPU execution is sufficient. Run commands from the repository root.

| Package | Pinned version |
|---|---|
| NumPy | 2.5.3 |
| pandas | 3.0.5 |
| scikit-learn | 1.9.1 |
| Matplotlib | 3.10.8 |
| seaborn | 0.13.2 |
| ipykernel | 7.4.0 |
| nbconvert | 7.17.2 |

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pip check
python -m ipykernel install --user --name wine-quality --display-name "Wine Quality"

jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=wine-quality --ExecutePreprocessor.timeout=-1 wine_quality_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=wine-quality --ExecutePreprocessor.timeout=-1 wine_rf_comparison.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=wine-quality --ExecutePreprocessor.timeout=-1 wine_svr_gridsearch_cv.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=wine-quality --ExecutePreprocessor.timeout=-1 wine_quality_gradient_boosting_model.ipynb
```

The named kernel ensures execution uses the installed environment. The cleaned CSV is committed; rerun EDA first when changing raw data. Run notebook cells from top to bottom. All comparison notebooks use `SEED = 42` for applicable random operations.

## 9. Disclosure of AI tool use

**Tools used and for what.** Claude (Anthropic). In this project it was used for: drafting this README (goal, data, method, results, conclusions, limitations, reproducibility, this disclosure); reading the repository through a GitHub connector so that the README matches the actual notebooks; translating `main.ipynb` and `wine_rf_comparison.ipynb` from Ukrainian into English; changing `wine_rf_comparison.ipynb` to load the cleaned `dataset/wine_clean.csv` instead of the raw files (the output tables are identical to the previous version); adding a library-version cell to `main.ipynb`; creating `requirements.txt`; and re-executing all three notebooks in one environment with pinned versions and committing the result.

**Influence on methodological decisions.** In this session the AI did not change any methodological decision: it described the design already present in the notebooks (scenarios, identical stratified folds, duplicate removal before splitting, hold-out permutation importance, error analysis). Its edits to the notebooks were limited to translation, the two typo fixes, the version cell and the switch to `wine_clean.csv`. After re-execution, all numeric results were identical to the earlier outputs.

**Parts written or substantially changed by me.** The research question and scenarios, the baseline and polynomial regression in `main.ipynb`, the choice of metrics, the interpretation of results.

**What I verified and understood independently.** Re-running the notebooks, checking that duplicates are removed before splitting and that no preprocessing is fitted on test folds, comparing README numbers with notebook outputs, and being able to explain permutation importance and stratified K-fold.

**Representative prompts.**
1. "write a README file for the project, covering the goal, the data, the method, the results, the conclusions and the limitations. Instructions for reproducing the work: the environment, the dependencies, and the commands to run."
2. "paste model comparisons into readme"
3. "change notebooks so that all libraries are the same, create a requirements.txt. let all notebooks be in english entirely. commit all changes"
4. "rewrite wine rf comparison so that it uses wine_clean and is therefore quicker. update readme accordingly"

## References

Cortez, P., Cerdeira, A., Almeida, F., Matos, T., Reis, J. (2009). *Modeling wine preferences by data mining from physicochemical properties.* Decision Support Systems, 47(4), 547–553. Dataset: UCI Machine Learning Repository, Wine Quality.
