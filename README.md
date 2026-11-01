# Wine Quality: Random Forest, SVM and Gradient Boosting

A reproducible study of quality-score regression on red and white Portuguese *vinho verde* from the UCI Wine Quality dataset.

## 1. Goal

Compare separate wine-type models with pooled models, and identify the chemical features most useful for predicting `quality` (integer scores 3–9). Predictions are continuous; rounded predictions are also evaluated with classification-style metrics.

The repository contains three model families: tuned Random Forest, tuned Support Vector Regression (SVR, the regression form of SVM), and fixed-parameter Gradient Boosting.

| Scenario | Training data | Evaluation rows |
|---|---|---|
| Red only | Red wines | Red wines |
| White only | White wines | White wines |
| Pooled + `wine_type` | All wines, including type | All wines, also reported by type |
| Pooled without `wine_type` | All wines, excluding type | All wines, also reported by type |

The two separate models together form the separate approach. All three comparison notebooks use the same five outer validation folds, so scenarios can be compared on identical rows.

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


## 3. Method

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

## 4. Results

RMSE is lower when predictions are better. Tables use four decimals; differences calculated before rounding may not match subtraction of displayed values exactly.

### 4.1 Tuned Random Forest

The saved executed results in [wine_rf_comparison.ipynb](wine_rf_comparison.ipynb) replace the earlier fixed-parameter RF results.

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

The existing `requirements.txt` covers every imported third-party package. SVR and Gradient Boosting are included in scikit-learn; no additional model library is needed. `ipykernel` supplies the kernel and IPython, while `nbconvert` supplies command-line notebook execution. The unchanged pins installed in a clean virtual environment and passed `python -m pip check` with no broken requirements. Transitive dependencies are resolved by pip.

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

The named kernel ensures execution uses the installed environment. The cleaned CSV is committed; rerun EDA first when changing raw data. Run notebook cells from top to bottom. All comparison notebooks use `SEED = 42` for applicable random operations. Nested searches and repeated permutation importance can take substantially longer than the old fixed-RF experiment; runtime depends on the machine. Unlimited cell timeouts accommodate these searches.

The SVR and Gradient Boosting figures/tables in this README were obtained by executing all code cells with their committed model and search settings; plotting ran headlessly. Their repository notebooks currently have no saved outputs. RF figures are taken from its saved executed notebook. The early `main.ipynb` experiments are outside the model comparison because their data and splits differ.

## 9. Disclosure of AI tool use

### Earlier contribution (historical record)

The following disclosure describes earlier work and predates the RF tuning and three-model README update. Claims about unchanged methods or identical outputs apply to that earlier work only.


**Tools used and for what.** Claude (Anthropic). In this project it was used for: drafting this README (goal, data, method, results, conclusions, limitations, reproducibility, this disclosure); reading the repository through a GitHub connector so that the README matches the actual notebooks; translating `main.ipynb` and `wine_rf_comparison.ipynb` from Ukrainian into English; changing `wine_rf_comparison.ipynb` to load the cleaned `dataset/wine_clean.csv` instead of the raw files (the output tables are identical to the previous version); fixing two typos in `main.ipynb` (the function name `polynomia_Regression` and a confusion-matrix title that said "White Wine" while the red-wine model was evaluated); adding a library-version cell to `main.ipynb`; creating `requirements.txt`; and re-executing all three notebooks in one environment with pinned versions and committing the result.

**Influence on methodological decisions.** In this session the AI did not change any methodological decision: it described the design already present in the notebooks (scenarios, identical stratified folds, duplicate removal before splitting, hold-out permutation importance, error analysis). Its edits to the notebooks were limited to translation, the two typo fixes, the version cell and the switch to `wine_clean.csv`. After re-execution, all numeric results were identical to the earlier outputs.

**Parts written or substantially changed by me.** The research question and scenarios, the baseline and polynomial regression in `main.ipynb`, the choice of metrics, the interpretation of results.

**What I verified and understood independently.** Re-running the notebooks, checking that duplicates are removed before splitting and that no preprocessing is fitted on test folds, comparing README numbers with notebook outputs, and being able to explain permutation importance and stratified K-fold.

**Representative prompts.**
1. "write a README file for the project, covering the goal, the data, the method, the results, the conclusions and the limitations. Instructions for reproducing the work: the environment, the dependencies, and the commands to run."
2. "paste model comparisons into readme"
3. "change notebooks so that all libraries are the same, create a requirements.txt. let all notebooks be in english entirely. commit all changes"
4. "rewrite wine rf comparison so that it uses wine_clean and is therefore quicker. update readme accordingly"


### Current ChatGPT / Codex contribution

ChatGPT, through Composio, added RF parameter search with nested CV and revised its conclusions from the user-executed outputs. In this update it inspected all notebook imports, installed the existing requirements in a clean environment, checked dependency consistency, executed the SVR and Gradient Boosting code cells with the full committed search/model settings, and revised this README with their measured results and reproduction instructions. CPU parallelism was limited for the validation runtime; model parameters, data and validation folds were preserved. The current update changes documentation only; requirements remained valid.

Representative prompts: “rewrite wine rf comparison so that it searches for the best parameters for random forest”; “I ran the script, change the conclusions accordingly”; “update readme with svm and gradient boosting conclusions, check that requierements.txt are still valid, update if needed”.

## References

Cortez, P., Cerdeira, A., Almeida, F., Matos, T., Reis, J. (2009). *Modeling wine preferences by data mining from physicochemical properties.* Decision Support Systems, 47(4), 547–553. Dataset: UCI Machine Learning Repository, Wine Quality.
