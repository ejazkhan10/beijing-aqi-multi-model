# CT046-3-M-AML — Applied Machine Learning Assignment
## Source Code: Air Quality Prediction and Analysis on Beijing Multi-Site Sensor Data

**Student:** [YOUR NAME] · **Student ID:** [YOUR ID]

---

## 1. Contents

| Notebook | Stage | Techniques |
|---|---|---|
| `01_Data_Preprocessing.ipynb` | Data loading, EDA, feature engineering, target derivation | Imputation, encoding, scaling, cyclical time features |
| `02_Multiclass_Classification.ipynb` | Six-class AQI categorisation | Naive Bayes, Logistic Regression, Decision Tree, SVM, MLP, Random Forest, Voting; **class balancing** (class weights, SMOTE, undersampling) |
| `03_Binary_Classification.ipynb` | Binary "unhealthy hour" detection | Same model family; ROC/AUC, precision–recall, decision-threshold analysis |
| `04_Regression.ipynb` | PM2.5 concentration prediction | MLR, Polynomial, Ridge, Lasso, ElasticNet, SVR, Decision Tree, Random Forest, MLP; VIF analysis, residual diagnostics |
| `05_KMeans_Clustering.ipynb` | Unsupervised station and daily-regime clustering | K-Means, elbow, silhouette, Davies–Bouldin, PCA |
| `06_Time_Series_Forecasting.ipynb` | Univariate daily PM2.5 forecasting | Decomposition, ADF test, ACF/PACF, exponential smoothing, ARIMA, SARIMA |
| `07_Final_Model_Export.ipynb` | Final model selection and persistence | `joblib` export, reload verification, inference demonstration |

---

## 2. Dataset

**Beijing Multi-Site Air Quality** — UCI Machine Learning Repository, Dataset ID 501.

- 420,768 hourly observations · 12 monitoring stations · March 2013 – February 2017
- Citation: Chen, S. (2017). *Beijing Multi-Site Air Quality* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5RK5G
- Licence: CC BY 4.0

Place the twelve station CSV files in a `data/` folder alongside the notebooks:

```
data/
  PRSA_Data_Aotizhongxin_20130301-20170228.csv
  PRSA_Data_Changping_20130301-20170228.csv
  ... (12 files total)
```

If `data/` is absent, Notebook 01 falls back to downloading the files from a public mirror, so it will still run given an internet connection.

---

## 3. Environment Setup

Python 3.11 or later. Using `uv`:

```bash
uv venv .venv
. .venv/bin/activate          # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

Or with pip:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

---

## 4. Execution Order

**Run `01_Data_Preprocessing.ipynb` first.** It writes `beijing_aqi_clean.parquet`, which every other notebook loads. Notebooks 02–07 can then be run in any order.

```
01  ──►  beijing_aqi_clean.parquet  ──►  02, 03, 04, 05, 06, 07
```

Notebook 07 additionally writes trained models to a `models/` folder.

---

## 5. Configuration Notes

**Sample size.** Notebooks 02–04 and 07 define `SAMPLE_SIZE` near the top (30,000 for classification, 20,000 for regression). Set it to `None` to train on the full ~412,000 rows. Two cautions:

- **RBF-kernel SVM should remain sampled regardless of hardware.** Its training cost scales approximately with the square of the sample size, so the full dataset is impractical by algorithm rather than by machine.
- Increasing the sample changes every reported metric, so the figures in the accompanying report would need regenerating.

**Reproducibility.** All stochastic operations use `random_state=42`. Re-running a notebook end to end reproduces the embedded outputs exactly.

**Runtime.** On a modern laptop each notebook completes in well under a minute, except Notebook 02, which takes a few minutes because of the cross-validated hyperparameter searches.

---

## 6. Key Methodological Points

These are the decisions most worth knowing when reading the code:

1. **Target leakage is prevented deliberately.** `AQI_Category` and `Unhealthy` are both derived from PM2.5, so **PM2.5 is excluded from the feature set for all classification models**. Including it would allow a model to invert the derivation and score near-perfectly without learning anything.

2. **All preprocessing lives inside `Pipeline` objects.** Imputation, scaling and encoding are therefore fitted on training folds only, keeping cross-validation and tuning free of leakage.

3. **Resampling uses `imblearn.Pipeline`.** SMOTE runs only on training folds; applying it before splitting would let synthetic minority points be interpolated from test-set neighbours.

4. **Time-series splitting is chronological, never random.** Shuffling a time series lets a model learn from the future to predict the past.

5. **SVR requires the target scaled, not just the features.** Its `epsilon` and `C` parameters are defined in the units of the response, so `TransformedTargetRegressor` is used. Without it, SVR underfits badly and appears — misleadingly — to be a poor model for this data.

---

## 7. Requirements

See `requirements.txt`. Core dependencies: `pandas`, `numpy`, `scikit-learn`, `imbalanced-learn`, `statsmodels`, `matplotlib`, `seaborn`, `pyarrow`, `joblib`.
