# Credit Risk Prediction

Machine learning project for predicting loan default using the Home Credit dataset.

The project explores the complete workflow of a binary classification problem under strong class imbalance, including exploratory data analysis, preprocessing, model development, evaluation, threshold analysis, and model interpretability with SHAP.

---

## Project Overview

Credit risk prediction is a binary classification problem in which the objective is to identify loan applications associated with a higher probability of default.

This project uses the Home Credit application dataset to develop and evaluate machine learning models capable of predicting loan default.

The workflow covers:

- Exploratory Data Analysis (EDA)
- Data quality and missing-value analysis
- Numerical and categorical feature preprocessing
- Logistic Regression as a baseline
- XGBoost classification
- ROC-AUC and PR-AUC evaluation
- Precision, Recall, and F1-score analysis
- Decision-threshold optimization
- SHAP-based model interpretation
- Individual prediction analysis

The main objective is not only to build a predictive model, but also to understand how the model makes its predictions.

---

## Dataset

The project uses the Home Credit dataset, containing:

- **307,511 loan applications**
- **121 explanatory variables**
- Numerical and categorical features
- A binary target variable:
  - `0` → No default
  - `1` → Default

The dataset presents a strong class imbalance, making metrics such as accuracy insufficient as the primary evaluation measure.

---

## Exploratory Data Analysis

The EDA investigates:

- Dataset structure and variable types
- Missing values
- Target distribution
- Numerical feature distributions
- Categorical variables
- Correlations with the target
- External credit indicators (`EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3`)
- Default rates across categorical groups

### Key EDA findings

Several variables contain substantial amounts of missing data, requiring an appropriate preprocessing strategy.

Financial variables such as:

- `AMT_INCOME_TOTAL`
- `AMT_CREDIT`
- `AMT_ANNUITY`
- `AMT_GOODS_PRICE`

show skewed distributions and potential extreme values.

The three `EXT_SOURCE` variables showed the strongest negative Pearson correlations with the target among the numerical variables:

| Feature | Correlation with TARGET |
|---|---:|
| `EXT_SOURCE_3` | -0.1789 |
| `EXT_SOURCE_2` | -0.1605 |
| `EXT_SOURCE_1` | -0.1553 |

These correlations indicate an association between higher external credit scores and lower observed default rates. Correlation should not be interpreted as causality.

---

## Data Preprocessing

Numerical and categorical variables are processed separately using a `ColumnTransformer`.

### Numerical features

- Median imputation
- Standardization using `StandardScaler`

### Categorical features

- Most-frequent-value imputation
- One-hot encoding
- `handle_unknown="ignore"`

The preprocessing pipeline is fitted exclusively on the training data and subsequently applied to the validation set to avoid data leakage.

---

## Machine Learning Models

Two classification models were evaluated.

### 1. Logistic Regression

Logistic Regression was used as a baseline model.

Because of the strong class imbalance, `class_weight="balanced"` was used during training.

### 2. XGBoost

XGBoost was used as a non-linear tree-based model capable of capturing more complex relationships and interactions between features.

---

## Model Performance

The models were evaluated on a held-out validation set using ROC-AUC, PR-AUC, Precision, Recall, and F1-score.

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7487 | 0.2280 | 0.1618 | 0.6790 | 0.2613 |
| XGBoost | **0.7616** | **0.2556** | **0.5792** | 0.0236 | 0.0453 |

XGBoost achieved higher ROC-AUC and PR-AUC than the Logistic Regression baseline, indicating stronger ranking performance on the validation set.

However, at the default probability threshold of `0.50`, its recall was very low. This illustrates why evaluating a single classification threshold is not sufficient for an imbalanced credit-risk problem.

---

## Threshold Analysis

The XGBoost probability threshold was varied to study the trade-off between precision and recall.

| Threshold | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 0.10 | 0.1925 | 0.5986 | 0.2913 |
| **0.15** | **0.2469** | **0.4272** | **0.3130** |
| 0.20 | 0.2989 | 0.3011 | 0.3000 |
| 0.25 | 0.3461 | 0.2087 | 0.2604 |
| 0.50 | 0.5792 | 0.0236 | 0.0453 |

Among the evaluated thresholds, `0.15` produced the highest F1-score (`0.3130`).

The threshold should not be considered universally optimal. In a real credit-risk application, the appropriate threshold would depend on the relative business costs of false positives and false negatives.

---

## Model Interpretability with SHAP

SHAP was used to understand how the XGBoost model contributes to individual predictions and which variables have the greatest influence on the model output.

### Top features by mean absolute SHAP value

| Feature | Mean |SHAP| |
|---|---:|
| `EXT_SOURCE_3` | 0.3758 |
| `EXT_SOURCE_2` | 0.3313 |
| `AMT_GOODS_PRICE` | 0.2356 |
| `AMT_CREDIT` | 0.1966 |
| `EXT_SOURCE_1` | 0.1519 |
| `DAYS_EMPLOYED` | 0.1040 |
| `DAYS_BIRTH` | 0.0981 |
| `AMT_ANNUITY` | 0.0966 |

The SHAP analysis confirms that the `EXT_SOURCE` variables are among the most influential features in the trained XGBoost model.

For example, higher values of the `EXT_SOURCE` variables generally contribute toward lower model-estimated default risk, while lower values tend to contribute toward higher risk.

SHAP values are interpreted as contributions to the model output and should not be interpreted as causal effects.

---

## Individual Prediction Analysis

The project also examines individual predictions using SHAP.

For a selected validation observation:

- Actual target: `0`
- Predicted probability: `0.0456`
- Predicted class at threshold `0.50`: `0`

SHAP values were used to identify the features that pushed the prediction toward higher or lower model output.

This provides a more transparent view of how individual predictions are constructed rather than relying only on global feature importance.

---

## Project Structure

```text
Credit-Risk-Prediction/
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Modeling.ipynb
│   └── 03_SHAP_Interpretation.ipynb
│
├── figures/
│   ├── eda_target_distribution.png
│   ├── eda_numeric_distributions.png
│   ├── eda_target_correlations.png
│   ├── eda_categorical_distributions.png
│   ├── eda_categorical_default_rates.png
│   ├── ...
│
├── models/
│   ├── xgboost_model.joblib
│   └── preprocessor.joblib
│
└── README.md
