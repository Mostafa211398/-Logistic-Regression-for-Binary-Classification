# Loan Approval Prediction — Binary Classification

A machine-learning notebook for predicting **loan approval status** from applicant and loan information. The project explores data preprocessing, regularized logistic regression, decision trees, random forests, and hyperparameter tuning, then compares the models using common classification metrics.

> **Project type:** Supervised machine learning · Binary classification · Tabular data  
> **Notebook:** [`Logistic Regression for Binary Classification.ipynb`](./Logistic%20Regression%20for%20Binary%20Classification.ipynb)

## Overview

The goal is to predict `Loan_Status`, a binary outcome indicating whether an application was approved (`Y`) or not (`N`). The notebook uses a loan-application dataset and compares a logistic regression baseline with tree-based classifiers.

### Highlights

- Preprocesses applicant information and loan-related attributes.
- Encodes categorical features with one-hot encoding.
- Removes incomplete rows for selected numerical fields.
- Uses a **70/30 stratified train/test split**.
- Explores logistic regression with **L1, L2, and ElasticNet** regularization.
- Compares logistic regression, decision tree, and random forest classifiers.
- Uses **5-fold `GridSearchCV`** to tune random forest hyperparameters.
- Reports **accuracy, precision, recall, and F1-score**.

## Dataset

The notebook loads a local CSV named:

```text
train_u6lujuX_CVtuZ9i (1).csv
```

The file is **required to rerun the notebook** but is not included in this README. Place it in the directory where the notebook runs, or change the `pd.read_csv(...)` path.

| Attribute | Value |
|---|---|
| Original size | 614 records, 13 columns |
| After dropping `Loan_ID` | 12 columns |
| After the notebook's missing-value filtering | 529 records |
| Target | `Loan_Status` (`Y` / `N`) |

**Input features** include gender, marital status, number of dependents, education, self-employment status, applicant income, co-applicant income, loan amount, loan term, credit history, and property area.

## Methodology

### 1. Data preparation

1. Load the CSV with pandas and inspect the dataset.
2. Drop `Loan_ID`, an identifier that is not used for prediction.
3. One-hot encode categorical variables, including the target.
4. One-hot encode `Dependents` separately.
5. Drop rows missing `Credit_History`, `Loan_Amount_Term`, or `LoanAmount`.
6. Separate features from the binary `Loan_Status_Y` target.
7. Split the data into training and testing sets (70% / 30%, stratified by target).

### 2. Model training and evaluation

The notebook experiments with:

- **Logistic Regression:** L1, L2, and ElasticNet variants.
- **Decision Tree:** including a constrained tree (`max_depth=5`, `min_samples_leaf=10`).
- **Random Forest:** a baseline estimator and a tuned estimator.

For the **final three-model comparison**, logistic regression inputs are standardized using `StandardScaler` fitted on the training partition; the decision tree and random forest are trained on the unscaled features.

### 3. Random forest hyperparameter tuning

A five-fold grid search evaluates 16 hyperparameter combinations (**80 fits total**) across these settings:

| Parameter | Values explored |
|---|---|
| `n_estimators` | 100, 200 |
| `max_depth` | `None`, 10 |
| `min_samples_split` | 2, 5 |
| `min_samples_leaf` | 1, 2 |

## Results

The following metrics are from the notebook's **final model-comparison output**:

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| **Logistic Regression** | **79.87%** | 77.86% | **99.09%** | **87.20%** |
| Decision Tree | 75.47% | **81.98%** | 82.73% | 82.35% |
| Random Forest | 78.62% | 77.94% | 96.36% | 86.18% |

**Additional random forest runs shown in the notebook:**

- Baseline random forest accuracy: **79.25%**.
- GridSearchCV-tuned random forest test accuracy: **78.62%**.

**Takeaway:** Logistic regression achieves the highest accuracy and F1-score in the final comparison, while the decision tree has the highest precision. Metrics reflect one saved notebook run and may vary when cells are rerun because the train/test split does not specify a fixed `random_state`.

## Tech stack

- Python
- pandas and NumPy
- scikit-learn
- Jupyter Notebook

## Getting started

### 1. Clone this repository

```bash
git clone https://github.com/Mostafa211398/-Logistic-Regression-for-Binary-Classification.git
cd -- -Logistic-Regression-for-Binary-Classification
```

### 2. Create an environment (recommended)

```bash
python -m venv .venv
```

Activate it:

- **Windows (PowerShell):** `.venv\Scripts\Activate.ps1`
- **macOS / Linux:** `source .venv/bin/activate`

### 3. Install dependencies

```bash
pip install numpy pandas scikit-learn jupyter
```

### 4. Add the dataset

Place `train_u6lujuX_CVtuZ9i (1).csv` beside the notebook, or update the notebook's file path.

### 5. Launch Jupyter

```bash
jupyter notebook
```

Open **`Logistic Regression for Binary Classification.ipynb`** and run the cells in order.

## Repository structure

```text
.
├── Logistic Regression for Binary Classification.ipynb
├── README.md
└── train_u6lujuX_CVtuZ9i (1).csv  # Required locally; add only if permitted to share
```

## Notes and limitations

- Some notebook outputs were produced in an interactive session; one intermediate results cell refers to a `results` variable that is populated later in the visible notebook. A clean **Restart & Run All** may therefore require reorganizing that cell.
- A convergence warning appears in a saved output for logistic regression. Consider checking convergence after preprocessing and standardization.
- The original notebook does not set `random_state` in its train/test split, so reproducing exact saved metrics is not guaranteed.
- Results are from a small educational dataset, not a validated credit decision system. Real-world lending applications would require additional fairness, explainability, calibration, and regulatory review.

## Future improvements

- Set a fixed random seed for reproducibility.
- Organize preprocessing and modeling into scikit-learn pipelines.
- Add cross-validated model comparison and a confusion matrix.
- Investigate class imbalance, calibration, and fairness across applicant groups.
- Save the best model and build a small inference demo.

---

**Disclaimer:** This repository is an educational machine-learning project and is not intended for making actual lending decisions.
