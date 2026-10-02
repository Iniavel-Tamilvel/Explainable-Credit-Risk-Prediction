# Explainable Credit Risk Prediction

## Project Overview

This project develops an end-to-end machine learning framework for predicting credit default risk using a synthetic credit-risk dataset.

The project compares Logistic Regression and Random Forest classification models and applies SHAP (SHapley Additive exPlanations) to interpret both global model behaviour and individual customer predictions.

The workflow covers:

- Exploratory Data Analysis (EDA)
- Data quality assessment
- Feature engineering
- Classification modelling
- Model comparison
- Feature importance analysis
- SHAP explainability
- Individual customer risk explanation
- Responsible AI considerations

> **Responsible-use note:** The dataset is synthetic and the models are for educational and portfolio purposes only. The results must not be used for real credit decisions.

---

## Business Problem

Credit-risk models can help financial institutions assess the likelihood of loan default. However, predictive performance alone is not sufficient for responsible decision-making.

A useful credit-risk analytics solution should also provide insight into:

- Which characteristics influence predictions
- How different models perform
- Which customers are identified as higher risk
- Why an individual prediction was generated
- What limitations and governance considerations apply

This project therefore combines predictive modelling with explainable AI.

---

## Dataset

The project uses a synthetic dataset containing:

- **10,000 credit applications**
- **14 original variables**
- **2,297 default cases**
- **22.97% default rate**

Original variables include:

- Age
- Income
- Employment years
- Credit score
- Debt-to-income ratio
- Loan amount
- Loan term
- Existing loans
- Late payments
- Credit history
- Home ownership
- Loan purpose
- Loan default target

The dataset was generated specifically for this portfolio project and does not represent real customer data.

---

## Exploratory Data Analysis

The exploratory analysis examined:

- Dataset structure
- Missing values
- Duplicate records
- Target-class distribution
- Home ownership
- Loan purpose
- Default rates across categories
- Numerical feature relationships with default

The analysis identified debt-to-income ratio and credit score among the variables showing stronger relationships with the default target.

---

## Feature Engineering

Several features were engineered to provide additional credit-risk signals:

- `log_income`
- `log_loan_amount`
- `loan_to_income_ratio`
- `loan_credit_score_ratio`
- `payment_burden_indicator`
- `credit_history_age_ratio`
- `late_payment_rate`
- `high_loan_burden`
- `high_dti`
- `low_credit_score`
- `has_late_payments`
- `multiple_existing_loans`
- `risk_indicator_count`

The `risk_indicator_count` showed an increasing observed default rate as the number of risk indicators increased within the synthetic dataset.

---

## Machine Learning Models

Two classification models were developed:

### Logistic Regression

A baseline linear classification model was used to provide an interpretable benchmark.

### Random Forest

A Random Forest classifier was developed using:

- 300 trees
- Maximum depth of 12
- Minimum samples per leaf of 5
- Balanced class weighting

The dataset was split using an 80/20 stratified train-test split.

---

## Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.782 | 0.653 | 0.102 | 0.177 | 0.706 |
| Random Forest | 0.742 | 0.426 | 0.366 | 0.394 | 0.691 |

The models demonstrate different performance characteristics.

Logistic Regression produced higher accuracy, precision and ROC-AUC in this experiment, while Random Forest produced higher recall and F1 score for the default class.

This highlights the trade-off between identifying more potential defaults and limiting false-positive predictions.

These results should be interpreted only within the context of the synthetic dataset.

---

## Random Forest Feature Importance

Random Forest feature importance identified the following variables among the most influential:

1. Debt-to-income ratio
2. Credit score
3. Employment years
4. Payment burden indicator
5. Loan-to-credit-score ratio
6. Log income
7. Loan-to-income ratio
8. Credit-history-to-age ratio
9. Income
10. Credit-history years

The complete feature-importance results are available in:

```text
outputs/random_forest_feature_importance.csv
