# E-commerce Customer Analytics Using Machine Learning

## Project Overview

This project investigates two important customer analytics problems in e-commerce:

1. Predicting whether an online browsing session will result in a purchase.
2. Predicting whether an existing customer is likely to churn.

The analysis was developed as part of an individual Data Science assessment at RMIT University. The project demonstrates an end-to-end machine learning workflow including data inspection, preprocessing, model development, evaluation, comparison and interpretation.

## Datasets

Two publicly available datasets are used.

### 1. Online Shoppers Purchasing Intention Dataset

**Source:** UCI Machine Learning Repository

**URL:** https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset

The dataset contains 12,330 online shopping sessions and 18 attributes. The target variable, `Revenue`, indicates whether a session resulted in a purchase.

It contains behavioural and contextual attributes such as:

- Administrative page activity
- Product-related page activity
- Bounce rates
- Exit rates
- Page values
- Month
- Visitor type
- Weekend indicator

### 2. E-commerce Customer Churn Dataset

**Source:** Kaggle

**URL:** ADD_THE_EXACT_KAGGLE_DATASET_URL_USED_IN_THE_ASSIGNMENT

The dataset contains 3,941 customer records and 11 attributes. The target variable, `Churn`, indicates whether a customer churned.

The attributes include customer and service-related information such as:

- Tenure
- Preferred login device
- Warehouse-to-home distance
- Preferred payment mode
- Satisfaction score
- Number of registered devices
- Complaints
- Days since last order
- Cashback amount

The datasets are not redistributed in this repository. They should be accessed from their original public sources.

## Machine Learning Approach

Two classification algorithms were evaluated:

- Logistic Regression
- Random Forest

The datasets were processed using separate numerical and categorical preprocessing pipelines. Missing values were handled during preprocessing, categorical variables were encoded, and numerical variables were standardised where appropriate.

Stratified train-test splitting was used to preserve the class distributions. Class weighting was also applied because the target classes were imbalanced.

## Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC

Accuracy alone was not used to determine the best model because both datasets contain class imbalance.

Recall is particularly useful for determining how many actual purchasers or churners are identified, while precision measures the reliability of positive predictions. F1-score balances precision and recall, and ROC-AUC evaluates the overall ability of a model to distinguish between classes.

## Results

### Purchase Prediction

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.846 | 0.505 | 0.796 | 0.618 | 0.910 |
| Random Forest | 0.898 | 0.674 | 0.678 | 0.676 | 0.928 |

Random Forest achieved stronger overall F1 and ROC-AUC performance, while Logistic Regression achieved higher recall.

### Customer Churn Prediction

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.786 | 0.427 | 0.897 | 0.578 | 0.919 |
| Random Forest | 0.916 | 0.724 | 0.785 | 0.753 | 0.952 |

Random Forest again achieved the strongest overall F1 and ROC-AUC performance. Logistic Regression produced higher recall and therefore detected a greater proportion of actual churners.

## Key Insights

The two datasets provide complementary perspectives on the e-commerce customer lifecycle.

For purchase prediction, Page Value was the strongest Random Forest predictor. Exit Rate, Product Page Duration, Product Pages Viewed and Bounce Rate were also influential.

For customer churn, Tenure was the strongest predictor, followed by Cashback Amount, Warehouse Distance, Days Since Last Order, Number of Addresses and Satisfaction Score.

These feature-importance results represent predictive associations and should not be interpreted as causal relationships.

## Repository Structure

```text
ecommerce-customer-analytics/
│
├── README.md
├── ecommerce_customer_analytics.ipynb
│
└── figures/
    ├── model_comparison.png
    ├── shopping_feature_importance.png
    ├── churn_feature_importance.png
    ├── shopping_lr_confusion_matrix.png
    ├── shopping_rf_confusion_matrix.png
    ├── churn_lr_confusion_matrix.png
    └── churn_rf_confusion_matrix.png
