# E-commerce Customer Analytics Using Machine Learning

## Project Overview

This project applies machine learning to two customer analytics problems in e-commerce:

1. Predicting whether an online browsing session will result in a purchase.
2. Predicting whether an existing customer is likely to churn.

The project demonstrates an end-to-end data science workflow including data inspection, preprocessing, machine learning model development, model evaluation, comparison, visualisation, and interpretation of results.

The analysis was developed as part of an individual Data Science assessment at RMIT University.

---

## Business Context

E-commerce organisations generate large volumes of customer, transactional, and behavioural data. Analysing this information can help organisations better understand customer behaviour and support decisions relating to conversion, retention, marketing, and customer experience.

This project considers two stages of the customer lifecycle:

- **Purchase intention:** identifying browsing sessions that are likely to result in a purchase.
- **Customer churn:** identifying existing customers who may discontinue their relationship with the business.

Together, these tasks provide complementary perspectives on customer acquisition/conversion and retention.

---

## Datasets

Two publicly available datasets are used in this project.

### 1. Online Shoppers Purchasing Intention Dataset

**Source:** UCI Machine Learning Repository

**Dataset URL:**  
https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset

The dataset contains **12,330 online shopping sessions and 18 attributes**. The target variable, `Revenue`, indicates whether an online session resulted in a purchase.

The attributes include behavioural and contextual information such as:

- Administrative page activity
- Administrative page duration
- Informational page activity
- Product-related page activity
- Product-related page duration
- Bounce rates
- Exit rates
- Page values
- Special day
- Month
- Operating system
- Browser
- Region
- Traffic type
- Visitor type
- Weekend indicator
- Revenue

The dataset is used to investigate whether browsing behaviour can help predict purchase intention.

---

### 2. E-commerce Customer Churn Dataset

**Source:** Kaggle

**Dataset URL:**  (https://www.kaggle.com/datasets/samuelsemaya/e-commerce-customer-churn?resource=download)

The dataset used in this analysis contains **3,941 customer records and 11 attributes**. The target variable, `Churn`, indicates whether a customer churned.

The analysed attributes include customer and service-related information such as:

- Tenure
- Preferred login device
- Warehouse-to-home distance
- Preferred payment mode
- Satisfaction score
- Number of registered devices
- Number of addresses
- Complaints
- Days since last order
- Cashback amount
- Churn

This dataset is used to investigate customer characteristics and behaviours associated with churn.

> The original datasets are not redistributed through this repository. They should be accessed from their original public sources.

---

## Machine Learning Approach

Two supervised classification algorithms were applied to both datasets:

### Logistic Regression

Logistic Regression was used as a classification model that provides a relatively simple baseline for comparison.

### Random Forest

Random Forest was used to capture potentially non-linear relationships and interactions between customer or behavioural attributes.

The two algorithms were evaluated using the same general evaluation framework to enable meaningful comparison.

---

## Data Preparation

The analysis included:

- Initial dataset inspection
- Examination of dataset dimensions and attributes
- Identification and treatment of missing values
- Duplicate checking
- Separation of predictor and target variables
- Numerical and categorical preprocessing
- Categorical encoding
- Numerical scaling where required
- Stratified train-test splitting
- Class weighting to address target-class imbalance

A fixed random state was used where appropriate to improve reproducibility.

---

## Evaluation Metrics

The models were evaluated using multiple classification metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC

Accuracy was not considered sufficient by itself because the positive classes were less common than the negative classes.

**Precision** measures how many predicted positive cases are actually positive.

**Recall** measures how many actual positive cases are successfully identified.

**F1-score** balances precision and recall.

**ROC-AUC** evaluates the model's overall ability to distinguish between the two classes across classification thresholds.

**PR-AUC** provides additional information about precision-recall performance when the positive class is relatively uncommon.

---

## Model Results

### Online Purchase Prediction

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.846 | 0.505 | **0.796** | 0.618 | 0.910 |
| Random Forest | **0.898** | **0.674** | 0.678 | **0.676** | **0.928** |

Random Forest achieved the strongest overall F1-score and ROC-AUC for purchase prediction.

However, Logistic Regression achieved higher recall, identifying a larger proportion of actual purchasers.

This demonstrates an important model-selection trade-off. Random Forest provides a stronger overall balance, whereas Logistic Regression may be useful when missing a potential purchaser is considered particularly costly.

---

## Customer Churn Prediction

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.786 | 0.427 | **0.897** | 0.578 | 0.919 |
| Random Forest | **0.916** | **0.724** | 0.785 | **0.753** | **0.952** |

Random Forest again achieved the strongest overall F1-score and ROC-AUC.

Logistic Regression achieved higher recall and therefore detected a greater proportion of actual churners.

The results suggest that the preferred model depends on the business objective. If failing to identify an at-risk customer is particularly costly, higher recall may be prioritised. If the business also needs to limit unnecessary retention interventions, the stronger precision-recall balance of Random Forest may be preferable.

---

## Confusion Matrix Insights

For online purchase prediction:

- Logistic Regression correctly identified **304 purchases** and missed **78** actual purchases.
- Random Forest correctly identified **259 purchases** and missed **123**.
- Random Forest substantially reduced false-positive purchase predictions compared with Logistic Regression.

For customer churn prediction:

- Logistic Regression correctly identified **96 churners** and missed **11**.
- Random Forest correctly identified **84 churners** and missed **23**.
- Logistic Regression generated **129 false positives**, whereas Random Forest reduced this to **32**.

These results help explain why Logistic Regression achieved higher recall while Random Forest produced stronger overall F1 and precision performance.

---

## Key Predictive Insights

### Online Purchase Intention

Random Forest feature importance indicated that **Page Value** was the strongest predictor of online purchase intention.

Other influential predictors included:

- Exit Rate
- Product Page Duration
- Product Pages Viewed
- Bounce Rate
- Administrative Page Duration
- Administrative Pages
- Month-related information

These findings suggest that behavioural engagement within an online session contains useful predictive information about conversion.

---

### Customer Churn

For customer churn, **Tenure** was the strongest Random Forest predictor.

Other influential features included:

- Cashback Amount
- Warehouse Distance
- Days Since Last Order
- Number of Addresses
- Satisfaction Score
- Complaint information
- Number of registered devices

These results suggest that customer relationship history and aspects of the service experience contain useful predictive information for identifying churn risk.

Feature importance represents **predictive association rather than causation**. A feature being important to the model does not demonstrate that changing that feature would directly cause a change in purchasing or churn behaviour.

---

## Overall Findings

Both datasets produced useful and complementary insights.

The online shoppers dataset focuses on **short-term browsing and conversion behaviour**, whereas the churn dataset focuses on the **longer-term customer relationship**.

Together, the datasets provide insights into two different stages of the e-commerce customer lifecycle:

**Visitor behaviour → Purchase conversion → Customer relationship → Retention/churn**

Random Forest provided the strongest overall F1-score and ROC-AUC across both prediction tasks.

Logistic Regression, however, achieved higher recall for both positive classes.

Therefore, model selection should depend on the business objective and the relative cost of false-positive and false-negative predictions rather than relying on a single evaluation metric.

---

## Figures

The `figures` directory contains the visual outputs generated during the analysis.

These include:

- Model performance comparison
- Online purchase feature importance
- Customer churn feature importance
- Online purchase Logistic Regression confusion matrix
- Online purchase Random Forest confusion matrix
- Customer churn Logistic Regression confusion matrix
- Customer churn Random Forest confusion matrix

---

## Repository Structure

```text
ecommerce-customer-analytics/
│
├── README.md
│
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
