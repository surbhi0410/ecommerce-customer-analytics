# E-commerce Customer Analytics Using Machine Learning

## Project Overview

This repository contains an end-to-end machine learning analysis of two
customer analytics problems in e-commerce:

1. Predicting whether an online browsing session will result in a purchase.
2. Predicting whether an existing customer is likely to churn.

The project was initially developed for Individual Task 1 and was subsequently
extended in Individual Task 2 with additional model evaluation using
cross-validation, learning curves, and subgroup analysis.

The complete workflow includes:

- Data inspection and preprocessing
- Exploratory analysis
- Logistic Regression
- Random Forest
- Model evaluation and comparison
- Confusion matrix analysis
- Feature importance
- Stratified cross-validation
- Learning-curve analysis
- Fairness-aware subgroup analysis
- Visualisation and interpretation of results

The work was completed as part of a Data Science assessment at RMIT University.

---

# Individual Task 1 – Customer Analytics Analysis

## Business Context

E-commerce organisations generate large volumes of customer, transactional,
and behavioural data. Analysing this information can help organisations
understand customer behaviour and support decisions relating to conversion,
retention, marketing, and customer experience.

This project considers two stages of the customer lifecycle:

- **Purchase intention:** identifying browsing sessions that are likely to
  result in a purchase.
- **Customer churn:** identifying existing customers who may discontinue
  their relationship with the business.

Together, these tasks provide complementary perspectives on customer
conversion and retention.

---

## Datasets

Two publicly available datasets were used.

### 1. Online Shoppers Purchasing Intention Dataset

**Source:** UCI Machine Learning Repository

**Dataset URL:**  
https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset

The dataset contains **12,330 online shopping sessions and 18 attributes**.
The target variable, `Revenue`, indicates whether an online session resulted
in a purchase.

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

The dataset was used to investigate whether browsing behaviour can help
predict purchase intention.

---

### 2. E-commerce Customer Churn Dataset

**Source:** Kaggle

**Dataset URL:**  
https://www.kaggle.com/datasets/samuelsemaya/e-commerce-customer-churn

The dataset used in the analysis contains **3,941 customer records and
11 attributes**. The target variable, `Churn`, indicates whether a customer
churned.

The analysed attributes include:

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

The dataset was used to investigate customer characteristics and behaviours
associated with churn.

> The original datasets are not redistributed through this repository.
> They should be accessed from their original public sources.

---

## Machine Learning Approach

Two supervised classification algorithms were applied to both datasets.

### Logistic Regression

Logistic Regression was used as a relatively simple classification model
and baseline for comparison.

### Random Forest

Random Forest was used to capture potentially non-linear relationships and
interactions between customer or behavioural attributes.

Both algorithms were evaluated using the same general evaluation framework
to support meaningful comparison.

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

Accuracy was not considered sufficient by itself because the positive classes
were less common than the negative classes.

**Precision** measures how many predicted positive cases are actually positive.

**Recall** measures how many actual positive cases are successfully identified.

**F1-score** balances precision and recall.

**ROC-AUC** evaluates the model's overall ability to distinguish between the
two classes across classification thresholds.

**PR-AUC** provides additional information about precision-recall performance
when the positive class is relatively uncommon.

---

## Task 1 Model Results

### Online Purchase Prediction

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.846 | 0.505 | **0.796** | 0.618 | 0.910 |
| Random Forest | **0.898** | **0.674** | 0.678 | **0.676** | **0.928** |

Random Forest achieved the higher overall F1-score and ROC-AUC for purchase
prediction.

Logistic Regression, however, achieved higher recall, identifying a larger
proportion of actual purchasers.

This demonstrates an important model-selection trade-off. Random Forest
provides a stronger balance between precision and recall, whereas Logistic
Regression may be relevant when missing a potential purchaser carries a
greater cost.

---

### Customer Churn Prediction

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.786 | 0.427 | **0.897** | 0.578 | 0.919 |
| Random Forest | **0.916** | **0.724** | 0.785 | **0.753** | **0.952** |

Random Forest achieved the higher overall F1-score and ROC-AUC.

Logistic Regression achieved higher recall and therefore detected a greater
proportion of actual churners.

The results demonstrate that model selection depends on the business objective
and the relative cost of false-positive and false-negative predictions.

---

## Confusion Matrix Insights

### Online Purchase Prediction

- Logistic Regression correctly identified **304 purchases** and missed
  **78 actual purchases**.
- Random Forest correctly identified **259 purchases** and missed
  **123 actual purchases**.
- Random Forest generated fewer false-positive purchase predictions than
  Logistic Regression.

### Customer Churn Prediction

- Logistic Regression correctly identified **96 churners** and missed
  **11 churners**.
- Random Forest correctly identified **84 churners** and missed
  **23 churners**.
- Logistic Regression generated **129 false positives**.
- Random Forest reduced this to **32 false positives**.

These results help explain why Logistic Regression achieved higher recall
while Random Forest produced stronger overall F1 and precision performance.

---

## Feature Importance and Predictive Insights

### Online Purchase Intention

Random Forest feature importance indicated that **Page Value** was the
strongest predictor of online purchase intention.

Other influential predictors included:

- Exit Rate
- Product Page Duration
- Product Pages Viewed
- Bounce Rate
- Administrative Page Duration
- Administrative Pages
- Month-related information

These findings indicate that behavioural engagement within an online session
contains useful predictive information about conversion.

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

These results indicate that customer relationship history and aspects of the
service experience contain useful predictive information for identifying
churn risk.

Feature importance represents **predictive association rather than
causation**. A feature being important to a model does not demonstrate that
changing that feature would directly cause a change in purchasing or churn
behaviour.

---

## Task 1 Overall Findings

The two datasets provide complementary perspectives on customer behaviour.

The online shoppers dataset focuses on **short-term browsing and conversion
behaviour**, whereas the churn dataset focuses on the **longer-term customer
relationship**.

Together, they represent different stages of the e-commerce customer
lifecycle:

**Visitor behaviour → Purchase conversion → Customer relationship → Retention/churn**

Random Forest achieved higher overall F1-score and ROC-AUC across both
prediction tasks.

Logistic Regression achieved higher recall for both positive classes.

Therefore, model selection should consider the business objective and the
relative cost of false-positive and false-negative predictions rather than
relying on a single evaluation metric.

---

# Individual Task 2 – Additional Model Evaluation

The original Task 1 methodology was extended with additional evaluation to
examine model stability, behaviour across different training-set sizes, and
subgroup-level performance.

The additional analysis included:

1. Five-fold stratified cross-validation
2. Learning-curve analysis
3. Fairness-aware subgroup analysis using Fairlearn

---

## Stratified Cross-Validation

Five-fold stratified cross-validation was applied to Logistic Regression and
Random Forest for both prediction tasks.

| Model | F1-score | ROC-AUC | PR-AUC |
|---|---:|---:|---:|
| Shopping Logistic Regression | 0.607 ± 0.019 | 0.902 ± 0.007 | 0.651 ± 0.010 |
| Shopping Random Forest | 0.679 ± 0.006 | 0.930 ± 0.009 | 0.743 ± 0.018 |
| Churn Logistic Regression | 0.563 ± 0.015 | 0.881 ± 0.011 | 0.667 ± 0.023 |
| Churn Random Forest | 0.712 ± 0.038 | 0.939 ± 0.012 | 0.777 ± 0.028 |

The cross-validation analysis was used to examine whether the performance
observed in the original train-test evaluation remained stable across
different partitions of the data.

The results were consistent with the broad pattern observed in the original
evaluation, while also providing information about variation between folds.

---

## Learning-Curve Analysis

Learning curves were generated for Logistic Regression and Random Forest using
progressively larger training-set sizes.

### Online Purchase Prediction

At the largest training-set size of **9,764 observations**:

| Model | Training F1 | Validation F1 |
|---|---:|---:|
| Logistic Regression | 0.619 | 0.607 |
| Random Forest | 0.942 | 0.678 |

The Logistic Regression training and validation scores became relatively
close at the largest training size.

Random Forest achieved a higher validation F1-score, while a larger gap
remained between its training and validation performance.

---

### Customer Churn Prediction

At the largest training-set size of **2,616 observations**:

| Model | Training F1 | Validation F1 |
|---|---:|---:|
| Logistic Regression | 0.569 | 0.563 |
| Random Forest | 0.948 | 0.712 |

The Logistic Regression training and validation scores were again relatively
close.

Random Forest achieved a higher validation F1-score, while retaining a larger
training-validation gap.

The learning curves provide additional evidence about how each model behaves
as the amount of available training data increases.

---

## Fairness and Subgroup Analysis

Fairlearn was used to examine the online purchase Random Forest model across
`VisitorType` groups.

`VisitorType` was used as an **operational subgroup for diagnostic analysis**.
It was not treated as a protected demographic attribute.

### Overall Held-Out Performance

| Metric | Value |
|---|---:|
| Accuracy | 0.898 |
| Precision | 0.674 |
| Recall / TPR | 0.678 |
| F1-score | 0.676 |
| Selection Rate | 0.157 |
| False Positive Rate | 0.061 |

---

### Performance by Visitor Type

| Visitor Type | Precision | Recall | F1 | Selection Rate | FPR |
|---|---:|---:|---:|---:|---:|
| New Visitor | 0.892 | 0.763 | 0.822 | 0.241 | 0.036 |
| Returning Visitor | 0.617 | 0.649 | 0.632 | 0.144 | 0.064 |

The held-out test partition contained:

- **344 New Visitors**, including 97 actual purchases
- **2,084 Returning Visitors**, including 285 actual purchases

The corresponding observed purchase rates were:

- New Visitor: **0.282**
- Returning Visitor: **0.137**

---

### Small Subgroup Limitation

The `Other` visitor category contained only **13 observations** and no actual
purchases.

Its precision, recall, and F1 values were therefore zero. Because this group
was very small and contained no positive cases, these metrics were interpreted
cautiously rather than being treated as strong evidence of model bias.

The dataset also does not contain standard protected demographic attributes.
Therefore, the `VisitorType` analysis should not be interpreted as establishing
demographic fairness or unfairness.

---

## Task 2 Evaluation Summary

The additional Task 2 analysis extends the original evaluation in three ways.

**Cross-validation** provides information about how model performance varies
across different partitions of the available data.

**Learning curves** provide evidence about model behaviour as training-set
size increases and make training-validation performance gaps visible.

**Subgroup analysis** demonstrates that aggregate model metrics may not fully
describe performance across operational groups.

These analyses complement the original train-test evaluation and provide a
broader view of model performance, stability, and limitations.

---

# Figures

The `figures` directory contains visual outputs generated during the Task 1
and Task 2 analyses.

These include:

### Task 1 Figures

- Model performance comparison
- Online purchase feature importance
- Customer churn feature importance
- Online purchase Logistic Regression confusion matrix
- Online purchase Random Forest confusion matrix
- Customer churn Logistic Regression confusion matrix
- Customer churn Random Forest confusion matrix

### Task 2 Figures

- Online purchase learning curves
- Customer churn learning curves
- Online purchase `VisitorType` subgroup performance

---

# Repository Structure

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
    ├── churn_rf_confusion_matrix.png
    ├── shopping_learning_curve.png
    ├── churn_learning_curve.png
    └── shopping_fairness_by_visitor_type.png
```

---

# Reproducibility

A fixed random state of **42** was used where appropriate.

The additional Task 2 evaluation used:

- Stratified train-test sampling
- Five-fold stratified cross-validation
- Multiple training-set sizes for learning-curve analysis
- Fairlearn for subgroup-level evaluation

The notebook contains the preprocessing, model development, evaluation, and
additional Task 2 analysis required to reproduce the reported results.

---

# Tools and Libraries

The analysis was implemented in Python using libraries including:

- pandas
- NumPy
- Matplotlib
- scikit-learn
- Fairlearn

---

# Limitations

The analysis has several limitations that should be considered when
interpreting the results:

- Both datasets are public historical datasets and may not represent current
  production e-commerce environments.
- Model performance may change when applied to new populations or data
  distributions.
- Feature importance indicates predictive association rather than causation.
- A single model metric is insufficient for determining suitability for a
  business application.
- The subgroup analysis uses `VisitorType`, which is not a protected
  demographic characteristic.
- The `Other` visitor subgroup contains very few observations.
- Production deployment would require additional validation, monitoring,
  privacy controls, security controls, and appropriate governance.

---

# Author

**Surbhi Chauhan**  
Master of Data Science  
RMIT University
