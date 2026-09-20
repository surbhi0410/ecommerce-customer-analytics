---

## Task 2 – Additional Model Evaluation

The original analysis was extended with additional evaluation to examine
model stability, behaviour across different training-set sizes, and
subgroup-level performance.

### Stratified Cross-Validation

Five-fold stratified cross-validation was applied to Logistic Regression
and Random Forest for both prediction tasks.

| Model | F1-score | ROC-AUC | PR-AUC |
|---|---:|---:|---:|
| Shopping Logistic Regression | 0.607 ± 0.019 | 0.902 ± 0.007 | 0.651 ± 0.010 |
| Shopping Random Forest | 0.679 ± 0.006 | 0.930 ± 0.009 | 0.743 ± 0.018 |
| Churn Logistic Regression | 0.563 ± 0.015 | 0.881 ± 0.011 | 0.667 ± 0.023 |
| Churn Random Forest | 0.712 ± 0.038 | 0.939 ± 0.012 | 0.777 ± 0.028 |

The cross-validation results were used to examine whether the performance
observed in the original train-test split remained stable across different
data partitions.

### Learning-Curve Analysis

Learning curves were generated for Logistic Regression and Random Forest
using progressively larger training-set sizes.

For online purchase prediction, Logistic Regression reached training and
validation F1-scores of 0.619 and 0.607 at the largest training size.
Random Forest reached training and validation F1-scores of 0.942 and
0.678.

For customer churn prediction, Logistic Regression reached training and
validation F1-scores of 0.569 and 0.563. Random Forest reached training
and validation F1-scores of 0.948 and 0.712.

The learning curves provide additional evidence about model stability and
the gap between training and validation performance as the amount of
training data increases.

### Fairness and Subgroup Analysis

Fairlearn was used to examine the online purchase Random Forest model
across `VisitorType` groups.

`VisitorType` was treated as an operational subgroup for diagnostic
analysis rather than as a protected demographic attribute.

Overall held-out Random Forest performance:

| Metric | Value |
|---|---:|
| Accuracy | 0.898 |
| Precision | 0.674 |
| Recall / TPR | 0.678 |
| F1-score | 0.676 |
| Selection Rate | 0.157 |
| False Positive Rate | 0.061 |

Performance for the two major visitor groups:

| Visitor Type | Precision | Recall | F1 | Selection Rate | FPR |
|---|---:|---:|---:|---:|---:|
| New Visitor | 0.892 | 0.763 | 0.822 | 0.241 | 0.036 |
| Returning Visitor | 0.617 | 0.649 | 0.632 | 0.144 | 0.064 |

The held-out test set contained 344 new visitors and 2,084 returning
visitors. Their observed purchase rates were 0.282 and 0.137,
respectively.

The `Other` category contained only 13 observations and no positive
purchase cases. Its subgroup metrics were therefore interpreted
cautiously.

Because the dataset does not contain standard protected demographic
attributes, this analysis should not be interpreted as establishing
demographic fairness or unfairness.

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

---
