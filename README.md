# Analyzing Classification Algorithms for Subscription Predicition

**Machine Learning in Finance – University of Deusto**

## About the Project

This project was developed individually for the Machine Learning in Finance course at the University of Deusto.

The objective was to compare different classification algorithms to predict whether a bank customer would subscribe to a term deposit. The analysis uses a dataset of 3,000 customers, including demographic, financial, and marketing-related information.

The aim was to identify which model performed best while also considering how easily its predictions could be interpreted.

## Methodology

The analysis was conducted in R using five classification models:

- **Logistic Regression:** Estimates the probability of a customer subscribing to a term deposit.
- **CHAID:** Uses chi-squared tests to build decision trees.
- **CART:** Creates binary decision trees based on the Gini index.
- **C5.0:** Builds decision trees using entropy-based splitting.
- **Random Forest:** Combines multiple decision trees to improve prediction accuracy.

The models were evaluated using accuracy, sensitivity, specificity, and AUC.

## Results

The original report obtained the following results:

| Model | Accuracy | AUC |
|---|---|---|
| Logistic Regression | 79.50% | 0.8712 |
| CHAID | 77.33% | 0.7718 |
| CART | 76.17% | 0.7589 |
| C5.0 | 78.17% | 0.7792 |
| Random Forest | 89.37% | 0.9646 |

Random Forest showed the highest reported predictive performance, while Logistic Regression offered greater interpretability. Among the individual decision tree models, C5.0 achieved the best results.

The analysis highlighted the importance of balancing predictive accuracy with interpretability when using machine learning to support financial marketing decisions.

*The results reflect the original academic report. The Random Forest evaluation requires further validation because part of the analysis uses a model trained on the full dataset.*

## Files

- **Assignment 3(2).R:** R code used for data preparation, classification models, and performance evaluation.
- **Analyzing Classification Algorithms for Subscription Prediction (2).pdf:** Full report explaining the methodology, results, and conclusions.
- **t_01(1).rds:** Dataset used in the analysis.

## Tools and Libraries

- **R**
- **caret** – Model evaluation
- **pROC** – ROC curves and AUC
- **CHAID, rpart, C50** – Decision trees
- **randomForest** – Random Forest classification
- **ggplot2** – Data visualization

## How to Run

1. Download the R script and dataset.
2. Open the script in RStudio.
3. Install the required packages.
4. Rename the dataset to `t_01.rds` and keep it in the working directory.
5. Run the script to reproduce the analysis.
