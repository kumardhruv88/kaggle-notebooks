# Credit Risk: Flagging Bad Loans

## Overview
This repository contains a comprehensive data science notebook (`flagging-loans.ipynb`) dedicated to predicting credit risk and flagging "bad loans." The analysis walks through a complete end-to-end machine learning pipeline, moving from raw data ingestion and cleaning all the way to business-driven threshold optimization.

The project emphasizes handling imbalanced datasets and translating standard machine learning metrics into actionable business logic.

## Key Phases

### 1. Data Ingestion & Sanity Checks
- Loads the credit risk dataset and standardizes the target variable to `bad_flag`.
- Inspects data types, missing values, and the target variable's class imbalance (e.g., 76% good loans vs 24% bad loans).

### 2. Exploratory Data Analysis (EDA)
- Identifies "hidden" missing values and data leakage risks.
- Drops non-predictive identifiers (like `customer_id`).
- Analyzes summary statistics, categorical distributions, and calculates numerical correlations against the target variable.

### 3. Feature Engineering & Preprocessing
- **One-Hot Encoding:** Converts categorical variables (`gender`, `employment_status`) into ML-ready numerical arrays, dropping the first category to avoid multicollinearity.
- **Stratified Split:** Splits the data into training and testing sets while preserving the original class imbalance ratio using `stratify=y`.
- **Feature Scaling:** Applies `StandardScaler` fitted strictly on the training data to prevent data leakage.

### 4. Baseline Modeling
- Establishes performance baselines using **Logistic Regression** and **Random Forest Classifiers**.
- Integrates native imbalance correction via `class_weight='balanced'`.
- Evaluates models using comprehensive classification reports and visualizes class separation capabilities through Yellowbrick ROC-AUC curves.

### 5. Hyperparameter Tuning with Optuna
- Optimizes the Logistic Regression model's regularization strength (`C`) logarithmically.
- Uses 5-fold cross-validation maximizing the **ROC-AUC score** to ensure robust generalization on imbalanced data.

### 6. Business Scenario Analysis (Threshold Tuning)
- Extracts continuous probability scores (`predict_proba`) rather than relying on standard static predictions (0.5 threshold).
- Simulates multiple risk tolerance scenarios (approval thresholds from 10% to 90%).
- Presents a final business table demonstrating the direct trade-off between the **Approval Rate** (volume of business) and the **Expected Default Rate** (portfolio risk).

## Tools & Libraries Used
- **Data Manipulation:** `pandas`, `numpy`
- **Machine Learning:** `scikit-learn` (Logistic Regression, Random Forest, Preprocessing, Metrics)
- **Hyperparameter Optimization:** `optuna`
- **Visualization:** `matplotlib`, `seaborn`, `yellowbrick` (ROC-AUC analysis)

## How to Run
Ensure you have the required dependencies installed. You can execute `flagging-loans.ipynb` sequentially in Jupyter Notebook, JupyterLab, or VS Code. The final output generates an intuitive business decision matrix for risk assessment.
