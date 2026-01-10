# Credit Risk Prediction – Kaggle Competition

## 🏆 Competition Overview
- **Problem Type:** Binary Classification  
- **Objective:** Predict whether a customer will default on a loan  
- **Evaluation Metric:** AUC / Accuracy  

---

## 📊 Dataset Description
The dataset contains customer-level financial and demographic information such as:
- Age, income, employment status
- Credit history and loan details
- Target variable indicating default risk

---

## 🔍 Approach

### 1. Exploratory Data Analysis (EDA)
- Target class imbalance analysis
- Feature distribution and outlier detection
- Correlation analysis

### 2. Data Preprocessing
- Missing value handling
- Encoding categorical features
- Feature scaling where required

### 3. Feature Engineering
- Credit utilization ratios
- Aggregated financial indicators

---

## 🤖 Models Used
- Logistic Regression (baseline)
- Random Forest
- XGBoost (final model)

---

## 📈 Results & Observations
- Tree-based models significantly outperformed linear models
- Feature engineering had a major impact on final performance
- Handling class imbalance improved model stability

---

## 🧠 Key Learnings
- Importance of EDA in understanding financial risk
- Model selection based on business context
- Evaluation metrics matter more than accuracy alone

---

## 📎 Notebook
➡️ `credit-risk-prediction.ipynb`
