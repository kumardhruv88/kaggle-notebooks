# Indian Real Estate Click Prediction Pipeline

## Overview
This repository contains an end-to-end Machine Learning pipeline to predict whether a real estate listing will be clicked by a user. The project utilizes the "Real Estate Data from 7 Indian Cities" dataset and covers comprehensive data auditing, structural cleaning, target engineering, and exploratory data analysis (EDA).

## Problem Statement
The primary objective is to build a classification model to predict user clicks on property listings. The raw dataset features:
- Zero missing values initially, but significant structural messiness.
- High-cardinality categorical features (e.g., 7050 unique locations).
- Embedded numerical data within text strings (e.g., Prices in "₹60.0 L" or "₹1.0 Cr").
- Extreme outliers or sentinel values masquerading as valid data.

## Dataset
- **Source**: Indian Real Estate - 7 Cities (Kaggle).
- **Structure**: 14,528 listings across 7 cities, initially with 9 columns.

## Pipeline Phases & Key Learnings

### Phase 1: Raw Data Audit & Structural Cleaning
- **Price Parsing**: Converted string prices (e.g., "₹1.5 Cr") into a consistent numeric Lakhs format.
- **Outlier Detection**: Identified `Price_per_SQFT` with values up to 999,000 (likely sentinel values or entry errors) and capped them using the 99th percentile (Winsorization) rather than dropping them.
- **Location Splitting**: Extracted the City and Locality from a combined Location string using the last-comma strategy.

### Phase 2: Feature Engineering & Imputation
- **BHK Extraction**: Extracted the number of bedrooms (BHK) from the `Property Title`. For listings missing the "BHK" keyword (e.g., Villas, Independent Houses), fallbacks to the `Description` column and median imputation were used to preserve non-apartment listings.
- **High-Cardinality Handling**: Planned Target Encoding for the highly granular `Locality` feature to avoid massive sparsity and prevent target leakage by encoding strictly within cross-validation folds.
- **Log Transformation**: Applied log transformations to `Price_Lakhs`, `Price_per_SQFT`, and `Total_Area` to handle right-skewness and extreme luxury outliers.

### Phase 3: Synthetic Click Target Engineering
- **Target Variable Creation**: Simulated a binary `clicked` target variable based on domain-informed business logic (affordability, BHK popularity, balcony presence, and city demand premiums).
- **Class Imbalance**: Resulted in an overall CTR of ~11% (8.1:1 class ratio).
- **Handling Imbalance Strategy**: Planned to use `scale_pos_weight` in XGBoost, PR-AUC for evaluation, and custom classification threshold tuning.

### Phase 4: Exploratory Data Analysis (EDA)
- Generated multiple visualizations to analyze class imbalance, CTR by BHK and City, Price distributions (log-scale), and feature relationships.
- Validated business assumptions, such as finding that highly expensive markets (Mumbai) had lower CTRs due to affordability constraints compared to active IT hubs (Bangalore).
