# DoorDash ETA Prediction Pipeline

## Overview
This repository contains a full Machine Learning pipeline for predicting food delivery ETAs (Estimated Time of Arrival) based on historical DoorDash data. The notebook explores data preprocessing, feature engineering, handling data leakage, and model evaluation using business-centric metrics.

## Problem Statement
The goal is to predict the `delivery_duration_seconds` (the time between order creation and actual delivery) using various features such as store IDs, market IDs, dasher availability, and estimated driving durations.

## Dataset
- **historical_data.csv**: Contains nearly 200,000 records of historical DoorDash orders.

## Pipeline Phases & Key Learnings

### Phase 1: Data Integrity & Outlier Removal
- **Target Variable Creation**: Computed delivery duration in seconds.
- **Outlier Trap**: Extreme outliers (e.g., 98-day deliveries due to system glitches) heavily skew models optimizing for RMSE.
- **Solution**: Applied a 99th-percentile hard-cap to filter out system glitches while preserving legitimate edge cases. Imputed missing dasher metrics with medians.

### Phase 2: Missing Data & High-Cardinality
- **Missing Data**: Dropped rows with missing estimated driving durations to preserve high data integrity rather than imputing with fabricated noise.
- **High-Cardinality Trap**: One-Hot Encoding the `store_id` (representing thousands of restaurants) would crash RAM and create extreme sparsity.
- **Solution**: Used **Target Encoding**. Replaced raw `store_id`s with the historical average delivery duration for that specific store, transforming a useless identifier into a highly predictive feature.

### Phase 3: The Data Leakage Trap
- **Data Leakage**: Target encoding using the entire dataset before splitting causes the model to peek into the test set's future.
- **Solution**: Implemented strict Train/Test splitting *first*, calculated historical averages strictly on the training set, and mapped them to both sets.
- **Cold Start Problem**: Addressed new stores with no history by imputing their missing historical averages with the global train median.

### Phase 4: Modeling & Dimensionality
- **Curse of Dimensionality**: One-Hot Encoding categorical variables expanded the feature space to 96 sparse columns.
- **Algorithm Choice**: Selected **XGBoost (Gradient Boosting)** over Multiple Linear Regression because tree-based models handle sparse, non-linear matrices exceptionally well without suffering from multicollinearity.
- **Business Metrics**: Optimized the model using **RMSE** to penalize large errors, but translated the final evaluation into **MAE (Mean Absolute Error)** for business stakeholders (e.g., "Predictions are off by ~10 minutes on average").

### Phase 5: Asymmetric Business Reality & Feature Pruning
- **Asymmetric Objectives**: Identified that predicting an ETA 10 minutes *early* is worse for customer satisfaction than 10 minutes *late*. Discussed the need for a custom asymmetric objective function in production.
- **Feature Importance**: Extracted XGBoost feature importances to identify the most predictive features and isolate zero-variance noise columns for future pruning.
