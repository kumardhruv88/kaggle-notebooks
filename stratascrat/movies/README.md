# Rotten Tomatoes Rating Prediction

## Overview
This repository contains a full Machine Learning pipeline to predict whether a movie is labeled **'Rotten'**, **'Fresh'**, or **'Certified-Fresh'**. The prediction is based on both structured movie metadata and unstructured critic review texts.

## Problem Statement
This is a 3-class classification problem featuring:
- Mixed data types (structured tabular data + unstructured text)
- Class imbalance (with 'Certified-Fresh' being the minority)
- Data leakage risks (rating columns that directly define the label need to be dropped)
- Multi-value categorical columns (e.g., genres, directors, actors)
- Free text that requires NLP preprocessing

## Data Sources
The project utilizes the Clapper Massive Rotten Tomatoes Movies and Reviews datasets:
- `rotten_tomatoes_movies.csv`: Contains structured movie features and the target variables. (143,258 rows)
- `rotten_tomatoes_movie_reviews.csv`: Contains critic reviews used for NLP feature extraction. (1.44 million rows)

## Pipeline Steps
1. **Exploratory Data Analysis (EDA) & Data Audit**
   - Deep data audit for nulls and unique values.
   - Target engineering: Derived the 3 classes from the `tomatoMeter` continuous score (`< 60` = Rotten, `60-74` = Fresh, `>= 75` = Certified-Fresh).
   - Identification and removal of leakage columns (e.g., `tomatoMeter`, `audienceScore`).
2. **Feature Engineering**
   - Handling multi-value categorical columns like genres and writers.
   - Processing missing data (NaNs).
3. **Natural Language Processing (NLP)**
   - Preprocessing the critic reviews for text modeling.
4. **Modelling**
   - Training classification models on the combined structured and NLP features.
5. **Evaluation**
   - Evaluating model performance on predicting the three target classes.
