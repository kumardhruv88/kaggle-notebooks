# Air Quality Analysis: New York vs Bogota

## Overview
This repository contains an in-depth data analysis notebook (`stratascratch.ipynb`) that compares PM2.5 air pollution levels between **New York (Queens College)** and **Bogota (San Cristobal)** over a 7-month period (September 2016 – March 2017). 

The analysis focuses on cleaning raw environmental data, performing exploratory data analysis (EDA), detecting anomalies using rolling Z-scores, and evaluating air quality against World Health Organization (WHO) standards.

## Key Findings

Through rigorous statistical and visual analysis, the notebook reveals fundamentally different pollution profiles for the two cities:

* **Pollution Type:** 
  * **New York:** Experiences **episodic spikes** of pollution, with the vast majority of readings sitting at low baseline levels. Spikes are primarily driven by winter heating and weather inversions.
  * **Bogota:** Suffers from a **chronic baseline** of elevated pollution with very little variance.
* **WHO Violations:**
  * **New York:** In violation of the WHO 24-hour limit (15 µg/m³) only **3.2%** of the time. The annual mean is 5.86 µg/m³, which is 1.2x the WHO annual limit (5 µg/m³).
  * **Bogota:** In violation of the WHO 24-hour limit **99.7%** of the time. The annual mean is 24.00 µg/m³, which is an alarming 4.8x the WHO annual limit.
* **Traffic Impact:**
  * **New York:** Exhibits a flat diurnal pattern, indicating that PM2.5 is not primarily traffic-driven at this station.
  * **Bogota:** Shows a sharp morning peak (8-10 AM) that perfectly aligns with rush hour, revealing traffic as a primary driver of pollution.
* **Correlation:** The two cities' pollution levels move completely independently (Pearson r = 0.055), confirming different pollution sources, weather systems, and geographic influences. New York's PM2.5 levels only exceed Bogota's roughly **0.8%** of the time.

## Methodology & Pipeline

The notebook implements a complete data science pipeline for time-series sensor data:

1. **Data Acquisition:** 
   * NY Data: Real hourly PM2.5 data downloaded directly from the EPA AQS API (2016 & 2017 datasets).
   * Bogota Data: A synthetic dataset modeled to represent realistic high-altitude developing city characteristics with diurnal traffic signatures.
2. **Data Cleaning & Imputation:**
   * Handled invalid negative readings (instrument noise) by converting them to NaN.
   * Conducted a thorough gap analysis to differentiate between short interpolatable dropouts and long sensor outages (e.g., an 8-day winter outage in NY).
   * Applied time-based interpolation for gaps ≤ 24 hours while intentionally preserving long outages to avoid distorting rolling statistics.
3. **Exploratory Data Analysis (EDA):**
   * Multi-perspective visualisations including violin distributions, diurnal (hourly) patterns, weekly patterns, and monthly trends.
4. **WHO Standard Evaluation:**
   * Calculated 24-hour rolling means to accurately track WHO 24-hour limit violations.
5. **Anomaly Detection:**
   * Implemented a rolling 7-day Z-score method (threshold=3σ) to detect local anomalies. This adaptive baseline correctly identified NY's pollution as heavy-tailed (3.7x expected anomalies) while showing Bogota's pollution as highly consistent.

## Visualizations
The analysis generates several professional-grade visualizations, including:
- Missingness heatmaps and imputation comparisons.
- A 4-panel EDA deep dive (Violin, Diurnal, Weekly, Monthly).
- 24-hour rolling mean plots with shaded WHO violation periods.
- Rolling Z-score anomaly detection plots.
- A comprehensive final dashboard summarizing all key metrics and findings.

## How to Run

1. Ensure you have a Python environment with the following libraries installed:
   - `pandas`
   - `numpy`
   - `matplotlib`
   - `scipy`
2. Open and run `stratascratch.ipynb` in Jupyter Notebook, JupyterLab, or any compatible environment (like Kaggle or VS Code). The script automatically downloads the required EPA zip files, processes the data, and generates all visualizations inline and as saved PNG files.
