# 🌫️ Beijing Air Quality — PM2.5 Prediction with ANN

> Predicting fine particulate matter concentration using a deep Artificial Neural Network
> trained on 4 years of real sensor data from 12 monitoring stations across Beijing.

---

## 📌 Problem Statement

Given hourly readings of weather conditions and pollutant gases from Beijing monitoring stations,
predict the **PM2.5 concentration** and classify the air quality into WHO-standard AQI categories.

This is a real-world regression problem where the output directly maps to
**how safe the air is to breathe at any given hour.**

---

## 📂 Dataset

| Property | Detail |
|---|---|
| Source | Beijing Multi-Site Air Quality Data — UCI / Kaggle |
| Stations | 12 monitoring stations across Beijing |
| Time Period | March 2013 — February 2017 |
| Raw Rows | 420,768 hourly readings |
| Features | PM10, SO2, NO2, CO, O3, TEMP, PRES, DEWP, RAIN, WSPM, wd |
| Target | PM2.5 (µg/m³) |

---

## 🧠 ANN Architecture
```
Input Layer        →    15 features
Hidden Layer 1     →    128 neurons  +  ReLU  +  BatchNorm  +  Dropout(0.2)
Hidden Layer 2     →    64 neurons   +  ReLU  +  BatchNorm  +  Dropout(0.2)
Hidden Layer 3     →    32 neurons   +  ReLU  +  BatchNorm
Output Layer       →    1 neuron     (PM2.5 value, no activation)

Total Parameters   →    13,313
Trainable Params   →    12,865
```

**Why this shape?**
Funnel structure — wide entry to capture complex feature interactions,
gradually compressed into a focused prediction. Same principle as
summarizing a book into a single paragraph.

---

## ⚙️ Full Pipeline
```
Raw CSV (12 files)
      ↓
Combine all stations → 420,768 rows
      ↓
Drop missing PM2.5 + Remove sensor outliers (PM2.5 > 500, CO > 5000)
      ↓
Label encode wind direction (wd) → 17 categories to 0–16
      ↓
Train-Test Split → 80% train / 20% test
      ↓
StandardScaler → normalize all 15 features to mean=0, std=1
      ↓
log1p transform on PM2.5 → compress skewed target distribution
      ↓
ANN Training → 50 epochs, batch=512, EarlyStopping(patience=5)
      ↓
Reverse log transform → real PM2.5 predictions
      ↓
Evaluate + AQI Classification
```

---

## 📊 Results

| Metric | Value |
|---|---|
| MAE | 11.78 µg/m³ |
| RMSE | 19.64 µg/m³ |
| R² Score | **0.9240** |
| Training Epochs | 50 |
| Best Epoch | 48 |

> R² of 0.924 means the model explains **92.4% of PM2.5 variance**
> on 77,951 completely unseen hourly readings.

---

## 🏙️ Sample Prediction
```
Input  →  January 15, 2017 | 9am | TEMP: 2°C | WSPM: 1.5 | CO: 1800 | NO2: 80

Output →  PM2.5: 108.78 µg/m³
          AQI Category: Unhealthy 🔴
```

Cold winter morning + low wind + high CO = pollutants trapped near ground.
The model captured this physical relationship purely from data.

---

## 📈 Training Behavior

- Loss dropped from ~3.5 → ~0.10 within first 5 epochs
- Train and validation curves ran parallel with no divergence → zero overfitting
- Validation loss slightly lower than training loss → strong generalization

---

## 🗂️ Key Design Decisions

| Decision | Reason |
|---|---|
| ReLU activation | Introduces non-linearity cheaply, works best for regression hidden layers |
| No output activation | PM2.5 is unbounded continuous value, sigmoid/ReLU would clip it |
| log1p on target | PM2.5 is right-skewed (mean 75, max 500) — log compresses the range for stable learning |
| StandardScaler | PRES (~1010) would dominate RAIN (~0.07) without scaling |
| BatchNormalization | Stabilizes learning between layers, allows higher learning rate |
| Dropout 0.2 | Randomly drops 20% of neurons per batch — prevents co-adaptation and overfitting |

---

*Built as part of a hands-on deep learning practice series.*
*Framework: TensorFlow 2.x + Keras*
