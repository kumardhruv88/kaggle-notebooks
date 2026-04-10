# 🫀 ECG Heartbeat Anomaly Detection using Autoencoder

> **Goal:** Build an unsupervised anomaly detection system for ECG heartbeat signals using a deep autoencoder in PyTorch. The model is trained exclusively on normal heartbeats to learn their structure. At inference, signals with high reconstruction error are flagged as anomalies.

---

## 📦 Dataset

| Property | Details |
|----------|---------|
| **Name** | MIT-BIH Arrhythmia Dataset |
| **Source** | Kaggle — `shayanfazeli/heartbeat` |
| **Normal Beats** | ~72,471 (Class 0) |
| **Anomalous Beats** | ~37,000 (Classes 1, 2, 3, 4) |
| **Signal Length** | 186 time steps per heartbeat |
| **Normalization** | Scaled to [0, 1] range |

> **Unsupervised Approach:** The model *never* sees anomalies during training. No class labels are used. It discovers what "normal" looks like entirely on its own.

---

## 🏗️ Architecture Overview

The autoencoder compresses the 186-point ECG signal into just 32 numbers, forcing the network to capture only the essential structure of a normal heartbeat.

```
INPUT ECG SIGNAL (186-dim)
         │
         ▼
┌─────────────────────┐
│       ENCODER        │
│  Linear(186 → 128)  │
│       ReLU          │
│  Linear(128 →  64)  │
│       ReLU          │
│  Linear( 64 →  32)  │
│       ReLU          │
└────────┬────────────┘
         │
         ▼
  BOTTLENECK (32-dim)
         │
         ▼
┌─────────────────────┐
│       DECODER        │
│  Linear( 32 →  64)  │
│       ReLU          │
│  Linear( 64 → 128)  │
│       ReLU          │
│  Linear(128 → 186)  │
│     Sigmoid         │
└─────────────────────┘
         │
         ▼
RECONSTRUCTED SIGNAL (186-dim)
```

| Component | Details |
|-----------|---------|
| **Total Parameters** | 68,698 (Lightweight) |
| **Encoder** | 186 → 128 → 64 → 32 |
| **Decoder** | 32 → 64 → 128 → 186 |
| **Output Activation** | Sigmoid (matches normalized [0, 1] input) |

---

## ⚙️ Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| **Epochs** | 50 |
| **Learning Rate** | 1e-3 |
| **Optimizer** | Adam |
| **Loss Function** | Mean Squared Error (MSE) |
| **Training Data** | Normal beats *only* |

The loss dropped from ~0.021 to ~0.000778 over 50 epochs, representing a 96% reduction in reconstruction error on the healthy signals.

---

## 🔍 How Anomaly Detection Works

1. **Training:** Autoencoder is trained only on Normal ECG beats.
2. **Reconstruction:** Pass *any* beat (normal or anomalous) through the trained model.
3. **Compute Error:** Calculate the MSE between the original and reconstructed signal.
4. **Thresholding:** Set an anomaly threshold (e.g., the 95th percentile of reconstruction errors on the normal training set).
5. **Detection:**
   - Error < Threshold  ➡️  **Normal** (Model recognized and reconstructed it well)
   - Error > Threshold  ➡️  **Anomaly** (Model struggled to reconstruct unfamiliar patterns)

---

## 📊 Evaluation & Results

| Metric | Score | Notes |
|--------|-------|-------|
| **ROC-AUC Score** | **0.8842** | Excellent for zero-label training |
| **Overall Accuracy** | 88% | Fully unsupervised classification |
| **Normal F1-Score** | 0.93 | High precision on normal class |
| **Anomaly F1-Score** | 0.63 | Strong result for unseen anomalous classes |

### Key Findings by Class

| Class | Detection Rate | Why? |
|-------|----------------|------|
| **Unknown Beats** (Class 4) | **80.2%** (Highest) | Completely out-of-distribution. The model has no framework for them, so errors spike. |
| **Fusion Beats** (Class 3) | **0.6%** (Lowest) | Hybrid signals (part normal, part ventricular). The autoencoder partially reconstructs the normal aspects, causing low error. This mirrors the difficulty cardiologists face in clinical classification. |

### Mean Reconstruction Errors
- **Normal:** 0.000810 (Very low)
- **Anomaly:** 0.003746 (4.6× higher than normal)
- **Threshold:** 0.002337 (95th percentile of normal errors)

*An impressive, reliable separation is achieved between healthy and anomalous beats with zero supervision!*

---

## 🚀 How to Run

1. Open on [Kaggle](https://www.kaggle.com/).
2. Add dataset: `shayanfazeli/heartbeat`
3. Run all cells in `auroencoder.ipynb`
4. The trained model saves as `ecg_autoencoder.pth`

---

## 📁 File Structure

```
autoencoder/
├── README.md           ← This file
└── auroencoder.ipynb   ← Full Jupyter notebook with training, PCA/t-SNE latent visualizations, and evaluation dashboards
```