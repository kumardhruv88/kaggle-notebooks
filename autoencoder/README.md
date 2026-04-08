# 🫀 ECG Anomaly Detection using Autoencoder

<p align="center">
  <img src="assets/evaluation_dashboard.png" width="900"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.0-EE4C2C?style=flat&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Kaggle-T4%20GPU-20BEFF?style=flat&logo=kaggle&logoColor=white"/>
  <img src="https://img.shields.io/badge/ROC--AUC-0.8842-2ecc71?style=flat"/>
  <img src="https://img.shields.io/badge/Accuracy-88%25-2ecc71?style=flat"/>
</p>

---

## 📌 Overview

An **unsupervised anomaly detection system** for ECG heartbeat signals built with a deep autoencoder in PyTorch.  
The model is trained **exclusively on normal heartbeat signals** and learns to reconstruct them with minimal error.  
At inference time, signals that deviate from normal patterns produce **high reconstruction error** — which serves directly as the anomaly score.

> **No labels are used during training.** The model discovers what "normal" looks like entirely on its own.

---

## 📊 Results

| Metric | Value |
|---|---|
| ROC-AUC Score | **0.8842** |
| Overall Accuracy | **88%** |
| Normal F1-Score | **0.93** |
| Anomaly F1-Score | **0.63** |
| Normal Mean Reconstruction Error | `0.000810` |
| Anomaly Mean Reconstruction Error | `0.003746` |
| Error Ratio (Anomaly / Normal) | **4.6×** |
| Labels used during training | **None (fully unsupervised)** |

---

## 💡 How It Works

```
Training  →  Normal beats only  →  Autoencoder learns normal ECG structure

                         ┌─────────────┐
Any ECG beat  ──────────▶│ Autoencoder │──────────▶  Reconstruction
                         └─────────────┘
                                │
                         Compute MSE error
                                │
               ┌────────────────┴────────────────┐
          MSE < threshold                   MSE > threshold
               │                                 │
           ✅ Normal                        🚨 Anomaly
```

The **threshold is set at the 95th percentile** of reconstruction errors on normal training samples.  
Anything the model reconstructs worse than 95% of known-normal beats gets flagged.

---

## 🏗️ Architecture

```
 Input      Encoder                    Bottleneck    Decoder                   Output
(186-dim)                              (32-dim)                               (186-dim)

  [186] ──▶ Linear(128) ──▶ ReLU ──▶ Linear(64) ──▶ ReLU ──▶ Linear(32) ──▶ ReLU
                                                                   │
                                          Sigmoid ◀── Linear(186) ◀── ReLU ◀── Linear(128) ◀── ReLU ◀── Linear(64)
```

| Component | Dimensions | Activation |
|---|---|---|
| Encoder | 186 → 128 → 64 → 32 | ReLU |
| Bottleneck | **32** | ReLU |
| Decoder | 32 → 64 → 128 → 186 | ReLU + Sigmoid |
| **Total Parameters** | **68,698** | — |

The bottleneck compresses a 186-point ECG signal into just **32 numbers**, forcing the network to capture only the essential structure of a normal heartbeat.

---

## 📁 Dataset

**MIT-BIH Arrhythmia Dataset** — [shayanfazeli/heartbeat on Kaggle](https://www.kaggle.com/datasets/shayanfazeli/heartbeat)

| Class | Label | Train Samples | Role |
|---|---|---|---|
| Normal | 0 | 72,471 | ✅ Training only |
| Supraventricular | 1 | 2,223 | 🔍 Test (never seen) |
| Ventricular | 2 | 5,788 | 🔍 Test (never seen) |
| Fusion | 3 | 641 | 🔍 Test (never seen) |
| Unknown | 4 | 6,431 | 🔍 Test (never seen) |

Each sample = one heartbeat represented as **186 time steps**, normalized to [0, 1].

---

## 🗂️ Notebook Structure

| Step | Description |
|---|---|
| 1 | Dataset loading and class distribution |
| 2 | ECG signal visualization per class |
| 3 | Preprocessing, normalization, DataLoader setup |
| 4 | Autoencoder architecture definition (PyTorch) |
| 5 | Training loop + loss convergence curve |
| 6 | Reconstruction error computation on test set |
| 7 | Threshold selection + evaluation dashboard (ROC, confusion matrix, graph network) |
| 8 | Original vs reconstructed signal comparison with error shading |
| 9 | Latent space visualization — PCA and t-SNE of 32D bottleneck |
| 10 | Per-class violin plots and detection rate analysis |
| 11 | Summary table and model export |

---

## 🔬 Key Findings

**Fusion beats are the hardest to detect (0.6% detection rate)**  
Fusion beats are hybrid signals — part normal, part ventricular. The autoencoder partially recognizes them as normal, producing low reconstruction error. This mirrors the difficulty cardiologists face when classifying Fusion beats clinically.

**Unknown beats are the easiest to detect (80.2% detection rate)**  
These signals are completely out-of-distribution. The model has no framework for reconstructing them, so errors spike — exactly the behaviour we want from an anomaly detector.

**The latent space is interpretable**  
t-SNE of the 32D bottleneck shows Normal beats forming structured ribbon-like manifolds while anomalous classes scatter at the edges — despite the model never receiving class labels.

**4.6× error gap between Normal and Anomaly**  
Normal mean MSE: `0.000810` vs Anomaly mean MSE: `0.003746`. A clear, reliable separation achieved with zero supervision.

---

## 🛠️ Tech Stack

| Library | Usage |
|---|---|
| `PyTorch` | Autoencoder model, training loop, GPU inference |
| `NumPy` | Array operations, threshold computation |
| `Pandas` | Data loading and class filtering |
| `Scikit-learn` | PCA, t-SNE, ROC-AUC, classification report |
| `Matplotlib` | All visualizations |
| `Seaborn` | Confusion matrix heatmap |
| `NetworkX` | Autoencoder graph network diagram |

---



