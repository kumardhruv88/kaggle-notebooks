Copy🫀 ECG Anomaly Detection using Autoencoder
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

📌 Overview
An unsupervised anomaly detection system for ECG heartbeat signals built with a deep autoencoder in PyTorch.
The model is trained exclusively on normal heartbeat signals and learns to reconstruct them with minimal error.
At inference time, signals that deviate from normal patterns produce high reconstruction error — which serves directly as the anomaly score.

No labels are used during training. The model discovers what "normal" looks like entirely on its own.


📊 Results
MetricValueROC-AUC Score0.8842Overall Accuracy88%Normal F1-Score0.93Anomaly F1-Score0.63Normal Mean Reconstruction Error0.000810Anomaly Mean Reconstruction Error0.003746Error Ratio (Anomaly / Normal)4.6×Labels used during trainingNone (fully unsupervised)

💡 How It Works
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
The threshold is set at the 95th percentile of reconstruction errors on normal training samples.
Anything the model reconstructs worse than 95% of known-normal beats gets flagged.

🏗️ Architecture
 Input      Encoder                    Bottleneck    Decoder                   Output
(186-dim)                              (32-dim)                               (186-dim)

  [186] ──▶ Linear(128) ──▶ ReLU ──▶ Linear(64) ──▶ ReLU ──▶ Linear(32) ──▶ ReLU
                                                                   │
                                          Sigmoid ◀── Linear(186) ◀── ReLU ◀── Linear(128) ◀── ReLU ◀── Linear(64)
ComponentDimensionsActivationEncoder186 → 128 → 64 → 32ReLUBottleneck32ReLUDecoder32 → 64 → 128 → 186ReLU + SigmoidTotal Parameters68,698—
The bottleneck compresses a 186-point ECG signal into just 32 numbers, forcing the network to capture only the essential structure of a normal heartbeat.

📁 Dataset
MIT-BIH Arrhythmia Dataset — shayanfazeli/heartbeat on Kaggle
ClassLabelTrain SamplesRoleNormal072,471✅ Training onlySupraventricular12,223🔍 Test (never seen)Ventricular25,788🔍 Test (never seen)Fusion3641🔍 Test (never seen)Unknown46,431🔍 Test (never seen)
Each sample = one heartbeat represented as 186 time steps, normalized to [0, 1].

🗂️ Notebook Structure
StepDescription1Dataset loading and class distribution2ECG signal visualization per class3Preprocessing, normalization, DataLoader setup4Autoencoder architecture definition (PyTorch)5Training loop + loss convergence curve6Reconstruction error computation on test set7Threshold selection + evaluation dashboard (ROC, confusion matrix, graph network)8Original vs reconstructed signal comparison with error shading9Latent space visualization — PCA and t-SNE of 32D bottleneck10Per-class violin plots and detection rate analysis11Summary table and model export

📈 Visualizations
ECG Signal Shapes by Class
<p align="center">
  <img src="assets/ecg_classes.png" width="850"/>
</p>

Training Loss Convergence
<p align="center">
  <img src="assets/training_loss.png" width="650"/>
</p>

Loss dropped from 0.021 → 0.000778 over 50 epochs — a 96% reduction.


Original vs Reconstructed ECG Signals

The pink shaded region is the reconstruction error gap. Wider gap = higher anomaly score.

<p align="center">
  <img src="assets/reconstruction_comparison.png" width="850"/>
</p>

Full Evaluation Dashboard
<p align="center">
  <img src="assets/evaluation_dashboard.png" width="900"/>
</p>

Latent Space — PCA and t-SNE of 32D Bottleneck

No class labels used. Clusters emerge purely from reconstruction learning.

<p align="center">
  <img src="assets/latent_space.png" width="850"/>
</p>

Per-Class Anomaly Detection Rate
<p align="center">
  <img src="assets/per_class_analysis.png" width="850"/>
</p>

🔬 Key Findings
Fusion beats are the hardest to detect (0.6% detection rate)
Fusion beats are hybrid signals — part normal, part ventricular. The autoencoder partially recognizes them as normal, producing low reconstruction error. This mirrors the difficulty cardiologists face when classifying Fusion beats clinically.
Unknown beats are the easiest to detect (80.2% detection rate)
These signals are completely out-of-distribution. The model has no framework for reconstructing them, so errors spike — exactly the behaviour we want from an anomaly detector.
The latent space is interpretable
t-SNE of the 32D bottleneck shows Normal beats forming structured ribbon-like manifolds while anomalous classes scatter at the edges — despite the model never receiving class labels.
4.6× error gap between Normal and Anomaly
Normal mean MSE: 0.000810 vs Anomaly mean MSE: 0.003746. A clear, reliable separation achieved with zero supervision.

🛠️ Tech Stack
LibraryUsagePyTorchAutoencoder model, training loop, GPU inferenceNumPyArray operations, threshold computationPandasData loading and class filteringScikit-learnPCA, t-SNE, ROC-AUC, classification reportMatplotlibAll visualizationsSeabornConfusion matrix heatmapNetworkXAutoencoder graph network diagram