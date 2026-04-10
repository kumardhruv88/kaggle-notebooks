# 🧠 Variational Autoencoder (VAE) on CelebA

> **Goal:** Build a Variational Autoencoder from scratch to learn a compressed, continuous latent representation of 200k+ celebrity faces — and then generate & interpolate new ones.

---

## 📦 Dataset

| Property | Details |
|----------|---------|
| **Name** | CelebA (Large-scale Face Attributes) |
| **Source** | Kaggle — `jessicali9530/celeba-dataset` |
| **Total Images** | 202,599 face images |
| **Image Format** | JPEG (cropped & aligned) |
| **Resolution (raw)** | ~178×218 |
| **Resolution (used)** | 64×64 (resized + center-cropped) |
| **Normalization** | Pixel values scaled to [-1, 1] |

---

## 🏗️ Architecture Overview

```
INPUT IMAGE (3 × 64 × 64)
         │
         ▼
┌─────────────────────┐
│       ENCODER        │
│  Conv2d(3→32)  s=2  │  → (32, 32, 32)
│  Conv2d(32→64) s=2  │  → (64, 16, 16)
│  Conv2d(64→128) s=2 │  → (128, 8, 8)
│  Conv2d(128→256) s=2│  → (256, 4, 4)
│  Flatten → Linear   │
│  BatchNorm + ReLU   │
└────────┬────────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
 μ (128)  logσ² (128)
    │         │
    └────┬────┘
         │
         ▼ REPARAMETERIZATION TRICK
    z = μ + ε · σ     (ε ~ N(0, I))
         │
         ▼
┌─────────────────────┐
│       DECODER        │
│  Linear → Unflatten │  → (256, 4, 4)
│  ConvT(256→128) s=2 │  → (128, 8, 8)
│  ConvT(128→64)  s=2 │  → (64, 16, 16)
│  ConvT(64→32)   s=2 │  → (32, 32, 32)
│  ConvT(32→3)    s=2 │  → (3, 64, 64)
│  Tanh activation    │
└─────────────────────┘
         │
         ▼
RECONSTRUCTED IMAGE (3 × 64 × 64)
```

---

## 📐 Model Parameters

| Component | Details |
|-----------|---------|
| **Architecture** | Convolutional VAE (from scratch) |
| **Latent Dimension** | 128 |
| **Total Parameters** | ~2,957,251 |
| **Encoder** | 4× Conv2d + BN + ReLU → Linear(μ, logσ²) |
| **Decoder** | Linear → 4× ConvTranspose2d + Tanh |
| **Framework** | PyTorch |
| **GPU** | NVIDIA Tesla T4 × 2 (Kaggle) |

---

## 📉 Loss Function

The VAE optimizes two competing objectives simultaneously:

```
Total Loss = Reconstruction Loss + β × KL Divergence
```

### 1. Reconstruction Loss (MSE)
```
L_recon = MSE(x_reconstructed, x_original)  [summed, per-sample normalized]
```
- Measures pixel-level accuracy of the decoder output
- Forces the decoder to faithfully recreate input images

### 2. KL Divergence
```
L_KL = -0.5 × Σ(1 + logσ² - μ² - exp(logσ²))
```
- Regularizes the latent space to follow N(0, I)
- Forces the latent space to be continuous and smooth
- Makes random sampling → valid face generation possible

### β parameter
| β | Effect |
|---|--------|
| β = 1.0 | Standard VAE (this notebook) |
| β > 1.0 | β-VAE: disentangled representations |
| β < 1.0 | More reconstruction, less regularization |

---

## ⚙️ Training Configuration

```
Pipeline:
  Dataset loading
       │
       ▼
  Transforms: Resize(64) → CenterCrop(64) → ToTensor → Normalize(0.5)
       │
       ▼
  DataLoader: batch_size=128, shuffle=True, num_workers=2
       │
       ▼
  Model: VAE(latent_dim=128) → DataParallel (2× T4 GPU)
       │
       ▼
  Optimizer: Adam (lr=1e-3)
       │
       ▼
  Scheduler: StepLR (step_size=3, gamma=0.5)
       │
       ▼
  Training: 5 Epochs
       │
       ▼
  Gradient Clipping: max_norm=1.0
       │
       ▼
  Model saved → /kaggle/working/vae_celeba.pth
```

| Hyperparameter | Value |
|----------------|-------|
| **Epochs** | 5 |
| **Batch Size** | 128 |
| **Learning Rate** | 1e-3 |
| **Optimizer** | Adam |
| **Scheduler** | StepLR (step=3, γ=0.5) |
| **Gradient Clip** | max norm = 1.0 |
| **β (KL weight)** | 1.0 |

---

## 🔄 Training Loop (Step-by-Step)

```
For each epoch:
    For each batch:
        1. Load batch of images → GPU
        2. Forward pass: recon, μ, logσ² = VAE(images)
        3. Compute loss: L_recon + β * L_KL
        4. Backward pass + clip gradients
        5. Adam step
    End batch
    Log epoch summary (total, recon, KL losses)
    Scheduler step (LR decay)
End epoch
Save model checkpoint
```

---

## 📊 Experiments & Results

### Experiment 1 — Image Reconstruction

```
Input Image (real face) ──→ Encoder ──→ z (128-dim) ──→ Decoder ──→ Reconstructed
```
- Feed 8 real CelebA faces through the VAE
- Compare original vs. decoded output
- Result: **Blurry but identity-preserving** (expected for VAE — blurriness is a known limitation due to MSE loss)

---

### Experiment 2 — Random Face Generation

```
z ~ N(0, I)   [128-dim random vector]
     │
     ▼
  Decoder
     │
     ▼
New Generated Face (never seen in training)
```
- Sample 8 random latent vectors z ~ N(0, 1)
- Decode each to a novel face image
- Result: **Diverse, realistic-looking new faces** — proves the latent space is well-regularized

---

### Experiment 3 — Latent Space Interpolation

```
Real Face A  ──→ Encoder ──→ z_A (128)
                                │
                      Linear interpolation:
                      z_t = (1-t)*z_A + t*z_B
                      for t ∈ [0.0, 0.14, 0.28, ..., 1.0]
                                │
                              Decoder
                                │
Real Face B  ──→ Encoder ──→ z_B (128)        8 smoothly morphing faces
```
- Encode two real faces to get μ_A and μ_B
- Linearly walk from z_A → z_B across 8 steps
- Result: **Smooth morphing confirms continuous latent space** — this is only possible because KL divergence enforces structure

---

## 📈 Training Metrics (Tracked Per Epoch)

| Metric | Description |
|--------|-------------|
| **Total Loss** | Combined VAE objective (recon + β·KL) |
| **Reconstruction Loss** | MSE between input and output |
| **KL Divergence** | Distance from learned q(z|x) to N(0, I) |

Training curves plotted across all epochs — loss converges steadily.

---

## 🏁 Final Summary

| Statistic | Value |
|-----------|-------|
| Dataset | CelebA — 202,599 faces |
| Image Size | 64 × 64 × 3 |
| Latent Dimension | 128 |
| Model Parameters | ~2,957,251 |
| Epochs Trained | 5 |
| Optimizer | Adam |
| Hardware | 2× NVIDIA Tesla T4 (Kaggle) |
| Saved Model | `vae_celeba.pth` |

---

## 🔑 Key Observations

| Observation | Explanation |
|-------------|-------------|
| Reconstructions are blurry | VAE uses MSE loss → averages pixel distributions → blur |
| Generations look realistic | Well-regularized latent space via KL divergence |
| Interpolation is smooth | Continuous latent space — no gaps or jumps |
| VAE ≠ GAN | VAE optimizes a lower bound (ELBO); GAN uses adversarial training for sharper images |

---

## ➡️ Next Step
> **Diffusion Model (DDPM) on CelebA-HQ 256×256** — for sharper, higher-resolution face generation using score-matching and iterative denoising.

See the [`diffusion/`](../diffusion/) notebook.

---

## 📁 File Structure

```
vae/
├── README.md                    ← This file
└── variational-autoencor.ipynb  ← Full Kaggle notebook
```

---

## 🚀 How to Run

1. Open on [Kaggle](https://www.kaggle.com/) with GPU enabled (T4 ×2)
2. Add dataset: `jessicali9530/celeba-dataset`
3. Run all cells in order
4. Model saves to `/kaggle/working/vae_celeba.pth`
