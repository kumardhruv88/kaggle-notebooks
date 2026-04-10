# 🌊 Diffusion Model from Scratch on CelebA-HQ 256×256

> **Goal:** Build a complete diffusion model pipeline from first principles — implementing DDPM, DDIM fast sampling, and understanding how modern image generators like Stable Diffusion work under the hood.

---

## 📦 Dataset

| Property | Details |
|----------|---------|
| **Name** | CelebA-HQ (High Quality) |
| **Source** | Kaggle — `badasstechie/celebahq-resized-256x256` |
| **Total Images** | ~30,000 high-quality face images |
| **Raw Resolution** | 256 × 256 |
| **Training Resolution** | 64 × 64 (resized for Kaggle T4×2 feasibility) |
| **Architecture** | Fully scalable to 256×256 with more compute |
| **GPU** | NVIDIA Tesla T4 × 2 (Kaggle) |

---

## 🗺️ What This Notebook Covers

This notebook implements 4 progressive concepts — from core DDPM to the idea behind Stable Diffusion:

| Part | Concept | Paper |
|------|---------|-------|
| **1** | DDPM — Denoising Diffusion Probabilistic Models | Ho et al. 2020 |
| **2** | DDIM — Fast sampling in 50 steps instead of 1000 | Song et al. 2020 |
| **3** | Score Matching + Noise Conditioning | Song et al. 2021 |
| **4** | Classifier-Free Guidance (core idea of Stable Diffusion) | Ho & Salimans, 2022 |

---

## 🔢 Part 1 — The Forward Noising Process (DDPM Theory)

The core idea of diffusion: **gradually destroy an image by adding Gaussian noise** over T=1000 timesteps.

```
FORWARD PROCESS (Training signal — adds noise):

x_0 (clean image)
    │
    ▼  q(x_1 | x_0) = N(x_1; √(1-β_1)·x_0, β_1·I)
x_1 (slightly noisy)
    │
    ▼  q(x_2 | x_1) = N(x_2; √(1-β_2)·x_1, β_2·I)
x_2 (noisier)
    │
   ...
    │
    ▼
x_T (pure Gaussian noise ≈ N(0, I))
```

**Magic shortcut (reparameterization):**
```
q(x_t | x_0) = N(x_t; √ᾱ_t · x_0, (1 - ᾱ_t) · I)

So:  x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε,   where ε ~ N(0, I)

This means we can jump to ANY timestep t in ONE STEP — no iteration needed!
```

### Noise Schedule (Linear)
```
β_1 = 0.0001  →  β_1000 = 0.02  (linearly increasing)

ᾱ_t = cumulative product of (1 - β) from 1 to t

ᾱ_0 ≈ 1.0  (image preserved)
ᾱ_999 ≈ 0.0 (pure noise)
```

---

## 🏗️ Architecture — The UNet Backbone

The UNet is the neural network that **learns to reverse the noising process**.

It takes two inputs: **noisy image x_t** AND **timestep t**, and predicts the noise that was added.

```
INPUT: x_t (noisy image, 3×64×64)  +  t (integer timestep)
                     │                         │
                     │            SinusoidalTimeEmbedding(t)
                     │                   → MLP → time_emb
                     │
         ┌───────────▼──────────────┐
         │        ENCODER           │
         │                          │
         │  ResNetBlock + time_emb  │ → (64, 64, 64)    ───┐ skip
         │  ↓ Downsample ×2         │ → (64, 32, 32)    ───┤ connections
         │  ResNetBlock + time_emb  │ → (128, 32, 32)   ───┤
         │  ↓ Downsample ×2         │ → (128, 16, 16)   ───┤
         │  ResNetBlock + time_emb  │ → (256, 16, 16)   ───┤
         │  ↓ Downsample ×2         │ → (256, 8, 8)     ───┘
         └───────────────┬──────────┘
                         │
         ┌───────────────▼──────────────┐
         │          BOTTLENECK           │
         │  ResNetBlock + Attention      │ → (256, 8, 8)
         └───────────────┬──────────────┘
                         │
         ┌───────────────▼──────────────┐
         │           DECODER            │
         │                              │
         │  ↑ Upsample + skip concat    │ ← skip from encoder
         │  ResNetBlock + time_emb ×3   │
         └───────────────┬──────────────┘
                         │
                    Conv2d → 3×64×64
                         │
                         ▼
              Predicted Noise ε_θ(x_t, t)
```

### Time Embedding (Sinusoidal → MLP)
```
t (scalar integer)
    │
    ▼
SinusoidalEmbedding(dim=256)
    │  [sin, cos of different frequencies — like positional encoding in Transformers]
    ▼
Linear(256 → 256) → SiLU → Linear(256 → 256)
    │
    ▼ injected into every ResNet block via addition
```

### ResNet Block (with time injection)
```
x ──→ GroupNorm → SiLU → Conv2d → + time_emb → GroupNorm → SiLU → Conv2d ──→ output
│                                                                               │
└───────────────────────── skip connection (1×1 Conv if dim changes) ──────────┘
```

---

## ⚙️ Training Configuration

```
DDPM Training Pipeline:
───────────────────────

Dataset: CelebA-HQ 256×256 (resized to 64×64)
         │
         ▼
Transform: Resize(64) → RandomHFlip → ToTensor → Normalize(0.5, 0.5)
         │
         ▼
DataLoader: batch_size=64, shuffle=True
         │
         ▼
Model: UNet → DataParallel (2× T4 GPU)
         │
         ▼
DDPM: T=1000, β_start=1e-4, β_end=0.02 (linear schedule)
         │
         ▼
Optimizer: AdamW (lr=2e-4, weight_decay=1e-4)
         │
         ▼
Scheduler: CosineAnnealingLR (T_max=30, eta_min=1e-6)
         │
         ▼
Epochs: 30
         │
         ▼
Gradient Clipping: max_norm=1.0
         │
         ▼
Checkpoint: every 5 epochs → /kaggle/working/checkpoints/
Samples: every epoch → /kaggle/working/samples/
```

| Hyperparameter | Value |
|----------------|-------|
| **Epochs** | 30 |
| **Batch Size** | 64 |
| **Learning Rate** | 2e-4 |
| **Optimizer** | AdamW |
| **Scheduler** | CosineAnnealingLR |
| **Gradient Clip** | 1.0 |
| **Diffusion Steps T** | 1000 |
| **β range** | 1e-4 → 0.02 |

---

## 🔄 Training Loop — DDPM

```
For each epoch (30 total):
    For each batch:
        1. Sample random timesteps: t ~ Uniform(0, 1000)  [one per image]
        2. Sample noise:            ε ~ N(0, I)
        3. Corrupt image:           x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε
        4. UNet forward pass:       ε̂ = UNet(x_t, t)
        5. Loss:                    L = MSE(ε̂, ε)
        6. Backward + clip grads + AdamW step
    End batch
    Log avg loss + LR
    Generate sample images (saved every epoch)
    Save checkpoint (every 5 epochs)
End epoch
```

**Training objective:**
```
L = E[‖ε − ε_θ(√ᾱ_t · x_0 + √(1-ᾱ_t) · ε, t)‖²]

Predict the noise — not the image directly.
```

---

## 🔁 Reverse Process — DDPM Sampling (Inference)

```
Start: x_T ~ N(0, I)   [pure Gaussian noise]
         │
         ▼ for t = T, T-1, T-2, ..., 1:
    ε̂ = UNet(x_t, t)                      ← predict noise
    x_0_pred = (x_t - √(1-ᾱ_t)·ε̂) / √ᾱ_t  ← estimate clean image
    x_{t-1} = μ_θ(x_t, t) + σ_t · z       ← add controlled noise
         │
         ▼ (after 1000 steps)
    x_0  [generated face image]
```
- **1000 sequential UNet forward passes** — slow (~30–60s per image)

---

## ⚡ Part 2 — DDIM Fast Sampling (Song et al. 2020)

**Problem:** DDPM needs 1000 UNet passes per image = slow

**DDIM Fix:** Skip timesteps deterministically using only **50 steps**

```
DDIM Sampling (50 steps instead of 1000):

timesteps: [999, 949, 899, 849, ..., 49, 0]   ← jump in steps of 20

For each step:
    ε̂ = UNet(x_t, t)
    x̂_0 = (x_t - √(1-ᾱ_t)·ε̂) / √ᾱ_t    ← predict clean image
    dir_xt = √(1-ᾱ_{t-1} - σ²)·ε̂           ← direction toward x_t
    x_{t-1} = √ᾱ_{t-1}·x̂_0 + dir_xt + σ·noise

With η=0 (our setting): σ=0 → fully deterministic trajectory
```

### DDPM vs DDIM Speed Comparison

| Sampler | Steps | ~Time |  Mode |
|---------|-------|-------|-------|
| DDPM | 1000 | ~45s / image | Stochastic |
| DDIM (η=0) | 50 | ~3–4s / image | Deterministic |
| **Speedup** | **20×** | **~12–15×** | **Same model** |

> ✅ No retraining needed — DDIM uses the same trained UNet, just with smarter step selection.

---

## 🔀 DDIM Interpolation in Noise Space

```
Noise A (seed=42):  z_A ~ N(0, I)
Noise B (seed=99):  z_B ~ N(0, I)

Interpolation: z_α = (1-α)·z_A + α·z_B    for α ∈ [0.0, 0.14, ..., 1.0]
     │
     ▼ DDIM decode (50 steps) for each z_α
     │
     ▼
8 smoothly interpolated face images
```

- Unlike VAE (which interpolates in learned latent space), DDIM interpolates in **raw noise space**
- Works because DDIM is deterministic — same noise → same image

---

## 🔬 Concept Coverage Summary

```
Part 1: DDPM ──────────────────────────────────────────
  ✅ Forward noising: x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε
  ✅ UNet: predicts noise ε_θ(x_t, t)
  ✅ Training: random t, add noise, predict it back
  ✅ Sampling: 1000-step iterative reverse diffusion

Part 2: DDIM ───────────────────────────────────────────
  ✅ 50-step deterministic sampling (20× speedup)
  ✅ Same trained model — no retraining
  ✅ Noise-space interpolation
  ✅ DDPM vs DDIM visual comparison

Part 3: Score Matching ─────────────────────────────────
  ✅ Score function: ∇_x log p(x)
  ✅ UNet implicitly parameterizes score via noise prediction

Part 4: Classifier-Free Guidance ─────────────────────
  ✅ Conceptual understanding of CFG
  ✅ Foundation of Stable Diffusion / DALL-E 2
```

---

## 📊 Monitoring & Checkpoints

| Event | Frequency | Saved To |
|-------|-----------|---------|
| Loss logged | Every 50 batches | Console |
| Epoch summary | Every epoch | Console |
| Sample images | Every epoch | `/kaggle/working/samples/epoch_NNN.png` |
| Model checkpoint | Every 5 epochs | `/kaggle/working/checkpoints/ddpm_epoch_N.pth` |

---

## 🏁 Final Summary

| Statistic | Value |
|-----------|-------|
| **Dataset** | CelebA-HQ (256×256, resized to 64×64) |
| **Architecture** | UNet with ResNet blocks + Sinusoidal Time Embedding |
| **Framework** | PyTorch |
| **GPU** | 2× NVIDIA Tesla T4 (Kaggle) |
| **Diffusion Steps** | T = 1000 (train), 50 (DDIM inference) |
| **Epochs** | 30 |
| **Loss** | MSE between predicted and actual noise |
| **DDIM Speedup** | ~12–20× over DDPM |

---

## 🔑 Key Insights

| Insight | Explanation |
|---------|-------------|
| **Predict noise, not image** | Simpler optimization — UNet learns a smoother function |
| **Any timestep in one shot** | Reparameterization trick: x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε |
| **DDIM is deterministic** | Same starting noise → same output image (reproducible) |
| **VAE vs Diffusion** | VAE compresses to latent; Diffusion iteratively denoises |
| **Sharp vs blurry** | Diffusion >> VAE for image sharpness (no MSE averaging in pixel space) |

---

## 📁 File Structure

```
diffusion/
├── README.md              ← This file
└── diffusion (1).ipynb    ← Full Kaggle notebook
```

---

## 🚀 How to Run

1. Open on [Kaggle](https://www.kaggle.com/) with **GPU T4 × 2** enabled
2. Add dataset: `badasstechie/celebahq-resized-256x256`
3. Run all cells in order
4. Training takes ~30 epochs; samples saved every epoch
5. DDIM comparison runs after training

---

## 🧩 Connection to Larger Picture

```
CelebA VAE  ──→  CelebA-HQ DDPM/DDIM  ──→  Score Matching  ──→  CFG  ──→  Stable Diffusion
(this repo)        (this notebook)
```

This notebook is the foundation for understanding modern text-to-image generation systems (Stable Diffusion, DALL-E 2, Imagen).
