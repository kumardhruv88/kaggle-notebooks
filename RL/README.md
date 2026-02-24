# Reinforcement Learning for Neural Architecture Search

This folder contains a Jupyter Notebook (`reinforcement_learning.ipynb`) that
demonstrates how to use Reinforcement Learning (RL) to automate Neural
Architecture Search (NAS).

## Overview

The notebook implements an approach inspired by the paper _Neural Architecture
Search with Reinforcement Learning_ (Zoph & Le, ICLR 2017). It dynamically
builds and discovers optimal Multi-Layer Perceptron (MLP) architectures for the
MNIST dataset.

## Architecture and Approach

1. **Controller RNN (The RL Agent)**:
   - An LSTM network acts as the agent that sequentially samples architectural
     decisions as discrete tokens.
   - For each architecture, the agent makes exactly 4 decisions:
     - Number of layers (from `[2, 3, 4]`)
     - Hidden layer size (from `[32, 64, 128, 256]`)
     - Activation function (from `['relu', 'tanh', 'gelu', 'leaky_relu']`)
     - Dropout rate (from `[0.0, 0.1, 0.3, 0.5]`)

2. **Child Network (The Evaluator)**:
   - A dynamic PyTorch model builder takes the decisions sampled by the
     Controller RNN and constructs an MLP.
   - The child network is trained on a subset of the MNIST dataset for a few
     epochs to evaluate its performance.
   - The validation accuracy of this child network acts as the **Reward** signal
     for the RL Agent.

3. **Optimization Step**:
   - Using the REINFORCE algorithm (Policy Gradient), the Controller RNN updates
     its weights based on the reward received.
   - Architectures that yield higher validation accuracy result in a higher
     reward, making the agent more likely to sample similar architectures in
     future episodes.

## Key Features

- **Dynamic PyTorch Graph Building**: Dynamically instantiates PyTorch layers
  based on parametric inputs.
- **Custom Policy Gradient Implementation**: A custom implementation to update
  the LSTM controller.
- **Live Visualization**: Provides live plots during training with `matplotlib`
  to track:
  - Reward curves (Validation Accuracy) vs Baseline
  - Policy Loss
  - Parameter count vs. Accuracy
  - Architecture choices distribution
  - Details of the best architecture discovered so far

## Requirements

- `torch`
- `torchvision`
- `matplotlib`
- `numpy`
