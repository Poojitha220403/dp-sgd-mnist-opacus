# Differentially Private Digit Recognition (DP-SGD on MNIST)

Trains a convolutional neural network on MNIST with **differentially private SGD** using **PyTorch + Opacus**. It reaches **90.0% test accuracy under a strong privacy budget of ε = 0.30 (δ = 10⁻⁵)**.

![CI](https://github.com/Poojitha220403/dp-sgd-mnist-opacus/actions/workflows/ci.yml/badge.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?logo=pytorch&logoColor=white)
![Opacus](https://img.shields.io/badge/Opacus-1.4-blue)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)

## Problem

Neural networks can memorize their training data, and membership-inference and model-inversion attacks can recover facts about individual training examples. **Differential privacy (DP)** bounds how much any single example can influence the trained model. This project measures the practical cost: how accurate can a digit classifier be when every gradient step is clipped and noised to give a formal (ε, δ) guarantee?

## Architecture

```mermaid
flowchart LR
    D[(MNIST<br/>60k train / 10k test)] --> L[Poisson-sampled<br/>DataLoader, batch 64]
    L --> M[CNN<br/>2× Conv-ReLU-MaxPool → FC 128 → FC 10]
    M --> G[Per-sample gradients]
    G --> C[Clip each to L2 norm ≤ 1.0]
    C --> N[Add Gaussian noise<br/>σ = 1.1 × clip norm]
    N --> O[Adam update]
    O --> M
    N --> A[PRV accountant<br/>tracks ε at δ = 1e-5]
    M --> T[Test accuracy per epoch]
```

## Tech stack

| Area | Tools |
|---|---|
| Deep learning | PyTorch 2.0, torchvision |
| Privacy | Opacus 1.4 (`PrivacyEngine`, `ModuleValidator`, PRV privacy accounting, Opacus's default) |
| Environment | Jupyter / Google Colab, CPU or CUDA |

## How it works

1. `ModuleValidator.fix()` makes the CNN DP-compatible (Opacus needs layers that support per-sample gradients).
2. `PrivacyEngine.make_private()` wraps the model, optimizer and data loader, with **noise multiplier 1.1** and **max per-sample gradient norm 1.0**.
3. Training runs for **5 epochs** with Adam (lr = 1e-3) and reports test accuracy after each epoch.
4. The engine's accountant reports the total privacy spent, **ε at δ = 10⁻⁵**.

## How to run

```bash
pip install -r requirements.txt
jupyter notebook dp_sgd_mnist_opacus.ipynb
```

MNIST downloads automatically to `./data/`. The notebook also runs as is on Google Colab.

## Testing

`pytest` runs on every push (CPU-only PyTorch, random tensors, no MNIST download):

- The CNN maps `(N, 1, 28, 28)` to `(N, 10)` logits and passes Opacus's strict `ModuleValidator`.
- The DP settings match the reported experiment (noise multiplier 1.1, clipping norm 1.0, δ = 1e-5).
- **Privacy accounting:** ε is finite and positive, and it grows as more DP-SGD steps are taken.
- **Clipping:** before noise is added, the summed gradient is bounded by `max_grad_norm × batch size`.
- The notebook uses the tested module, and its recorded results (90.02%, ε = 0.30) are intact.

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest -q
```

## Results

| Epoch | Train accuracy | Test accuracy |
|---|---|---|
| 1 | 62.81% | 80.68% |
| 2 | 83.60% | 86.29% |
| 3 | 86.75% | 88.58% |
| 4 | 88.35% | 88.90% |
| 5 | 88.85% | **90.02%** |

**Final privacy budget: ε = 0.30, δ = 10⁻⁵.** Single-digit ε values are generally considered meaningful protection, so ε = 0.30 is a strong guarantee.

Training accuracy is measured during the noisy DP-SGD updates, which is why it trails test accuracy. The remaining gap to non-private CNNs (~99% on MNIST) is the price of privacy. A larger budget, more epochs or bigger batches would close part of that gap.

## Screenshots

![Accuracy per epoch](images/accuracy_per_epoch.png)

## Repo structure

```
.
├── dp_sgd.py                   # CNN + DP-SGD setup (noise 1.1, clip 1.0)
├── dp_sgd_mnist_opacus.ipynb   # data, training loop, privacy accounting (with outputs)
├── tests/                      # pytest suite
├── images/accuracy_per_epoch.png
└── requirements.txt            # pinned: numpy 1.24.4, torch 2.0.1, torchvision 0.15.2, opacus 1.4.0
```
