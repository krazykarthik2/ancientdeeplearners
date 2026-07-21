# FlowMatching3D — 3D Chromatin Reconstruction

## Overview

FlowMatching3D reconstructs 3D chromatin structure from DNA sequence using **Optimal Transport Conditional Flow Matching (OT-CFM)**. A denoising neural network learns to map Gaussian noise to valid 3D structures conditioned on sequence-derived bilinear logits.

### Two-tier architecture

1. **GEMINITiny** (5,061 params) — Binary E-P loop detection via Modern Hopfield Networks + Bilinear Interaction
2. **FlowMatching3D** (656,800 params) — Conditional flow matching for 3D coordinate generation

---

## Architecture

| Component | Parameters | Role |
|-----------|-----------|------|
| Contractive Autoencoder | 320 | Per-position embedding (5->16) |
| Conv1D (k=7, pad=3) | 3,584 | Sequence context per channel |
| ModernHopfieldNetwork (x2) | 1,056 | Motif detection (8 slots, 16 dim) |
| BilinearInteraction | 1,025 | Pairwise logits (dual W1+W2 + bias) |
| **GEMINITiny total** | **5,061** | |
| FlowMatching3D logit encoder | 9,632 | CNN + global pooling (32-dim cond) |
| FlowMatching3D velocity net | 647,136 | 4-layer MLP (256 hid) |
| FlowMatching3D time embed | 32 | Sinusoidal positional encoding |
| **FlowMatching3D total** | **656,800** | |
| **Grand total** | **661,861** | |

### Pipeline

```
Sequence -> GEMINITiny -> bilinear logits -> FlowMatching3D (100-200 Euler steps) -> 3D coords
```

---

## Training Data

| Property | Detail |
|----------|--------|
| Source | Synthetic helix-with-loop |
| Window | 32 bp |
| Training | 5,000 samples (seed=42) |
| Validation | 200 samples (seed=100) |
| Input | One-hot DNA (5 channels: A, C, G, T, pos) x 32 |
| Target | 3D coordinates (32 x 3) |
| Contact model | P = 1 / (1 + (d/2)^3) |

---

## Performance

| Metric | PC-3D (old, 0 params) | FlowMatching3D (new, 657K params) |
|--------|----------------------|----------------------------------|
| Contact correlation | 0.812 +/- 0.117 | **0.977 +/- 0.032** |
| 3D reconstruction error | 0.358 | **0.101** |
| Near-perfect (>0.99 corr) | 0/200 | **85/200** |
| Best single sample | ~0.94 | **0.9993** |
| Worst single sample | ~0.40 | **0.803** |
| Wins (corr) | — | 198/200 |
| Wins (err) | — | 199/200 |

---

## Data Flow

```
DNA (5 channels x 32 bp)
  |
  v
CAE (5->16, per position)
  |
  v
Conv1D (k=7, enhancer + promoter)
  |
  v
MHN1_e -- MHN1_p + BilinearInteraction
  |
  +---> 2D logits (32 x 32)
  |
  v
FlowMatching3D (100-200 Euler steps)
  |
  v
3D coordinates (32 x 3)
```

---

## Usage

```bash
# Full pipeline: train GEMINITiny + train FlowMatching3D + evaluate
python run_flow3d.py
```

---

*Last updated: July 2026*
