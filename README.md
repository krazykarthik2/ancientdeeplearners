# FlowMatching3D: 3D Chromatin Reconstruction via Flow Matching

**Contact correlation: 0.977 | 3D error: 0.101 | Best sample: 0.9993**

Learned 3D chromatin reconstruction from DNA sequence using **Optimal Transport Conditional Flow Matching (OT-CFM)**. The pipeline: GEMINITiny (5K params) extracts bilinear logits from DNA → FlowMatching3D (657K params, trained) generates 3D coordinates via learned denoising.

---

## Architecture

```
DNA Sequence (one-hot 5 x L)
      |
      v
GEMINITiny (5,061 params, trained for E-P loop detection)
  - Contractive Autoencoder (5->16 per position)
  - Conv1D context (kernel 7, enhancer + promoter channels)
  - 2x Modern Hopfield Networks (8 slots, learnable beta)
  - BilinearInteraction (dual W1+W2)
      |
      +---> MHN bilinear logits (L x L, continuous scores)
      |
      v
FlowMatching3D (656,800 params, trained with OT-CFM)
  - CNN logit encoder (32-dim conditioning vector)
  - Sinusoidal time embedding (32-dim)
  - MLP velocity field (256 hidden, 4 layers)
  - OT-CFM training: MSE on velocity field
  - Inference: 100-200 Euler steps from Gaussian noise
      |
      v
  3D coordinates (L x 3)
```

---

## Training

| Phase | Model | Data | Epochs | Optimizer |
|-------|-------|------|--------|-----------|
| 1 | GEMINITiny | Synthetic (5K) | 80 | AdamW (1e-3) |
| 2 | FlowMatching3D | Synthetic (5K) | 500 | AdamW (1e-3, cosine) |

## Key Results

| Metric | PC-3D (0 params) | FlowMatching3D (657K params) |
|--------|------------------|------------------------------|
| Contact correlation | 0.812 +/- 0.117 | **0.977 +/- 0.032** |
| 3D reconstruction error | 0.358 | **0.101** |
| Near-perfect (>0.99) | 0/200 | **85/200** |
| Best sample | ~0.94 | **0.9993** |
| Wins vs PC-3D | — | 198/200 (corr), 199/200 (err) |

---

## Files

| File | Description |
|------|-------------|
| `model.py` | GEMINITiny — Modern Hopfield + Bilinear E-P detection |
| `flow3d.py` | FlowMatching3D — OT-CFM model + sampling |
| `utils.py` | Dataset, metrics, PC-3D baseline, helix generator |
| `run_flow3d.py` | Full pipeline: train GEMINITiny + train FlowMatching3D + eval |
| `dataset.py` | RealGenomicEPDataset (Ensembl Chromosome 22) |

---

## Usage

```bash
# Train GEMINITiny + FlowMatching3D + full evaluation
python run_flow3d.py
```

---

## Why Flow Matching?

- **Learned** from data, not hand-designed parametric helix
- **Fast inference**: 100-200 Euler steps (no gradient descent on latents)
- **Accurate**: 0.977 mean contact correlation, 0.9993 on best samples
- **Generative**: can sample multiple plausible structures per sequence
- **Scales**: more data + compute directly improves results
