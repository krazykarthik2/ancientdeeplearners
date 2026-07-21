# Walkthrough: FlowMatching3D

This document details the architecture and validation of **FlowMatching3D**: a conditional flow matching model for 3D chromatin reconstruction from DNA sequence.

---

## 1. Architecture

Two trained components:

### GEMINITiny (5,061 params)
Binary enhancer-promoter loop detection:
- Contractive Autoencoder (5->16 dim per base)
- Conv1D context integration (kernel size 7)
- Two Modern Hopfield Networks (8 slots, learnable beta)
- Bilinear Interaction (dual W1 + W2, learnable bias -7)

### FlowMatching3D (656,800 params)
Conditional flow matching for 3D coordinate generation:
- **Logit encoder**: 3-layer CNN (8->16->32 channels) + global average pooling -> 64-dim conditioning vector
- **Time embedding**: Sinusoidal positional encoding (32-dim)
- **Velocity network**: 4-layer MLP (256 hidden dim) with SiLU activations
- **Training**: OT-CFM loss (MSE on learned velocity field v_t(x | cond))
- **Inference**: 100-200 Euler steps from N(0, I)

---

## 2. Training

Flow matching learns a vector field v_t(x | cond) that transports noise to data:

```
Training:
  t ~ Uniform[0, 1]
  noise ~ N(0, I)
  x_t = (1-t) * noise + t * coords
  v_target = coords - noise
  L = MSE(model(x_t, logits, t), v_target)

Inference:
  x_0 ~ N(0, I)
  for t = 0 to 1: x_{t+dt} = x_t + model(x_t, logits, t) * dt
```

The model is trained for 500 epochs on 5,000 synthetic samples with EMA (decay=0.9995) and cosine learning rate decay.

---

## 3. Validation Results

| Metric | PC-3D (old) | FlowMatching3D |
|--------|-------------|----------------|
| Contact correlation | 0.812 | **0.977** |
| 3D reconstruction error | 0.358 | **0.101** |
| Samples > 0.99 corr | 0/200 | **85/200** |
| Best sample | ~0.94 | **0.9993** |

The best sample achieves **0.9993 contact correlation** and the worst is **0.803** — even the worst case is competitive with PC-3D's average.

---

## 4. Why Flow Matching Over PC-3D?

| Aspect | PC-3D | FlowMatching3D |
|--------|-------|----------------|
| Parameters | 0 (pure optimization) | 656,800 (learned) |
| 3D prior | Hand-designed helix | Learned from data |
| Inference | 80-step GD on 4 latents | 100-200 Euler steps |
| Accuracy | 0.81 corr | **0.977 corr** |
| Generativity | Single output | Multiple samples |

Flow matching learns a **generative model** of 3D structure from data, enabling both higher accuracy and multi-sample diversity.

---

## 5. Visual Results

Key visualizations:
- `flow_v3_best_median_worst.png`: Contact maps + 3D overlay for best (0.999 corr), median, and worst samples
- `flow_v3_histograms.png`: Distribution of contact correlations and 3D errors across 200 validation samples
- `flow_v3_training.png`: Training loss curve and validation contact correlation over 500 epochs
