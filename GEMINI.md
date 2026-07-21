# GEMINI-X: FlowMatching3D for Genomic 3D Structure Prediction

GEMINI-X reconstructs 3D chromatin structure from DNA sequence using **Optimal Transport Conditional Flow Matching (OT-CFM)**. The architecture has two trained components: GEMINITiny (E-P loop detection) and FlowMatching3D (3D generation).

---

## 1. GEMINITiny (5,061 params)

### A. Contractive Autoencoder (CAE)
Per-position feature extraction: one-hot DNA (5 channels) -> 16-dim embedding.

### B. Modern Hopfield Network (MHN)
Associative memory for motif detection:
- **8 memory slots** per channel (enhancer, promoter)
- **Learnable beta**: temperature for slot retrieval
- **Learnable bias**: sparsity control

### C. Bilinear Interaction
Symmetric pair scoring with dual weight matrices:
```
logits_ij = (z_i^T W1 z_j + z_j^T W2 z_i) / 2 + bias
```
- Learnable bias (init -7) for background suppression

---

## 2. FlowMatching3D (656,800 params)

### Architecture
- **Logit encoder**: 3x Conv2D (8->16->32 channels) + global avg pool + linear to 64-dim
- **Time embedding**: Sinusoidal positional encoding (32-dim)
- **Velocity network**: 4-layer MLP (256 hidden) predicting v_t from (x_t, cond, t)

### Training (OT-CFM Loss)
```
t ~ Uniform[0, 1]
noise ~ N(0, I)
x_t = (1-t) * noise + t * coords
v_target = coords - noise
L = MSE(v_pred(t, x_t, logits), v_target)
```

### Inference (Euler Sampling)
```
x_0 ~ N(0, I)
for t = 0 to 1 step dt:
    v = model(x_t, logits, t)
    x_{t+dt} = x_t + v * dt
return x_1
```

---

## 3. Configuration

| Parameter | Value |
|-----------|-------|
| Input length | 32 bp |
| GEMINITiny params | 5,061 |
| FlowMatching3D params | 656,800 |
| Flow hidden dim | 256 |
| Training epochs | 500 |
| Training samples | 5,000 (synthetic) |
| Optimizer | AdamW (cosine decay) |
| EMA decay | 0.9995 |
| Inference steps | 100-200 |

---

## 4. Validation Results

| Test | Result |
|------|--------|
| FlowMatching3D contact corr | **0.977 +/- 0.032** |
| FlowMatching3D 3D error | **0.101** |
| Samples > 0.99 corr | 85/200 |
| Best single sample | **0.9993** |
| PC-3D comparison wins | 198/200 (corr) |
| Total params | **661,861** |
