# GEMINI-X: Spatial-Interaction Foundation Model for Genomic Sequences

GEMINI-X integrates Contractive Autoencoding, Modern Hopfield associative retrieval, Hierarchical Predictive Coding, and Boltzmann energy-based coupling into a unified genomic model.

---

## 1. Core Architecture Modules

### A. Contractive Autoencoder (CAE)
Acts as the initial noise-robust embedding layer mapping one-hot DNA to a continuous manifold.
- **Input:** $x \in \{0, 1\}^{L \times 4}$
- **Encoder:** Linear layer $4 \rightarrow d_{model}$ with Sigmoid activation.
- **Objective Contractive Penalty:** $\lambda \| J_f(x) \|_F^2$ where $J_f$ is the Jacobian matrix of the encoder.

### B. Modern Hopfield Network (MHN)
Provides associative lookup mapping sequence vectors to motif memory slots.
- **Memory Matrix ($M$):** $N_{slots} \times d_{model}$ containing static motif vectors.
- **Query Projection ($W_q$):** $d_{model} \times d_{model}$
- **Retrieval Formula:** $\text{Retrieve}(Q, M) = M^T \cdot \text{softmax}(\beta \cdot M \cdot Q)$

### C. Hierarchical Predictive Coding Backbone (PC)
Estimates levels of spatial arrangement through predictive coding states ($\mu_l$).
- **Level 1 State ($\mu_1$):** Local feature states.
- **Level 2 State ($\mu_2$):** Motif combination states.
- **Generative weights ($W_l$):** Transposed 1D convolutions representing top-down generative predictions:
  $$\hat{\mu}_{l-1} = \text{ReLU}(W_l * \mu_l + b_l)$$
- **Error Neurons:** $\epsilon_{l-1} = \mu_{l-1} - \hat{\mu}_{l-1}$

### D. Boltzmann Interaction Fusion Head (BM)
Translates top-level representations to physically consistent symmetric 3D contact matrices.
- **Energy Function:**
  $$E(y) = -\sum_{i,j} y_{ij} (z_i^T J z_j)$$
  where $J$ is a learnable symmetric coupling matrix ($J = \frac{1}{2}(J + J^T)$).
- **Contact Probability:**
  $$P(y_{ij} = 1) = \sigma(z_i^T J z_j)$$

---

## 2. Validation Model: GEMINI-Tiny
Designed for verification on the "Synthetic Loop-32" validation task.
- **Input Length ($L$):** 32
- **CAE dimension:** $4 \rightarrow 16$
- **MHN Memory:** 8 slots of size 16
- **PC States:**
  - $\mu_1$: $16 \times 32$ (channels $\times$ length)
  - $\mu_2$: $8 \times 64$ (channels $\times$ length)
- **BM coupling dimension:** 8
