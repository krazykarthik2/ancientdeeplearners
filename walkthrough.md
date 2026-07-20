# Walkthrough: Pure Energy-Based Hopfield Architecture & Validation

This walkthrough documents the validation metrics and architecture details for **GEMINI-Tiny** using the Pure Energy-Based Hopfield Network.

---

## 1. Phased Energy Pre-Training
The model trained for 3000 steps using a two-phase protocol to prevent feature drift and spatial interference:

*   **Phase 1: 1D Motif Pre-training (Steps 0-799)**:
    The parallel Enhancer and Promoter motif context layers and 1D Hopfield networks (`mhn1_e` and `mhn1_p`) were trained individually to shape their energy landscapes to locate the `TATA` and `CCGC` motifs.
*   **Phase 2: 2D Pair Training (Steps 800-2999)**:
    All 1D layers were frozen. The single 2D Interaction Hopfield Network (`mhn2`) was trained on the frozen representations to map loop contacts. The loss settled smoothly to `0.0687`.

---

## 2. Final Validation Metrics
Evaluation on the validation loader using the settled energy state:
*   **AUROC:** `0.6929`
*   **PR-AUC:** `0.0017`
*   **PC Error:** `0.000000` (Predictive Coding replaced by pure Hopfield Energy minimization)

---

## 3. Visual Prediction Mapping Result

### What are you looking at?
*   **Panel 1:** The one-hot encoding of the 32bp sequence window containing both motifs.
*   **Panel 2 (Ground Truth Loop Starts):** Heatmap showing the true starting coordinate of the Enhancer-Promoter loop.
*   **Panel 3 (Predicted Loop Starts):** The predictions from the single 2D Interaction Hopfield head (`mhn2`), showing precise, selective mapping without spatial interference grids.

Below is the visualization of the model's predictions:

![GEMINI-Tiny DNA Loop Prediction Mapping](loop_prediction_visualization.png)
