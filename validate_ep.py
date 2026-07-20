import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from model import GEMINITiny
from dataset import RealGenomicEPDataset
from train import train_step, evaluate_model

device = 'cpu'
torch.manual_seed(42)
np.random.seed(42)

print("=== GEMINI-Tiny E-P Loop Validation ===")
print("Training model...")
train_ds = RealGenomicEPDataset(num_samples=1000, seed=42)
val_ds = RealGenomicEPDataset(num_samples=200, seed=100)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

model = GEMINITiny().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([60.0]))

pts = (2000, 5000)
gs = 0
while gs < 5000:
    for b in train_loader:
        if gs >= 5000: break
        train_step(model, b, gs, opt, criterion, pts, device)
        gs += 1

model.eval()
metrics = evaluate_model(model, val_loader, device)
print(f"Validation AUROC: {metrics['auroc']:.4f}, PR-AUC: {metrics['prauc']:.6f}")

print("\n=== E-P Loop Predictions ===")
test_ds = RealGenomicEPDataset(num_samples=10, seed=200)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()

for idx, batch in enumerate(test_loader):
    x = batch['sequence'].to(device)
    target = batch['target_starts'][0]
    
    with torch.no_grad():
        (logits, probs), _ = model(x, settle=False)
        p = probs[0].cpu().numpy()
    
    seq = x[0, :, :4].cpu().numpy()
    seq_str = ''.join(['ACGT'[np.argmax(seq[j])] if np.max(seq[j]) > 0.5 else 'N' for j in range(32)])
    
    rows, cols = torch.where(target > 0)
    true_pairs = [(r.item(), c.item()) for r, c in zip(rows, cols)]
    
    pred_mask = p > 0.5
    pred_rows, pred_cols = np.where(pred_mask)
    pred_pairs = list(zip(pred_rows.tolist(), pred_cols.tolist()))
    
    true_set = set(true_pairs)
    pred_set = set(pred_pairs)
    correct = pred_set & true_set
    fp = pred_set - true_set
    fn = true_set - pred_set
    
    ax = axes[idx]
    ax.imshow(p, cmap='Greys_r', vmin=0, vmax=1, interpolation='nearest')
    for r, c in correct:
        ax.plot(c, r, 'o', color='lime', markersize=8, markeredgewidth=2, markeredgecolor='white')
    for r, c in fp:
        ax.plot(c, r, 'x', color='red', markersize=8, markeredgewidth=2)
    for r, c in fn:
        ax.plot(c, r, 's', color='cyan', markersize=8, markeredgewidth=2, markerfacecolor='none')
    ax.set_title(f"Sample {idx+1}\nSeq: {seq_str[:16]}...", fontsize=9)
    ax.set_xlabel("Promoter position", fontsize=8)
    ax.set_ylabel("Enhancer position", fontsize=8)
    
    print(f"\nSample {idx+1}: {seq_str}")
    print(f"  True E-P pairs: {true_pairs}")
    print(f"  Predicted (>0.5): {pred_pairs}")
    print(f"  Correct: {len(correct)} | FP: {len(fp)} | FN: {len(fn)}")
    for r, c in pred_pairs:
        print(f"    ({r:2d},{c:2d}): p={p[r,c]:.4f} {'*** MATCH' if (r,c) in true_set else 'FP'}")

fig.suptitle("GEMINI-Tiny E-P Loop Validation\nGreen○=Correct | Red✗=FP | Cyan□=FN", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('ep_loop_validation.png', dpi=150, bbox_inches='tight')
print(f"\nSaved to ep_loop_validation.png")
plt.close()
