import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from model import GEMINITiny
from dataset import RealGenomicEPDataset
from train import train_step

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}...")
    
    dataset = RealGenomicEPDataset(num_samples=400, seed=42)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
    
    model = GEMINITiny().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    pos_weight = torch.tensor([60.0], device=device)
    criterion_bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    phase_thresholds = (2000, 6000)
    total_steps = 6000
    global_step = 0
    
    while global_step < total_steps:
        for batch in loader:
            if global_step >= total_steps:
                break
            train_step(model, batch, global_step, optimizer, criterion_bce, phase_thresholds, device)
            if global_step % 500 == 0:
                print(f"Step {global_step}/{total_steps}")
            global_step += 1
    
    print("Finding positive samples...")
    model.eval()
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    loop_samples = []
    with torch.no_grad():
        for batch in test_loader:
            if torch.sum(batch['target_starts']) > 0:
                loop_samples.append(batch)
                if len(loop_samples) >= 3:
                    break
    while len(loop_samples) < 3:
        loop_samples.append(next(iter(test_loader)))
    
    fig, axes = plt.subplots(3, 4, figsize=(22, 15))
    
    for row_idx, sample in enumerate(loop_samples):
        x = sample['sequence'].to(device)
        
        with torch.no_grad():
            (logits_starts, probs_starts), ((p1e, p1e_p), (p1p, p1p_p)) = model(x, settle=False)
            
            one_hot = x[0, :, :4].cpu().numpy()
            true_map = sample['target_starts'][0].cpu().numpy()
            prob_map = probs_starts[0].detach().cpu().numpy()
            
            seq_str = ''.join(['ACGT'[np.argmax(one_hot[j])] if np.max(one_hot[j]) > 0.5 else 'N' for j in range(32)])
            
            mot_e = p1e_p[0].detach().cpu().numpy()
            mot_p = p1p_p[0].detach().cpu().numpy()
        
        # Panel 1: DNA sequence as text heatmap
        ax = axes[row_idx, 0]
        im = ax.imshow(one_hot.T, aspect='auto', cmap='Blues', interpolation='nearest')
        ax.set_yticks([0,1,2,3])
        ax.set_yticklabels(['A','C','G','T'])
        ax.set_xlabel('Position')
        ax.set_ylabel('Base')
        ax.set_title(f'Sample {row_idx+1}\n{seq_str}', fontsize=9)
        fig.colorbar(im, ax=ax)
        
        # Panel 2: 1D motif predictions
        ax = axes[row_idx, 1]
        x_pos = np.arange(32)
        ax.bar(x_pos-0.15, mot_e, width=0.3, alpha=0.7, color='blue', label='TATA')
        ax.bar(x_pos+0.15, mot_p, width=0.3, alpha=0.7, color='red', label='CCGC')
        ax.set_xlabel('Position')
        ax.set_ylabel('Motif Prob')
        ax.set_title('1D Motif Detection')
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1)
        
        # Panel 3: Ground truth (2 red dots)
        ax = axes[row_idx, 2]
        ax.imshow(true_map, cmap='Reds', interpolation='nearest', vmin=0, vmax=1)
        ax.set_xlabel('Position')
        ax.set_ylabel('Position')
        ax.set_title('Ground Truth\nLoop Positions')
        
        # Panel 4: Raw probability heatmap
        ax = axes[row_idx, 3]
        im = ax.imshow(prob_map, cmap='Greys_r', interpolation='nearest', vmin=0, vmax=1.0)
        ax.set_xlabel('Position')
        ax.set_ylabel('Position')
        ax.set_title(f'Prediction\nmean={np.mean(prob_map):.4f} max={np.max(prob_map):.4f}')
        fig.colorbar(im, ax=ax)
        
        true_rows, true_cols = np.where(true_map > 0)
        for tr, tc in zip(true_rows, true_cols):
            ax.plot(tc, tr, 'o', color='lime', markersize=8, markeredgecolor='white', markeredgewidth=1.5)
        
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), "loop_prediction_visualization.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()