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
    device = torch.device("cpu")
    print("Preparing data and initializing model for visualization...")
    
    # 1. Prepare small dataset
    dataset = RealGenomicEPDataset(num_samples=400, seed=42)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    model = GEMINITiny().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # Add positive weight (250x) to counter class imbalance (2 active cells vs 1022 inactive)
    pos_weight = torch.tensor([250.0], device=device)
    criterion_bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # 2. Train for 3000 steps to get perfectly clear, sparse predictions
    print("Training model for 3000 steps to learn motif-anchored loop mapping...")
    global_step = 0
    total_steps = 3000
    phase_thresholds = (50, 200)
    
    while global_step < total_steps:
        for batch in loader:
            if global_step >= total_steps:
                break
            train_step(
                model=model,
                batch=batch,
                global_step=global_step,
                optimizer=optimizer,
                criterion_bce=criterion_bce,
                phase_thresholds=phase_thresholds,
                device=device
            )
            global_step += 1
            
    print("Training finished. Extracting 3 loop-positive test samples...")
    
    # 3. Find 3 samples that contain active loops
    model.eval()
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    loop_samples = []
    with torch.no_grad():
        for batch in test_loader:
            if torch.sum(batch['target_starts']) > 0: # Check if there is an active loop
                loop_samples.append(batch)
                if len(loop_samples) >= 3:
                    break
                    
    # Fallback if we don't have 3 positive samples
    while len(loop_samples) < 3:
        loop_samples.append(next(iter(test_loader)))
        
    # 4. Predict and Plot all 3 samples in a 3x3 grid
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    
    for row_idx, sample in enumerate(loop_samples):
        x = sample['sequence'].to(device)
        
        # Predict
        with torch.no_grad():
            (logits_starts, probs_starts), (logits_ends, probs_ends) = model(x)
            
            # Detach tensors for plotting
            one_hot_seq = x[0, :, :4].cpu().numpy() # [32, 4]
            true_starts = sample['target_starts'][0].cpu().numpy()
            true_ends = sample['target_ends'][0].cpu().numpy()
            
            pred_starts = (probs_starts[0].detach().cpu().numpy() > 0.5).astype(float)
            pred_ends = (probs_ends[0].detach().cpu().numpy() > 0.5).astype(float)
            
            # Create RGB composite for Ground Truth
            # Red channel = Starts, Blue channel = Ends
            true_rgb = np.zeros((32, 32, 3), dtype=np.float32)
            true_rgb[:, :, 0] = true_starts
            true_rgb[:, :, 2] = true_ends
            
            # Create RGB composite for Prediction
            pred_rgb = np.zeros((32, 32, 3), dtype=np.float32)
            pred_rgb[:, :, 0] = pred_starts
            pred_rgb[:, :, 2] = pred_ends
            
        # Panel 1: Sequence One-hot (first 4 channels)
        im1 = axes[row_idx, 0].imshow(one_hot_seq.T, aspect='auto', cmap='Blues', interpolation='nearest')
        axes[row_idx, 0].set_title(f"Sample {row_idx+1} DNA Input sequence\n[A, C, G, T]")
        axes[row_idx, 0].set_xlabel("Genomic Sequence Index")
        axes[row_idx, 0].set_ylabel("Nucleotide Channels")
        axes[row_idx, 0].set_yticks([0, 1, 2, 3])
        axes[row_idx, 0].set_yticklabels(['A', 'C', 'G', 'T'])
        fig.colorbar(im1, ax=axes[row_idx, 0])
        
        # Panel 2: Ground Truth Interaction Map
        axes[row_idx, 1].imshow(true_rgb, interpolation='nearest')
        axes[row_idx, 1].set_title(f"Sample {row_idx+1} Ground-Truth\n(Red=Starts, Blue=Ends)")
        axes[row_idx, 1].set_xlabel("Genomic Sequence Coordinate")
        axes[row_idx, 1].set_ylabel("Genomic Sequence Coordinate")
        
        # Panel 3: Predicted Exact Binary Map
        axes[row_idx, 2].imshow(pred_rgb, interpolation='nearest')
        axes[row_idx, 2].set_title(f"Sample {row_idx+1} Binary Prediction\n(Red=Starts, Blue=Ends)")
        axes[row_idx, 2].set_xlabel("Genomic Sequence Coordinate")
        axes[row_idx, 2].set_ylabel("Genomic Sequence Coordinate")
        
    plt.tight_layout()
    
    # Save image to artifact folder
    output_path = r"C:\Users\karthikkrazy\.gemini\antigravity\brain\59e92e74-dab5-4ef0-beff-ca5eb4dcfde1\loop_prediction_visualization.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Successfully generated and saved visualization to {output_path}")

if __name__ == "__main__":
    main()
