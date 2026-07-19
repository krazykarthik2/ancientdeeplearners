import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import GEMINITiny
from dataset import RealGenomicEPDataset
from train import train_step, evaluate_model

def run_pc_residual_proving_test(model, device):
    """
    Runs a test on a single sample to show that the PC state settling
    minimizes the residual predictive error over the T=10 steps.
    """
    model.eval()
    dataset = RealGenomicEPDataset(num_samples=1, seed=999)
    loader = DataLoader(dataset, batch_size=1)
    batch = next(iter(loader))
    x = batch['sequence'].to(device)
    
    # 1. Encoding & Lookup
    cae_out = model.cae(x)
    x_lookup, _ = model.mhn(cae_out)
    x_lookup_t = x_lookup.transpose(1, 2)
    
    # 2. Init states
    mu1, mu2 = model.pc.initialize_states(x_lookup_t)
    
    # Track error progression
    errors = []
    eta = 0.05
    optimizer = optim.SGD([mu1, mu2], lr=eta)
    
    for step in range(11):
        x_pred, mu1_pred = model.pc.forward_prediction(mu1, mu2)
        eps0 = x_lookup_t - x_pred
        eps1 = mu1 - mu1_pred
        eps2 = mu2
        
        loss_pc = torch.mean(eps0 ** 2) + torch.mean(eps1 ** 2) + 0.01 * torch.mean(eps2 ** 2)
        errors.append(loss_pc.item())
        
        if step < 10:
            optimizer.zero_grad()
            loss_pc.backward(retain_graph=True)
            optimizer.step()
            
    print("\n--- PC Residual Decay Verification Test ---")
    for i, err in enumerate(errors):
        print(f"  Step {i:2d} Error: {err:.6f}")
    improvement = (errors[0] - errors[-1]) / (errors[0] + 1e-9) * 100
    print(f"  PC Residual Decay Improvement: {improvement:.2f}%")
    print("-------------------------------------------\n")
    return errors

def main():
    device = torch.device("cpu")
    print("=== Configuration ===")
    print("Model: GEMINI-Tiny")
    print("Dataset: RealGenomicEPDataset (Human Chromosome 22)")
    print(f"Device: {device}")
    print("=====================\n")

    # 1. Datasets & Dataloaders
    print("Preparing Real Human Genomic datasets...")
    train_dataset = RealGenomicEPDataset(num_samples=1000, seed=42)
    val_dataset = RealGenomicEPDataset(num_samples=200, seed=100)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 2. Instantiate GEMINI-Tiny
    model = GEMINITiny().to(device)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion_bce = nn.BCEWithLogitsLoss()

    # 3. Training Loop (Multi-phase)
    # Thresholds: Warmup=50 steps, Settling=200 steps, Total=300 steps
    phase_thresholds = (50, 200)
    total_steps = 300
    global_step = 0
    
    print(f"Starting Multi-phase Training Loop (Total {total_steps} steps)...")
    
    while global_step < total_steps:
        for batch in train_loader:
            if global_step >= total_steps:
                break
                
            loss_val, phase_name = train_step(
                model=model,
                batch=batch,
                global_step=global_step,
                optimizer=optimizer,
                criterion_bce=criterion_bce,
                phase_thresholds=phase_thresholds
            )
            
            if global_step % 50 == 0 or global_step == total_steps - 1:
                print(f"Step {global_step:4d}/{total_steps} | {phase_name:<25} | Loss: {loss_val:.6f}")
                
            global_step += 1
            
    # 4. Proving Test
    run_pc_residual_proving_test(model, device)
    
    # 5. Evaluate on Validation Set
    print("Evaluating on validation set...")
    metrics = evaluate_model(model, val_loader, device)
    print("=== Validation Metrics ===")
    print(f"  AUROC:    {metrics['auroc']:.4f}")
    print(f"  PR-AUC:   {metrics['prauc']:.4f}")
    print(f"  PC Error: {metrics['pc_error']:.6f}")
    print("==========================")

if __name__ == "__main__":
    main()
