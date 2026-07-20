import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import GEMINITiny
from dataset import RealGenomicEPDataset
from train import train_step, evaluate_model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=== GEMINI-Tiny Training Pipeline ===")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=====================================\n")

    try:
        print("Preparing Real Human Genomic datasets...")
        train_dataset = RealGenomicEPDataset(num_samples=2000, seed=42)
        val_dataset = RealGenomicEPDataset(num_samples=500, seed=100)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
        
        print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
        
        model = GEMINITiny().to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        
        pos_weight = torch.tensor([60.0], device=device)
        criterion_bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        phase_thresholds = (3000, 8000)
        total_steps = 8000
        global_step = 0
        best_auroc = 0.0
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)
        
        print(f"\nStarting Multi-phase Training Loop (Total {total_steps} steps)...")
        print("=" * 60)
        
        start_time = time.time()
        
        while global_step < total_steps:
            for batch in train_loader:
                if global_step >= total_steps:
                    break
                
                try:
                    loss_val, phase_name = train_step(
                        model=model,
                        batch=batch,
                        global_step=global_step,
                        optimizer=optimizer,
                        criterion_bce=criterion_bce,
                        phase_thresholds=phase_thresholds,
                        device=device
                    )
                    
                    if global_step >= phase_thresholds[0]:
                        scheduler.step()
                    
                    if global_step % 50 == 0 or global_step == total_steps - 1:
                        elapsed = time.time() - start_time
                        lr = optimizer.param_groups[0]['lr']
                        print(f"Step {global_step:4d}/{total_steps} | {phase_name:<25} | "
                              f"Loss: {loss_val:.6f} | LR: {lr:.6f} | "
                              f"Time: {elapsed/60:.1f}m")
                    
                    if global_step % 200 == 0 and global_step > 0:
                        metrics = evaluate_model(model, val_loader, device)
                        print(f"  >> Validation - AUROC: {metrics['auroc']:.4f}, PR-AUC: {metrics['prauc']:.6f}")
                        if metrics['auroc'] > best_auroc:
                            best_auroc = metrics['auroc']
                            model.eval()
                        model.train()
                        
                except Exception as e:
                    print(f"Error at step {global_step}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue
                
                global_step += 1
        
        elapsed = time.time() - start_time
        print(f"\nTraining completed in {elapsed/60:.1f}m")
        print(f"Best validation AUROC: {best_auroc:.4f}")
        
        print("\nEvaluating on validation set...")
        metrics = evaluate_model(model, val_loader, device)
        print("=== Final Validation Metrics ===")
        print(f"  AUROC:    {metrics['auroc']:.4f}")
        print(f"  PR-AUC:   {metrics['prauc']:.6f}")
        print(f"  PC Error: {metrics['pc_error']:.6f}")
        print("================================")
        
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()