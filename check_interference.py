import torch
from model import GEMINITiny
from dataset import RealGenomicEPDataset
from torch.utils.data import DataLoader
from train import train_step

def main():
    device = torch.device("cpu")
    model = GEMINITiny().to(device)
    
    dataset = RealGenomicEPDataset(num_samples=400, seed=42)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    pos_weight = torch.tensor([60.0], device=device)
    criterion_bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    global_step = 0
    total_steps = 3000
    phase_thresholds = (1200, 3000)
    
    print("Training model...")
    while global_step < total_steps:
        for batch in loader:
            if global_step >= total_steps:
                break
            loss_val, phase_name = train_step(
                model=model,
                batch=batch,
                global_step=global_step,
                optimizer=optimizer,
                criterion_bce=criterion_bce,
                phase_thresholds=phase_thresholds,
                device=device
            )
            if global_step % 100 == 0:
                print(f"Step {global_step}: {phase_name} - Loss: {loss_val:.6f}")
            global_step += 1
            
    print("Done training. Checking positive sample predictions...")
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    for batch in test_loader:
        if torch.sum(batch['target_starts']) > 0:
            x = batch['sequence'].to(device)
            with torch.no_grad():
                (logits_starts, probs_starts), _ = model(x, steps=10, eta=0.05)
                
                print("\n--- Predicted Loop Starts (prob > 0.5) ---")
                pred_starts = (probs_starts[0].cpu().numpy() > 0.5).astype(float)
                for i in range(32):
                    for j in range(32):
                        if pred_starts[i, j] > 0:
                            print(f"  Active at ({i:2d}, {j:2d}) with prob {probs_starts[0, i, j]:.4f}")
                
                print("\n--- Ground Truth ---")
                true_starts = batch['target_starts'][0].cpu().numpy()
                for i in range(32):
                    for j in range(32):
                        if true_starts[i, j] > 0:
                            print(f"  True loop at ({i:2d}, {j:2d})")
            break

if __name__ == "__main__":
    main()