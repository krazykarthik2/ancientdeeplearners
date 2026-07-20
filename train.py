import os
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score

def train_step(model, batch, global_step, optimizer, criterion_bce, phase_thresholds, device, lambda_cae=1e-4, lambda_pc=0.1):
    """
    Executes a training step for the Dual Hopfield Architecture.
    """
    x = batch['sequence'].to(device)
    target_starts = batch['target_starts'].to(device)
    target_ends = batch['target_ends'].to(device)
    
    optimizer.zero_grad()
    
    # Forward pass through Dual Hopfield Architecture
    (logits_starts, probs_starts), (logits_ends, probs_ends) = model(x)
    
    # 1. BCE Loss for both Interaction heads
    loss_bce = criterion_bce(logits_starts, target_starts) + criterion_bce(logits_ends, target_ends)
    
    # 2. L1 Sparsity regularization
    loss_sparsity = 0.05 * (torch.mean(probs_starts) + torch.mean(probs_ends))
    
    # 3. Contractive Autoencoder penalty
    penalty = model.cae.contractive_penalty(x)
    
    # Total Loss
    loss = loss_bce + loss_sparsity + 0.1 * penalty
    
    loss.backward()
    optimizer.step()
    
    return loss.item(), "Dual Hopfield Training"

def evaluate_model(model, dataloader, device):
    """
    Evaluates the dual-head Hopfield model on the interaction prediction task.
    """
    model.eval()
    all_targets_starts = []
    all_preds_starts = []
    all_targets_ends = []
    all_preds_ends = []
    
    with torch.no_grad():
        for batch in dataloader:
            x = batch['sequence'].to(device)
            target_starts = batch['target_starts'].to(device)
            target_ends = batch['target_ends'].to(device)
            
            # Predict contact maps directly
            (logits_starts, probs_starts), (logits_ends, probs_ends) = model(x)
            
            # Flatten predictions and targets for metric computation
            all_targets_starts.extend(target_starts.view(-1).cpu().tolist())
            all_preds_starts.extend(probs_starts.view(-1).cpu().tolist())
            
            all_targets_ends.extend(target_ends.view(-1).cpu().tolist())
            all_preds_ends.extend(probs_ends.view(-1).cpu().tolist())
            
    try:
        auroc_starts = roc_auc_score(all_targets_starts, all_preds_starts)
        prauc_starts = average_precision_score(all_targets_starts, all_preds_starts)
        
        auroc_ends = roc_auc_score(all_targets_ends, all_preds_ends)
        prauc_ends = average_precision_score(all_targets_ends, all_preds_ends)
        
        # Average the metrics across both heads
        auroc = (auroc_starts + auroc_ends) / 2.0
        prauc = (prauc_starts + prauc_ends) / 2.0
    except Exception:
        auroc, prauc = 0.5, 0.5
        
    return {
        'auroc': auroc,
        'prauc': prauc,
        'pc_error': 0.0 # No longer applicable
    }
