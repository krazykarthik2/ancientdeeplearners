import os
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score

def get_1d_motif_targets(x):
    # x: [B, L, 5]
    B, L, _ = x.shape
    device = x.device
    target_e = torch.zeros(B, L, device=device)
    target_p = torch.zeros(B, L, device=device)
    
    # Identify TATA (T=3, A=0, T=3, A=0) and CCGC (C=1, C=1, G=2, C=1) in 1D sequence
    for i in range(L - 3):
        # A=0, C=1, G=2, T=3
        is_tata = (x[:, i, 3] > 0.8) & (x[:, i+1, 0] > 0.8) & (x[:, i+2, 3] > 0.8) & (x[:, i+3, 0] > 0.8)
        is_ccgc = (x[:, i, 1] > 0.8) & (x[:, i+1, 1] > 0.8) & (x[:, i+2, 2] > 0.8) & (x[:, i+3, 1] > 0.8)
        
        target_e[:, i] = torch.where(is_tata, 1.0, 0.0)
        target_p[:, i] = torch.where(is_ccgc, 1.0, 0.0)
        
    return target_e, target_p

def train_step(model, batch, global_step, optimizer, criterion_bce, phase_thresholds, device, lambda_cae=1e-4, lambda_pc=0.1):
    """
    Executes a training step for the Pure Energy Hopfield Architecture.
    """
    x = batch['sequence'].to(device)
    target_starts = batch['target_starts'].to(device)
    
    # Extract 1D motif targets dynamically from sequence channels
    target_e, target_p = get_1d_motif_targets(x)
    
    optimizer.zero_grad()
    
    # Unpack thresholds (Phase 1: pretrain steps)
    motif_pretrain_steps, _ = phase_thresholds
    
    # Forward pass (only include interaction energy in Phase 2)
    include_interaction = (global_step >= motif_pretrain_steps)
    (logits_starts, probs_starts), (probs_1d_e, probs_1d_p) = model(x, include_interaction_energy=include_interaction)
    
    if global_step < motif_pretrain_steps:
        # --- PHASE 1: PRE-TRAIN 1D MOTIF Hopfields ---
        criterion_1d = nn.BCELoss()
        
        # Clamp probabilities to avoid any log(0) NaN issues
        probs_1d_e_clamped = torch.clamp(probs_1d_e, 1e-7, 1.0 - 1e-7)
        probs_1d_p_clamped = torch.clamp(probs_1d_p, 1e-7, 1.0 - 1e-7)
        
        loss_e = criterion_1d(probs_1d_e_clamped, target_e)
        loss_p = criterion_1d(probs_1d_p_clamped, target_p)
        
        penalty = model.cae.contractive_penalty(x)
        loss = loss_e + loss_p + 0.1 * penalty
        phase_name = "Phase 1: Motif Pre-training"
        
    else:
        # --- PHASE 2: TRAIN 2D PAIR Hopfield ---
        # Freeze 1D feature extractors and motif networks once pre-training finishes
        if global_step == motif_pretrain_steps:
            print("\n>>> Freezing 1D motif identification layers and starting Pair training...")
            for param in model.cae.parameters():
                param.requires_grad = False
            for param in model.sequence_context_e.parameters():
                param.requires_grad = False
            for param in model.sequence_context_p.parameters():
                param.requires_grad = False
            for param in model.mhn1_e.parameters():
                param.requires_grad = False
            for param in model.mhn1_p.parameters():
                param.requires_grad = False
                
        loss_bce = criterion_bce(logits_starts, target_starts)
        loss_sparsity = 0.15 * torch.mean(probs_starts)
        loss = loss_bce + loss_sparsity
        phase_name = "Phase 2: Pair Training"
        
    loss.backward()
    optimizer.step()
    
    return loss.item(), phase_name

def evaluate_model(model, dataloader, device):
    """
    Evaluates the single-head Hopfield model on the interaction prediction task.
    """
    model.eval()
    all_targets_starts = []
    all_preds_starts = []
    
    with torch.no_grad():
        for batch in dataloader:
            x = batch['sequence'].to(device)
            target_starts = batch['target_starts'].to(device)
            
            # Predict contact maps directly
            (logits_starts, probs_starts), _ = model(x)
            
            # Flatten predictions and targets for metric computation
            all_targets_starts.extend(target_starts.view(-1).cpu().tolist())
            all_preds_starts.extend(probs_starts.view(-1).cpu().tolist())
            
    try:
        auroc = roc_auc_score(all_targets_starts, all_preds_starts)
        prauc = average_precision_score(all_targets_starts, all_preds_starts)
    except Exception:
        auroc, prauc = 0.5, 0.5
        
    return {
        'auroc': auroc,
        'prauc': prauc,
        'pc_error': 0.0 # No longer applicable
    }
