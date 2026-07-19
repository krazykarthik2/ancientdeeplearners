import os
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score

def train_step(model, batch, global_step, optimizer, criterion_bce, phase_thresholds, lambda_cae=1e-4, lambda_pc=0.1):
    """
    Executes a single multi-task training step based on the current global step phase.
    
    phase_thresholds: tuple of (warmup_steps, settling_steps)
    """
    warmup_steps, settling_steps = phase_thresholds
    
    x = batch['sequence'] # [B, 32, 4]
    target = batch['target'] # [B, 32, 32]
    
    optimizer.zero_grad()
    
    # Check Phase
    if global_step < warmup_steps:
        # --- PHASE 1: WARM-UP ---
        # Train only CAE and MHN to reconstruct the input sequence
        cae_out = model.cae(x)
        x_lookup, attn = model.mhn(cae_out)
        recon_x = model.cae.reconstruct(x_lookup)
        
        loss_recon = nn.MSELoss()(recon_x, x)
        penalty = model.cae.contractive_penalty(x)
        loss = loss_recon + lambda_cae * penalty
        phase_name = "Phase 1: CAE/MHN Warm-up"
        
    elif global_step < settling_steps:
        # --- PHASE 2: INFERENCE SETTLING ---
        # Train CAE, MHN, and PC layers to align states hierarchically
        mu2_settled, mu1_settled, x_lookup_t, _ = model.forward_inference(x, steps=10, eta=0.05)
        
        # Calculate reconstruction error under settled states
        x_pred, mu1_pred = model.pc.forward_prediction(mu1_settled, mu2_settled)
        eps0 = x_lookup_t - x_pred
        eps1 = mu1_settled - mu1_pred
        eps2 = mu2_settled
        
        loss_pc_val = torch.mean(eps0 ** 2) + torch.mean(eps1 ** 2) + 0.01 * torch.mean(eps2 ** 2)
        penalty = model.cae.contractive_penalty(x)
        loss = loss_pc_val + lambda_cae * penalty
        phase_name = "Phase 2: PC State Settling"
        
    else:
        # --- PHASE 3: FULL COUPLING ---
        # Train all components including Boltzmann Head to predict interactions
        mu2_settled, mu1_settled, x_lookup_t, _ = model.forward_inference(x, steps=10, eta=0.05)
        logits, probs = model.bm(mu2_settled)
        
        # 1. Boltzmann BCE Loss & L1 Sparsity regularization (to completely eliminate background cross-talk/interference stripes)
        loss_bce = criterion_bce(logits, target)
        loss_sparsity = 0.05 * torch.mean(probs)
        
        # 2. Predictive coding residuals
        x_pred, mu1_pred = model.pc.forward_prediction(mu1_settled, mu2_settled)
        eps0 = x_lookup_t - x_pred
        eps1 = mu1_settled - mu1_pred
        loss_pc_val = torch.mean(eps0 ** 2) + torch.mean(eps1 ** 2)
        
        # 3. Contractive Autoencoder penalty
        penalty = model.cae.contractive_penalty(x)
        
        # Total Joint Loss
        loss = loss_bce + loss_sparsity + lambda_pc * loss_pc_val + lambda_cae * penalty
        phase_name = "Phase 3: Full Coupling"
        
    loss.backward()
    optimizer.step()
    
    # Enforce Boltzmann J symmetry after each step
    with torch.no_grad():
        model.bm.J.copy_(0.5 * (model.bm.J + model.bm.J.T))
        
    return loss.item(), phase_name

def evaluate_model(model, dataloader, device):
    """
    Evaluates the model on the interaction prediction task.
    """
    model.eval()
    all_targets = []
    all_preds = []
    total_pc_error = 0.0
    
    with torch.no_grad():
        for batch in dataloader:
            x = batch['sequence'].to(device)
            target = batch['target'].to(device)
            
            # Run inference settling
            mu2_settled, mu1_settled, x_lookup_t, _ = model.forward_inference(x, steps=10, eta=0.05)
            
            # Predict contact map
            logits, probs = model.bm(mu2_settled)
            
            # Track PC error
            x_pred, mu1_pred = model.pc.forward_prediction(mu1_settled, mu2_settled)
            eps0 = x_lookup_t - x_pred
            eps1 = mu1_settled - mu1_pred
            total_pc_error += (torch.mean(eps0 ** 2) + torch.mean(eps1 ** 2)).item()
            
            # Flatten predictions and targets for metric computation
            all_targets.extend(target.view(-1).cpu().tolist())
            all_preds.extend(probs.view(-1).cpu().tolist())
            
    try:
        auroc = roc_auc_score(all_targets, all_preds)
        prauc = average_precision_score(all_targets, all_preds)
    except Exception:
        auroc, prauc = 0.5, 0.5
        
    return {
        'auroc': auroc,
        'prauc': prauc,
        'pc_error': total_pc_error / len(dataloader)
    }
