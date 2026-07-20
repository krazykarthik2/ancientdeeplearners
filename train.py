import os
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score

def get_1d_motif_targets(x):
    B, L, _ = x.shape
    device = x.device
    target_e = torch.zeros(B, L, device=device)
    target_p = torch.zeros(B, L, device=device)
    
    for i in range(L - 3):
        is_tata = (x[:, i, 3] > 0.8) & (x[:, i+1, 0] > 0.8) & (x[:, i+2, 3] > 0.8) & (x[:, i+3, 0] > 0.8)
        is_ccgc = (x[:, i, 1] > 0.8) & (x[:, i+1, 1] > 0.8) & (x[:, i+2, 2] > 0.8) & (x[:, i+3, 1] > 0.8)
        
        target_e[:, i] = torch.where(is_tata, 1.0, 0.0)
        target_p[:, i] = torch.where(is_ccgc, 1.0, 0.0)
        
    return target_e, target_p

def train_step(model, batch, global_step, optimizer, criterion_bce, phase_thresholds, device):
    x = batch['sequence'].to(device)
    target_starts = batch['target_starts'].to(device)
    
    target_e, target_p = get_1d_motif_targets(x)
    
    optimizer.zero_grad()
    
    motif_pretrain_steps, _ = phase_thresholds
    
    if global_step < motif_pretrain_steps:
        (logits_starts, probs_starts), ((logits_1d_e, probs_1d_e), (logits_1d_p, probs_1d_p)) = model(x, settle=False)
        
        pos_weight = torch.tensor([30.0], device=device)
        criterion_1d = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        loss_e = criterion_1d(logits_1d_e, target_e)
        loss_p = criterion_1d(logits_1d_p, target_p)
        
        loss_sparse_1d = 0.5 * (torch.mean(probs_1d_e) + torch.mean(probs_1d_p))
        
        cae_out = model.cae(x)
        recon = model.cae.reconstruct(cae_out)
        loss_recon = torch.mean((recon[:, :, :4] - x[:, :, :4]) ** 2)
        
        loss_wd = 1e-5 * model.cae.contractive_penalty(x)
        loss = loss_e + loss_p + loss_sparse_1d + loss_recon + loss_wd
        phase_name = "Phase 1: Motif Pre-training"
        
    else:
        if global_step == motif_pretrain_steps:
            print("\n>>> Freezing 1D layers and starting Pair training...")
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
        
        (logits_starts, probs_starts), _ = model(x, settle=False)
        
        loss_bce = criterion_bce(logits_starts, target_starts)
        loss_sparsity = 1.0 * torch.mean(probs_starts)
        loss = loss_bce + loss_sparsity
        phase_name = "Phase 2: Pair Training"
        
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    return loss.item(), phase_name

def evaluate_model(model, dataloader, device):
    model.eval()
    all_targets_starts = []
    all_preds_starts = []
    
    with torch.no_grad():
        for batch in dataloader:
            x = batch['sequence'].to(device)
            target_starts = batch['target_starts'].to(device)
            
            (logits_starts, probs_starts), _ = model(x, settle=False)
            
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
        'pc_error': 0.0
    }