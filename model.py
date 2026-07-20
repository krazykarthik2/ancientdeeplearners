import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ContractiveAutoencoder(nn.Module):
    def __init__(self, in_dim=4, out_dim=16):
        super().__init__()
        self.encoder = nn.Linear(in_dim, out_dim)
        self.decoder = nn.Linear(out_dim, in_dim)
        
    def forward(self, x):
        # x: [B, L, 4]
        # output: [B, L, 16]
        h = self.encoder(x)
        return torch.sigmoid(h)

    def reconstruct(self, h):
        # h: [B, L, 16]
        # output: [B, L, 4]
        return self.decoder(h)

    def contractive_penalty(self, x):
        # h = sigmoid(x W^T + b)
        # J_j = h_j * (1 - h_j) * W_j
        # ||J||_F^2 = sum_j (h_j * (1 - h_j))^2 * sum_i W_ji^2
        h = self.forward(x) # [B, L, 16]
        sigmoid_deriv = h * (1.0 - h) # [B, L, 16]
        
        W = self.encoder.weight # [16, 4]
        W_sq_sum = torch.sum(W ** 2, dim=-1) # [16]
        
        # Multiply squared derivatives by squared weight sums
        penalty = (sigmoid_deriv ** 2) * W_sq_sum.view(1, 1, -1) # [B, L, 16]
        return torch.sum(penalty)

class ModernHopfieldNetwork(nn.Module):
    def __init__(self, num_slots=8, d_model=16):
        super().__init__()
        self.num_slots = num_slots
        self.d_model = d_model
        # Trainable motif slots M
        self.M = nn.Parameter(torch.randn(num_slots, d_model))
        # Query projection W_q
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.beta = 1.0 / math.sqrt(d_model)
        self.bias = nn.Parameter(torch.tensor(-3.0))

    def forward(self, x):
        # x: [B, L, 16]
        Q = self.W_q(x) # [B, L, 16]
        
        # LogSumExp energy formulation for Modern Hopfield Network
        dot_products = torch.matmul(Q, self.M.transpose(-2, -1)) # [B, L, num_slots]
        energy = - (1.0 / self.beta) * torch.logsumexp(self.beta * dot_products, dim=-1) # [B, L]
        probs = torch.sigmoid(-energy + self.bias)
        return energy, probs

class InteractionHopfieldNetwork(nn.Module):
    def __init__(self, d_pair=32, num_slots=1):
        super().__init__()
        # Memory slots for valid interactions
        self.M = nn.Parameter(torch.randn(num_slots, d_pair))
        self.W_q = nn.Linear(d_pair, d_pair, bias=False)
        self.beta = 1.0 / math.sqrt(d_pair)
        # Strong negative bias to cleanly suppress background pairs
        self.bias = nn.Parameter(torch.tensor(-4.5))

    def forward(self, pair_features):
        # pair_features: [B, L*L, d_pair]
        Q = self.W_q(pair_features) # [B, L*L, d_pair]
        
        dot_products = torch.matmul(Q, self.M.transpose(-2, -1)) # [B, L*L, num_slots]
        # LogSumExp energy formulation
        energy = - (1.0 / self.beta) * torch.logsumexp(self.beta * dot_products, dim=-1) # [B, L*L]
        
        # Reshape to [B, L, L]
        B, LL = energy.shape
        L = int(math.sqrt(LL))
        energy_2d = energy.view(B, L, L)
        
        # Logits derived directly from negative energy of the pairing state
        logits = -energy_2d + self.bias
        probs = torch.sigmoid(logits)
        return logits, probs, energy_2d


class GEMINITiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.cae = ContractiveAutoencoder(in_dim=5, out_dim=16)
        
        # Parallel Sequence Context Layers for Enhancer (E) and Promoter (P)
        self.sequence_context_e = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.sequence_context_p = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=5, stride=1, padding=2)
        
        # Parallel 1D Motif Hopfield Networks (MHN-1)
        self.mhn1_e = ModernHopfieldNetwork(num_slots=8, d_model=16)
        self.mhn1_p = ModernHopfieldNetwork(num_slots=8, d_model=16)
        
        # Single 2D Interaction Hopfield Network (MHN-2)
        # Pair features will be concatenation of (H_E_i, H_P_j) so d_pair = 16 + 16 = 32
        self.mhn2 = InteractionHopfieldNetwork(d_pair=32, num_slots=1)

    def forward(self, x, steps=10, eta=0.05, include_interaction_energy=True):
        # x: [B, 32, 5]
        B, L, _ = x.shape
        
        # 1. Obtain initial states using CAE and Parallel Sequence Contexts
        cae_out = self.cae(x) # [B, L, 16]
        cae_out_t = cae_out.transpose(1, 2)
        
        context_out_e = F.leaky_relu(self.sequence_context_e(cae_out_t)).transpose(1, 2) # [B, L, 16]
        context_out_p = F.leaky_relu(self.sequence_context_p(cae_out_t)).transpose(1, 2) # [B, L, 16]
            
        # Initialize continuous states z_e and z_p for the energy model
        z_e = context_out_e.detach()
        z_p = context_out_p.detach()
        
        # 2. Iterative Energy Minimization (State Settling)
        with torch.enable_grad():
            for step in range(steps):
                # Detach and require grad to make states leaf tensors for this step
                z_e = z_e.detach().requires_grad_(True)
                z_p = z_p.detach().requires_grad_(True)
                
                # 1D motif energies
                E1_e, _ = self.mhn1_e(z_e) # [B, L]
                E1_p, _ = self.mhn1_p(z_p) # [B, L]
                
                if include_interaction_energy:
                    # Construct 2D Cross-Pair Features
                    z_e_i = z_e.unsqueeze(2).expand(B, L, L, 16)
                    z_p_j = z_p.unsqueeze(1).expand(B, L, L, 16)
                    pair_features = torch.cat([z_e_i, z_p_j], dim=-1).view(B, L*L, 32)
                    
                    # 2D interaction energy
                    _, _, E2 = self.mhn2(pair_features) # [B, L, L]
                    
                    # Total energy to minimize (Motifs + Interaction)
                    loss_energy = torch.mean(E1_e) + torch.mean(E1_p) + torch.mean(E2)
                else:
                    # Total energy to minimize (Motifs Only during pre-training)
                    loss_energy = torch.mean(E1_e) + torch.mean(E1_p)
                
                # Gradient update on both states
                grad_z_e, grad_z_p = torch.autograd.grad(loss_energy, [z_e, z_p])
                
                # Update out-of-graph
                with torch.no_grad():
                    z_e = z_e - eta * grad_z_e
                    z_p = z_p - eta * grad_z_p
                
        # 3. Final pairing pass under settled states
        z_e_i = z_e.unsqueeze(2).expand(B, L, L, 16)
        z_p_j = z_p.unsqueeze(1).expand(B, L, L, 16)
        pair_features = torch.cat([z_e_i, z_p_j], dim=-1).view(B, L*L, 32)
        
        logits, probs, _ = self.mhn2(pair_features)
        
        # Get 1D predictions under settled states for motif pre-training
        _, probs_1d_e = self.mhn1_e(z_e)
        _, probs_1d_p = self.mhn1_p(z_p)
        
        return (logits, probs), (probs_1d_e, probs_1d_p)
