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

    def forward(self, x):
        # x: [B, L, 16]
        Q = self.W_q(x) # [B, L, 16]
        
        # M: [8, 16]
        # Q @ M.T: [B, L, 8]
        scores = torch.matmul(Q, self.M.transpose(-2, -1)) * self.beta
        attn = torch.softmax(scores, dim=-1) # [B, L, 8]
        
        # Retrieve: [B, L, 16]
        retrieved = torch.matmul(attn, self.M)
        return retrieved, attn

class InteractionHopfieldNetwork(nn.Module):
    def __init__(self, d_pair=32, num_slots=4):
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
        
        # scores: [B, L*L, num_slots]
        scores = torch.matmul(Q, self.M.transpose(-2, -1)) * self.beta
        
        # Aggregate across slots to get a single logit per pair
        logits = torch.sum(scores, dim=-1) + self.bias # [B, L*L]
        probs = torch.sigmoid(logits)
        
        # Reshape to [B, L, L]
        B, LL = logits.shape
        L = int(math.sqrt(LL))
        return logits.view(B, L, L), probs.view(B, L, L)


class GEMINITiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.cae = ContractiveAutoencoder(in_dim=5, out_dim=16)
        
        # Sequence Context Layer to provide a spatial receptive field of 4 (e.g. for "TATA")
        self.sequence_context = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=4, stride=1, padding=3)
        
        # MHN-1: Memorize and identify individual E and P motifs from 1D sequence
        self.mhn1 = ModernHopfieldNetwork(num_slots=8, d_model=16)
        
        # MHN-2: Memorize and identify 2D Interaction Pairs (Starts and Ends)
        # Pair features will be concatenation of (H_i, H_j) so d_pair = 16 + 16 = 32
        self.mhn2_starts = InteractionHopfieldNetwork(d_pair=32, num_slots=4)
        self.mhn2_ends = InteractionHopfieldNetwork(d_pair=32, num_slots=4)

    def forward(self, x):
        # x: [B, 32, 5]
        B, L, _ = x.shape
        
        # 1. 1D Motif Identification (MHN-1) with Receptive Field
        cae_out = self.cae(x) # [B, L, 16]
        
        # Apply 1D convolution over sequence length
        cae_out_t = cae_out.transpose(1, 2) # [B, 16, L]
        context_out = F.leaky_relu(self.sequence_context(cae_out_t)) # [B, 16, L+3]
        
        # Crop padding to keep length exactly L (take the first L elements)
        context_out = context_out[:, :, :L] # [B, 16, L]
        context_out = context_out.transpose(1, 2) # [B, L, 16]
        
        H1, _ = self.mhn1(context_out) # [B, L, 16]
        
        # 2. Construct 2D Pair Features
        # H1_i: [B, L, 1, 16], H1_j: [B, 1, L, 16]
        H1_i = H1.unsqueeze(2).expand(B, L, L, 16)
        H1_j = H1.unsqueeze(1).expand(B, L, L, 16)
        pair_features = torch.cat([H1_i, H1_j], dim=-1) # [B, L, L, 32]
        pair_features = pair_features.view(B, L*L, 32)
        
        # 3. 2D Interaction Identification (MHN-2)
        logits_starts, probs_starts = self.mhn2_starts(pair_features)
        logits_ends, probs_ends = self.mhn2_ends(pair_features)
        
        return (logits_starts, probs_starts), (logits_ends, probs_ends)
