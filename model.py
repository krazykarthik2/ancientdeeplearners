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

class PredictiveCodingBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        # Generative Weights (Transposed Convolutions)
        # Level 2 state: [B, 64, 8] -> predicts Level 1 state: [B, 32, 16]
        self.W2 = nn.ConvTranspose1d(in_channels=64, out_channels=32, 
                                     kernel_size=4, stride=2, padding=1)
        # Level 1 state: [B, 32, 16] -> predicts Level 0 lookup: [B, 16, 32]
        self.W1 = nn.ConvTranspose1d(in_channels=32, out_channels=16, 
                                     kernel_size=4, stride=2, padding=1)
        
        # Bottom-up helper layers to initialize PC states quickly
        self.init_conv1 = nn.Conv1d(in_channels=16, out_channels=32, 
                                    kernel_size=4, stride=2, padding=1)
        self.init_conv2 = nn.Conv1d(in_channels=32, out_channels=64, 
                                    kernel_size=4, stride=2, padding=1)

    def initialize_states(self, x_lookup):
        # x_lookup: [B, 16, 32] (channels=16, length=32)
        # Initialize mu1 and mu2 via bottom-up helper convs
        mu1 = F.leaky_relu(self.init_conv1(x_lookup)) # [B, 32, 16]
        mu2 = F.leaky_relu(self.init_conv2(mu1))      # [B, 64, 8]
        
        # Track gradients on intermediate tensors without detaching from bottom-up graph
        mu1.requires_grad_(True)
        mu2.requires_grad_(True)
        return mu1, mu2

    def forward_prediction(self, mu1, mu2):
        # Generative predictions
        mu1_pred = F.leaky_relu(self.W2(mu2)) # [B, 32, 16]
        x_pred = F.leaky_relu(self.W1(mu1))   # [B, 16, 32]
        return x_pred, mu1_pred

class BoltzmannFusionHead(nn.Module):
    def __init__(self, in_features=64, latent_dim=8):
        super().__init__()
        self.W_proj = nn.Linear(in_features, latent_dim)
        self.J = nn.Parameter(torch.randn(latent_dim, latent_dim))
        
        # Learnable sequence upsampler to sharply map length 8 -> 32 without linear smoothing/blurring
        self.upsample_seq = nn.ConvTranspose1d(
            in_channels=latent_dim, 
            out_channels=latent_dim, 
            kernel_size=4, 
            stride=4, 
            padding=0
        )
        
        # Initial negative bias to cleanly suppress sparse background contacts
        self.bias = nn.Parameter(torch.tensor(-4.5))
        
        # Initialize symmetric coupling matrix J
        with torch.no_grad():
            self.J.copy_(0.5 * (self.J + self.J.T))

    def forward(self, mu2):
        # mu2 state: [B, 64, 8]
        B = mu2.shape[0]
        
        # 1. Project channels 64 -> 8
        # Permute to apply Linear on channel dimension
        z = mu2.transpose(1, 2) # [B, 8, 64] (length=8, channels=64)
        z = self.W_proj(z) # [B, 8, 8] (length=8, channels=8)
        z = z.transpose(1, 2) # [B, 8, 8] (channels=8, length=8)
        
        # 2. Sharply upsample sequence length 8 -> 32 and apply ReLU to enforce sparsity
        z_upsampled = F.relu(self.upsample_seq(z)) # [B, 8, 32] (channels=8, length=32)
        z_upsampled = z_upsampled.transpose(1, 2) # [B, 32, 8] (length=32, channels=8)
        
        # 3. Symmetric Coupling Matrix J
        J_sym = 0.5 * (self.J + self.J.T)
        
        # 4. Energy calculation to generate final 32x32 interaction matrix with bias
        # logits_ij = z_i^T J_sym z_j + bias
        # z_upsampled: [B, L, 8]
        logits = torch.matmul(torch.matmul(z_upsampled, J_sym), z_upsampled.transpose(-2, -1)) + self.bias
        probs = torch.sigmoid(logits)
        return logits, probs

class GEMINITiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.cae = ContractiveAutoencoder(in_dim=5, out_dim=16)
        self.mhn = ModernHopfieldNetwork(num_slots=8, d_model=16)
        self.pc = PredictiveCodingBackbone()
        self.bm = BoltzmannFusionHead(in_features=64, latent_dim=8)

    def forward_inference(self, x, steps=10, eta=0.05):
        # x: [B, 32, 5]
        # 1. CAE Encoding
        cae_out = self.cae(x) # [B, 32, 16]
        
        # 2. MHN Dictionary lookup
        x_lookup, attn = self.mhn(cae_out) # [B, 32, 16], [B, 32, 8]
        
        # Reshape to match ConvTranspose1d sequence input expectation: [B, 16, 32]
        x_lookup_t = x_lookup.transpose(1, 2)
        
        # 3. Initialize Predictive Coding States
        mu1, mu2 = self.pc.initialize_states(x_lookup_t)
        
        # 4. Iterative Inference (Unrolled State Settling)
        with torch.enable_grad():
            for step in range(steps):
                x_pred, mu1_pred = self.pc.forward_prediction(mu1, mu2)
                
                # Compute predictive residuals (errors)
                eps0 = x_lookup_t - x_pred
                eps1 = mu1 - mu1_pred
                eps2 = mu2 # Prior regularization
                
                # Mean Squared Error sum
                loss_pc = torch.mean(eps0 ** 2) + torch.mean(eps1 ** 2) + 0.01 * torch.mean(eps2 ** 2)
                
                # Compute gradients of PC loss w.r.t state tensors
                grad_mu1, grad_mu2 = torch.autograd.grad(
                    loss_pc, [mu1, mu2], 
                    create_graph=True,
                    retain_graph=True
                )
                
                # Explicit unrolled gradient update steps
                mu1 = mu1 - eta * grad_mu1
                mu2 = mu2 - eta * grad_mu2
            
        # Return settled states
        if self.training:
            return mu2, mu1, x_lookup_t, attn
        else:
            return mu2.detach(), mu1.detach(), x_lookup_t, attn

    def forward(self, x, steps=10, eta=0.05):
        # Run inference state settling
        mu2_settled, _, _, _ = self.forward_inference(x, steps=steps, eta=eta)
        
        # 5. Boltzmann Interaction Mapping
        logits, probs = self.bm(mu2_settled)
        return logits, probs
