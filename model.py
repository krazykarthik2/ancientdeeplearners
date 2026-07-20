import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ContractiveAutoencoder(nn.Module):
    def __init__(self, in_dim=4, out_dim=16):
        super().__init__()
        self.encoder = nn.Linear(in_dim, out_dim, bias=False)
        self.decoder = nn.Linear(out_dim, in_dim, bias=False)
        
    def forward(self, x):
        return self.encoder(x)

    def reconstruct(self, h):
        return self.decoder(h)

    def contractive_penalty(self, x):
        return torch.sum(self.encoder.weight ** 2)

class ModernHopfieldNetwork(nn.Module):
    def __init__(self, num_slots=8, d_model=16):
        super().__init__()
        self.num_slots = num_slots
        self.d_model = d_model
        self.M = nn.Parameter(torch.randn(num_slots, d_model))
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.log_beta = nn.Parameter(torch.tensor(math.log(1.0 / math.sqrt(d_model))))
        self.bias = nn.Parameter(torch.tensor(-10.0))

    @property
    def beta(self):
        return torch.exp(self.log_beta)

    def forward(self, x):
        Q = self.W_q(x)
        dot_products = torch.matmul(Q, self.M.transpose(-2, -1))
        energy = - (1.0 / self.beta) * torch.logsumexp(self.beta * dot_products, dim=-1)
        logits = -energy + self.bias
        probs = torch.sigmoid(logits)
        return energy, logits, probs

class BilinearInteraction(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(d, d) * 0.01)
        self.W2 = nn.Parameter(torch.randn(d, d) * 0.01)
        self.bias = nn.Parameter(torch.tensor(-7.0))

    def forward(self, z_e, z_p):
        z_e_W1 = torch.matmul(z_e, self.W1)
        z_p_W2 = torch.matmul(z_p, self.W2)
        logits = (torch.bmm(z_e_W1, z_p.transpose(-2, -1)) +
                  torch.bmm(z_p_W2, z_e.transpose(-2, -1))) / 2 + self.bias
        probs = torch.sigmoid(logits)
        return logits, probs


class GEMINITiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.cae = ContractiveAutoencoder(in_dim=5, out_dim=16)
        
        self.sequence_context_e = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=7, stride=1, padding=3)
        self.sequence_context_p = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=7, stride=1, padding=3)
        
        self.mhn1_e = ModernHopfieldNetwork(num_slots=8, d_model=16)
        self.mhn1_p = ModernHopfieldNetwork(num_slots=8, d_model=16)
        
        self.mhn2 = BilinearInteraction(d=16)

    def forward(self, x, steps=10, eta=0.05, settle=False):
        B, L, _ = x.shape
        
        cae_out = self.cae(x)
        cae_out_t = cae_out.transpose(1, 2)
        
        context_out_e = F.leaky_relu(self.sequence_context_e(cae_out_t)).transpose(1, 2)
        context_out_p = F.leaky_relu(self.sequence_context_p(cae_out_t)).transpose(1, 2)
        
        if settle:
            z_e = context_out_e.detach()
            z_p = context_out_p.detach()
            
            with torch.enable_grad():
                for step in range(steps):
                    z_e = z_e.detach().requires_grad_(True)
                    z_p = z_p.detach().requires_grad_(True)
                    
                    E1_e, _, _ = self.mhn1_e(z_e)
                    E1_p, _, _ = self.mhn1_p(z_p)
                    
                    loss_anchor = 1.0 * (
                        torch.mean((z_e - context_out_e.detach()) ** 2) + 
                        torch.mean((z_p - context_out_p.detach()) ** 2)
                    )
                    loss_energy = torch.sum(E1_e.mean(dim=-1) + E1_p.mean(dim=-1)) + loss_anchor
                    
                    grad_z_e, grad_z_p = torch.autograd.grad(loss_energy, [z_e, z_p])
                    
                    with torch.no_grad():
                        z_e = z_e - eta * grad_z_e
                        z_p = z_p - eta * grad_z_p
                        torch.clamp_(z_e, -5.0, 5.0)
                        torch.clamp_(z_p, -5.0, 5.0)
        else:
            z_e = context_out_e
            z_p = context_out_p
        
        logits, probs = self.mhn2(z_e, z_p)
        
        _, logits_1d_e, probs_1d_e = self.mhn1_e(z_e)
        _, logits_1d_p, probs_1d_p = self.mhn1_p(z_p)
        
        return (logits, probs), ((logits_1d_e, probs_1d_e), (logits_1d_p, probs_1d_p))