import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.manifold import MDS

class ContractiveAutoencoder3D(nn.Module):
    def __init__(self, in_dim=5, out_dim=32):
        super().__init__()
        self.encoder = nn.Linear(in_dim, out_dim, bias=False)
        self.decoder = nn.Linear(out_dim, in_dim, bias=False)

    def forward(self, x):
        return self.encoder(x)

    def reconstruct(self, h):
        return self.decoder(h)

    def contractive_penalty(self, x):
        return torch.sum(self.encoder.weight ** 2)

class BilinearInteraction3D(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(d, d) * 0.01)
        self.W2 = nn.Parameter(torch.randn(d, d) * 0.01)
        self.bias = nn.Parameter(torch.tensor(-7.0))

    def forward(self, z):
        z_W1 = torch.matmul(z, self.W1)
        z_W2 = torch.matmul(z, self.W2)
        logits = (torch.bmm(z_W1, z.transpose(-2, -1)) +
                  torch.bmm(z_W2, z.transpose(-2, -1))) / 2 + self.bias
        probs = torch.sigmoid(logits)
        return logits, probs


class RBM(nn.Module):
    def __init__(self, n_visible, n_hidden):
        super().__init__()
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.W = nn.Parameter(torch.randn(n_visible, n_hidden) * 0.01)
        self.bv = nn.Parameter(torch.zeros(n_visible))
        self.bh = nn.Parameter(torch.zeros(n_hidden))

    def forward(self, v):
        h_prob = torch.sigmoid(v @ self.W + self.bh)
        h = torch.bernoulli(h_prob)
        v_prob = torch.sigmoid(h @ self.W.t() + self.bv)
        return v_prob, h_prob, h

    def free_energy(self, v):
        v_term = -torch.sum(v * self.bv, dim=-1)
        h_term = -torch.sum(F.softplus(v @ self.W + self.bh), dim=-1)
        return v_term + h_term

    def cd_k(self, v, k=1):
        h_prob = torch.sigmoid(v @ self.W + self.bh)
        h_sample = torch.bernoulli(h_prob)
        chain_v = v
        chain_h = h_sample
        for _ in range(k):
            v_prob = torch.sigmoid(chain_h @ self.W.t() + self.bv)
            chain_v = torch.bernoulli(v_prob)
            h_prob_k = torch.sigmoid(chain_v @ self.W + self.bh)
            chain_h = torch.bernoulli(h_prob_k)
        v_k, h_k = chain_v, chain_h
        positive_grad = (v.unsqueeze(-1) @ h_prob.unsqueeze(1)).mean(0)
        negative_grad = (v_k.unsqueeze(-1) @ h_k.unsqueeze(1)).mean(0)
        return positive_grad, negative_grad, v.mean(0), v_k.mean(0), h_prob.mean(0), h_k.mean(0)


class DeepBoltzmannMachine(nn.Module):
    def __init__(self, n_visible, n_hidden_1, n_hidden_2):
        super().__init__()
        self.rbm1 = RBM(n_visible, n_hidden_1)
        self.rbm2 = RBM(n_hidden_1, n_hidden_2)

    def forward(self, v):
        v_prob, h1_prob, h1 = self.rbm1(v)
        h1_prob_2, h2_prob, h2 = self.rbm2(h1_prob)
        return v_prob, h1_prob, h2_prob

    def pretrain(self, v, lr=0.01, k=1):
        pos1, neg1, bv_d, bv_m, bh_d, bh_m = self.rbm1.cd_k(v, k)
        self.rbm1.W.data += lr * (pos1 - neg1)
        self.rbm1.bv.data += lr * (bv_d - bv_m)
        self.rbm1.bh.data += lr * (bh_d - bh_m)

        h1_prob = torch.sigmoid(v @ self.rbm1.W + self.rbm1.bh)
        pos2, neg2, _, _, _, _ = self.rbm2.cd_k(h1_prob, k)
        self.rbm2.W.data += lr * (pos2 - neg2)
        return (pos1 - neg1).norm().item(), (pos2 - neg2).norm().item()

    def gibbs_sample(self, v, steps=10):
        h1 = torch.sigmoid(v @ self.rbm1.W + self.rbm1.bh)
        h2 = torch.sigmoid(h1 @ self.rbm2.W + self.rbm2.bh)
        for _ in range(steps):
            h1 = torch.sigmoid(v @ self.rbm1.W + self.rbm1.bh)
            h2 = torch.sigmoid(h1 @ self.rbm2.W + self.rbm2.bh)
            h1 = torch.sigmoid(h2 @ self.rbm2.W.t() + v @ self.rbm1.W + self.rbm1.bh)
            v = torch.sigmoid(h1 @ self.rbm1.W.t() + self.rbm1.bv)
        return v


class GEMINI3D(nn.Module):
    def __init__(self, window=64, dim=32):
        super().__init__()
        self.window = window
        self.dim = dim

        self.cae = ContractiveAutoencoder3D(in_dim=5, out_dim=dim)
        self.conv = nn.Conv1d(in_channels=dim, out_channels=dim,
                              kernel_size=7, stride=1, padding=3)
        self.contact = BilinearInteraction3D(d=dim)

        n_visible = window * (window - 1) // 2
        self.dbm = DeepBoltzmannMachine(
            n_visible=n_visible,
            n_hidden_1=n_visible // 4,
            n_hidden_2=n_visible // 8
        )

    def forward(self, x, refine=False):
        B, L, _ = x.shape
        z = self.cae(x)
        z = F.leaky_relu(self.conv(z.transpose(1, 2))).transpose(1, 2)
        logits, probs = self.contact(z)

        if refine:
            triu_indices = torch.triu_indices(L, L, offset=1)
            v = probs[:, triu_indices[0], triu_indices[1]]
            v_refined = self.dbm.gibbs_sample(v, steps=5)
            probs_refined = probs.clone()
            probs_refined[:, triu_indices[0], triu_indices[1]] = v_refined
            probs_refined[:, triu_indices[1], triu_indices[0]] = v_refined
            return logits, probs, probs_refined

        return logits, probs, probs

    def predict_3d(self, x):
        self.eval()
        with torch.no_grad():
            _, probs, probs_refined = self(x, refine=True)
            p = probs_refined[0].cpu().numpy()
        d = 1.0 / (p + 0.001)
        np.fill_diagonal(d, 0)
        n = d.shape[0]
        dist = d.copy()
        for k in range(n):
            dk = dist[k:k+1, :]
            dist_kk = dist[:, k:k+1]
            dist = np.minimum(dist, dk + dist_kk)
        coords_3d = self._mds(dist, n_components=3)
        return coords_3d, p

    def _mds(self, dist, n_components=3):
        n = dist.shape[0]
        J = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * J @ (dist ** 2) @ J
        eigvals, eigvecs = np.linalg.eigh(B)
        idx = np.argsort(eigvals)[::-1][:n_components]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        coords = eigvecs * np.sqrt(np.maximum(eigvals, 0))
        return coords
