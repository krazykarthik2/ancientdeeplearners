import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ContractiveAutoencoder3D(nn.Module):
    def __init__(self, in_dim=5, out_dim=32):
        super().__init__()
        self.encoder = nn.Linear(in_dim, out_dim, bias=False)
        self.decoder = nn.Linear(out_dim, in_dim, bias=False)

    def forward(self, x):
        return self.encoder(x)

    def reconstruct(self, h):
        return self.decoder(h)

    def contractive_penalty(self):
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
        self.W = nn.Parameter(torch.randn(n_visible, n_hidden) * 0.01)
        self.bv = nn.Parameter(torch.zeros(n_visible))
        self.bh = nn.Parameter(torch.zeros(n_hidden))

    def cd_k(self, v, k=1):
        h_prob = torch.sigmoid(v @ self.W + self.bh)
        h_sample = torch.bernoulli(h_prob)
        chain_v = v; chain_h = h_sample
        for _ in range(k):
            v_prob = torch.sigmoid(chain_h @ self.W.t() + self.bv)
            chain_v = torch.bernoulli(v_prob)
            h_prob_k = torch.sigmoid(chain_v @ self.W + self.bh)
            chain_h = torch.bernoulli(h_prob_k)
        v_k, h_k = chain_v, chain_h
        pos_grad = (v.unsqueeze(-1) @ h_prob.unsqueeze(1)).mean(0)
        neg_grad = (v_k.unsqueeze(-1) @ h_k.unsqueeze(1)).mean(0)
        return pos_grad, neg_grad, v.mean(0), v_k.mean(0), h_prob.mean(0), h_k.mean(0)


class DeepBoltzmannMachine(nn.Module):
    def __init__(self, n_visible, n_hidden_1, n_hidden_2):
        super().__init__()
        self.rbm1 = RBM(n_visible, n_hidden_1)
        self.rbm2 = RBM(n_hidden_1, n_hidden_2)

    def pretrain(self, v, lr=0.01, k=1):
        pos1, neg1, bv_d, bv_m, bh_d, bh_m = self.rbm1.cd_k(v, k)
        self.rbm1.W.data += lr * (pos1 - neg1)
        self.rbm1.bv.data += lr * (bv_d - bv_m)
        self.rbm1.bh.data += lr * (bh_d - bh_m)
        h1_prob = torch.sigmoid(v @ self.rbm1.W + self.rbm1.bh)
        pos2, neg2, _, _, _, _ = self.rbm2.cd_k(h1_prob, k)
        self.rbm2.W.data += lr * (pos2 - neg2)
        return (pos1 - neg1).norm().item(), (pos2 - neg2).norm().item()

    def gibbs_sample(self, v, steps=3):
        h1 = torch.sigmoid(v @ self.rbm1.W + self.rbm1.bh)
        h2 = torch.sigmoid(h1 @ self.rbm2.W + self.rbm2.bh)
        for _ in range(steps):
            h1 = torch.sigmoid(v @ self.rbm1.W + self.rbm1.bh)
            h2 = torch.sigmoid(h1 @ self.rbm2.W + self.rbm2.bh)
            h1 = torch.sigmoid(h2 @ self.rbm2.W.t() + v @ self.rbm1.W + self.rbm1.bh)
            v = torch.sigmoid(h1 @ self.rbm1.W.t() + self.rbm1.bv)
        return v


class BoltzmannForceRefinement(nn.Module):
    def __init__(self, steps=30, lr=0.5):
        super().__init__()
        self.steps = steps
        self.lr = lr

    def forward(self, contacts, coords_init):
        d_target = 1.0 / (contacts + 0.01)
        x = coords_init
        for _ in range(self.steps):
            diffs = x.unsqueeze(1) - x.unsqueeze(2)
            dists = torch.norm(diffs, dim=-1) + 1e-8
            grad = 2 * ((dists - d_target) / dists).unsqueeze(-1) * diffs
            force = grad.sum(dim=2)
            x = x - self.lr * torch.tanh(force * 0.1)
        return x


class GEMINI3D(nn.Module):
    def __init__(self, window=64, dim=32, refine_steps=30):
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

        self.refine = BoltzmannForceRefinement(steps=refine_steps, lr=0.5)

    def forward(self, x, coords_init=None, refine_3d=False):
        B, L, _ = x.shape
        z = self.cae(x)
        z = F.leaky_relu(self.conv(z.transpose(1, 2))).transpose(1, 2)
        logits, probs = self.contact(z)

        triu_indices = torch.triu_indices(L, L, offset=1)
        v = probs[:, triu_indices[0], triu_indices[1]]
        v_refined = self.dbm.gibbs_sample(v, steps=3)
        probs_refined = probs.clone()
        probs_refined[:, triu_indices[0], triu_indices[1]] = v_refined
        probs_refined[:, triu_indices[1], triu_indices[0]] = v_refined

        coords = None
        if refine_3d and coords_init is not None:
            coords = self.refine(probs_refined, coords_init)

        return logits, probs, probs_refined, coords

    def predict_3d(self, x, coords_init=None):
        self.eval()
        B, L = x.shape[0], x.shape[1]
        with torch.no_grad():
            _, _, probs_refined, _ = self(x, coords_init=None)
            p = probs_refined[0].cpu().numpy()

        d = 1.0 / (p + 0.001)
        np.fill_diagonal(d, 0)
        n = d.shape[0]
        dist = d.copy()
        for k in range(n):
            dk = dist[k:k+1, :]
            dist_kk = dist[:, k:k+1]
            dist = np.minimum(dist, dk + dist_kk)
        J = np.eye(n) - np.ones((n, n)) / n
        Bmat = -0.5 * J @ (dist ** 2) @ J
        eigvals, eigvecs = np.linalg.eigh(Bmat)
        idx = np.argsort(eigvals)[::-1][:3]
        x_init = eigvecs[:, idx] * np.sqrt(np.maximum(eigvals[idx], 0))

        d_target = torch.from_numpy(1.0 / (p + 0.01)).float()
        x = torch.from_numpy(x_init).float().unsqueeze(0)
        for _ in range(self.refine.steps):
            diffs = x.unsqueeze(1) - x.unsqueeze(2)
            dists = torch.norm(diffs, dim=-1) + 1e-8
            grad = 2 * ((dists - d_target) / dists).unsqueeze(-1) * diffs
            x = x - self.refine.lr * torch.tanh(grad.sum(dim=2) * 0.1)
        return x[0].numpy(), p
