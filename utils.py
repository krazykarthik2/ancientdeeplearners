import math, numpy as np, torch
from torch.utils.data import Dataset


class Synthetic3DDataset(Dataset):
    def __init__(self, num_samples=500, window=32, seed=42):
        self.window = window
        self.rng = np.random.RandomState(seed)
        self.scale = 2.0; self.alpha = 3.0
        self.samples = []
        for _ in range(num_samples):
            seq, contact, coords, e_pos, p_pos = self._generate()
            self.samples.append((seq, contact, coords, e_pos, p_pos))

    def _generate(self):
        L = self.window
        seq = np.zeros((L, 4), dtype=np.float32)
        bases = ['A','C','G','T']; ni = {'A':0,'C':1,'G':2,'T':3}
        for i in range(L):
            n = self.rng.choice(bases, p=[0.26,0.24,0.24,0.26])
            seq[i, ni[n]] = 1.0
        gap = self.rng.randint(8, L // 2)
        e_pos = self.rng.randint(2, L//3)
        p_pos = min(e_pos + gap, L - 6)
        if p_pos == L - 6: e_pos = p_pos - gap
        for off, n in enumerate([3,0,3,0]):
            seq[e_pos+off, :] = 0; seq[e_pos+off, n] = 1.0
        for off, n in enumerate([1,1,2,1]):
            seq[p_pos+off, :] = 0; seq[p_pos+off, n] = 1.0
        pos_coord = np.arange(L, dtype=np.float32) / (L - 1)
        seq_5ch = np.zeros((L, 5), dtype=np.float32)
        seq_5ch[:, :4] = seq
        seq_5ch[:, 4] = pos_coord
        t = np.linspace(0, 4*np.pi, L)
        coords = np.zeros((L, 3), dtype=np.float32)
        coords[:, 0] = np.sin(t) * 5
        coords[:, 1] = np.cos(t * 0.7) * 4
        coords[:, 2] = t * 0.3
        lc = (coords[e_pos] + coords[p_pos]) / 2
        for i in range(L):
            pull = 0.8 * np.exp(-min(abs(i-e_pos), abs(i-p_pos)) / 5.0)
            coords[i] = coords[i] * (1 - pull) + lc * pull
        coords -= coords.mean(axis=0, keepdims=True)
        coords /= coords.std() + 0.01
        diffs = coords[:, None] - coords[None, :]
        dists = np.linalg.norm(diffs, axis=-1)
        contact = 1.0 / (1.0 + (dists / self.scale) ** self.alpha)
        np.fill_diagonal(contact, 1.0)
        return seq_5ch, contact, coords, e_pos, p_pos

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s, c, crd, e, p = self.samples[idx]
        L = self.window
        ep_target = torch.zeros(L, L)
        ep_target[e, p] = 1.0; ep_target[p, e] = 1.0
        return {'sequence': torch.FloatTensor(s),
                'contact': torch.FloatTensor(c),
                'coords': torch.FloatTensor(crd),
                'ep_target': ep_target,
                'e_pos': torch.tensor(float(e)),
                'p_pos': torch.tensor(float(p))}


def contact_corr(p, t):
    n = p.shape[0]
    pu = p[np.triu_indices(n, 1)]
    tu = t[np.triu_indices(n, 1)]
    return np.corrcoef(pu, tu)[0, 1]


def procrustes(X, Y):
    X -= X.mean(axis=0, keepdims=True); Y -= Y.mean(axis=0, keepdims=True)
    U, _, Vt = np.linalg.svd(Y.T @ X, full_matrices=False)
    return Y @ (U @ Vt)


def reconstruction_error(true_coords, pred_coords):
    pa = procrustes(true_coords, pred_coords)
    err = np.linalg.norm(pa - true_coords) / np.linalg.norm(true_coords - true_coords.mean(0))
    return float(err)


def contacts_from_coords(coords, scale=2.0, alpha=3.0):
    diffs = coords[:, None] - coords[None, :]
    dists = np.linalg.norm(diffs, axis=-1)
    return 1.0 / (1.0 + (dists / scale) ** alpha)


def build_helix(e_pos, p_pos, radius, pull, L=32):
    B = e_pos.shape[0]
    device = e_pos.device
    t = torch.linspace(0, 4 * math.pi, L, device=device).view(1, L).expand(B, L)
    x = torch.sin(t) * radius.view(B, 1)
    y = torch.cos(t * 0.7) * (radius.view(B, 1) * 0.8)
    z = t * 0.3
    coords = torch.stack([x, y, z], dim=-1)
    positions = torch.arange(L, device=device).float().view(1, L)
    attn_e = torch.exp(-((positions - e_pos.view(B, 1)) ** 2) / 1.0)
    attn_e = attn_e / (attn_e.sum(dim=1, keepdim=True) + 1e-8)
    e_center = (attn_e.unsqueeze(-1) * coords).sum(dim=1)
    attn_p = torch.exp(-((positions - p_pos.view(B, 1)) ** 2) / 1.0)
    attn_p = attn_p / (attn_p.sum(dim=1, keepdim=True) + 1e-8)
    p_center = (attn_p.unsqueeze(-1) * coords).sum(dim=1)
    lc = (e_center + p_center) / 2
    de = (positions - e_pos.view(B, 1)).abs()
    dp = (positions - p_pos.view(B, 1)).abs()
    pull_factor = pull.view(B, 1) * torch.exp(-torch.min(de, dp) / 5.0)
    coords = coords * (1 - pull_factor.unsqueeze(-1)) + lc.unsqueeze(1) * pull_factor.unsqueeze(-1)
    coords = coords - coords.mean(dim=1, keepdim=True)
    coords = coords / (coords.std(dim=(1, 2), keepdim=True) + 0.01)
    return coords


class PredictiveCoding3D(torch.nn.Module):
    def __init__(self, steps=80, lr=0.3, scale=2.0, alpha=3.0):
        super().__init__()
        self.steps = steps
        self.lr = lr
        self.scale = scale
        self.alpha = alpha

    def forward(self, logits_obs, L=32, params_init=None):
        B = logits_obs.shape[0]
        device = logits_obs.device
        with torch.enable_grad():
            if params_init is None:
                max_pair = logits_obs.view(B, -1).argmax(dim=1)
                ep = (max_pair // L).float()
                pp = (max_pair % L).float()
                ep = ep.clamp(2, L - 8)
                pp = (pp - ep).clamp(min=4) + ep
                pp = pp.clamp(max=L - 4)
                rd = torch.full((B,), 5.0, device=device)
                pl = torch.full((B,), 0.5, device=device)
            else:
                ep, pp, rd, pl = params_init
            ep = ep.clone().detach().requires_grad_(True)
            pp = pp.clone().detach().requires_grad_(True)
            rd = rd.clone().detach().requires_grad_(True)
            pl = pl.clone().detach().requires_grad_(True)
            eye = torch.eye(L, device=device).view(1, L, L)
            for _ in range(self.steps):
                coords = build_helix(ep, pp, rd, pl, L=L)
                diffs = coords.unsqueeze(1) - coords.unsqueeze(2)
                dists = torch.norm(diffs, dim=-1) + 1e-8
                contacts_pred = 1.0 / (1.0 + (dists / self.scale) ** self.alpha)
                p_safe = contacts_pred.clamp(1e-6, 1 - 1e-6)
                logits_pred = torch.log(p_safe / (1 - p_safe))
                logits_pred = logits_pred * (1 - eye)
                err = ((logits_pred - logits_obs.detach()) ** 2) * (1 - eye)
                loss = err.sum() / (B * L * (L - 1))
                grads = torch.autograd.grad(loss, [ep, pp, rd, pl])
                with torch.no_grad():
                    ep = (ep - self.lr * grads[0]).clamp(2, L - 8)
                    pp = (pp - self.lr * grads[1]).clamp(min=ep + 4).clamp(max=L - 4)
                    rd = rd - self.lr * grads[2]
                    pl = pl - self.lr * grads[3]
                    rd.data = rd.data.clamp(0.5, 10.0)
                    pl.data = pl.data.clamp(0.01, 0.99)
                ep = ep.detach().requires_grad_(True)
                pp = pp.detach().requires_grad_(True)
                rd = rd.detach().requires_grad_(True)
                pl = pl.detach().requires_grad_(True)
        return ep.detach(), pp.detach(), rd.detach(), pl.detach()
