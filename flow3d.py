import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(torch.linspace(0, math.log(10000), half, device=t.device))
        emb = t * freqs[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class FlowMatching3D(nn.Module):
    def __init__(self, L=32, d_logit_hid=64, d_time=32, d_hid=256):
        super().__init__()
        self.L = L

        self.logit_encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.SiLU(),
            nn.Conv2d(8, 16, 3, padding=1), nn.SiLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(32, d_logit_hid)
        )

        self.time_embed = SinusoidalTimeEmbedding(d_time)
        self.cond_proj = nn.Linear(d_logit_hid, d_hid)
        self.time_proj = nn.Linear(d_time, d_hid)

        self.net = nn.Sequential(
            nn.Linear(L * 3 + d_hid + d_hid, d_hid * 2), nn.SiLU(),
            nn.Linear(d_hid * 2, d_hid * 2), nn.SiLU(),
            nn.Linear(d_hid * 2, L * 3)
        )

    @torch.no_grad()
    def sample(self, logits, steps=100):
        B = logits.shape[0]
        device = logits.device
        x = torch.randn(B, self.L, 3, device=device)
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((B, 1), i * dt, device=device)
            v = self.forward(x, logits, t)
            x = x + v * dt
        return x

    def forward(self, coords, logits, t):
        B = coords.shape[0]
        if t.dim() == 0:
            t = t.view(1).expand(B, 1)
        elif t.dim() == 1:
            t = t.view(B, 1)

        cond = self.logit_encoder(logits.unsqueeze(1))
        cond_h = self.cond_proj(cond)
        t_h = self.time_proj(self.time_embed(t))

        x_flat = coords.view(B, -1)
        h = torch.cat([x_flat, cond_h, t_h], dim=-1)
        return self.net(h).view(B, self.L, 3)


class FlowMatchingTrainer:
    def __init__(self, model, lr=1e-3, weight_decay=1e-4):
        self.model = model
        self.opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def loss_fn(self, coords, logits):
        B = coords.shape[0]
        t = torch.rand(B, 1, device=coords.device)
        noise = torch.randn_like(coords)
        xt = (1 - t[:, :, None]) * noise + t[:, :, None] * coords
        v_target = coords - noise
        v_pred = self.model(xt, logits, t)
        return F.mse_loss(v_pred, v_target)

    def train_step(self, coords, logits):
        self.opt.zero_grad()
        loss = self.loss_fn(coords, logits)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()
        return loss.item()


