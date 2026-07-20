import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import Dataset, DataLoader
from model3d import GEMINI3D
import random

class Synthetic3DDataset(Dataset):
    def __init__(self, num_samples=500, window=64, seed=42):
        self.num_samples = num_samples
        self.window = window
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.samples = []
        for s in range(num_samples):
            seq, contact_3d, coords_3d = self._generate_3d_sample()
            self.samples.append((seq, contact_3d, coords_3d))

    def _generate_3d_sample(self):
        L = self.window
        seq = np.zeros((L, 5), dtype=np.float32)
        bases = ['A', 'C', 'G', 'T']
        nuc_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        for i in range(L):
            nuc = self.rng.choice(bases, p=[0.26, 0.24, 0.24, 0.26])
            seq[i, nuc_to_idx[nuc]] = 1.0
        seq[:, 4] = 1.0 - seq[:, :4].sum(axis=1)

        e_pos = self.rng.randint(0, L // 4 - 4)
        p_pos = self.rng.randint(L // 2, L - 4)
        for off in range(4):
            if e_pos + off < L:
                seq[e_pos + off, :4] = 0
                seq[e_pos + off, 4] = 1.0
            if p_pos + off < L:
                seq[p_pos + off, :4] = 0
                seq[p_pos + off, 4] = 1.0
        if e_pos + 4 <= L:
            tata = [3, 0, 3, 0]
            for off, n in enumerate(tata):
                seq[e_pos + off, :4] = 0
                seq[e_pos + off, n] = 1.0
        if p_pos + 4 <= L:
            ccgc = [1, 1, 2, 1]
            for off, n in enumerate(ccgc):
                seq[p_pos + off, :4] = 0
                seq[p_pos + off, n] = 1.0
        seq[:, 4] = 1.0 - seq[:, :4].sum(axis=1)

        coords = np.zeros((L, 3), dtype=np.float32)
        theta = 0.0
        for i in range(L):
            theta += self.rng.normal(0.3, 0.3)
            phi = self.rng.uniform(0, 2 * np.pi)
            step = 1.0
            dx = step * np.sin(theta) * np.cos(phi)
            dy = step * np.sin(theta) * np.sin(phi)
            dz = step * np.cos(theta)
            if i == 0:
                coords[i] = [0, 0, 0]
            else:
                coords[i] = coords[i-1] + [dx, dy, dz]
        loop_center = (coords[e_pos] + coords[p_pos]) / 2
        for i in range(L):
            dist_to_ep = min(abs(i - e_pos), abs(i - p_pos))
            pull = 0.5 * np.exp(-dist_to_ep / 8.0)
            coords[i] = coords[i] * (1 - pull) + loop_center * pull

        dist = np.zeros((L, L), dtype=np.float32)
        for i in range(L):
            for j in range(L):
                d = np.linalg.norm(coords[i] - coords[j])
                dist[i, j] = d
        contact = 1.0 / (1.0 + (dist / 3.0) ** 3)
        contact += 0.5 * np.exp(-(((np.arange(L)[:, None] - np.arange(L)[None, :]) / 20.0) ** 2))
        contact = np.clip(contact, 0.0, 1.0)

        return seq, contact, coords

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        seq, contact, coords = self.samples[idx]
        return {
            'sequence': torch.FloatTensor(seq),
            'contact': torch.FloatTensor(contact),
            'coords': torch.FloatTensor(coords),
        }


def train_3d(model, loader, opt, epochs=20, device='cpu'):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in loader:
            x = batch['sequence'].to(device)
            target = batch['contact'].to(device)
            opt.zero_grad()
            logits, probs, _ = model(x)
            loss = F.binary_cross_entropy(probs, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.6f}")


def evaluate_3d(model, dataset, device='cpu'):
    model.eval()
    idx = np.random.randint(len(dataset))
    sample = dataset[idx]
    x = sample['sequence'].unsqueeze(0).to(device)
    true_coords = sample['coords'].numpy()
    true_contact = sample['contact'].numpy()

    pred_coords, pred_contact = model.predict_3d(x)

    n = pred_contact.shape[0]
    true_upper = true_contact[np.triu_indices(n, 1)]
    pred_upper = pred_contact[np.triu_indices(n, 1)]
    corr = np.corrcoef(true_upper, pred_upper)[0, 1]
    print(f"  Contact correlation: {corr:.4f}")

    fig = plt.figure(figsize=(18, 6))
    ax1 = fig.add_subplot(131)
    ax1.imshow(true_contact, cmap='Greys_r', vmin=0, vmax=1)
    ax1.set_title("True Contact Map")
    ax1.set_xlabel("Position"); ax1.set_ylabel("Position")

    ax2 = fig.add_subplot(132)
    ax2.imshow(pred_contact, cmap='Greys_r', vmin=0, vmax=1)
    ax2.set_title(f"Predicted (r={corr:.3f})")
    ax2.set_xlabel("Position"); ax2.set_ylabel("Position")

    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(true_coords[:, 0], true_coords[:, 1], true_coords[:, 2],
                c=range(n), cmap='viridis', s=40, label='True')
    ax3.scatter(pred_coords[:, 0], pred_coords[:, 1], pred_coords[:, 2],
                c=range(n), cmap='plasma', s=20, marker='x', label='Predicted')
    ax3.set_title("3D Reconstruction")
    ax3.legend()

    plt.tight_layout()
    plt.savefig('3d_reconstruction.png', dpi=150, bbox_inches='tight')
    print(f"Saved to 3d_reconstruction.png")
    plt.close()
    return corr


if __name__ == '__main__':
    device = 'cpu'
    window = 64
    print(f"=== GEMINI-3D: Deep Boltzmann Machine for Hi-C 3D Reconstruction ===")
    print(f"Window: {window}bp, Device: {device}")

    print("\nGenerating synthetic 3D Hi-C data...")
    train_ds = Synthetic3DDataset(num_samples=500, window=window, seed=42)
    val_ds = Synthetic3DDataset(num_samples=50, window=window, seed=100)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)

    model = GEMINI3D(window=window, dim=32).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}, Trainable: {trainable:,}")

    print("\nPhase 1: Training contact predictor...")
    opt1 = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_3d(model, train_loader, opt1, epochs=30, device=device)

    print("\nPhase 2: Pre-training Deep Boltzmann Machine...")
    model.dbm.rbm1.W.data *= 0.1
    model.dbm.rbm2.W.data *= 0.1
    for epoch in range(10):
        avg_grad1, avg_grad2 = 0.0, 0.0
        for batch in train_loader:
            x = batch['sequence'].to(device)
            target = batch['contact'].to(device)
            with torch.no_grad():
                _, probs, _ = model(x)
            L = probs.shape[1]
            triu = torch.triu_indices(L, L, offset=1)
            v = target[:, triu[0], triu[1]]
            v_pred = probs[:, triu[0], triu[1]]
            v_combined = (v + v_pred) / 2
            g1, g2 = model.dbm.pretrain(v_combined, lr=0.005, k=1)
            avg_grad1 += g1; avg_grad2 += g2
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}, |dW1|={avg_grad1/len(train_loader):.4f}, |dW2|={avg_grad2/len(train_loader):.4f}")

    print("\nPhase 3: Fine-tuning with refinement...")
    opt3 = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for epoch in range(10):
        total_loss = 0.0
        for batch in train_loader:
            x = batch['sequence'].to(device)
            target = batch['contact'].to(device)
            opt3.zero_grad()
            _, _, probs_refined = model(x, refine=True)
            loss = F.binary_cross_entropy(probs_refined, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt3.step()
            total_loss += loss.item()
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.6f}")

    print("\nEvaluating 3D reconstruction...")
    corr = evaluate_3d(model, val_ds, device=device)
    print(f"\nDone. Final contact correlation: {corr:.4f}")
