import torch, torch.nn as nn
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from model import GEMINITiny
from flow3d import FlowMatching3D
from utils import Synthetic3DDataset, contact_corr, procrustes, reconstruction_error, contacts_from_coords, build_helix, PredictiveCoding3D


def norm_coords(c):
    c = c - c.mean(dim=1, keepdim=True)
    c = c / (c.std(dim=(1, 2), keepdim=True) + 0.01)
    return c


class EMA:
    def __init__(self, model, decay=0.9995):
        self.model = model
        self.decay = decay
        self.shadow = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        self.backup = None

    def update(self):
        for k, v in self.model.state_dict().items():
            self.shadow[k] = self.decay * self.shadow[k] + (1 - self.decay) * v.cpu()

    def apply(self):
        self.backup = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
        self.model.load_state_dict(self.shadow)

    def restore(self):
        if self.backup is not None:
            self.model.load_state_dict(self.backup)


if __name__ == '__main__':
    torch.set_num_threads(1)
    torch.manual_seed(42)
    np.random.seed(42)
    device = 'cpu'
    window = 32

    print("=" * 60)
    print("  Flow Matching 3D v3 — Going for 0.999")
    print("  5000 samples, 500 epochs, EMA, 200-step inference")
    print("=" * 60)

    train_ds = Synthetic3DDataset(num_samples=5000, window=window, seed=42)
    val_ds = Synthetic3DDataset(num_samples=200, window=window, seed=100)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    print("\nPhase 1: Training GEMINITiny...")
    tiny = GEMINITiny().to(device)
    opt_t = torch.optim.AdamW(tiny.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([30.0]).to(device))
    for ep in range(80):
        tiny.train()
        for b in train_loader:
            x = b['sequence'].to(device)
            t2d = b['ep_target'].to(device)
            opt_t.zero_grad()
            (logits, probs), _ = tiny(x)
            loss = crit(logits, t2d) + 0.5 * torch.mean(probs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tiny.parameters(), 1.0)
            opt_t.step()
    tiny.eval()
    print("  Done.")

    print("\nPhase 2: Generating conditioning...")
    def gen(ds):
        ll, cc = [], []
        for idx in range(len(ds)):
            s = ds[idx]
            x = s['sequence'].unsqueeze(0).to(device)
            with torch.no_grad():
                (lg, _), _ = tiny(x)
            ll.append(lg.cpu())
            cc.append(s['coords'].unsqueeze(0))
        return torch.cat(ll), norm_coords(torch.cat(cc))

    train_logits, train_coords = gen(train_ds)
    val_logits, val_coords = gen(val_ds)
    print(f"  Train: {train_logits.shape[0]}, Val: {val_logits.shape[0]}")

    print("\nPhase 3: Training FlowMatching3D (v1 config, 500 epochs, EMA)...")
    model = FlowMatching3D(L=window, d_hid=256).to(device)
    ema = EMA(model, decay=0.9995)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    n_epochs = 500
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    n_train = train_logits.shape[0]
    batch_size = 64
    best_corr = 0.0
    best_state = None
    loss_trace, corr_trace = [], []

    for ep in range(n_epochs):
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        n_batches = 0
        model.train()

        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            cb = train_coords[idx].to(device)
            lb = train_logits[idx].to(device)

            t = torch.rand(cb.shape[0], 1, device=device)
            noise = torch.randn_like(cb)
            xt = (1 - t[:, :, None]) * noise + t[:, :, None] * cb
            v_target = cb - noise
            v_pred = model(xt, lb, t)
            loss = nn.functional.mse_loss(v_pred, v_target)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ema.update()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        loss_trace.append(epoch_loss / n_batches)

        if (ep + 1) % 50 == 0 or ep == 0:
            model.eval()
            ema.apply()
            with torch.no_grad():
                samples = model.sample(val_logits[:64].to(device), steps=200)
                corrs = []
                for j in range(samples.shape[0]):
                    cp = contacts_from_coords(samples[j].cpu().numpy())
                    tc = val_coords[j].cpu().numpy()
                    cr = contact_corr(cp, contacts_from_coords(tc))
                    corrs.append(cr)
                mc = float(np.mean(corrs))
                corr_trace.append(mc)
                if mc > best_corr:
                    best_corr = mc
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            ema.restore()
            print(f"  Ep {ep+1:4d}/{n_epochs} L={loss_trace[-1]:.6f} "
                  f"corr={mc:.6f} best={best_corr:.6f} lr={scheduler.get_last_lr()[0]:.1e}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    torch.save(best_state, 'flow_v3_checkpoint.pt')
    print("  Saved flow_v3_checkpoint.pt")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(loss_trace)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    v_ep = list(range(0, n_epochs + 1, 50))[:len(corr_trace)]
    axes[1].plot(v_ep, corr_trace, marker='o')
    axes[1].axhline(best_corr, color='g', linestyle='--', label=f'Best={best_corr:.6f}')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Val Contact Corr')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig('flow_v3_training.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Best val corr: {best_corr:.6f}")

    # Phase 4: Full evaluation
    print("\nPhase 4: Full validation (200 samples, 200 inference steps)...")
    pc3d = PredictiveCoding3D(steps=80, lr=0.3, scale=2.0, alpha=3.0).to(device)

    # Inference mode: try 200 steps
    results = []
    for idx in range(len(val_ds)):
        lb = val_logits[idx:idx+1].to(device)
        tc = val_coords[idx].numpy()
        true_contact = contacts_from_coords(tc)

        # PC-3D
        with torch.enable_grad():
            ep, pp, rd, pl = pc3d(lb, L=window)
        pc3d_c = build_helix(ep, pp, rd, pl, L=window)[0].detach().cpu().numpy()
        pc3d_c = pc3d_c - pc3d_c.mean(axis=0, keepdims=True)
        pc3d_c = pc3d_c / (pc3d_c.std() + 0.01)

        # Flow (200 steps)
        with torch.no_grad():
            fc = model.sample(lb, steps=200)[0].cpu().numpy()

        results.append({
            'pc3d_corr': contact_corr(contacts_from_coords(pc3d_c), true_contact),
            'pc3d_err': reconstruction_error(tc, pc3d_c),
            'flow_corr': contact_corr(contacts_from_coords(fc), true_contact),
            'flow_err': reconstruction_error(tc, fc),
            'pc3d_c': pc3d_c, 'flow_c': fc, 'true_c': tc,
            'true_contact': true_contact
        })

    pc3d_corrs = np.array([r['pc3d_corr'] for r in results])
    flow_corrs = np.array([r['flow_corr'] for r in results])
    pc3d_errs = np.array([r['pc3d_err'] for r in results])
    flow_errs = np.array([r['flow_err'] for r in results])

    print(f"\n{'='*60}")
    print(f"  COMPARISON: PC-3D vs FlowMatching3D v3")
    print(f"{'='*60}")
    print(f"  {'Method':<24} {'Contact corr':<22} {'3D error':<18}")
    print(f"  {'-'*24} {'-'*22} {'-'*18}")
    print(f"  {'PC-3D':<24} {pc3d_corrs.mean():.6f} +/- {pc3d_corrs.std():.6f}  {pc3d_errs.mean():.6f}")
    print(f"  {'FlowMatching3D':<24} {flow_corrs.mean():.6f} +/- {flow_corrs.std():.6f}  {flow_errs.mean():.6f}")

    wc = (flow_corrs > pc3d_corrs).sum()
    we = (flow_errs < pc3d_errs).sum()
    print(f"\n  Flow wins: {wc}/200 (corr) {we}/200 (err)")

    p99 = (flow_corrs > 0.99).sum()
    p98 = (flow_corrs > 0.98).sum()
    p97 = (flow_corrs > 0.97).sum()
    print(f"  corr > 0.99: {p99}/200")
    print(f"  corr > 0.98: {p98}/200")
    print(f"  corr > 0.97: {p97}/200")
    print(f"  Range: [{flow_corrs.min():.6f}, {flow_corrs.max():.6f}]")

    # Phase 5: Visualizations
    print("\nPhase 5: Visualizations...")
    # Best/median/worst contact maps
    best_i = flow_corrs.argmax()
    worst_i = flow_corrs.argmin()
    mid_i = len(results) // 2
    sorted_i = np.argsort(flow_corrs)
    med_i = sorted_i[len(sorted_i)//2]

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3)
    titles = ['Best', 'Median', 'Worst']
    picks = [best_i, med_i, worst_i]
    for col, (idx, title) in enumerate(zip(picks, titles)):
        r = results[idx]
        tc = r['true_contact']
        fc = contacts_from_coords(r['flow_c'])
        pc = contacts_from_coords(r['pc3d_c'])
        tcoords = r['true_c']
        fcoords = r['flow_c']

        ax = fig.add_subplot(gs[0, col])
        ax.imshow(tc, cmap='Greys_r', vmin=0, vmax=1)
        ax.set_title(f'{title}: TRUE', fontsize=10)

        ax = fig.add_subplot(gs[1, col])
        ax.imshow(fc, cmap='Greys_r', vmin=0, vmax=1)
        ax.set_title(f'Flow corr={r["flow_corr"]:.6f}', fontsize=10)

        ax3d = fig.add_subplot(gs[2, col], projection='3d')
        pa = procrustes(tcoords, fcoords)
        ax3d.scatter(tcoords[:, 0], tcoords[:, 1], tcoords[:, 2],
                    c=np.arange(32), cmap='viridis', s=60, alpha=0.9, label='TRUE')
        ax3d.scatter(pa[:, 0], pa[:, 1], pa[:, 2],
                    c=np.arange(32), cmap='cool', s=20, marker='^', alpha=0.8, label=f'err={r["flow_err"]:.6f}')
        ax3d.set_title(f'3D err={r["flow_err"]:.6f}', fontsize=10)
        ax3d.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig('flow_v3_best_median_worst.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved flow_v3_best_median_worst.png')

    # Summary histograms
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].hist(flow_corrs, bins=40, alpha=0.7, color='blue')
    axes[0, 0].axvline(flow_corrs.mean(), color='k', linestyle='--', label=f'Mean={flow_corrs.mean():.4f}')
    axes[0, 0].axvline(0.99, color='g', linestyle=':', label='r=0.99')
    axes[0, 0].set_xlabel('Contact correlation')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title(f'Flow Matching 3D v3 (n={len(results)})')
    axes[0, 0].legend()

    axes[0, 1].hist(flow_errs, bins=40, alpha=0.7, color='purple')
    axes[0, 1].axvline(flow_errs.mean(), color='k', linestyle='--', label=f'Mean={flow_errs.mean():.4f}')
    axes[0, 1].set_xlabel('3D error')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('3D Reconstruction Error')
    axes[0, 1].legend()

    axes[1, 0].scatter(pc3d_corrs, flow_corrs, alpha=0.4, s=12)
    axes[1, 0].plot([0, 1], [0, 1], 'k--', alpha=0.3)
    axes[1, 0].set_xlabel('PC-3D corr')
    axes[1, 0].set_ylabel('Flow corr')
    axes[1, 0].set_title('Flow vs PC-3D (above = Flow wins)')

    axes[1, 1].scatter(pc3d_errs, flow_errs, alpha=0.4, s=12)
    axes[1, 1].plot([0, 1], [0, 1], 'k--', alpha=0.3)
    axes[1, 1].set_xlabel('PC-3D err')
    axes[1, 1].set_ylabel('Flow err')
    axes[1, 1].set_title('Flow vs PC-3D (below = Flow wins)')
    plt.tight_layout()
    plt.savefig('flow_v3_histograms.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved flow_v3_histograms.png')

    # Final
    print(f"\n{'='*60}")
    print(f"  FINAL v3")
    print(f"{'='*60}")
    print(f"  PC-3D:         corr={pc3d_corrs.mean():.6f} err={pc3d_errs.mean():.6f}")
    print(f"  FlowMatching3D: corr={flow_corrs.mean():.6f} err={flow_errs.mean():.6f}")
    print(f"  Best val:       {best_corr:.6f}")
    print(f"  >0.99: {p99}/200  >0.98: {p98}/200  >0.97: {p97}/200")
    print(f"  Range: [{flow_corrs.min():.6f}, {flow_corrs.max():.6f}]")

    # Try more inference steps
    print("\n  Inference step ablation:")
    for n_steps in [100, 200, 500, 1000]:
        with torch.no_grad():
            samples = model.sample(val_logits.to(device), steps=n_steps)
        corrs = []
        for j in range(200):
            cp = contacts_from_coords(samples[j].cpu().numpy())
            tc = val_coords[j].numpy()
            cr = contact_corr(cp, contacts_from_coords(tc))
            corrs.append(cr)
        corrs = np.array(corrs)
        print(f"    {n_steps:4d} steps: mean={corrs.mean():.6f} max={corrs.max():.6f} >0.99={(corrs>0.99).sum()}/200")
