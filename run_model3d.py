import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import Dataset, DataLoader
from model3d import GEMINI3D

class Synthetic3DDataset(Dataset):
    def __init__(self, num_samples=500, window=64, seed=42):
        self.window = window
        self.rng = np.random.RandomState(seed)
        self.samples = []
        for _ in range(num_samples):
            seq, contact, coords = self._generate()
            self.samples.append((seq, contact, coords))

    def _generate(self):
        L = self.window
        seq = np.zeros((L, 5), dtype=np.float32)
        bases, ni = ['A','C','G','T'], {'A':0,'C':1,'G':2,'T':3}
        for i in range(L):
            n = self.rng.choice(bases, p=[0.26,0.24,0.24,0.26])
            seq[i, ni[n]] = 1.0
        gap = self.rng.randint(8, 24)
        e_pos = self.rng.randint(2, L//3)
        p_pos = min(e_pos + gap, L - 6)
        if p_pos == L - 6: e_pos = p_pos - gap
        for off,n in enumerate([3,0,3,0]):
            seq[e_pos+off,:4]=0; seq[e_pos+off,n]=1.0
        for off,n in enumerate([1,1,2,1]):
            seq[p_pos+off,:4]=0; seq[p_pos+off,n]=1.0
        seq[:,4] = 1.0 - seq[:,:4].sum(axis=1)

        contact = np.zeros((L,L), dtype=np.float32)
        for i in range(L):
            for j in range(L):
                d = abs(i-j)
                contact[i,j] = np.exp(-d/15.0) + 0.3*np.exp(-d/3.0)
        for peak in [3.0]:
            contact[e_pos,p_pos] += peak; contact[p_pos,e_pos] += peak
            for i in range(L):
                for j in range(L):
                    de = abs(i-e_pos)+abs(j-p_pos)
                    dp = abs(i-p_pos)+abs(j-e_pos)
                    contact[i,j] += peak*np.exp(-min(de,dp)/6.0)
        contact = np.clip(contact, 0, 1)

        coords = np.zeros((L,3), dtype=np.float32)
        t = np.linspace(0, 4*np.pi, L)
        coords[:,0] = np.sin(t)*5 + np.random.randn(L)*0.5
        coords[:,1] = np.cos(t*0.7)*4 + np.random.randn(L)*0.5
        coords[:,2] = t*0.3 + np.random.randn(L)*0.3
        lc = (coords[e_pos]+coords[p_pos])/2
        for i in range(L):
            pull = 0.8*np.exp(-min(abs(i-e_pos),abs(i-p_pos))/5.0)
            coords[i] = coords[i]*(1-pull) + lc*pull
        coords -= coords.mean(axis=0, keepdims=True)
        coords /= coords.std() + 0.01
        return seq, contact, coords

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        s,c,crd = self.samples[idx]
        return {'sequence': torch.FloatTensor(s),
                'contact': torch.FloatTensor(c),
                'coords': torch.FloatTensor(crd)}

def contact_corr(p,t):
    n=p.shape[0]; pu=p[np.triu_indices(n,1)]; tu=t[np.triu_indices(n,1)]
    return np.corrcoef(pu,tu)[0,1]

def procrustes(X,Y):
    X-=X.mean(axis=0,keepdims=True); Y-=Y.mean(axis=0,keepdims=True)
    U,_,Vt=np.linalg.svd(Y.T@X,full_matrices=False)
    return Y@(U@Vt)

def eval_contact(model,loader,device):
    model.eval(); corrs=[]
    with torch.no_grad():
        for b in loader:
            x=b['sequence'].to(device); t=b['contact'].numpy()
            _,_,pr,_=model(x)
            for bb in range(pr.shape[0]): corrs.append(contact_corr(pr[bb].cpu().numpy(),t[bb]))
    return float(np.mean(corrs))

if __name__ == '__main__':
    device='cpu'; window=64
    print(f"=== GEMINI-3D: Boltzmann Force Refinement ===")

    train_ds=Synthetic3DDataset(num_samples=500,window=window,seed=42)
    val_ds=Synthetic3DDataset(num_samples=100,window=window,seed=100)
    tl=DataLoader(train_ds,batch_size=16,shuffle=True)
    vl=DataLoader(val_ds,batch_size=16,shuffle=False)

    model=GEMINI3D(window=window,dim=32,refine_steps=30).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    print("Phase 1: Contact predictor (40 epochs)...")
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    for ep in range(40):
        model.train(); tot=0.0
        for b in tl:
            x=b['sequence'].to(device); tc=b['contact'].to(device)
            opt.zero_grad()
            _,_,pr,_=model(x)
            loss=F.binary_cross_entropy(pr,tc); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step(); tot+=loss.item()
        if (ep+1)%10==0 or ep==0:
            vc=eval_contact(model,vl,device)
            print(f"  Ep {ep+1:2d}/40 Loss:{tot/len(tl):.4f} Val:{vc:.4f}")

    print("Phase 2: DBM pretrain (15 epochs)...")
    model.dbm.rbm1.W.data.mul_(0.5); model.dbm.rbm2.W.data.mul_(0.5)
    for ep in range(15):
        g1,g2=0.0,0.0
        for b in tl:
            x=b['sequence'].to(device); tc=b['contact'].to(device)
            with torch.no_grad(): _,_,pr,_=model(x)
            L=pr.shape[1]; triu=torch.triu_indices(L,L,offset=1)
            v=tc[:,triu[0],triu[1]]; vp=pr[:,triu[0],triu[1]]
            gg1,gg2=model.dbm.pretrain((v+vp)/2,lr=0.002,k=1)
            g1+=gg1; g2+=gg2
        if (ep+1)%5==0: print(f"  Ep {ep+1:2d}/15 |dW1|={g1/len(tl):.1f} |dW2|={g2/len(tl):.1f}")

    print("Final evaluation..."); model.eval()
    fig,axes=plt.subplots(5,3,figsize=(15,20)); corrs=[]
    for idx in range(5):
        s=val_ds[np.random.randint(len(val_ds))]
        x=s['sequence'].unsqueeze(0).to(device)
        tc=s['contact'].numpy(); ty=s['coords'].numpy()
        py,pc=model.predict_3d(x); r=contact_corr(pc,tc); corrs.append(r)
        pa=procrustes(ty,py)
        axes[idx][0].imshow(tc,cmap='Greys_r',vmin=0,vmax=1); axes[idx][0].set_title(f"True [{idx+1}]")
        axes[idx][1].imshow(pc,cmap='Greys_r',vmin=0,vmax=1); axes[idx][1].set_title(f"Pred (r={r:.3f})")
        a2=fig.add_subplot(5,3,idx*3+3,projection='3d')
        a2.scatter(ty[:,0],ty[:,1],ty[:,2],c=range(len(ty)),cmap='viridis',s=30,label='True')
        a2.scatter(pa[:,0],pa[:,1],pa[:,2],c=range(len(pa)),cmap='plasma',s=15,marker='x',label='Pred')
        a2.set_title(f"3D (r={r:.3f})")
    plt.tight_layout(); plt.savefig('3d_reconstruction.png',dpi=150,bbox_inches='tight'); plt.close()
    print(f"Mean contact corr: {np.mean(corrs):.4f}")
