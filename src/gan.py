 """
gan.py
A CPU-friendly GAN to generate PSD-like feature vectors.
This is a classical (non-WGAN) stable MLP-GAN tuned for CPU.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class GAN_Generator(nn.Module):
    def __init__(self, zdim, out_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(zdim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, z):
        return self.net(z)

class GAN_Discriminator(nn.Module):
    def __init__(self, in_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden//2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden//2, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

def train_gan(features, zdim=64, epochs=80, batch=64, lr=2e-4, device="cpu", verbose=True):
    """
    features: numpy array (n_samples, feat_dim)
    returns trained generator (on CPU)
    """
    device = torch.device(device)
    X = torch.from_numpy(features.astype(np.float32))
    ds = TensorDataset(X)
    loader = DataLoader(ds, batch_size=batch, shuffle=True, drop_last=True, num_workers=0)
    feat_dim = features.shape[1]
    G = GAN_Generator(zdim, feat_dim).to(device)
    D = GAN_Discriminator(feat_dim).to(device)
    optG = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    optD = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    bce = nn.BCELoss()
    for ep in range(epochs):
        for (xb,) in loader:
            xb = xb.to(device)
            bsize = xb.size(0)
            # train D
            z = torch.randn(bsize, zdim, device=device)
            fake = G(z).detach()
            pred_real = D(xb)
            pred_fake = D(fake)
            lossD = bce(pred_real, torch.ones_like(pred_real)) + bce(pred_fake, torch.zeros_like(pred_fake))
            optD.zero_grad(); lossD.backward(); optD.step()
            # train G
            z = torch.randn(bsize, zdim, device=device)
            fake2 = G(z)
            pred = D(fake2)
            lossG = bce(pred, torch.ones_like(pred))
            optG.zero_grad(); lossG.backward(); optG.step()
        if verbose and ((ep+1) % 10 == 0 or ep == 0):
            print(f"[GAN] Epoch {ep+1}/{epochs} lossD={lossD.item():.4f} lossG={lossG.item():.4f}")
    return G.cpu()

def generate_from_g(G, n_samples, zdim=64, device="cpu", batch=1024):
    G.to(device); G.eval()
    out = []
    with torch.no_grad():
        for i in range(0, n_samples, batch):
            b = min(batch, n_samples - i)
            z = torch.randn(b, zdim, device=device)
            out.append(G(z).cpu().numpy())
    return np.vstack(out)
