 """
autoencoder.py
Simple fully-connected autoencoder for feature compression.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class DenseAutoencoder(nn.Module):
    def __init__(self, in_dim, latent_dim=64, hidden=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, in_dim)
        )
    def forward(self, x):
        z = self.encoder(x)
        xr = self.decoder(z)
        return xr, z

def train_autoencoder(features, latent_dim=64, epochs=80, batch=128, lr=1e-3, device="cpu", verbose=True):
    device = torch.device(device)
    X = torch.from_numpy(features.astype(np.float32))
    ds = TensorDataset(X)
    loader = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=0)
    model = DenseAutoencoder(features.shape[1], latent_dim=latent_dim).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for ep in range(epochs):
        total = 0.0
        for (xb,) in loader:
            xb = xb.to(device)
            xr, _ = model(xb)
            loss = loss_fn(xr, xb)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * xb.size(0)
        if verbose and ((ep+1) % 10 == 0 or ep == 0):
            print(f"[AE] Epoch {ep+1}/{epochs} MSE={total/len(X):.6f}")
    model.cpu(); model.eval()
    return model

def compress_with_autoencoder(model, features, device="cpu", batch=1024):
    model.to(device); model.eval()
    X = torch.from_numpy(features.astype(np.float32))
    ds = TensorDataset(X)
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0)
    zs = []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            _, z = model(xb)
            zs.append(z.cpu().numpy())
    return np.vstack(zs)

