"""Fully-connected autoencoder for reconstruction-based anomaly scoring."""

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden: tuple[int, ...] = (128, 64), latent: int = 16):
        super().__init__()
        enc, dim = [], input_dim
        for h in hidden:
            enc += [nn.Linear(dim, h), nn.ReLU()]
            dim = h
        enc.append(nn.Linear(dim, latent))
        self.encoder = nn.Sequential(*enc)

        dec, dim = [], latent
        for h in reversed(hidden):
            dec += [nn.Linear(dim, h), nn.ReLU()]
            dim = h
        dec.append(nn.Linear(dim, input_dim))
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        return self.decoder(self.encoder(x))


def train_autoencoder(
    X_train: np.ndarray,
    epochs: int = 12,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = 0,
    verbose: bool = False,
) -> Autoencoder:
    torch.manual_seed(seed)
    model = Autoencoder(X_train.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train.astype(np.float32))),
        batch_size=batch_size,
        shuffle=True,
    )
    model.train()
    for epoch in range(epochs):
        total = 0.0
        for (batch,) in loader:
            opt.zero_grad()
            loss = loss_fn(model(batch), batch)
            loss.backward()
            opt.step()
            total += loss.item() * len(batch)
        if verbose:
            print(f"    epoch {epoch + 1:2d}/{epochs}  mse={total / len(X_train):.5f}")
    return model


@torch.no_grad()
def reconstruction_errors(model: Autoencoder, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    """Per-sample mean squared reconstruction error."""
    model.eval()
    errs = []
    for i in range(0, len(X), batch_size):
        batch = torch.from_numpy(X[i : i + batch_size].astype(np.float32))
        errs.append(((model(batch) - batch) ** 2).mean(dim=1).numpy())
    return np.concatenate(errs)


def empirical_threshold(train_errors: np.ndarray, quantile: float = 0.99) -> float:
    """Anomaly threshold from the empirical training-loss distribution."""
    return float(np.quantile(train_errors, quantile))
