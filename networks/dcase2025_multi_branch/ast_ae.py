"""
Simplified Anomalous Sound Detection model for DCASE2025 Task 2
================================================================

Architecture
------------
Raw Audio → Log‑Mel Spectrogram → **Fine‑tuned AST Encoder** → latent **z**  
                                                               ↘            
                                                       **Decoder** → Reconstructed Spectrogram  

Anomaly scoring:
* **Mahalanobis distance** in latent space measures global deviation of *z* from the normal cluster.
* **Reconstruction MSE** measures low‑level signal mismatch.
* Combined score  `S = α · M(z) + (1 – α) · E(x, \hat x)` with α∈[0,1].

This single‑branch model keeps the strengths of discriminative and generative
approaches while remaining easy to train on a single GPU.
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import ASTModel
from models.branch_decoder import SpectroDecoder
from models.branch_astencoder import ASTEncoder


# -----------------------------------------------------------------------------
# 3. Full Autoencoder + scoring utilities
# -----------------------------------------------------------------------------
class ASTAutoencoderASD(nn.Module):
    """End‑to‑end model producing latent embeddings, reconstructions, and scores."""

    def __init__(
        self,
        latent_dim: int = 128,
        n_mels: int = 128,
        alpha: float = 0.5,
    ) -> None:
        super().__init__()
        self.encoder = ASTEncoder(latent_dim=latent_dim)
        self.decoder = SpectroDecoder(latent_dim=latent_dim, n_mels=n_mels)
        self.alpha = alpha

        # Mean and precision for Mahalanobis – initialised later via `fit_stats`.
        self.register_buffer("mu", torch.zeros(latent_dim))
        self.register_buffer("inv_cov", torch.eye(latent_dim))

    # ------------------------------------------------------------------
    # Forward pass (training) returns reconstruction loss.
    # ------------------------------------------------------------------
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Return recon, latent *z*, and per‑sample MSE reconstruction error."""
        B, _, _, T = x.shape
        z = self.encoder(x)                # [B, latent]
        recon = self.decoder(z, T)         # [B, 1, n_mels, T]
        mse = F.mse_loss(recon, x, reduction="none")
        mse = mse.mean(dim=[1, 2, 3])      # [B]
        return recon, z, mse

    # ------------------------------------------------------------------
    # Statistics fitting – call once on *normal* training data.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def fit_stats(self, dataloader: torch.utils.data.DataLoader) -> None:
        """Compute mean and inverse covariance of latent *z* on NORMAL data."""
        zs: List[Tensor] = []
        device = next(self.parameters()).device
        for xb, _ in dataloader:
            xb = xb.to(device)
            z = self.encoder(xb)
            zs.append(z)
        z_all = torch.cat(zs, dim=0)
        self.mu = z_all.mean(0, keepdim=False)
        cov = torch.cov(z_all.T) + 1e-6 * torch.eye(z_all.size(1), device=device)
        self.inv_cov = torch.linalg.inv(cov)

    # ------------------------------------------------------------------
    # Scoring – used at inference.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def anomaly_score(self, x: Tensor) -> Tensor:
        """Compute combined anomaly score for input batch."""
        recon, z, mse = self.forward(x)

        # Mahalanobis distance D_M(z)
        delta = z - self.mu
        m_dist = torch.einsum("bi,ij,bj->b", delta, self.inv_cov, delta)

        score = self.alpha * m_dist + (1.0 - self.alpha) * mse
        return score


# -----------------------------------------------------------------------------
# 4. Example training loop skeleton (pseudo‑code)
# -----------------------------------------------------------------------------

def train_one_epoch(model: ASTAutoencoderASD, loader, optim, device: torch.device):
    """This is the pseudo-code for a single training epoch."""
    model.train()
    total_loss = 0.0
    for xb, _ in loader:
        xb = xb.to(device)
        rc, z, mse = model(xb)
        loss = mse.mean()
        optim.zero_grad()
        loss.backward()
        optim.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)


def validate(model: ASTAutoencoderASD, loader, device: torch.device):
    model.eval()
    scores = []
    labels = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            score = model.anomaly_score(xb)
            scores.extend(score.cpu().tolist())
            labels.extend(yb.cpu().tolist())
    # Compute AUC / pAUC, etc. – left as exercise
    return scores, labels


# -----------------------------------------------------------------------------
# 5. Quick usage example (not executed here)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight between latent and recon scores")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ASTAutoencoderASD(alpha=args.alpha).to(device)

    # Replace the following with your dataset / loader implementation
    train_loader = DataLoader(...)
    val_loader   = DataLoader(...)

    optim = torch.optim.Adam(model.parameters(), lr=3e-4)

    # Training
    for epoch in range(args.epochs):
        loss = train_one_epoch(model, train_loader, optim, device)
        print(f"Epoch {epoch+1}: recon‑only loss={loss:.4f}")

    # Fit latent distribution on the full normal training set
    model.fit_stats(train_loader)

    # Validation / inference
    scores, labels = validate(model, val_loader, device)
    # Compute metrics ↗
