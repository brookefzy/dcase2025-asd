import torch
import torch.nn as nn
from torch import Tensor

# -----------------------------------------------------------------------------
# 2. Decoder: a lightweight fully connected network that reconstructs the input
#    log‑mel spectrogram from latent *z*.
# -----------------------------------------------------------------------------
class SpectroDecoder(nn.Module):
    """Slightly less naive decoder reconstructing an entire spectrogram."""

    def __init__(
        self,
        latent_dim: int = 128,
        n_mels: int = 128,
        time_steps: int = 512,
    ) -> None:
        super().__init__()
        self.n_mels = n_mels
        self.time_steps = time_steps
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim * 2, n_mels * time_steps),
        )

    def forward(self, z: Tensor, t: int | None = None) -> Tensor:
        """Reconstruct ``time_steps`` frames irrespective of ``t``."""
        f = self.net(z)  # [B, n_mels * T_fix]
        recon = f.view(z.size(0), 1, self.n_mels, self.time_steps)
        if t is not None and t != self.time_steps:
            recon = recon[..., :t]
        return recon
