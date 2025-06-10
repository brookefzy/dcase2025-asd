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
            nn.Linear(latent_dim, 64 * (time_steps // 4)),
            nn.ReLU(True),
            nn.Unflatten(1, (64, time_steps // 4)),
            nn.ConvTranspose1d(64, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose1d(32, n_mels, 4, 2, 1),
        )

    def forward(self, z: Tensor, t: int | None = None) -> Tensor:
        """Reconstruct ``time_steps`` frames irrespective of ``t``."""
        f = self.net(z)  # [B, n_mels, T_fix]
        recon = f.unsqueeze(1)
        if t is not None and t != self.time_steps:
            recon = recon[..., :t]
        return recon
