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
        assert (
            time_steps % 64 == 0
        ), "time_steps must be divisible by 64 (decoder upsample)."
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256 * (time_steps // 64)),
            nn.ReLU(True),
            nn.Unflatten(1, (256, time_steps // 64)),          # [B,256,T/64]
            nn.ConvTranspose1d(256, 128, 4, 2, 1), nn.ReLU(True),  # T/32
            nn.ConvTranspose1d(128, 64, 4, 2, 1), nn.ReLU(True),   # T/16
            nn.ConvTranspose1d(64, 64, 4, 2, 1), nn.ReLU(True),    # T/8
            nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(True),        # mix freq
            nn.ConvTranspose1d(64, 32, 4, 2, 1), nn.ReLU(True),    # T/4
            nn.ConvTranspose1d(32, n_mels, 4, 2, 1), nn.ReLU(True),# T/2
            nn.ConvTranspose1d(n_mels, n_mels, 4, 2, 1),           # T
        )

    def forward(self, z: Tensor, t: int | None = None) -> Tensor:
        """Reconstruct ``time_steps`` frames irrespective of ``t``."""
        f = self.net(z)  # [B, n_mels, T_fix]
        recon = f.unsqueeze(1)
        return recon
