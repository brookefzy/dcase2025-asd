import torch
import torch.nn as nn
from torch import Tensor

# -----------------------------------------------------------------------------
# 2. Decoder: a lightweight fully connected network that reconstructs the input
#    log‑mel spectrogram from latent *z*.
# -----------------------------------------------------------------------------
class SpectroDecoder(nn.Module):
    """Very small decoder – enough to learn normal patterns without overfitting."""

    def __init__(self, latent_dim: int = 128, n_mels: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim * 2, n_mels),
        )

    def forward(self, z: Tensor, t: int) -> Tensor:
        """
        Reconstruct spectrogram with *T* time frames by repeating frequency
        vector along the time axis.
        """
        f_vec = self.net(z)              # [B, n_mels]
        recon = f_vec.unsqueeze(2).repeat(1, 1, t)  # [B, n_mels, T]
        recon = recon.unsqueeze(1)       # [B, 1, n_mels, T]
        return recon
