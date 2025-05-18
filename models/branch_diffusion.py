# models/branch_diffusion.py

import torch
import torch.nn as nn
from denoising_diffusion_pytorch import Unet, GaussianDiffusion

class BranchDiffusion(nn.Module):
    """
    Branch 4: Denoising-diffusion ASD (ASD-Diffusion).
    Returns a diffusion‐reconstruction loss.
    """
    def __init__(
        self,
        image_size: int,
        unet_dim: int = 64,
        unet_dim_mults: tuple = (1, 2, 4),
        timesteps: int = 1000,
    ):
        super().__init__()
        # Build a single-channel UNet
        self.unet = Unet(
            dim       = unet_dim,
            dim_mults = unet_dim_mults,
            channels  = 1           # spectrograms are mono
        )                          # <— correct channels here :contentReference[oaicite:2]{index=2}

        # Wrap in GaussianDiffusion (no 'channels' or 'loss_type' args)
        self.diffusion = GaussianDiffusion(
            self.unet,
            image_size = image_size,  # e.g. n_mels
            timesteps  = timesteps    # e.g. 1000
        )                          # <— matches README usage :contentReference[oaicite:3]{index=3}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,1,H,W] Tensor (normalized to [0,1])
        returns: scalar loss averaged over batch
        """
        return self.diffusion(x)   # single scalar loss :contentReference[oaicite:4]{index=4}
