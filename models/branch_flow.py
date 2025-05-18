# RealNVP from earlier
from models.flow import NormalizingFlow
import torch.nn as nn

class BranchFlow(nn.Module):
    """
    Branch 5: Normalizing Flow on concatenated embeddings
    """
    def __init__(self, latent_dim):
        super().__init__()
        self.flow = NormalizingFlow(dim=latent_dim, block_count=6)
    def forward(self, z):
        logp = self.flow(z)

        return -logp  # anomaly score
