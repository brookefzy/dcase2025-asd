import torch.nn as nn
import torch
from torch.nn import functional as F

class BranchContrastive(nn.Module):
    """
    Branch 3: Machine-ID contrastive encoder (two-stage as in ICASSP’23) :contentReference[oaicite:11]{index=11}
    """
    def __init__(self, latent_dim):
        super().__init__()
        # Base encoder: small CNN / AST / ResNet block
        self.encoder = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), nn.ReLU(),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc = nn.Linear(64, latent_dim)
        self.projector = nn.Sequential(nn.Linear(latent_dim, latent_dim),
                                       nn.ReLU(), nn.Linear(latent_dim, latent_dim))
    def forward(self, x_i, x_j=None, labels=None):
        """
        If x_j is provided, compute contrastive loss between x_i and x_j 
        grouped by machine ID in labels; otherwise just return embeddings.
        """
        h_i = self.encoder(x_i).view(x_i.size(0), -1)
        z_i = self.fc(h_i)
        if x_j is not None:
            h_j = self.encoder(x_j).view(x_j.size(0), -1)
            z_j = self.fc(h_j)
            p_i = self.projector(z_i); p_j = self.projector(z_j)
            # InfoNCE loss across IDs
            return z_i, self.contrastive_loss(p_i, p_j, labels)
        return z_i

    def contrastive_loss(self, p_i, p_j, labels):
        # simple InfoNCE
        batch_size = p_i.size(0)
        p = torch.cat([p_i, p_j], dim=0)
        sim = F.cosine_similarity(p.unsqueeze(1), p.unsqueeze(0), dim=2)
        # mask positives by matching labels :contentReference[oaicite:12]{index=12}
        mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        # compute loss...
        return nn.CrossEntropyLoss()(sim, mask.argmax(dim=1))
