import torch.nn as nn
import torch
from torch.nn import functional as F


class BranchContrastive(nn.Module):
    """
    Branch 3: Machine-ID contrastive encoder (two-stage as in ICASSP’23) :contentReference[oaicite:11]{index=11}
    """
    def __init__(self, latent_dim, cfg):
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
        self.cfg = cfg
    def forward(self, x, labels):
        """
        If x_j is provided, compute contrastive loss between x_i and x_j 
        grouped by machine ID in labels; otherwise just return embeddings.
        """
        h = self.encoder(x).view(x.size(0), -1)  # [B, 64, 1, 1] → [B, 64]
        z = self.fc(h)  # [B, 64] → [B, latent_dim]
        p = self.projector(z)
        #2) Compute InfoNCE loss per CL-Meta (eq 2)
        loss = self.contrastive_loss(p, labels, tau=self.cfg['tau'])
        return z, loss


    def contrastive_loss(self, p, labels, tau=0.05):
        """
        p: Tensor[N, D] – projected embeddings
        labels: LongTensor[N] – machine ID labels
        """
        # 1) Pairwise cosine similarities
        sim = F.cosine_similarity(p.unsqueeze(1), p.unsqueeze(0), dim=2)   # [N, N]
        sim = sim / tau                                                   # apply temperature

        # 2) Mask positives (same ID) and exclude diagonal
        mask = labels.unsqueeze(0).eq(labels.unsqueeze(1)).float()        # [N, N]
        mask.fill_diagonal_(0)                                            # remove self-pairs

        # 3) Exponentiate and build denominators/numerators
        exp_sim = torch.exp(sim) * (1 - torch.eye(sim.size(0), device=sim.device))
        denom   = exp_sim.sum(dim=1)                                      # sum over all j≠i
        pos_sum = (exp_sim * mask).sum(dim=1)                             # sum over k∈K(i)
        pos_cnt = mask.sum(dim=1)                                         # |K(i)| for each i

        # 4) Compute per-sample loss and average
        loss_i = - (1.0 / pos_cnt) * torch.log(pos_sum / denom)
        return loss_i.mean()

