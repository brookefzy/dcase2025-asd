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
        Returns: Tensor[N] of per-sample contrastive losses (zero if no positives).
        """
        # 1) Pairwise cosine similarities [N,N]
        sim = F.cosine_similarity(p.unsqueeze(1), p.unsqueeze(0), dim=2) / tau

        # 2) Build mask for positives (same ID), zero out diagonal
        mask = labels.unsqueeze(0).eq(labels.unsqueeze(1)).float()
        mask.fill_diagonal_(0)

        # 3) Exponentiate and build sums
        exp_sim = torch.exp(sim) * (1 - torch.eye(sim.size(0), device=sim.device))

        denom   = exp_sim.sum(dim=1)              # sum over all j≠i
        pos_sum = (exp_sim * mask).sum(dim=1)     # sum over k∈K(i)
        pos_cnt = mask.sum(dim=1)                 # number of positives per i

        # 4) clamp to avoid log(0) and div-by-zero
        eps = 1e-8
        denom   = denom.clamp(min=eps)
        pos_sum = pos_sum.clamp(min=eps)

        # avoid zeros in pos_cnt
        pos_cnt_safe = pos_cnt.clone()
        pos_cnt_safe[pos_cnt_safe == 0] = 1.0

        # 5) per-sample loss
        loss_i = - (1.0 / pos_cnt_safe) * torch.log(pos_sum / denom)

        # 6) set loss to 0 where there were no positives
        loss_i = loss_i.masked_fill(pos_cnt == 0, 0.0)

        return loss_i


