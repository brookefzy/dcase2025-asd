# ./models/branch_contrastive.py
import torch.nn as nn
import torch
from torch.nn import functional as F
import math


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
    def forward(self, x, tau=None):
        """Embed both views and compute InfoNCE loss.

        Parameters
        ----------
        x : Tensor[B,2,1,n_mels,T]
            Augmented views of the same input.

        Returns
        -------
        Tensor[B, latent_dim] : Embedding for the first view
        Tensor[B] : Negative positive-pair similarity (higher = more anomalous)
        Tensor[B] : Per-sample InfoNCE loss (same scalar repeated)
        """
        if tau is None:
            tau = self.cfg.get("tau", 0.1)

        B, V, C, M, T = x.shape  # V should be 2
        x = x.view(B * V, C, M, T)
        h = self.encoder(x).view(B * V, -1)
        z = self.fc(h)
        z = F.normalize(z, dim=1)

        # InfoNCE loss over both views
        loss_ce = self.info_nce(z, tau)
        B2 = z.size(0) // 2
        # Normalise cross-entropy loss by the log of the batch size
        if B2 > 1:
            loss_ce = loss_ce / math.log(float(B2))

        z1, z2 = z[:B2], z[B2:]
        sim_pos = (z1 * z2).sum(1) / tau

        return z1, -sim_pos, loss_ce.repeat(B2)


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

    def info_nce(self, z, tau=0.1):
        """Standard NT-Xent loss over 2*B embeddings."""
        z = F.normalize(z, dim=1)
        B = z.size(0) // 2
        z1, z2 = z[:B], z[B:]
        logits = torch.mm(z1, z2.t()) / tau
        labels = torch.arange(B, device=z.device)
        loss = F.cross_entropy(logits, labels)
        loss = (loss + F.cross_entropy(logits.t(), labels)) / 2.0
        return loss


