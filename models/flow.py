import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

class RealNVPBlock(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        # Split half dims
        self.net_s = nn.Sequential(
            nn.Linear(dim//2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, dim//2), nn.Tanh()
        )
        self.net_t = nn.Sequential(
            nn.Linear(dim//2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, dim//2)
        )

    def forward(self, x, reverse=False):
        # x: [B, dim]
        x1, x2 = x.chunk(2, dim=1)
        if not reverse:
            s = self.net_s(x1)
            t = self.net_t(x1)
            y2 = x2 * torch.exp(s) + t
            y = torch.cat([x1, y2], dim=1)
            log_det = s.sum(dim=1)
            return y, log_det
        else:
            s = self.net_s(x1)
            t = self.net_t(x1)
            y2 = (x2 - t) * torch.exp(-s)
            y = torch.cat([x1, y2], dim=1)
            log_det = -s.sum(dim=1)
            return y, log_det

class NormalizingFlow(nn.Module):
    def __init__(self, dim, block_count=6, hidden_dim=512):
        super().__init__()
        self.blocks = nn.ModuleList([RealNVPBlock(dim, hidden_dim) for _ in range(block_count)])
        self.register_buffer('prior_mean', torch.zeros(dim))
        self.register_buffer('prior_cov', torch.eye(dim))
        
        self.prior = MultivariateNormal(
            loc=self.prior_mean,
            scale_tril=self.prior_cov,
        )
        

    def forward(self, x):
        log_det_sum = 0
        for blk in self.blocks:
            x, log_det = blk(x)
            log_det_sum += log_det
        # x is now in base space
        x = x.to(self.prior_mean.device)
        prior = MultivariateNormal(
            loc=self.prior_mean.to(x.device),
            scale_tril=self.prior_cov.to(x.device),
        )
        log_prob = prior.log_prob(x)

        # send to cuda if needed
        log_prob = log_prob + log_det_sum

        return log_prob

    def sample(self, num_samples):
        z = self.prior.sample((num_samples,))
        for blk in reversed(self.blocks):
            z, _ = blk(z, reverse=True)
        return z
