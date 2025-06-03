import torch
import torch.nn as nn

class FusionAttention(nn.Module):
    """
    Learnable attention-based fusion of branch scores 
    """
    def __init__(self, num_branches):
        super().__init__()
        # learnable branch weights
        self.attn = nn.Parameter(torch.ones(num_branches))
    def forward(self, scores):
        # scores: [B, num_branches]
        weights = torch.softmax(self.attn, dim=0)  # [num_branches]
        fused = (weights * scores).sum(dim=1)  # [B]
        return fused
