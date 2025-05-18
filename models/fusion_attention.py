import torch.nn as nn

class FusionAttention(nn.Module):
    """
    Learnable attention-based fusion of branch scores 
    (replaces median ensemble) :contentReference[oaicite:14]{index=14}
    """
    def __init__(self, num_branches):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(num_branches, num_branches),
            nn.Softmax(dim=1)
        )
    def forward(self, scores):
        # scores: [B, num_branches]
        weights = self.attn(scores)            # [B, num_branches]
        fused = (weights * scores).sum(dim=1)  # [B]
        return fused
