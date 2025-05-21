import torch

class BranchAttrs(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x):
        # x: [B, input_dim]
        return self.mlp(x)  # → [B, latent_dim]
