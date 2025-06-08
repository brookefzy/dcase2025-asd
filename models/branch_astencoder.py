import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import ASTModel

# -----------------------------------------------------------------------------
# 1. Encoder: Pre‑trained AST (Audio Spectrogram Transformer) fine‑tuned on the
#    normal machine‑sound domain.
# -----------------------------------------------------------------------------
class ASTEncoder(nn.Module):
    """Wrapper that outputs a compressed latent vector *z* from a log‑mel tensor.

    Input shape : [B, 1, n_mels, T]
    Output      : [B, latent_dim]
    """

    def __init__(
        self,
        pretrained_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
        latent_dim: int = 128,
        fine_tune: bool = True,
    ) -> None:
        super().__init__()
        self.ast = ASTModel.from_pretrained(pretrained_name)
        self.proj = nn.Linear(self.ast.config.hidden_size, latent_dim)

        # Optionally freeze AST to use it as a fixed feature extractor
        if not fine_tune:
            for p in self.ast.parameters():
                p.requires_grad = False

    def forward(self, x: Tensor) -> Tensor:
        x = x.squeeze(1)  # [B, n_mels, T] – AST expects channel dim last
        out = self.ast(input_values=x)
        cls_emb = out.last_hidden_state[:, 0]  # CLS token
        z = self.proj(cls_emb)
        return z