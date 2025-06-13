import torch
import torch.nn as nn
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
        freeze_layers: int = 0,
        n_mels: int = 128,
        T_fix: int = 512,
    ) -> None:
        super().__init__()
        self.ast = ASTModel.from_pretrained(pretrained_name)

        # Affine transform to undo dataset-wise normalisation when fine-tuning
        self.input_scale = nn.Parameter(torch.tensor(1.0))
        self.input_bias = nn.Parameter(torch.tensor(0.0))

        # Crop positional embeddings to match the mel/time dimensions
        ps = self.ast.config.patch_size
        fs = self.ast.config.frequency_stride
        ts = self.ast.config.time_stride

        H = (n_mels - ps) // fs + 1
        W = (T_fix - ps) // ts + 1
        new_len = 2 + H * W
        old_pos = self.ast.embeddings.position_embeddings
        self.ast.embeddings.position_embeddings = nn.Parameter(
            old_pos[:, :new_len].clone()
        )

        self.proj = nn.Linear(self.ast.config.hidden_size, latent_dim)

        # Freeze parameters according to transformer block index
        for name, p in self.ast.named_parameters():
            if not fine_tune:
                p.requires_grad = False
                continue

            if name.startswith("encoder.layer."):
                try:
                    layer_idx = int(name.split(".")[2])
                except (IndexError, ValueError):
                    layer_idx = None
                if layer_idx is not None:
                    p.requires_grad = layer_idx >= freeze_layers
                else:
                    p.requires_grad = freeze_layers <= 0
            else:
                p.requires_grad = freeze_layers <= 0

    def forward(self, x: Tensor) -> Tensor:
        x = x.float()
        # Undo zero‑mean/unit‑var normalisation with a learnable affine
        x = x * self.input_scale + self.input_bias
        x = x.squeeze(1)  # [B, n_mels, T] – AST expects channel dim last

        # Positional embeddings have been resized once in ``__init__``.
        # All inputs are expected to be padded/cropped to ``T_fix`` beforehand.

        out = self.ast(input_values=x)
        cls_emb = out.last_hidden_state[:, 0]  # CLS token
        z = self.proj(cls_emb)
        return z

