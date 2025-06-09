import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import ASTModel

def _resize_position_embeddings(old_emb: torch.Tensor, seq_len: int) -> torch.Tensor:
    """Interpolate or crop positional embeddings to the desired length."""
    old_len = old_emb.size(1)
    if seq_len == old_len:
        return old_emb
    if seq_len < old_len:
        return old_emb[:, :seq_len, :].clone()
    # interpolate to longer length
    emb_t = old_emb.permute(0, 2, 1)  # [1, d, L]
    new_emb = F.interpolate(emb_t, size=seq_len, mode="linear", align_corners=False)
    return new_emb.permute(0, 2, 1)

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

        # Crop positional embeddings to match the default mel/time dimensions
        H = (n_mels - 16) // 10 + 1
        W = (T_fix - 16) // 10 + 1
        new_len = 2 + H * W
        old_pos = self.ast.embeddings.position_embeddings
        self.ast.embeddings.position_embeddings = nn.Parameter(
            old_pos[:, :new_len].clone()
        )

        self.proj = nn.Linear(self.ast.config.hidden_size, latent_dim)

        # keep a copy of the original positional embeddings for resizing later
        self.register_buffer(
            "_orig_pos",
            self.ast.embeddings.position_embeddings.detach().clone(),
            persistent=False,
        )

        # Optionally freeze AST to use it as a fixed feature extractor
        if not fine_tune:
            for p in self.ast.parameters():
                p.requires_grad = False

        # Freeze initial parameters when requested
        if freeze_layers > 0:
            for i, (_, param) in enumerate(self.ast.named_parameters()):
                if i < freeze_layers:
                    param.requires_grad = False

    def forward(self, x: Tensor) -> Tensor:
        # Undo zero‑mean/unit‑var normalisation with a learnable affine
        x = x * self.input_scale + self.input_bias
        x = x.squeeze(1)  # [B, n_mels, T] – AST expects channel dim last

        # Adapt positional embeddings dynamically to the sequence length
        _, n_mels, T = x.shape
        ps = self.ast.config.patch_size
        fs = self.ast.config.frequency_stride
        ts = self.ast.config.time_stride
        H = (n_mels - ps) // fs + 1
        W = (T - ps) // ts + 1
        seq_len = 2 + H * W

        if self.ast.embeddings.position_embeddings.size(1) != seq_len:
            resized = _resize_position_embeddings(self._orig_pos, seq_len)
            self.ast.embeddings.position_embeddings = nn.Parameter(resized)

        out = self.ast(input_values=x)
        cls_emb = out.last_hidden_state[:, 0]  # CLS token
        z = self.proj(cls_emb)
        return z
