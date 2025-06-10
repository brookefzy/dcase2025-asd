# models/beats_encoder.py
from __future__ import annotations
import torch
import torchaudio


class BEATSEncoder(torch.nn.Module):
    """
    Minimal wrapper around the BEATs models shipped with torchaudio ≥ 2.2.
    Produces one 768- (base) or 1024-d (large) embedding per clip.
    """
    def __init__(
        self,
        variant: str = "base",       # "base" or "large"
        output_layer: int = -1,      # which transformer block to tap
        pooling: str = "mean",       # "mean" | "cls"
        freeze: bool = True,         # usually freeze for first-shot ASD
    ):
        super().__init__()

        bundle = {
            "base":  torchaudio.pipelines.BEATS_BASE,
            "large": torchaudio.pipelines.BEATS_LARGE,
        }[variant]
        self.model = bundle.get_model()
        self.sample_rate = bundle.sample_rate        # 16 kHz
        self.out_dim = self.model.embed_dim          # 768 / 1024
        self.output_layer = output_layer
        self.pooling = pooling

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

    @torch.inference_mode()
    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Args
          wav  (B, T) float32 in [-1, 1] @ 16 kHz
        Returns
          emb  (B, out_dim)  – pooled transformer representation
        """
        feats = self.model.extract_features(
            wav, layers=[self.output_layer]
        )["x"]               # (B, frames, out_dim)

        if self.pooling == "mean":
            emb = feats.mean(dim=1)
        elif self.pooling == "cls":
            emb = feats[:, 0]
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        return emb
