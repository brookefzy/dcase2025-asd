# models/branch_pretrained.py

from transformers import ASTModel
import torch.nn as nn

class BranchPretrained(nn.Module):
    """
    Branch 1: Audio Spectrogram Transformer (AST) embeddings
    """
    def __init__(self, model_name, cfg):
        super().__init__()
        self.ast  = ASTModel.from_pretrained(model_name)
        self.proj = nn.Linear(self.ast.config.hidden_size, cfg['latent_dim'])
        
        ps = self.ast.config.patch_size
        fs = self.ast.config.frequency_stride
        ts = self.ast.config.time_stride
        
        time_dim = cfg.get('frames', cfg.get('time_steps'))
        H = (cfg['n_mels'] - ps) // fs + 1
        W = (time_dim        - ps) // ts + 1
        seq_len = 2 + H * W

        # Crop the pretrained positional embeddings
        old_pos = self.ast.embeddings.position_embeddings  # [1, old_seq_len, d]
        new_pos = old_pos[:, :seq_len, :].clone()
        self.ast.embeddings.position_embeddings = nn.Parameter(new_pos)
        print(f"AST pos-emb old_len={old_pos.size(1)}, new_len={seq_len}")

    def forward(self, x):
        """
        x: Tensor of shape [B, 1, n_mels, T]
        returns: z of shape [B, latent_dim]
        """
        # 1) remove that extra channel dim so AST sees [B, n_mels, T]
        x = x.squeeze(1)

        # 2) feed into AST via `input_values`; it will unsqueeze to [B,1,n_mels,T] internally
        outputs = self.ast(input_values=x)

        # 3) take the CLS token embedding (first token) from last_hidden_state
        cls_emb = outputs.last_hidden_state[:, 0]   # [B, hidden_size]

        # 4) project down to our anomaly‐score latent space
        z = self.proj(cls_emb)                     # [B, latent_dim]
        return z
