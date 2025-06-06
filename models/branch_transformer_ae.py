import torch
import torch.nn as nn
from transformers import ASTConfig, ASTModel

class BranchTransformerAE(nn.Module):
    """
    Branch 2: Spectrogram Transformer Autoencoder
    """
    def __init__(self, latent_dim, cfg):
        super().__init__()
        self.cfg = cfg

        # Build an ASTConfig from our YAML settings
        config = ASTConfig(
            hidden_size=cfg['transformer_hidden'],         # e.g. 512
            num_attention_heads=cfg['transformer_nhead'],  # e.g. 8
            intermediate_size=cfg['transformer_ff'],       # e.g. 2048
            num_hidden_layers=cfg['transformer_layers'],   # e.g. 6
            hidden_dropout_prob=cfg['transformer_dropout'] # e.g. 0.1
        )
        # Encoder backbone
        self.encoder = ASTModel(config)
        
        ps = config.patch_size
        fs = config.frequency_stride
        ts = config.time_stride
        
        # Compute new sequence length:
        time_dim = cfg.get('frames', cfg.get('time_steps'))
        H = (cfg['n_mels']     - ps) // fs + 1
        W = (time_dim - ps) // ts + 1    #
        seq_len = 2 + H * W

        # Crop the pretrained positional embeddings
        old_pos = self.encoder.embeddings.position_embeddings  # [1, old_seq_len, d]
        new_pos = old_pos[:, :seq_len, :].clone()
        self.encoder.embeddings.position_embeddings = nn.Parameter(new_pos)

        # Project CLS output into our low-dim latent space
        self.fc_enc = nn.Linear(config.hidden_size, latent_dim)
        # Map latent back to hidden for the decoder
        self.fc_dec = nn.Linear(latent_dim, config.hidden_size)

        hidden = config.hidden_size
        if cfg['decoder_layers'] > 0:
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=hidden,
                nhead=cfg['transformer_nhead'],
                dim_feedforward=cfg['transformer_ff'],
            )
            self.decoder = nn.TransformerDecoder(
                decoder_layer,
                num_layers=cfg['decoder_layers'],
            )
            self.use_transformer_decoder = True
        else:
            self.decoder = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
            )
            self.use_transformer_decoder = False

        # Final linear to reconstruct the mel-bin dimension
        self.out = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, cfg['n_mels'])
        )

    def forward(self, x):
        # x is [B, 1, n_mels, T] → [B, T, n_mels] for ASTModel
        x_flat = x.squeeze(1).transpose(1, 2)

        # Encoder pass
        enc_outputs = self.encoder(input_values=x_flat)
        enc_out     = enc_outputs.last_hidden_state  # [B, T, hidden_size]
        cls_emb = enc_out[:, 0, :]                       # CLS token → [B, hidden]
        z       = self.fc_enc(cls_emb)                   # [B, latent_dim]
        if self.training and self.cfg.get('latent_noise_std', 0) > 0:
            z = z + torch.randn_like(z) * self.cfg['latent_noise_std']
        


        # Prepare decoder input from latent
        dec_inp = self.fc_dec(z)
        if self.use_transformer_decoder:
            dec_inp_t = dec_inp.unsqueeze(0)
            enc_out_t = enc_out.transpose(0, 1)
            dec_out_t = self.decoder(tgt=enc_out_t, memory=dec_inp_t)
            dec_out = dec_out_t.transpose(0, 1)
        else:
            h = self.decoder(dec_inp)
            dec_out = h.unsqueeze(1).expand(-1, enc_out.size(1), -1)

        # Reconstruct mel-bins
        recon = self.out(dec_out)                        # [B, T, n_mels]
        recon = recon.transpose(1, 2).unsqueeze(1)       # [B, 1, n_mels, T]

        return recon, z
