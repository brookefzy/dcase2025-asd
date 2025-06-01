import os
import sys
import torch
from torch import optim
import torch.nn.functional as F
import numpy as np
import scipy
from sklearn import metrics
import csv
from tqdm import tqdm

from networks.base_model import BaseModel
from networks.criterion.mahala import cov_v, loss_function_mahala, calc_inv_cov
from torch.utils.data import DataLoader, ConcatDataset, Dataset, random_split
from torch.nn.functional import adaptive_avg_pool2d, pad

from tools.plot_anm_score import AnmScoreFigData
from tools.plot_loss_curve import csv_to_figdata
from models.branch_pretrained import BranchPretrained
from models.branch_transformer_ae import BranchTransformerAE
from models.branch_contrastive import BranchContrastive
from models.branch_flow import BranchFlow
from models.branch_attr import BranchAttrs
from models.fusion_attention import FusionAttention

class DCASE2025MultiBranch(BaseModel):
    """
    DCASE2025 Multi-Branch Model
    """
    def __init__(self, cfg, device):
        super().__init__(cfg)
        self.cfg = cfg

        # Branches
        # Instantiate sub-networks (branches)
        self.b1 = BranchPretrained(cfg["ast_model"], cfg).to(device)
        self.b2 = BranchTransformerAE(cfg["latent_dim"], cfg).to(device)
        self.b3 = BranchContrastive(cfg["latent_dim"], cfg).to(device)
        self.b5 = BranchFlow(cfg["flow_dim"]).to(device)
        self.b_attr = BranchAttrs(input_dim=cfg["attr_input_dim"], 
                                   hidden_dim=cfg["attr_hidden"], 
                                   latent_dim=cfg["attr_latent"]).to(device)
        self.fusion = FusionAttention(num_branches=3).to(device)
        # Optimizer setup
        self.optimizer = optim.Adam(self.parameters(), lr=cfg["learning_rate"])
        
    def forward(self, x, labels=None, attrs=None):
        """
        Forward pass that computes branch outputs and fused anomaly score.
        Returns: loss2, loss3, loss5, score (tensors)
        """
        # Branch 1: Pretrained feature extractor
        z1 = self.b1(x)
        # Branch 2: Autoencoder (reconstruction and latent)
        recon2, z2 = self.b2(x)
        # Compute reconstruction loss (MSE) for branch 2
        feats_ds = torch.nn.functional.adaptive_avg_pool2d(x, (self.cfg["n_mels"], recon2.shape[-1]))
        loss2 = ((recon2 - feats_ds)**2).reshape(x.size(0), -1).mean(dim=1)
        # Branch 3: Contrastive branch (returns latent and loss term)
        z3, loss3 = self.b3(x, labels)
        # Branch 5: Normalizing flow on concatenated embeddings
        z_flow = self.b5(torch.cat([z1, z2, z3], dim=1))
        # Attribute branch (if attribute data is provided)
        if attrs is not None:
            z_attr = self.b_attr(attrs)
            flow_input = torch.cat([z1, z2, z3, z_flow.unsqueeze(1), z_attr], dim=1)
            loss5 = self.b5(flow_input)   # Flow returns a loss when given full input
        else:
            loss5 = z_flow  # If no attr branch, use z_flow output as loss5
        # Fusion: combine losses into final anomaly score
        scores = self.fusion(torch.stack([loss2, loss3, loss5], dim=1))
        return loss2, loss3, loss5, scores

    
    def get_log_header(self):
        self.column_heading_list=[
                ["loss"],
                ["val_loss"],
                ["recon_loss"], 
                ["recon_loss_source", "recon_loss_target"],
        ]
        return "loss,val_loss,recon_loss,recon_loss_source,recon_loss_target"
    
    def train(self, epoch):
        """Run one training epoch.

        This routine closely follows the baseline implementation in
        ``DCASE2023T2AE.train``.  Each branch is trained jointly and the losses
        from all branches are fused via the attention module to obtain the final
        anomaly score.  The mean loss over the training and validation sets is
        logged to ``self.log_path`` and model parameters are saved to the
        checkpoint paths created in :class:`BaseModel`.

        Parameters
        ----------
        epoch : int
            Current epoch number.  Training starts from ``self.epoch + 1``.
            """
        if epoch <= getattr(self, "epoch", 0):
            return

        device = self.cfg.get("device", "cpu")

        # set modules to train mode
        self.b1.train()
        self.b2.train()
        self.b3.train()
        self.b5.train()
        self.b_attr.train()
        self.fusion.train()

        train_loss = 0.0
        y_pred = []

        for batch_idx, batch in enumerate(tqdm(self.train_loader)):
            feats = batch[0].to(device).float()
            labels = torch.argmax(batch[2], dim=1).long().to(device)

            # reshape flattened features to [B, 1, n_mels, frames]
            b, dim = feats.shape
            frames = dim // self.cfg["n_mels"]
            feats = feats.view(b, 1, self.cfg["n_mels"], frames)

            self.optimizer.zero_grad()
            loss2, loss3, loss5, scores = self.forward(feats, labels)

            loss = (
                self.cfg.get("w2", 1.0) * loss2.mean() +
                self.cfg.get("w3", 1.0) * loss3.mean() +
                self.cfg.get("w5", 1.0) * loss5.mean()
            )

            loss.backward()
            self.optimizer.step()

            train_loss += float(loss)
            y_pred.extend(scores.detach().cpu().numpy().tolist())

            if batch_idx % self.cfg.get("log_interval", 100) == 0:
                print(
                    f"Train Epoch: {epoch} [{batch_idx * len(feats)}/{len(self.train_loader.dataset)}]\t"
                    f"Loss: {loss.item():.6f}"
                )

        # ── validation ──────────────────────────────────────────────────
        val_loss = 0.0
        with torch.no_grad():
            self.b1.eval()
            self.b2.eval()
            self.b3.eval()
            self.b5.eval()
            self.b_attr.eval()
            self.fusion.eval()

            for batch in self.valid_loader:
                feats = batch[0].to(device).float()
                labels = torch.argmax(batch[2], dim=1).long().to(device)

                b, dim = feats.shape
                frames = dim // self.cfg["n_mels"]
                feats = feats.view(b, 1, self.cfg["n_mels"], frames)

                loss2, loss3, loss5, scores = self.forward(feats, labels)
                loss = (
                    self.cfg.get("w2", 1.0) * loss2.mean() +
                    self.cfg.get("w3", 1.0) * loss3.mean() +
                    self.cfg.get("w5", 1.0) * loss5.mean()
                )
                val_loss += float(loss)
                y_pred.extend(scores.detach().cpu().numpy().tolist())

        avg_train = train_loss / len(self.train_loader)
        avg_val = val_loss / len(self.valid_loader)

        print(
            f"====> Epoch: {epoch} Average loss: {avg_train:.4f} Validation loss: {avg_val:.4f}"
        )

        # log CSV
        with open(self.log_path, "a") as log:
            np.savetxt(log, [f"{avg_train},{avg_val},0,0,0"], fmt="%s")

        csv_to_figdata(
            file_path=self.log_path,
            column_heading_list=self.column_heading_list,
            ylabel="loss",
            fig_count=len(self.column_heading_list),
            cut_first_epoch=True,
        )

        # fit anomaly score distribution using fused scores
        self.fit_anomaly_score_distribution(y_pred=y_pred)

        # update epoch counter
        self.epoch = epoch

        # save parameters
        torch.save(
            {
                "epoch": epoch,
                "b1": self.b1.state_dict(),
                "b2": self.b2.state_dict(),
                "b3": self.b3.state_dict(),
                "b5": self.b5.state_dict(),
                "b_attr": self.b_attr.state_dict(),
                "fusion": self.fusion.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": avg_train,
            },
            self.checkpoint_path,
        )
        
        
    
    

