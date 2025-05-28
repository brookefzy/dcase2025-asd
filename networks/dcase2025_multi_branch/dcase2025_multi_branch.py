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
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg

        # Branches
        device = cfg["device"]
        self.branch_pretrained = BranchPretrained(cfg["ast_model"], cfg).to(device) # b1
        self.branch_transformer_ae = BranchTransformerAE(cfg["latent_dim"], cfg).to(device) # b2
        self.branch_contrastive = BranchContrastive(cfg["latent_dim"], cfg).to(device) # b3
        self.branch_flow = BranchFlow(cfg["flow_dim"]).to(device) # b5
        
        self.fusion = FusionAttention(num_branches=3).to(device)
        self.branch_attrs = BranchAttrs(
            input_dim=self.shared_attr_len,
            hidden_dim=cfg["attr_hidden"],
            latent_dim=cfg["attr_latent"]).to(device)
        self.shared_attr_len = cfg["shared_attr_len"]
        
        # Optimizer
        self.optimizer = optim.Adam(
            list(self.branch_pretrained.parameters()) +
            list(self.branch_transformer_ae.parameters()) +
            list(self.branch_contrastive.parameters()) +
            list(self.branch_flow.parameters()) +
            list(self.branch_attrs.parameters()),
                                    lr=float(cfg["lr"]))
        
    def _compute_branch_scores(self, x, labels, attrs=None):
        z1 = self.branch_pretrained(x)
        recon2, z2 = self.branch_transformer_ae(x)
        feats_ds = adaptive_avg_pool2d(x, (self.cfg["n_mels"], recon2.shape[-1]))
        loss2 = (recon2 - feats_ds).pow(2).reshape(x.size(0), -1).mean(dim=1)
        z3, loss3 = self.branch_contrastive(x, labels)
        z_flow = self.branch_flow(torch.cat([z1, z2, z3], dim=1))
        if attrs is not None:
            z_attr = self.branch_attrs(attrs)
            flow_input = torch.cat([z1, z2, z3, z_flow.unsqueeze(1), z_attr], dim=1)
            loss5 = self.branch_flow(flow_input)
        else:
            loss5 = z_flow
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
        """
        Train the model for a single epoch.
        Args:
            epoch (int): The current epoch number.
        Returns:
            None
        This method trains the model for one epoch, calculates the training loss. 
        It also evaluates the model on the validation set and 
        saves the model's state.
        Key Steps:
        - If the epoch is greater than the total number of epochs, calculate covariance 
          matrices for Mahalanobis distance-based anomaly detection.
        - For each batch in the training data:
            - Perform forward propagation to compute the reconstruction loss.
            - If calculating covariance, update the covariance matrices for source 
              and target domains.
            - Otherwise, compute the loss, backpropagate, and update the model parameters.
        - If calculating covariance, compute the inverse covariance matrices and fit 
          the anomaly score distribution.
        - Evaluate the model on the validation set.
        - Log training and validation losses.
        - Save the model and optimizer states."""
        
        
    
    

