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
        
        
    
    

