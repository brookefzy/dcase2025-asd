# ast_ae.py
from __future__ import annotations
from pathlib import Path
from typing import Tuple, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from models.branch_decoder import SpectroDecoder
from models.branch_astencoder import ASTEncoder
from networks.base_model import BaseModel
from datasets.datasets import pad_collate
from tools.plot_anm_score import AnmScoreFigData
from tools.plot_loss_curve import csv_to_figdata
from sklearn import metrics
import scipy
import numpy as np
import csv
import os


# -----------------------------------------------------------------------------
# 3. Full Autoencoder + scoring utilities
# -----------------------------------------------------------------------------
class ASTAutoencoder(nn.Module):
    """End‑to‑end model producing latent embeddings, reconstructions, and scores."""

    def __init__(
        self,
        latent_dim: int = 128,
        n_mels: int = 128,
        time_steps: int = 512,
        alpha: float = 0.9,
        latent_noise_std: float = 0.0,
        cfg: Dict = None,
        logmag_lambda: float = 0.0,
        *,
        attr_dim: int = 0,
        use_attribute: bool = False,
    ) -> None:
        super().__init__()
        freeze_layers = cfg.get("ast_freeze_layers", 6)
        self.encoder = ASTEncoder(
            latent_dim=latent_dim,
            freeze_layers=freeze_layers,
            n_mels=n_mels,
            T_fix=time_steps,
        )
        self.decoder = SpectroDecoder(latent_dim=latent_dim, n_mels=n_mels, time_steps=time_steps)
        self.alpha = alpha
        self.latent_noise_std = latent_noise_std
        self.logmag_lambda = logmag_lambda

        self.use_attribute = use_attribute
        if self.use_attribute and attr_dim > 0:
            self.attr_fc = nn.Sequential(
                nn.Linear(attr_dim, 64),
                nn.ReLU(),
            )
            self.fuse_fc = nn.Linear(latent_dim + 64, latent_dim)

        # Statistics for latent space whitening – initialised later via
        # ``fit_stats_streaming``.
        self.register_buffer("mu", torch.zeros(latent_dim))          # mean of z
        self.register_buffer("cov", torch.ones(latent_dim))          # diagonal var

        # Parameters of Mahalanobis distance distribution (mean/std) computed on
        # whitened distances.
        self.register_buffer("m_mean", torch.zeros(1))
        self.register_buffer("m_std", torch.ones(1))

        # Domain specific mean/std of whitened Mahalanobis distances.
        # index 0 -> source, 1 -> target
        self.register_buffer("m_mean_domain", torch.zeros(2))
        self.register_buffer("m_std_domain", torch.ones(2))

        # Robust scaling parameters for reconstruction error (median / MAD).
        self.register_buffer("mse_med", torch.zeros(1))
        self.register_buffer("mse_mad", torch.ones(1))
        

    # ------------------------------------------------------------------
    # Forward pass (training) returns reconstruction loss.
    # ------------------------------------------------------------------
    def forward(self, x: Tensor, attr_vec: Tensor | None = None) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return recon, latent *z*, MSE and log-magnitude reconstruction losses."""
        B, _, _, T = x.shape
        z = self.encoder(x)                # [B, latent]
        if self.training and getattr(self, "latent_noise_std", 0) > 0:
            z = z + torch.randn_like(z) * self.latent_noise_std

        if self.use_attribute and attr_vec is not None and attr_vec.numel() > 0:
            a = self.attr_fc(attr_vec.float())
            z = torch.cat([z, a], dim=-1)
            z = self.fuse_fc(z)

        recon = self.decoder(z, self.decoder.time_steps)         # [B, 1, n_mels, T]
        mse = F.mse_loss(recon[..., :T], x, reduction="none")
        mse = mse.mean(dim=[1, 2, 3])      # [B]
        log_recon = torch.log10(recon[..., :T].clamp(min=1e-8))
        log_gt = torch.log10(x.clamp(min=1e-8))
        log_mse = F.mse_loss(log_recon, log_gt, reduction="none")
        log_mse = log_mse.mean(dim=[1, 2, 3])
        return recon, z, mse, log_mse
    
    # --------------------------------------------------------------
    # Statistics helpers
    # --------------------------------------------------------------
    def reset_domain_stats(self) -> None:
        """Zero out statistics buffers used for domain adaptation."""
        self.mu.zero_()
        self.cov.fill_(1.0)
        self.m_mean.zero_()
        self.m_std.fill_(1.0)
        self.m_mean_domain.zero_()
        self.m_std_domain.fill_(1.0)
        self.mse_med.zero_()
        self.mse_mad.fill_(1.0)

    def mahalanobis(self, z: Tensor) -> Tensor:
        """Return whitened Mahalanobis distance for latent vectors ``z``."""
        delta = z - self.mu
        delta = delta / torch.sqrt(self.cov + 1e-6)
        return torch.linalg.norm(delta, dim=-1)

    # ------------------------------------------------------------------
    # Statistics fitting – call once on *normal* training data.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def fit_stats_streaming(self, loader):
        """Compute mean/std of Mahalanobis distance on normal clips."""
        import math

        self.eval()
        sum_, sum2, n = 0.0, 0.0, 0
        per_dom = {0: [0.0, 0.0, 0], 1: [0.0, 0.0, 0]}

        for batch in loader:
            feats = batch[0]
            names = batch[-1] if len(batch) > 1 else []
            # ``y_true`` can appear at index 1 or 2 depending on the dataset.
            if len(batch) > 1 and isinstance(batch[1], torch.Tensor) and batch[1].dim() == 1:
                y_true = batch[1]
            elif len(batch) > 2 and isinstance(batch[2], torch.Tensor) and batch[2].dim() == 1:
                y_true = batch[2]
            else:
                # fall back to zeros if labels are missing
                y_true = torch.zeros(len(feats), device=feats.device)

            mask = y_true == 0
            if not mask.any():
                continue

            feats = feats[mask].to(self.mu.device)
            if names:
                dom = torch.tensor(
                    [1 if "target" in n.lower() else 0 for n in names],
                    device=self.mu.device,
                )[mask]
            else:
                dom = torch.zeros(len(feats), dtype=torch.long, device=self.mu.device)

            _, z_full, _, _ = self.forward(feats)           # your forward already returns z
            # ``z_full`` may be [B, D] or [B, T, D]; pool over time only if needed
            z = z_full.mean(1) if z_full.dim() == 3 else z_full
            md_raw = self.mahalanobis(z)            # SAME call as in anomaly_score()

            sum_ += md_raw.sum().item()
            sum2 += (md_raw ** 2).sum().item()
            n += md_raw.numel()

            for d in (0, 1):
                m = md_raw[dom == d]
                if m.numel():
                    s_, s2_, k = per_dom[d]
                    per_dom[d] = [s_ + m.sum().item(), s2_ + (m ** 2).sum().item(), k + m.numel()]

        mu = sum_ / n if n else 0.0
        std = math.sqrt(max(sum2 / n - mu ** 2, 1e-12)) if n else 0.0

        self.m_mean.fill_(mu)
        self.m_std.fill_(std)

        for d in (0, 1):
            s_, s2_, k = per_dom[d]
            if k > 1:
                mu_d = s_ / k
                std_d = math.sqrt(max(s2_ / k - mu_d ** 2, 1e-12))
                self.m_mean_domain[d] = mu_d
                self.m_std_domain[d] = std_d
            else:
                self.m_mean_domain[d] = self.m_mean
                self.m_std_domain[d] = self.m_std
        print("μ_source =", self.m_mean_domain[0].item(),
                "μ_target =", self.m_mean_domain[1].item(),
                "global σ =", self.m_std.item())




    # ------------------------------------------------------------------
    # Scoring – used at inference.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def anomaly_score(
        self,
        x: Tensor,
        attr_vec: Tensor | None = None,
        names: list[str] | None = None,
    ) -> Tensor:
        """Compute combined anomaly score for input batch."""
        recon, z_full, mse, _ = self.forward(x, attr_vec=attr_vec)   # z_full may be B×T×D or B×D
        # If temporal dimension exists, average over it; otherwise use z as-is
        z = z_full.mean(1) if z_full.dim() == 3 else z_full

        # ---------- Mahalanobis distance on whitened latent ----------
        delta_raw = z - self.mu
        delta = delta_raw / torch.sqrt(self.cov + 1e-6)

        # (1) raw Mahalanobis distance – should always be non-negative
        m_dist_raw = torch.linalg.norm(delta, dim=-1)

        # (2) normalise per-element
        if names is not None:
            ids = torch.tensor(
                [1 if "target" in n.lower() else 0 for n in names],
                device=m_dist_raw.device,
            )
            mu  = self.m_mean_domain[ids]     # vector of same length as batch
            sig = self.m_std_domain[ids]
        else:
            mu  = self.m_mean_domain[0]
            sig = self.m_std_domain[0]


        m_dist_ctr = m_dist_raw - mu

        # (3) z-score on centred distances using domain-specific std
        m_norm = m_dist_ctr / (sig + 1e-9)
        mse_log = torch.log10(mse + 1e-8)
        mse_norm = (mse_log - self.mse_med) / (self.mse_mad + 1e-6)
        mse_norm = torch.clamp(mse_norm, -5, 5)

        # ---------- Weighted sum ----------

        score = self.alpha * m_norm + (1 - self.alpha) * mse_norm
        
        print(
            "[DEBUG] anomaly_score: "
            f"md_raw={m_dist_raw.mean():.3f}  "
            f"md_ctr={m_dist_ctr.mean():.3f}  "
            f"m_norm={m_norm.mean():.3f}, "
            f"mse={mse.mean().item():.4f}, "
            f"mse_norm={mse_norm.mean().item():.4f}, "
            f"score={score.mean().item():.4f}"
        )
        return score, m_dist_ctr, m_norm

# -----------------------------------------------------------------------------
# 4. Training and validation functions
# -----------------------------------------------------------------------------

def train_one_epoch(model: ASTAutoencoder, loader, optim, device: torch.device):
    """This is the pseudo-code for a single training epoch."""
    model.train()
    total_loss = 0.0
    for batch in loader:
        xb = batch[0].to(device)
        attr = batch[1].to(device) if model.use_attribute and len(batch) > 1 else None
        rc, z, mse, log_mse = model(xb, attr_vec=attr)
        loss = (mse + model.logmag_lambda * log_mse).mean()
        optim.zero_grad()
        loss.backward()
        optim.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)



class ASTAutoencoderASD(BaseModel):
    """Wrapper integrating ``ASTAutoencoder`` with the training framework."""

    def init_model(self):
        cfg = self.args.__dict__
        self.cfg = cfg
        self.device = "cuda" if self.args.use_cuda and torch.cuda.is_available() else "cpu"
        latent = cfg.get("latent_dim", 128)
        alpha = cfg.get("alpha", 0.8)
        time_steps = cfg.get("time_steps", 512)
        self._latent_noise_base = cfg.get("latent_noise_std", 0.0)
        attr_dim = cfg.get("attr_dim", 0)
        use_attribute = cfg.get("use_attribute", False)
        logmag_lambda = cfg.get("logmag_lambda", 0.0)
        return ASTAutoencoder(
            latent_dim=latent,
            n_mels=self.data.height,
            time_steps=time_steps,
            alpha=alpha,
            latent_noise_std=0.0,
            cfg = cfg,
            attr_dim=attr_dim,
            use_attribute=use_attribute,
            logmag_lambda=logmag_lambda,
        )

    def __init__(self, args, train, test):
        super().__init__(args=args, train=train, test=test)
        # ``BaseModel`` already initialises ``self.model`` and moves it to the
        # correct device, so avoid re-creating it here.  Re-initialising the
        # model would leave it on the CPU and lead to device mismatch errors
        # during training.
        # ---------- DEBUG: check how many AST params can learn ----------
        n_trainable = sum(p.numel() for p in self.model.encoder.ast.parameters() if p.requires_grad)
        print("[DEBUG] trainable AST params:", n_trainable)
        # should be > 0  (about 22 M if 12 of 24 layers are unfrozen)
        
        
        dec_params = list(self.model.decoder.parameters())
        proj_params = list(self.model.encoder.proj.parameters())
        enc_params = [p for p in self.model.encoder.ast.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(
            [
                {"params": dec_params + proj_params, "lr": 1e-4},
                {"params": enc_params, "lr": self.args.learning_rate},
            ]
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.3,
            patience=5,
        )
        self.ema_scores = []
        self.noise_enabled = False
        self.model.latent_noise_std = 0.0
        self._aug_backup = []  # store datasets and their augmentations

        # ----- domain weighting -----
        self._calc_domain_ratio()

        # ----- warm-up freezing -----
        self._orig_ast_requires_grad = {
            n: p.requires_grad for n, p in self.model.encoder.ast.named_parameters()
        }
        self._warmup_epochs = getattr(self.args, "warm_up_epochs", 0)
        self._ast_frozen = False
        if self._warmup_epochs > 0:
            self._freeze_ast()

        # directory to store reconstruction samples
        self.recon_dir = self.logs_dir / "recon_samples"
        self.recon_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------
    # Utility helpers to temporarily disable SpecAugment
    # --------------------------------------------------------------
    def _disable_aug(self, dataset):
        """Recursively disable augmentation on ``dataset``."""
        if hasattr(dataset, "datasets"):
            for d in dataset.datasets:
                self._disable_aug(d)
        elif hasattr(dataset, "dataset"):
            self._disable_aug(dataset.dataset)
        elif hasattr(dataset, "augment"):
            self._aug_backup.append((dataset, dataset.augment))
            dataset.augment = None

    def _restore_aug(self, dataset=None):
        """Restore previously disabled augmentation pipelines.

        If ``dataset`` is ``None``, restore all cached augmentations.  Otherwise
        only restore augmentation for the specified dataset.
        """
        if dataset is None:
            for ds, aug in self._aug_backup:
                ds.augment = aug
            self._aug_backup = []
        else:
            remaining = []
            for ds, aug in self._aug_backup:
                if ds is dataset:
                    ds.augment = aug
                else:
                    remaining.append((ds, aug))
            self._aug_backup = remaining
            
    def get_log_header(self):
        self.column_heading_list=[
                ["loss"],
                ["val_loss"],
                ["recon_loss"], 
                ["recon_loss_source", "recon_loss_target"],
        ]
        return "loss,val_loss,recon_loss,recon_loss_source,recon_loss_target"

    # --------------------------------------------------------------
    # Domain weighting and AST freezing helpers
    # --------------------------------------------------------------
    def _calc_domain_ratio(self) -> None:
        """Compute source/target clip ratio for loss weighting."""
        from torch.utils.data import ConcatDataset, Subset

        dataset = self.train_loader.dataset
        names: List[str] = []

        def gather(ds):
            if isinstance(ds, Subset):
                base = ds.dataset
                idxs = ds.indices
                if hasattr(base, "basenames"):
                    names.extend([base.basenames[i] for i in idxs])
            else:
                if hasattr(ds, "basenames"):
                    names.extend(list(ds.basenames))

        if isinstance(dataset, ConcatDataset):
            for d in dataset.datasets:
                gather(d)
        else:
            gather(dataset)
        n_target = sum(1 for n in names if "target" in n.lower())
        n_source = max(len(names) - n_target, 1)
        n_target = max(n_target, 1)
        # self._tgt_weight =  float(n_target) / float(n_source)
        # DEBUGGING
        self._tgt_weight = 0.5

    def _freeze_ast(self) -> None:
        for p in self.model.encoder.ast.parameters():
            p.requires_grad = False
        self._ast_frozen = True

    def _unfreeze_ast(self) -> None:
        for name, p in self.model.encoder.ast.named_parameters():
            p.requires_grad = self._orig_ast_requires_grad.get(name, True)
        self._ast_frozen = False

    def _update_latent_noise(self, epoch: int, recon_error: float) -> None:
        """Schedule latent noise activation based on epoch and reconstruction error."""
        if epoch >= 5:
            self.model.latent_noise_std = 0.03
        else:
            self.model.latent_noise_std = 0.0

    def _save_recon_samples(self, epoch: int, num_samples: int = 3) -> None:
        """Save input/reconstruction pairs for a few random training clips."""
        import random
        from torchvision.utils import save_image

        dataset = self.train_loader.dataset
        if len(dataset) == 0:
            return
        idxs = random.sample(range(len(dataset)), k=min(num_samples, len(dataset)))

        self.model.eval()
        imgs = []
        with torch.no_grad():
            for idx in idxs:
                sample = dataset[idx]
                feat = sample[0]
                if feat.shape[-1] != self.model.decoder.time_steps:
                    pad = self.model.decoder.time_steps - feat.shape[-1]
                    if pad > 0:
                        feat = F.pad(feat, (0, pad))
                    else:
                        feat = feat[..., : self.model.decoder.time_steps]
                feat = feat.to(self.device)
                attr = None
                if self.model.use_attribute and len(sample) > 1 and isinstance(sample[1], torch.Tensor) and sample[1].numel() > 0:
                    attr = sample[1].unsqueeze(0).to(self.device)

                recon, _, _, _ = self.model(feat.unsqueeze(0), attr_vec=attr)
                recon = recon[0, :, :, : feat.shape[-1]].cpu()
                cat = torch.cat([feat.cpu(), recon], dim=-1)
                imgs.append(cat)

        for i, img in enumerate(imgs):
            out_path = self.recon_dir / f"epoch{epoch}_{i}.png"
            save_image(img, out_path)

    def train(self, epoch):
        if epoch <= getattr(self, "epoch", 0):
            return
        if epoch > self._warmup_epochs:
            print("Unfreezing AST encoder after warm-up")
            self._unfreeze_ast()
        device = self.device
        if epoch == 1 or epoch == self._warmup_epochs + 1:   # print twice
            n_trainable = sum(p.numel() for p in self.model.encoder.ast.parameters() if p.requires_grad)
            print(f"[DEBUG] epoch {epoch}: trainable AST params = {n_trainable}")
            
        self.model.train()
        train_loss = 0.0
        train_recon_loss = 0.0
        train_recon_loss_source = 0.0
        train_recon_loss_target = 0.0

        for batch in self.train_loader:
            feats = batch[0].to(device).float()
            attr = batch[1].to(device) if self.model.use_attribute and len(batch) > 1 else None
            names = batch[-1] # always the last item in the batch
            _, _, mse, log_mse = self.model(feats, attr_vec=attr)
            if names:
                is_target = torch.tensor(
                    [("target" in n.lower()) for n in names],
                    device=mse.device,
                    dtype=torch.bool,
                )
            else:
                is_target = torch.zeros_like(mse, dtype=torch.bool)
            weights = torch.ones_like(mse)
            # weights[is_target] = self._tgt_weight
            loss = ((mse + self.model.logmag_lambda * log_mse) * weights).mean()
            self.optimizer.zero_grad()
            loss.backward()
            # if not self._ast_frozen:
            #     for n, p in self.model.encoder.ast.named_parameters():
            #         if p.grad is not None:
            #             print(n, p.grad.abs().mean())
            self.optimizer.step()

            is_source = ~is_target


            train_loss += float(loss)
            train_recon_loss += float(loss)
            if is_source.any():
                train_recon_loss_source += float(mse[is_source].mean())
            if is_target.any():
                train_recon_loss_target += float(mse[is_target].mean())

        val_loss = 0.0
        self.model.eval()
        with torch.no_grad():
            for batch in self.valid_loader:
                feats = batch[0].to(device).float()
                attr = batch[1].to(device) if self.model.use_attribute and len(batch) > 1 else None
                _, _, mse, log_mse = self.model(feats, attr_vec=attr)
                val_loss += float((mse + self.model.logmag_lambda * log_mse).mean())

        avg_train = train_loss / len(self.train_loader)
        avg_val = val_loss / len(self.valid_loader)
        avg_recon = train_recon_loss / len(self.train_loader)
        avg_recon_source = train_recon_loss_source / len(self.train_loader)
        avg_recon_target = train_recon_loss_target / len(self.train_loader)

        self._update_latent_noise(epoch, avg_recon)

        with open(self.log_path, "a") as log:
            np.savetxt(
                log,
                [
                    f"{avg_train},{avg_val},{avg_recon},{avg_recon_source},{avg_recon_target}"
                ],
                fmt="%s",
            )

        csv_to_figdata(
            file_path=self.log_path,
            column_heading_list=self.column_heading_list,
            ylabel="loss",
            fig_count=len(self.column_heading_list),
            cut_first_epoch=True,
        )
        self.scheduler.step(avg_val)
        self.epoch = epoch
        print(f"Epoch {epoch} complete: "
                  f"train_loss={avg_train:.4f}, "
                  f"val_loss={avg_val:.4f}, "
                  f"recon_loss={avg_recon:.4f}, "
                  f"recon_loss_source={avg_recon_source:.4f}, "
                  f"recon_loss_target={avg_recon_target:.4f}")
        self._save_recon_samples(epoch)
        # update μ and Σ only at the very last epoch
        if epoch == self.args.epochs:
            print("Now epoch is the last epoch, fitting statistics...")
            self.model.eval()                     # turn off dropout, BN updates
            with torch.no_grad():                 # no gradients needed
                # Temporarily disable SpecAugment when computing μ/Σ
                self._disable_aug(self.train_loader.dataset)

                train_sets = getattr(self.data, "datasets", [self.data])
                stats_dir = Path("stats")
                stats_dir.mkdir(exist_ok=True)
                for dset in train_sets:
                    # turn off SpecAug *just* for this dataset
                    self._disable_aug(dset.train_dataset)

                    # clear previous machine's domain stats
                    if hasattr(self.model, "reset_domain_stats"):
                        self.model.reset_domain_stats()
                    else:
                        self.model.mu.zero_()
                        self.model.cov.fill_(1.0)
                        self.model.m_mean_domain.zero_()
                        self.model.m_std_domain.fill_(1.0)
                        self.model.m_mean.zero_()
                        self.model.m_std.fill_(1.0)
                        self.model.mse_med.zero_()
                        self.model.mse_mad.fill_(1.0)
                    if hasattr(self.model, "n_seen"):
                        self.model.n_seen.zero_()

                    loader = DataLoader(
                        dset.train_dataset,
                        batch_size=32,
                        shuffle=False,
                        collate_fn=pad_collate,
                        num_workers=0,
                    )
                    if len(loader.dataset) == 0:
                        self._restore_aug(dset.train_dataset)
                        continue
                    self.model.latent_noise_std = 0.0
                    self.model.fit_stats_streaming(loader)
                    torch.save(
                        {
                            "mu": self.model.mu.clone(),
                            "cov": self.model.cov.clone(),
                            "m_m": self.model.m_mean_domain.clone(),
                            "m_s": self.model.m_std_domain.clone(),
                            "mse_med": self.model.mse_med.clone(),
                            "mse_mad": self.model.mse_mad.clone(),
                        },
                        stats_dir / f"{dset.machine_type}.pth",
                    )

                    # re-enable SpecAug for this dataset before next one
                    self._restore_aug(dset.train_dataset)

                

                # ── compute anomaly-score distribution on normal training clips ──
                y_pred = []
                domain_list = []
                m_dists_ls = []
                m_norms_ls = []

                for dset in train_sets:
                    loader = DataLoader(
                        dset.train_dataset,
                        batch_size=32,
                        shuffle=False,
                        collate_fn=pad_collate,
                        num_workers=0,
                    )
                    if len(loader.dataset) == 0:
                        continue
                    block = torch.load(stats_dir / f"{dset.machine_type}.pth")
                    self.model.mu.copy_(block["mu"].to(self.model.mu.device))
                    self.model.cov.copy_(block["cov"].to(self.model.cov.device))
                    self.model.m_mean_domain.copy_(
                        block["m_m"].to(self.model.m_mean_domain.device)
                    )
                    self.model.m_std_domain.copy_(
                        block["m_s"].to(self.model.m_std_domain.device)
                    )
                    for batch in loader:
                        feats = batch[0].to(self.device, dtype=torch.float32).float()
                        attr = batch[1].to(self.device) if self.model.use_attribute and len(batch) > 1 else None
                        scores, m_dists, m_norms = self.model.anomaly_score(
                            feats,
                            attr_vec=attr,
                            names=batch[-1],
                        )          # [B]
                        y_pred.extend(scores.cpu().numpy())               # list of floats
                        m_dists_ls.extend(m_dists.cpu().numpy())          # list of floats
                        m_norms_ls.extend(m_norms.cpu().numpy())          # list of floats

                        domain_list.extend(
                            [
                                "target" if "target" in name.lower() else "source"
                                for name in batch[-1]
                            ]
                        )
                # restore noise level for subsequent scoring
                self.model.latent_noise_std = self._latent_noise_base
                # Restore augmentation for subsequent epochs/tests
                self._restore_aug()
            print(
                    "m_norm mean on TRAIN normals:",
                    np.mean(m_norms_ls)
                )
            

            # additional per-machine calibration
            datasets = getattr(self.data, "datasets", None)
            if datasets:
                for dset in datasets:
                    loader = DataLoader(
                        dset.train_dataset,
                        batch_size=32,
                        shuffle=False,
                        collate_fn=pad_collate,
                        num_workers=0,
                    )
                    if len(loader.dataset) == 0:
                        continue
                    block = torch.load(Path("stats") / f"{dset.machine_type}.pth")
                    self.model.mu.copy_(block["mu"])
                    self.model.cov.copy_(block["cov"])
                    self.model.m_mean_domain.copy_(block["m_m"])
                    self.model.m_std_domain.copy_(block["m_s"])
                    self.model.mse_med.copy_(block["mse_med"])
                    self.model.mse_mad.copy_(block["mse_mad"])
                    y_mt = []
                    dlist_mt = []
                    for batch in loader:
                        feats = batch[0].to(self.device, dtype=torch.float32).float()
                        attr = batch[1].to(self.device) if self.model.use_attribute and len(batch) > 1 else None
                        scores, _, _ = self.model.anomaly_score(
                            feats,
                            attr_vec=attr,
                            names=batch[-1],
                        )
                        y_mt.extend(scores.cpu().numpy())
                        # if len(batch) > 3:
                        names = batch[-1]
                        dlist_mt.extend(
                            ["target" if "target" in name.lower() else "source" for name in batch[-1]]
                        )
                    self.fit_anomaly_score_distribution(
                        y_pred=y_mt,
                        domain_list=dlist_mt,
                        machine_type=dset.machine_type,
                    )
            # save a global distribution for all training data
            print("Fitting global anomaly score distribution...")  
            self.fit_anomaly_score_distribution(
                y_pred=y_pred,
                domain_list=None,           # <= one distribution
                percentile=self.args.decision_threshold
            )
                    
            # ── final export ────────────────────────────────────────────────
            print("Saving model and training statistics...")
            torch.save(self.model.state_dict(), self.model_path)  # for inference

        else:
            # save model and optimizer state for resuming training
            checkpoint_path = Path(self.checkpoint_path)
            if not checkpoint_path.parent.exists():
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                    "loss": avg_train,
                },
                checkpoint_path,
            )

    def test(self):
        device = self.device
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.load_state_dict(checkpoint)
            else:
                # adapt checkpoints saved as plain state_dict
                self.load_state_dict({
                    "model_state_dict": checkpoint,
                    "epoch": 0,
                    "loss": 0,
                })
        self.model.eval()

        mode = self.data.mode

        dir_name = "test"
        result_dir = self.result_dir if self.args.dev else self.eval_data_result_dir
        datasets = getattr(self.data, "datasets", [self.data])

        for d in datasets:
            csv_lines = []
            if mode:
                performance = []
                anm_score_figdata = AnmScoreFigData()

            dataset_str = getattr(d, "dataset_str", getattr(d, "machine_type", self.args.dataset))
            print(f"Testing dataset: {dataset_str}")
            machine_type = getattr(d, "machine_type", None)
            print(f"Machine type: {machine_type}")

            if machine_type:
                stats_path = Path("stats") / f"{machine_type}.pth"
                if stats_path.exists():
                    block = torch.load(stats_path, map_location=device)
                    self.model.mu.copy_(block["mu"])
                    self.model.cov.copy_(block["cov"])
                    self.model.m_mean_domain.copy_(block["m_m"])
                    self.model.m_std_domain.copy_(block["m_s"])
                    self.model.mse_med.copy_(block["mse_med"])
                    self.model.mse_mad.copy_(block["mse_mad"])

            for idx, test_loader in enumerate(d.test_loader):
                section_name = f"section_{d.section_id_list[idx]}"

                anomaly_score_csv = result_dir / (
                    f"anomaly_score_{dataset_str}_{section_name}_{dir_name}_seed{self.args.seed}{self.model_name_suffix}{self.eval_suffix}.csv"
                )
                decision_result_csv = result_dir / (
                    f"decision_result_{dataset_str}_{section_name}_{dir_name}_seed{self.args.seed}{self.model_name_suffix}{self.eval_suffix}.csv"
                )

                scores, domains, basenames = [], [], []
                y_true = []
                with torch.no_grad():
                    for batch in test_loader:
                        feats = batch[0].to(device, dtype=torch.float32).float()
                        attr_vec = None                     # default
                        # attribute tensor is the first 2-D tensor after feats
                        for t in batch[1:]:
                            if isinstance(t, torch.Tensor) and t.ndim == 2:
                                attr_vec = t.to(device)     # may be empty
                                break

                        clip_scores, _, _ = self.model.anomaly_score(
                            feats,
                            attr_vec=attr_vec,
                            names=batch[-1],
                        )
                        clip_scores = clip_scores.cpu().tolist()
                        scores.extend(clip_scores)
                        basenames.extend(batch[-1])         # always last element
                            
                        # one basename & domain per *clip* in the batch
                        domains.extend(
            ["target" if "target" in n.lower() else "source" for n in batch[-1]]
        )
                        if mode:
                            label_tensor = next(
                                    t for t in batch if isinstance(t, torch.Tensor) and t.ndim == 1
                                )
                            y_true.extend(label_tensor.int().tolist())

                from sklearn.metrics import roc_auc_score
                print("quick sanity AUC =", roc_auc_score(y_true, scores))
                print(len(scores), len(domains), len(y_true))
                print("number of anomaly in the true label: ", np.sum(y_true))
                if mode and y_true:
                    normal_mean = float(np.mean([s for s, l in zip(scores, y_true) if l == 0]))
                    anomaly_mean = float(np.mean([s for s, l in zip(scores, y_true) if l == 1]))
                    print("normal", normal_mean, "anomaly", anomaly_mean)
                # fit distribution for this machine section
                
                decision_thresholds = self.calc_decision_threshold()
                y_pred = scores

                anomaly_score_list = [[b, s] for b, s in zip(basenames, scores)]
                decision_result_list = []
                for b, s, domain in zip(basenames, scores, domains):
                    key = f"{machine_type}_{domain}" if machine_type else domain
                    thresh = decision_thresholds.get(
                        key,
                        decision_thresholds.get(
                            domain,
                            decision_thresholds.get("all", next(iter(decision_thresholds.values())))
                        ),
                    )
                    decision_result_list.append([b, 1 if s > thresh else 0])
                if mode:
                    domain_list = domains

                save_csv(anomaly_score_csv, anomaly_score_list)
                save_csv(decision_result_csv, decision_result_list)

                if mode:
                    y_true_s = [y_true[i] for i in range(len(y_true)) if domain_list[i] == "source"]
                    y_pred_s = [y_pred[i] for i in range(len(y_true)) if domain_list[i] == "source"]
                    y_true_t = [y_true[i] for i in range(len(y_true)) if domain_list[i] == "target"]
                    y_pred_t = [y_pred[i] for i in range(len(y_true)) if domain_list[i] == "target"]

                    auc_s = metrics.roc_auc_score(y_true_s, y_pred_s)
                    auc_t = metrics.roc_auc_score(y_true_t, y_pred_t)
                    p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=self.args.max_fpr)
                    p_auc_s = metrics.roc_auc_score(y_true_s, y_pred_s, max_fpr=self.args.max_fpr)
                    p_auc_t = metrics.roc_auc_score(y_true_t, y_pred_t, max_fpr=self.args.max_fpr)

                    thresh_s = decision_thresholds.get(
                        f"{machine_type}_source" if machine_type else "source",
                        decision_thresholds.get(
                            "source",
                            decision_thresholds.get("all", next(iter(decision_thresholds.values())))
                        ),
                    )
                    thresh_t = decision_thresholds.get(
                        f"{machine_type}_target" if machine_type else "target",
                        decision_thresholds.get(
                            "target",
                            decision_thresholds.get("all", next(iter(decision_thresholds.values())))
                        ),
                    )

                    tn, fp, fn, tp = metrics.confusion_matrix(
                        y_true_s, [1 if x > thresh_s else 0 for x in y_pred_s]
                    ).ravel()
                    prec_s = tp / np.maximum(tp + fp, np.finfo(float).eps)
                    recall_s = tp / np.maximum(tp + fn, np.finfo(float).eps)
                    f1_s = 2.0 * prec_s * recall_s / np.maximum(prec_s + recall_s, np.finfo(float).eps)

                    tn, fp, fn, tp = metrics.confusion_matrix(
                        y_true_t, [1 if x > thresh_t else 0 for x in y_pred_t]
                    ).ravel()
                    prec_t = tp / np.maximum(tp + fp, np.finfo(float).eps)
                    recall_t = tp / np.maximum(tp + fn, np.finfo(float).eps)
                    f1_t = 2.0 * prec_t * recall_t / np.maximum(prec_t + recall_t, np.finfo(float).eps)

                    if len(csv_lines) == 0:
                        csv_lines.append(self.result_column_dict["source_target"])
                    csv_lines.append([
                        section_name.split("_", 1)[1],
                        auc_s,
                        auc_t,
                        p_auc,
                        p_auc_s,
                        p_auc_t,
                        prec_s,
                        prec_t,
                        recall_s,
                        recall_t,
                        f1_s,
                        f1_t,
                    ])
                    performance.append([
                        auc_s,
                        auc_t,
                        p_auc,
                        p_auc_s,
                        p_auc_t,
                        prec_s,
                        prec_t,
                        recall_s,
                        recall_t,
                        f1_s,
                        f1_t,
                    ])

                    anm_score_figdata.append_figdata(
                        anm_score_figdata.anm_score_to_figdata(
                            scores=[[t, p] for t, p in zip(y_true_s, y_pred_s)],
                            title=f"{section_name}_source_AUC{auc_s}"
                        )
                    )
                    anm_score_figdata.append_figdata(
                        anm_score_figdata.anm_score_to_figdata(
                            scores=[[t, p] for t, p in zip(y_true_t, y_pred_t)],
                            title=f"{section_name}_target_AUC{auc_t}"
                        )
                    )

            if mode:
                mean_perf = np.mean(np.array(performance, dtype=float), axis=0)
                csv_lines.append(["arithmetic mean"] + list(mean_perf))
                hmean_perf = scipy.stats.hmean(np.maximum(np.array(performance, dtype=float), np.finfo(float).eps), axis=0)
                csv_lines.append(["harmonic mean"] + list(hmean_perf))
                csv_lines.append([])

                anm_score_figdata.show_fig(
                    title=self.args.model + "_" + dataset_str + self.model_name_suffix + self.eval_suffix + "_anm_score",
                    export_dir=result_dir,
                )

                result_path = result_dir / (
                    f"result_{dataset_str}_{dir_name}_seed{self.args.seed}{self.model_name_suffix}{self.eval_suffix}_roc.csv"
                )
                save_csv(result_path, csv_lines)


def save_csv(save_file_path, save_data):
    with open(save_file_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(save_data)


