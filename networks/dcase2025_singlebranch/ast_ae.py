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
        alpha: float = 0.8,
        latent_noise_std: float = 0.0,
        cfg: Dict = None,
        *,
        attr_dim: int = 0,
        use_attribute: bool = False,
    ) -> None:
        super().__init__()
        freeze_layers = cfg.get("ast_freeze_layers", 0)
        self.encoder = ASTEncoder(
            latent_dim=latent_dim,
            freeze_layers=freeze_layers,
            n_mels=n_mels,
            T_fix=time_steps,
        )
        self.decoder = SpectroDecoder(latent_dim=latent_dim, n_mels=n_mels, time_steps=time_steps)
        self.alpha = alpha
        self.latent_noise_std = latent_noise_std

        self.use_attribute = use_attribute
        if self.use_attribute and attr_dim > 0:
            self.attr_fc = nn.Sequential(
                nn.Linear(attr_dim, 64),
                nn.ReLU(),
            )
            self.fuse_fc = nn.Linear(latent_dim + 64, latent_dim)

        # Mean and precision for Mahalanobis – initialised later via `fit_stats`.
        self.register_buffer("mu", torch.zeros(latent_dim))
        self.register_buffer("inv_cov", torch.eye(latent_dim))
        # Parameters of Mahalanobis distance distribution (mean/std).
        self.register_buffer("m_mean", torch.zeros(1))
        self.register_buffer("m_std", torch.ones(1))
        
        self.register_buffer("mse_mean", torch.zeros(1))
        self.register_buffer("mse_std", torch.ones(1))
        

    # ------------------------------------------------------------------
    # Forward pass (training) returns reconstruction loss.
    # ------------------------------------------------------------------
    def forward(self, x: Tensor, attr_vec: Tensor | None = None) -> Tuple[Tensor, Tensor, Tensor]:
        """Return recon, latent *z*, and per‑sample MSE reconstruction error."""
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
        return recon, z, mse

    # ------------------------------------------------------------------
    # Statistics fitting – call once on *normal* training data.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def fit_stats(self, dataloader: torch.utils.data.DataLoader) -> None:
        """Compute mean and inverse covariance of latent *z* on NORMAL data."""
        zs: List[Tensor] = []
        device = next(self.parameters()).device
        for batch in dataloader:
            xb = batch[0].to(device).float()
            if self.use_attribute and len(batch) > 1:
                attr = batch[1].to(device)
            else:
                attr = None
            z = self.forward(xb, attr_vec=attr)[1]
            zs.append(z)
        z_all = torch.cat(zs, dim=0)
        self.mu = z_all.mean(0, keepdim=False)
        cov = torch.cov(z_all.T) + 1e-6 * torch.eye(z_all.size(1), device=device)
        self.inv_cov = torch.linalg.inv(cov)
        
    @torch.no_grad()
    def fit_stats_streaming(self, loader):
        mean = torch.zeros(self.mu.shape, device=self.mu.device)
        M2 = torch.zeros_like(self.inv_cov)
        n = 0
        for batch in loader:
            xb = batch[0].to(self.mu.device).float()
            if self.use_attribute and len(batch) > 1:
                attr = batch[1].to(self.mu.device)
            else:
                attr = None
            z = self.forward(xb, attr_vec=attr)[1]
            for zi in z:
                n += 1
                delta = zi - mean
                mean += delta / n
                M2   += torch.outer(delta, zi - mean)
        cov = M2 / max(n - 1, 1) + 1e-6 * torch.eye(mean.numel(), device=mean.device)
        self.mu.copy_(mean)
        self.inv_cov.copy_(torch.linalg.inv(cov))

        # Compute Mahalanobis distance statistics for z-scoring
        dists = []
        recon_errs = []
        for batch in loader:
            xb = batch[0].to(self.mu.device).float()
            if self.use_attribute and len(batch) > 1:
                attr = batch[1].to(self.mu.device)
            else:
                attr = None

            recon, z, mse = self.forward(xb, attr_vec=attr)
            recon_errs.append(mse)
            delta = z - self.mu
            md = torch.einsum("bi,ij,bj->b", delta, self.inv_cov, delta)
            dists.append(md)
        m_dist_train = torch.cat(dists)
        self.m_mean.copy_(m_dist_train.mean())
        self.m_std.copy_(m_dist_train.std() + 1e-9)
        
        recon_errs = torch.cat(recon_errs)
        self.mse_mean.copy_(recon_errs.mean())
        self.mse_std.copy_(recon_errs.std() + 1e-9)


    # ------------------------------------------------------------------
    # Scoring – used at inference.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def anomaly_score(self, x: Tensor, attr_vec: Tensor | None = None) -> Tensor:
        """Compute combined anomaly score for input batch."""
        recon, z, mse = self.forward(x, attr_vec=attr_vec)
        # Mahalanobis distance D_M(z)
        delta = z - self.mu
        m_dist = torch.einsum("bi,ij,bj->b", delta, self.inv_cov, delta)

        # Combine z-scored Mahalanobis distance with MSE
        m_norm = (m_dist - self.m_mean) / self.m_std
        mse_norm = (mse - self.mse_mean) / self.mse_std
        score = self.alpha * m_norm + (1.0 - self.alpha) * mse_norm
        return score


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
        rc, z, mse = model(xb, attr_vec=attr)
        loss = mse.mean()
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
        return ASTAutoencoder(
            latent_dim=latent,
            n_mels=self.data.height,
            time_steps=time_steps,
            alpha=alpha,
            latent_noise_std=0.0,
            cfg = cfg,
            attr_dim=attr_dim,
            use_attribute=use_attribute,
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

    def _restore_aug(self):
        """Restore previously disabled augmentation pipelines."""
        for ds, aug in self._aug_backup:
            ds.augment = aug
        self._aug_backup = []

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
        self._tgt_weight = float(n_source) / float(n_target)

    def _freeze_ast(self) -> None:
        for p in self.model.encoder.ast.parameters():
            p.requires_grad = False
        self._ast_frozen = True

    def _unfreeze_ast(self) -> None:
        for name, p in self.model.encoder.ast.named_parameters():
            p.requires_grad = self._orig_ast_requires_grad.get(name, True)
        self._ast_frozen = False

    def get_log_header(self):
        self.column_heading_list = [
            ["loss"],
            ["val_loss"],
            ["recon_loss"],
            ["recon_loss_source", "recon_loss_target"],
        ]
        return (
            "loss,val_loss,recon_loss,recon_loss_source,recon_loss_target"
        )

    def apply_ema(self, values, alpha=0.2):
        smoothed = []
        ema = None
        for v in values:
            ema = v if ema is None else alpha * v + (1 - alpha) * ema
            smoothed.append(ema)
        return smoothed

    def _update_latent_noise(self, epoch: int, recon_error: float) -> None:
        """Schedule latent noise activation based on epoch and reconstruction error."""
        if epoch <= 10:
            self.model.latent_noise_std = 0.0
        elif not self.noise_enabled:
            if recon_error < 0.5:
                self.model.latent_noise_std = self._latent_noise_base
                self.noise_enabled = True
            else:
                self.model.latent_noise_std = 0.0
        elif epoch >=50:
            self.model.latent_noise_std = 0.05
            self.noise_enabled = True

    def train(self, epoch):
        if epoch <= getattr(self, "epoch", 0):
            return
        if epoch > self._warmup_epochs:
            print("Unfreezing AST encoder after warm-up")
            self._unfreeze_ast()
        device = self.device
        # print(self.model.encoder.ast.patch_embed.proj.weight.mean())
        # print("AST parameter examples:")
        # for n, p in self.model.encoder.ast.named_parameters():
        #     print(" ", n, p.shape)
        #     break          # we only need the first one
        # w = self.model.encoder.ast.state_dict()['embeddings.cls_token']   # (1,1,768)
        # print("mean =", w.mean().item(), "std =", w.std().item())
        # print("DEBUGGING: AST encoder parameters:")
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
            names = batch[3] if len(batch) > 3 else []
            _, _, mse = self.model(feats, attr_vec=attr)
            if names:
                is_target = torch.tensor(
                    [("target" in n.lower()) for n in names],
                    device=mse.device,
                    dtype=torch.bool,
                )
            else:
                is_target = torch.zeros_like(mse, dtype=torch.bool)
            weights = torch.ones_like(mse)
            weights[is_target] = self._tgt_weight
            loss = (mse * weights).mean()
            self.optimizer.zero_grad()
            loss.backward()
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
                _, _, mse = self.model(feats, attr_vec=attr)
                val_loss += float(mse.mean())

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
        # update μ and Σ only at the very last epoch
        if epoch == self.args.epochs - 1:
            print("Now epoch is the last epoch, fitting statistics...")
            self.model.eval()                     # turn off dropout, BN updates
            with torch.no_grad():                 # no gradients needed
                # Temporarily disable SpecAugment when computing μ/Σ
                self._disable_aug(self.train_loader.dataset)
                clean_loader = DataLoader(
                    self.train_loader.dataset,
                    batch_size=32,
                    shuffle=False,
                    collate_fn=pad_collate,
                    num_workers=4,
                )
                self.model.latent_noise_std = self._latent_noise_base
                self.model.fit_stats_streaming(clean_loader)
                self.model.latent_noise_std = self._latent_noise_base # keep noise for testing

                # ── compute anomaly-score distribution on normal training clips ──
                y_pred = []
                domain_list = []
                for batch in clean_loader:
                    feats = batch[0].to(self.device).float()
                    attr = batch[1].to(self.device) if self.model.use_attribute and len(batch) > 1 else None
                    scores = self.model.anomaly_score(feats, attr_vec=attr)          # [B]
                    y_pred.extend(scores.cpu().numpy())               # list of floats
                    domain_list.extend(
                        [
                            "target" if "target" in name.lower() else "source"
                            for name in batch[3]
                        ]
                    )
                # Restore augmentation for subsequent epochs/tests
                self._restore_aug()

            # fit whichever parametric or percentile model you use for thresholds
            self.fit_anomaly_score_distribution(y_pred=y_pred, domain_list=domain_list)

            # additional per-machine calibration
            datasets = getattr(self.data, "datasets", None)
            if datasets:
                for dset in datasets:
                    loader = DataLoader(
                        dset.train_dataset,
                        batch_size=32,
                        shuffle=False,
                        collate_fn=pad_collate,
                        num_workers=4,
                    )
                    y_mt = []
                    dlist_mt = []
                    for batch in loader:
                        feats = batch[0].to(self.device).float()
                        attr = batch[1].to(self.device) if self.model.use_attribute and len(batch) > 1 else None
                        scores = self.model.anomaly_score(feats, attr_vec=attr)
                        y_mt.extend(scores.cpu().numpy())
                        if len(batch) > 3:
                            dlist_mt.extend(
                                ["target" if "target" in name.lower() else "source" for name in batch[3]]
                            )
                    self.fit_anomaly_score_distribution(
                        y_pred=y_mt,
                        domain_list=dlist_mt,
                        machine_type=dset.machine_type,
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
                        feats = batch[0].to(device).float()
                        attr_vec = None                     # default
                        # attribute tensor is the first 2-D tensor after feats
                        for t in batch[1:]:
                            if isinstance(t, torch.Tensor) and t.ndim == 2:
                                attr_vec = t.to(device)     # may be empty
                                break

                        clip_scores = self.model.anomaly_score(feats, attr_vec=attr_vec).cpu().tolist()
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
                            print("batch labels:", label_tensor.tolist())          # should mix 0 & 1
                            print("accumulated label set:", set(y_true))           # should be {0,1}

                from sklearn.metrics import roc_auc_score
                print("quick sanity AUC =", roc_auc_score(y_true, scores))
                print(len(scores), len(domains), len(y_true))
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
