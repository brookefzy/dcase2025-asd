from __future__ import annotations
from pathlib import Path
from typing import Tuple, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import ASTModel
from models.branch_decoder import SpectroDecoder
from models.branch_astencoder import ASTEncoder
from networks.base_model import BaseModel
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
        alpha: float = 0.5,
        latent_noise_std: float = 0.0,
    ) -> None:
        super().__init__()
        freeze_layers = cfg.get("ast_freeze_layers", 0)
        self.encoder = ASTEncoder(latent_dim=latent_dim, freeze_layers=freeze_layers)
        self.decoder = SpectroDecoder(latent_dim=latent_dim, n_mels=n_mels, time_steps=time_steps)
        self.alpha = alpha
        self.latent_noise_std = latent_noise_std

        # Mean and precision for Mahalanobis – initialised later via `fit_stats`.
        self.register_buffer("mu", torch.zeros(latent_dim))
        self.register_buffer("inv_cov", torch.eye(latent_dim))

    # ------------------------------------------------------------------
    # Forward pass (training) returns reconstruction loss.
    # ------------------------------------------------------------------
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Return recon, latent *z*, and per‑sample MSE reconstruction error."""
        B, _, _, T = x.shape
        z = self.encoder(x)                # [B, latent]
        if self.training and getattr(self, "latent_noise_std", 0) > 0:
            z = z + torch.randn_like(z) * self.latent_noise_std
        recon = self.decoder(z, T)         # [B, 1, n_mels, T]
        mse = F.mse_loss(recon, x, reduction="none")
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
        for xb, _ in dataloader:
            xb = xb.to(device)
            z = self.encoder(xb)
            zs.append(z)
        z_all = torch.cat(zs, dim=0)
        self.mu = z_all.mean(0, keepdim=False)
        cov = torch.cov(z_all.T) + 1e-6 * torch.eye(z_all.size(1), device=device)
        self.inv_cov = torch.linalg.inv(cov)
        
    @torch.no_grad()
    def fit_stats_streaming(self, loader):
        mean = torch.zeros(self.mu.shape, device=self.mu.device)
        M2   = torch.zeros_like(self.inv_cov)
        n = 0
        for xb, _ in loader:
            z = self.encoder(xb.to(self.mu.device))
            for zi in z:
                n += 1
                delta = zi - mean
                mean += delta / n
                M2   += torch.outer(delta, zi - mean)
        cov = M2 / max(n - 1, 1) + 1e-6 * torch.eye(mean.numel(), device=mean.device)
        self.mu.copy_(mean)
        self.inv_cov.copy_(torch.linalg.inv(cov))


    # ------------------------------------------------------------------
    # Scoring – used at inference.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def anomaly_score(self, x: Tensor) -> Tensor:
        """Compute combined anomaly score for input batch."""
        recon, z, mse = self.forward(x)

        # Mahalanobis distance D_M(z)
        delta = z - self.mu
        m_dist = torch.einsum("bi,ij,bj->b", delta, self.inv_cov, delta)

        score = self.alpha * m_dist + (1.0 - self.alpha) * mse
        return score


# -----------------------------------------------------------------------------
# 4. Example training loop skeleton (pseudo‑code)
# -----------------------------------------------------------------------------

def train_one_epoch(model: ASTAutoencoder, loader, optim, device: torch.device):
    """This is the pseudo-code for a single training epoch."""
    model.train()
    total_loss = 0.0
    for xb, _ in loader:
        xb = xb.to(device)
        rc, z, mse = model(xb)
        loss = mse.mean()
        optim.zero_grad()
        loss.backward()
        optim.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)


def validate(model: ASTAutoencoder, loader, device: torch.device):
    model.eval()
    scores = []
    labels = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            score = model.anomaly_score(xb)
            scores.extend(score.cpu().tolist())
            labels.extend(yb.cpu().tolist())
    # Compute AUC / pAUC, etc. – left as exercise
    return scores, labels


class ASTAutoencoderASD(BaseModel):
    """Wrapper integrating ``ASTAutoencoder`` with the training framework."""

    def init_model(self):
        cfg = self.args.__dict__
        self.cfg = cfg
        self.device = "cuda" if self.args.use_cuda and torch.cuda.is_available() else "cpu"
        latent = cfg.get("latent_dim", 128)
        alpha = cfg.get("alpha", 0.5)
        time_steps = cfg.get("time_steps", 512)
        self._latent_noise_base = cfg.get("latent_noise_std", 0.0)
        return ASTAutoencoder(
            latent_dim=latent,
            n_mels=self.data.height,
            time_steps=time_steps,
            alpha=alpha,
            latent_noise_std=0.0,
        )

    def __init__(self, args, train, test):
        super().__init__(args=args, train=train, test=test)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.ema_scores = []
        self.noise_enabled = False
        self.model.latent_noise_std = 0.0

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
            if recon_error < 20:
                self.model.latent_noise_std = self._latent_noise_base
                self.noise_enabled = True
            else:
                self.model.latent_noise_std = 0.0

    def train(self, epoch):
        if epoch <= getattr(self, "epoch", 0):
            return
        device = self.device
        self.model.train()
        train_loss = 0.0
        train_recon_loss = 0.0
        train_recon_loss_source = 0.0
        train_recon_loss_target = 0.0

        for batch in self.train_loader:
            feats = batch[0].to(device).float()
            _, _, mse = self.model(feats)
            loss = mse.mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            is_target = torch.tensor(
                [("target" in n.lower()) for n in batch[3]],
                device=mse.device, dtype=torch.bool

            )
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
                _, _, mse = self.model(feats)
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
        self.epoch = epoch
        # update μ and Σ only at the very last epoch
        if epoch == self.args.epochs - 1:
            self.model.eval()                     # turn off dropout, BN updates
            with torch.no_grad():                 # no gradients needed
                self.model.fit_stats_streaming(self.train_loader)

                # ── compute anomaly-score distribution on normal training clips ──
                y_pred = []
                for batch in self.train_loader:
                    feats = batch[0].to(self.device).float()
                    scores = self.model.anomaly_score(feats)          # [B]
                    y_pred.extend(scores.cpu().numpy())               # list of floats

            # fit whichever parametric or percentile model you use for thresholds
            self.fit_anomaly_score_distribution(y_pred=y_pred)
            # ── final export ────────────────────────────────────────────────
            torch.save(self.model.state_dict(), self.model_path)  # for inference

        else:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "loss": avg_train,
                },
                self.checkpoint_path,
            )

    def test(self):
        device = self.device
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path, map_location=device))
        self.model.eval()

        decision_threshold = self.calc_decision_threshold()
        anm_score_figdata = AnmScoreFigData()
        mode = self.data.mode
        csv_lines = []
        if mode:
            performance = []

        dir_name = "test"
        for idx, test_loader in enumerate(self.test_loader):
            section_name = f"section_{self.data.section_id_list[idx]}"
            result_dir = self.result_dir if self.args.dev else self.eval_data_result_dir

            anomaly_score_csv = result_dir / (
                f"anomaly_score_{self.args.dataset}_{section_name}_{dir_name}_seed{self.args.seed}{self.model_name_suffix}{self.eval_suffix}.csv"
            )
            decision_result_csv = result_dir / (
                f"decision_result_{self.args.dataset}_{section_name}_{dir_name}_seed{self.args.seed}{self.model_name_suffix}{self.eval_suffix}.csv"
            )

            anomaly_score_list = []
            decision_result_list = []
            domain_list = [] if mode else None
            y_pred = []
            y_true = []
            with torch.no_grad():
                for batch in test_loader:
                    feats = batch[0].to(device).float()
                    score = self.model.anomaly_score(feats).cpu().numpy()
                    basename = batch[3][0]
                    y_true.append(batch[1][0].item())
                    y_pred.append(score)
                    anomaly_score_list.append([basename, score])
                    decision_result_list.append([basename, 1 if score > decision_threshold else 0])
                    if mode:
                        domain_list.append("target" if "target" in basename.lower() else "source")

            save_csv(anomaly_score_csv, anomaly_score_list)
            save_csv(decision_result_csv, decision_result_list)

            if mode:
                y_true_s = [y_true[i] for i in range(len(y_true)) if domain_list[i] == "source"]
                y_pred_s = [y_pred[i] for i in range(len(y_true)) if domain_list[i] == "source"]
                auc_s = metrics.roc_auc_score(y_true_s, y_pred_s)
                p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=self.args.max_fpr)
                tn, fp, fn, tp = metrics.confusion_matrix(
                    y_true_s, [1 if x > decision_threshold else 0 for x in y_pred_s]
                ).ravel()
                prec = tp / np.maximum(tp + fp, np.finfo(float).eps)
                recall = tp / np.maximum(tp + fn, np.finfo(float).eps)
                f1 = 2.0 * prec * recall / np.maximum(prec + recall, np.finfo(float).eps)

                if len(csv_lines) == 0:
                    csv_lines.append(self.result_column_dict["single_domain"])
                csv_lines.append([section_name.split("_", 1)[1], auc_s, p_auc, prec, recall, f1])
                performance.append([auc_s, p_auc, prec, recall, f1])

                anm_score_figdata.append_figdata(
                    anm_score_figdata.anm_score_to_figdata(
                        scores=[[t, p] for t, p in zip(y_true_s, y_pred_s)],
                        title=f"{section_name}_source_AUC{auc_s}"
                    )
                )

        if mode:
            mean_perf = np.mean(np.array(performance, dtype=float), axis=0)
            csv_lines.append(["arithmetic mean"] + list(mean_perf))
            hmean_perf = scipy.stats.hmean(np.maximum(np.array(performance, dtype=float), np.finfo(float).eps), axis=0)
            csv_lines.append(["harmonic mean"] + list(hmean_perf))
            csv_lines.append([])

            anm_score_figdata.show_fig(
                title=self.args.model + "_" + self.args.dataset + self.model_name_suffix + self.eval_suffix + "_anm_score",
                export_dir=result_dir,
            )

            result_path = result_dir / (
                f"result_{self.args.dataset}_{dir_name}_seed{self.args.seed}{self.model_name_suffix}{self.eval_suffix}_roc.csv"
            )
            save_csv(result_path, csv_lines)


def save_csv(save_file_path, save_data):
    with open(save_file_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(save_data)
