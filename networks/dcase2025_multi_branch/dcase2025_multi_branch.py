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
import re

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
from models.meta_learner import MetaLearner

class DCASE2025MultiBranch(BaseModel):
    """
    DCASE2025 Multi-Branch Model
    """
    def init_model(self):
        """Return a dummy module for BaseModel initialisation.

        ``BaseModel`` expects :func:`init_model` to return a ``torch.nn.Module``
        which is then moved to the configured device.  The multi-branch model
        manages its sub-modules itself, so here we simply return an identity
        module to satisfy that requirement.
        """
        return torch.nn.Identity()
    def __init__(self, args, train, test):
        super().__init__(
            args=args,
            train=train,
            test=test
            
        )
        device = 'cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu'
        cfg = self.args.__dict__
        self.cfg = cfg
        self.device = device

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
        self.meta_learner = MetaLearner(self.fusion, lr_inner=cfg.get("maml_lr", 1e-2))
        self.maml_shots = cfg.get("maml_shots", 5)
        self.fusion_var_lambda = cfg.get("fusion_var_lambda", 0.1)
        self.w_fusion = cfg.get("w_fusion", 0.1)
        # running means for normalising branch losses
        self.mu2 = 1.0
        self.mu5 = 1.0
        # Optimizer setup
        # self.optimizer = optim.Adam(self.parameters(), lr=cfg["learning_rate"])
                # Optimizer setup - combine parameters from all submodules
        parameter_list = (
            list(self.b1.parameters())
            + list(self.b2.parameters())
            + list(self.b3.parameters())
            + list(self.b5.parameters())
            + list(self.b_attr.parameters())
            + list(self.fusion.parameters())
        )
        self.optimizer = optim.Adam(parameter_list, lr=cfg["learning_rate"])

        # Sanity check that ``BranchContrastive`` parameters are registered in
        # the optimiser.  ``param_groups`` may contain all parameters in a
        # single group, so comparing parameter counts directly is unreliable.
        b3_param_ids = {id(p) for p in self.b3.parameters()}
        opt_param_ids = {
            id(p)
            for pg in self.optimizer.param_groups
            for p in pg["params"]
        }
        assert b3_param_ids <= opt_param_ids, (
            "BranchContrastive parameters missing from optimiser!"
        )

    def _compute_branch_scores(self, x, labels=None, attrs=None, fusion_module=None):
        """Return branch losses and fused score for input batch."""
        # Use first augmented view for branches other than contrastive
        x_main = x[:, 0]

        # Branch 1: Pretrained feature extractor
        z1 = self.b1(x_main)
        # Branch 2: Autoencoder (reconstruction and latent)
        recon2, z2 = self.b2(x_main)
        feats_ds = torch.nn.functional.adaptive_avg_pool2d(
            x_main, (self.cfg["n_mels"], recon2.shape[-1])
        )
        loss2 = ((recon2 - feats_ds) ** 2).reshape(x_main.size(0), -1).mean(dim=1)

        # Branch 3 - InfoNCE contrastive loss
        z3, loss3, loss3_ce = self.b3(x)

        # Branch 5 - log-likelihood from the normalising flow
        logp = self.b5(torch.cat([z1, z2, z3], dim=1))  # <= 0
        loss5 = -logp  # >= 0
        if attrs is not None:
            z_attr = self.b_attr(attrs)
            flow_input = torch.cat([z1, z2, z3, logp.unsqueeze(1), z_attr], dim=1)
            logp = self.b5(flow_input)
            loss5 = -logp

        stacked = torch.stack([loss2, loss3, loss5], 1)
        stacked = (stacked - stacked.mean(0)) / (stacked.std(0) + 1e-6)
        fusion_net = fusion_module if fusion_module is not None else self.fusion
        scores = fusion_net(stacked)
        return loss2, loss3, loss5, scores, loss3_ce

    def sanity_check(self, x, labels=None, attrs=None):
        """Print branch losses and fused scores for debugging."""
        # ``DCASE2025MultiBranch`` does not inherit from ``torch.nn.Module`` so
        # it doesn't have the ``eval``/``train`` helpers.  Manually switch all
        # sub-modules to evaluation mode here and restore training mode after
        # the check.
        self.b1.eval()
        self.b2.eval()
        self.b3.eval()
        self.b5.eval()
        self.b_attr.eval()
        self.fusion.eval()
        with torch.no_grad():
            loss2, score3, loss5, fused, loss3_ce = self._compute_branch_scores(x, labels, attrs)
            print("loss2 (MSE) :", loss2[:8])
            print("score3      :", score3[:8])
            print("loss3 CE    :", loss3_ce.mean().item())
            print("loss5 (flow) :", loss5[:8])
            print("fused       :", fused[:8])
        self.b1.train()
        self.b2.train()
        self.b3.train()
        self.b5.train()
        self.b_attr.train()
        self.fusion.train()
        
    def forward(self, x, labels=None, attrs=None, fusion_module=None):
        """Compute branch losses and fused anomaly score."""
        return self._compute_branch_scores(x, labels, attrs, fusion_module)

    
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
        train_recon_loss = 0.0
        train_recon_loss_source = 0.0
        train_recon_loss_target = 0.0
        y_pred = []

        for batch_idx, batch in enumerate(tqdm(self.train_loader)):
            feats = batch[0].to(device).float()  # [B,2,1,n_mels,frames]
            labels = torch.argmax(batch[2], dim=1).long().to(device)

            self.optimizer.zero_grad()
            loss2, score3, loss5, scores, loss3_ce = self.forward(feats, labels)
            # Reconstruction losses for logging
            data_name_list = batch[3]
            is_target_list = [re.search(r"target", name, re.IGNORECASE) is not None for name in data_name_list]
            # is_source_list = [not f for f in is_target_list]
            is_source_list = np.logical_not(is_target_list).tolist()

            n_source = is_source_list.count(True)
            n_target = is_target_list.count(True)
            if epoch == 1 and batch_idx == 0 and n_target == 0:
                print("Warning: no target domain samples found in first batch")

            recon_loss = loss2.mean()
            recon_loss_source = (
                loss2[is_source_list].mean() if n_source > 0 else torch.tensor(0.0, device=device)
            )
            recon_loss_target = (
                loss2[is_target_list].mean() if n_target > 0 else torch.tensor(0.0, device=device)
            )

            # update running means for normalisation
            if epoch == 1 and batch_idx == 0:
                self.mu2 = loss2.mean().item()
                self.mu5 = loss5.mean().item()
            else:
                self.mu2 = 0.99 * self.mu2 + 0.01 * loss2.mean().item()
                self.mu5 = 0.99 * self.mu5 + 0.01 * loss5.mean().item()

            loss2_norm = loss2 / (self.mu2 + 1e-6)
            loss5_norm = loss5 / (self.mu5 + 1e-6)
            # assert (loss5 >= 0).all(), "loss5 sign error!"
            if (loss5 < -1e-6).any() and epoch == 1 and batch_idx == 0:
                print("info: some -log p(x) values are < 0; this is expected when p(x) > 1")

            fusion_loss = scores.var(unbiased=False)
            loss = (
                self.cfg.get("w2", 1.0) * loss2_norm.mean() +
                self.cfg.get("w3", 1.0) * loss3_ce.mean() +
                self.cfg.get("w5", 1.0) * loss5_norm.mean() +
                self.w_fusion * fusion_loss
            )

            if epoch %5==1 and batch_idx in [0,1,2,3,4]:
                self.sanity_check(feats, labels)

            loss.backward()
            print('b3 grad norm:', sum(p.grad.norm().item() for p in self.b3.parameters() if p.grad is not None))
            self.optimizer.step()

            train_loss += float(loss)
            train_recon_loss += float(recon_loss)
            train_recon_loss_source += float(recon_loss_source)
            train_recon_loss_target += float(recon_loss_target)
            y_pred.extend(scores.detach().cpu().numpy().tolist())

            if batch_idx % self.cfg.get("log_interval", 100) == 0:
                print(
                    f"Train Epoch: {epoch} [{batch_idx * len(feats)}/{len(self.train_loader.dataset)}]\t"
                    f"Loss: {loss.item():.6f}"
                )
                print(
                    f"  loss2: {loss2.mean().item():.3f}"
                    f"  loss3_ce: {loss3_ce.mean().item():.3f}"
                    f"  loss5: {loss5.mean().item():.3f}"
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

                loss2, score3, loss5, scores, loss3_ce = self.forward(feats, labels)
                loss2_norm = loss2 / (self.mu2 + 1e-6)
                loss5_norm = loss5 / (self.mu5 + 1e-6)
                # assert (loss5 >= 0).all(), "loss5 sign error!"
                if (loss5 < -1e-6).any() and epoch == 1 and batch_idx == 0:
                    print("info: some -log p(x) values are < 0; this is expected when p(x) > 1")

                fusion_loss = scores.var(unbiased=False)
                loss = (
                    self.cfg.get("w2", 1.0) * loss2_norm.mean() +
                    self.cfg.get("w3", 1.0) * loss3_ce.mean() +
                    self.cfg.get("w5", 1.0) * loss5_norm.mean() +
                    self.w_fusion * fusion_loss
                )
                val_loss += float(loss)
                y_pred.extend(scores.detach().cpu().numpy().tolist())

        avg_train = train_loss / len(self.train_loader)
        avg_val = val_loss / len(self.valid_loader)
        avg_recon = train_recon_loss / len(self.train_loader)
        avg_recon_source = train_recon_loss_source / len(self.train_loader)
        avg_recon_target = train_recon_loss_target / len(self.train_loader)

        print(
            f"====> Epoch: {epoch} Average loss: {avg_train:.4f} Validation loss: {avg_val:.4f}"
        )

        # log CSV
        with open(self.log_path, "a") as log:
                        np.savetxt(log, [f"{avg_train},{avg_val},{avg_recon},{avg_recon_source},{avg_recon_target}"], fmt="%s")

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
        
    def test(self):
        """Evaluate the model on the test set."""
        device = self.cfg.get("device", "cpu")

        # Put all modules in eval mode
        self.b1.eval()
        self.b2.eval()
        self.b3.eval()
        self.b5.eval()
        self.b_attr.eval()
        self.fusion.eval()

        # Load checkpoint if present
        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path, map_location=device)
            self.b1.load_state_dict(checkpoint.get("b1", {}))
            self.b2.load_state_dict(checkpoint.get("b2", {}))
            self.b3.load_state_dict(checkpoint.get("b3", {}))
            self.b5.load_state_dict(checkpoint.get("b5", {}))
            self.b_attr.load_state_dict(checkpoint.get("b_attr", {}))
            self.fusion.load_state_dict(checkpoint.get("fusion", {}))

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
            # ----- Meta-Learner adaptation on support shots -----
            test_batches = list(test_loader)
            support_batches = test_batches[: self.maml_shots]
            learner = self.meta_learner.maml.clone()
            for batch in support_batches:
                feats = batch[0].to(device).float()
                labels = torch.argmax(batch[2], dim=1).long().to(device)
                _, _, _, sc_tmp, _ = self.forward(feats, labels, fusion_module=learner.module)
                learner.adapt(sc_tmp.mean())
            adapted_fusion = learner.module if support_batches else self.fusion

            print("\n============== BEGIN TEST FOR A SECTION ==============")
            with torch.no_grad():
                for batch in test_batches:
                    feats = batch[0].to(device).float()
                    labels = torch.argmax(batch[2], dim=1).long().to(device)

                    _, _, _, scores, _ = self.forward(feats, labels, fusion_module=adapted_fusion)
                    score = scores.mean().item()

                    basename = batch[3][0]
                    y_true.append(batch[1][0].item())
                    y_pred.append(score)

                    anomaly_score_list.append([basename, score])
                    decision_result_list.append([basename, 1 if score > decision_threshold else 0])
                    if mode:
                        domain_list.append("target" if re.search(r"target", basename, re.IGNORECASE) else "source")

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
                prec = tp / np.maximum(tp + fp, sys.float_info.epsilon)
                recall = tp / np.maximum(tp + fn, sys.float_info.epsilon)
                f1 = 2.0 * prec * recall / np.maximum(prec + recall, sys.float_info.epsilon)

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

            print("\n============ END OF TEST FOR A SECTION ============")

        if mode:
            mean_perf = np.mean(np.array(performance, dtype=float), axis=0)
            csv_lines.append(["arithmetic mean"] + list(mean_perf))
            hmean_perf = scipy.stats.hmean(np.maximum(np.array(performance, dtype=float), sys.float_info.epsilon), axis=0)
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
            
    
    

