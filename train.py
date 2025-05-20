#!/usr/bin/env python3
# %%
import argparse
import os
import yaml
import torch
import pandas as pd
import gc
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from torch.nn.functional import adaptive_avg_pool2d
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

from datasets.dataset_spec import SpectrogramDataset
from datasets.loader_common import select_dirs, get_machine_type_dict
from models.branch_pretrained    import BranchPretrained
from models.branch_transformer_ae import BranchTransformerAE
from models.branch_contrastive   import BranchContrastive
# from models.branch_diffusion     import BranchDiffusion  # unused
from models.branch_flow          import BranchFlow
from models.fusion_attention     import FusionAttention

# result columns for metrics CSV
result_column_dict = {
    "single_domain": [
        "section", "AUC", "pAUC", "precision", "recall", "F1 score"
    ],
    "source_target": [
        "section",
        "AUC (source)", "AUC (target)",
        "pAUC",
        "pAUC (source)", "pAUC (target)",
        "precision (source)", "precision (target)",
        "recall (source)", "recall (target)",
        "F1 score (source)", "F1 score (target)"
    ]
}


def save_checkpoint(model, optimizer, epoch, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch':       epoch,
        'model_state': model.state_dict(),
        'optim_state': optimizer.state_dict()
    }, path)


def evaluate(fusion_model, loader, device):
    """
    Evaluate the fusion_model on data in loader and return per-section metric dicts.
    """
    fusion_model.eval()
    secs, scores, labels = [], [], []

    with torch.no_grad():
        for feats, labs, sections in loader:
            x    = feats.to(device).squeeze().unsqueeze(1)    # [B,1,H,W]
            labs = labs.to(device)

            # ── Branch 2 per-sample MSE ────────────────────────────
            recon2, z2 = b2(x)
            feats_ds   = adaptive_avg_pool2d(x, (cfg['n_mels'], recon2.shape[-1]))
            # get a [B] vector of MSE per sample
            per_pixel2 = (recon2 - feats_ds).pow(2)
            loss2 = per_pixel2.reshape(per_pixel2.size(0), -1).mean(dim=1)

            # ── Branch 3: assume this now returns per-sample anomaly score as a [B] tensor
            z3, loss3 = b3(x, labs)     # make sure loss3 is shape [B]

            # ── Branch 5: flow gives you a [B] tensor already
            z1   = b1(x)
            z_cat= torch.cat([z1, z2, z3], dim=1)
            loss5 = b5(z_cat)           # [B]

            # ── Stack into [B×3] ───────────────────────────────────
            anomaly_vector = torch.stack([loss2, loss3, loss5], dim=1)  # now valid

            scores_batch = fusion(anomaly_vector)  # [B] or [B×1], depending on fusion

            # collect
            scores.extend(scores_batch.cpu().tolist())
            labels.extend(labs.cpu().tolist())
            secs.extend(sections)

    # per-section metrics
    df      = pd.DataFrame({'section': secs, 'score': scores, 'label': labels})
    results = []
    for sec, grp in df.groupby('section'):
        y_true  = grp['label'].values
        y_score = grp['score'].values

        auc_val   = roc_auc_score(y_true, y_score)
        p_auc_val = roc_auc_score(y_true, y_score, max_fpr=0.1)
        preds     = (y_score >= 0.5).astype(int)

        results.append({
            'section':   sec,
            'AUC':       auc_val,
            'pAUC':      p_auc_val,
            'precision': precision_score(y_true, preds, zero_division=0),
            'recall':    recall_score(y_true, preds, zero_division=0),
            'F1 score':  f1_score(y_true, preds, zero_division=0)
        })

    return results


class WrappedSpecDS(Dataset):
    """Wrap SpectrogramDataset to attach binary label and section"""
    def __init__(self, ds, is_train: bool):
        self.ds    = ds
        self.train = is_train

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        spec, fname, *rest = self.ds[idx]
        lbl = 0 if self.train else int("_anomaly_" in fname)
        sec = fname.split("_")[1]
        return spec, lbl, sec


def pad_collate(batch):
    specs, labels, secs = zip(*batch)
    max_W = max(s.shape[-1] for s in specs)
    padded = [F.pad(s, (0, max_W - s.shape[-1])) for s in specs]
    specs_tensor = torch.stack(padded, dim=0)
    labels_tensor = torch.tensor(labels)
    return specs_tensor, labels_tensor, list(secs)


def main():
    # parse arguments
    parser = argparse.ArgumentParser(description="Train and evaluate ASD model")
    parser.add_argument('--mode', choices=['dev', 'test'], default='dev',
                        help="'dev' for dev-set evaluation, 'test' for test-set evaluation")
    parser.add_argument('--config', type=str, default='config.yaml',
                        help="path to experiment config YAML")
    parser.add_argument('--baseline-config', type=str, default='baseline.yaml',
                        help="path to baseline config YAML")
    args = parser.parse_args()

    mode            = args.mode
    config_path     = args.config
    baseline_config = args.baseline_config
    device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load configs
    global cfg
    cfg = yaml.safe_load(open(config_path))
    param = yaml.safe_load(open(baseline_config))

    # data roots
    train_root = cfg['dev_data_root']
    eval_root  = cfg['dev_data_root'] if mode == 'dev' else cfg['eval_data_root']

    # helper dicts (sections selection)
    name       = 'DCASE2025T2'
    param["dev_directory"] = train_root
    base_dirs  = select_dirs(param, mode=(mode=='dev'))
    mt_dict    = get_machine_type_dict(name, mode=(mode=='dev'))

    # build datasets
    train_dsets, eval_dsets = [], []
    for mt, sect_info in mt_dict['machine_type'].items():
        for sec in sect_info['dev']:
            ds_train_raw = SpectrogramDataset(base_dir=train_root,
                                              machine_type=mt,
                                              section=sec,
                                              mode='train',
                                              config=cfg)
            ds_sup_raw   = SpectrogramDataset(base_dir=train_root,
                                              machine_type=mt,
                                              section=sec,
                                              mode='supplemental',
                                              config=cfg)
            ds_test_raw  = SpectrogramDataset(base_dir=eval_root,
                                              machine_type=mt,
                                              section=sec,
                                              mode='test',
                                              config=cfg)

            if len(ds_train_raw):
                train_dsets.append(WrappedSpecDS(ds_train_raw, is_train=True))
            if len(ds_sup_raw) and mode=='dev':
                train_dsets.append(WrappedSpecDS(ds_sup_raw,   is_train=True))
            if len(ds_test_raw):
                eval_dsets.append(WrappedSpecDS(ds_test_raw,  is_train=False))

    full_train_ds = ConcatDataset(train_dsets)
    full_eval_ds  = ConcatDataset(eval_dsets)
    # check the total number of samples
    print(f"Train dataset size: {len(full_train_ds)}")
    print(f"Eval dataset size: {len(full_eval_ds)}")

    train_loader = DataLoader(full_train_ds,
                              batch_size=cfg['batch_size'],
                              shuffle=True,
                              num_workers=cfg.get('num_workers', 4),
                              collate_fn=pad_collate)

    eval_loader = DataLoader(full_eval_ds,
                             batch_size=cfg['batch_size'],
                             shuffle=False,
                             num_workers=cfg.get('num_workers', 4),
                             collate_fn=pad_collate)

    # instantiate model branches and fusion
    global b1, b2, b3, b5, fusion
    b1     = BranchPretrained(cfg['ast_model'], cfg).to(device)
    b2     = BranchTransformerAE(cfg['latent_dim'], cfg).to(device)
    b3     = BranchContrastive(cfg['latent_dim'], cfg).to(device)
    b5     = BranchFlow(cfg['flow_dim']).to(device)
    fusion = FusionAttention(num_branches=3).to(device)

    print("AE pos-emb len:", b2.encoder.embeddings.position_embeddings.shape[1])

    optimizer = optim.Adam(
        list(b1.parameters()) +
        list(b2.parameters()) +
        list(b3.parameters()) +
        list(b5.parameters()) +
        list(fusion.parameters()),
        lr=float(cfg['lr'])
    )

    os.makedirs(cfg['save_dir'], exist_ok=True)
    metrics_csv = os.path.join(cfg['save_dir'], 'metrics_all_epochs.csv')

    # training + evaluation
    best_auc = 0.0
    for epoch in range(1, cfg['epochs']+1):
        # train
        b1.train(); b2.train(); b3.train(); b5.train(); fusion.train()
        total_loss = 0.0

        for feats, labels, _ in train_loader:
            feats = feats.squeeze(1).unsqueeze(1).to(device)
            labels = labels.to(device)

            z1 = b1(feats)
            recon2, z2 = b2(feats)
            feats_ds = adaptive_avg_pool2d(feats, (cfg['n_mels'], recon2.shape[-1]))
            loss2 = F.mse_loss(recon2, feats_ds)
            z3, loss3 = b3(feats, labels)
            z_cat = torch.cat([z1, z2, z3], dim=1)
            loss5 = b5(z_cat)

            total_branch_loss = (cfg['w2']*loss2 + cfg['w3']*loss3 + cfg['w5']*loss5)
            total_branch_loss = total_branch_loss.mean()

            optimizer.zero_grad()
            total_branch_loss.backward()
            optimizer.step()

            total_loss += total_branch_loss.item()

        # evaluate on dev or test set
        epoch_results = evaluate(fusion, eval_loader, device)
        epoch_auc     = sum(r['AUC'] for r in epoch_results) / len(epoch_results)

        # save
        save_checkpoint(fusion, optimizer, epoch,
                        os.path.join(cfg['save_dir'], 'checkpoint_last.pth'))
        if epoch_auc > best_auc:
            best_auc = epoch_auc
            save_checkpoint(fusion, optimizer, epoch,
                            os.path.join(cfg['save_dir'], 'checkpoint_best.pth'))

        # dump metrics
        df = pd.DataFrame(epoch_results)[ result_column_dict['single_domain'] ]
        df['epoch'] = epoch
        if epoch == 1:
            df.to_csv(metrics_csv, index=False)
        else:
            df.to_csv(metrics_csv, mode='a', header=False, index=False)

        print(f"Epoch {epoch}/{cfg['epochs']} ({mode}) — TrainLoss: {total_loss/len(train_loader):.4f} — Eval AUC: {epoch_auc:.4f}")

    print("Training complete.")


if __name__ == '__main__':
    main()
