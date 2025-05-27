#!/usr/bin/env python3
import argparse
import os
import yaml
import torch
import pandas as pd
import gc
from torch.utils.data import DataLoader, ConcatDataset, Dataset, random_split
from torch.nn.functional import adaptive_avg_pool2d, pad
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from collections import defaultdict
from datasets.dataset_spec import SpectrogramDataset
from datasets.loader_common import select_dirs, get_machine_type_dict
from models.branch_pretrained import BranchPretrained
from models.branch_transformer_ae import BranchTransformerAE
from models.branch_contrastive import BranchContrastive
from models.branch_flow import BranchFlow
from models.branch_attr import BranchAttrs
from models.fusion_attention import FusionAttention
import csv

# Metrics column mapping
RESULT_COLUMNS = {
    "single_domain": [
        "machine_type",
        "section",
        "AUC",
        "pAUC",
        "precision",
        "recall",
        "F1 score",
    ],
    "source_target": [
        "machine_type",
        "section",
        "AUC (source)",
        "AUC (target)",
        "pAUC",
        "pAUC (source)",
        "pAUC (target)",
        "precision (source)",
        "precision (target)",
        "recall (source)",
        "recall (target)",
        "F1 score (source)",
        "F1 score (target)",
    ],
}


def load_attributes(root: str, machine_type: str, section: str):
    print(f"Loading attributes for {machine_type} - {section}...")
    csv_path = os.path.join(root, machine_type, f"attributes_{section}.csv")
    mapping = {}
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            fname = row[0]
            vals = [
                float(v) if v not in ("", "noAttributes") else 0.0
                for i, v in enumerate(row[2:], start=2)
                if i % 2 == 0
            ]
            mapping[fname] = vals
    max_len = max(len(v) for v in mapping.values())
    for k, v in mapping.items():
        if len(v) < max_len:
            mapping[k] = v + [0.0] * (max_len - len(v))
    return mapping


def save_checkpoint(model, optimizer, epoch, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
        },
        path,
    )


def _compute_branch_scores(x, labels, attrs=None):
    z1 = b1(x)
    recon2, z2 = b2(x)
    feats_ds = adaptive_avg_pool2d(x, (cfg["n_mels"], recon2.shape[-1]))
    loss2 = (recon2 - feats_ds).pow(2).reshape(x.size(0), -1).mean(dim=1)
    z3, loss3 = b3(x, labels)
    z_flow = b5(torch.cat([z1, z2, z3], dim=1))
    if attrs is not None:
        z_attr = b_attr(attrs)
        flow_input = torch.cat([z1, z2, z3, z_flow.unsqueeze(1), z_attr], dim=1)
        loss5 = b5(flow_input)
    else:
        loss5 = z_flow
    scores = fusion(torch.stack([loss2, loss3, loss5], dim=1))
    return loss2, loss3, loss5, scores


def evaluate_single(fusion_model, loader, device):
    fusion_model.eval()
    records = []
    with torch.no_grad():
        for feats, labs, fnames, mts, attrs in loader:
            x = feats.to(device).unsqueeze(1)
            labs = labs.to(device)
            _, _, _, scores = _compute_branch_scores(x, labs, attrs.to(device))
            for fname, mt, lab, score in zip(
                fnames, mts, labs.cpu().tolist(), scores.cpu().tolist()
            ):
                records.append((mt, fname.split("_")[1], lab, score))
    df = pd.DataFrame(records, columns=["machine_type", "section", "label", "score"])
    results = []
    for sec, grp in df.groupby("section"):
        y_true, y_score = grp["label"].values, grp["score"].values
        preds = (y_score >= cfg.get("threshold", 0.5)).astype(int)
        results.append(
            {
                "machine_type": grp["machine_type"].iloc[0],
                "section": sec,
                "AUC": roc_auc_score(y_true, y_score),
                "pAUC": roc_auc_score(y_true, y_score, max_fpr=0.1),
                "precision": precision_score(y_true, preds, zero_division=0),
                "recall": recall_score(y_true, preds, zero_division=0),
                "F1 score": f1_score(y_true, preds, zero_division=0),
            }
        )
    return results


def evaluate_source_target(fusion_model, loader, device):
    fusion_model.eval()
    records = []
    with torch.no_grad():
        for feats, labs, fnames, mts, attrs in loader:
            x = feats.to(device).unsqueeze(1)
            labs = labs.to(device)
            _, _, _, scores = _compute_branch_scores(x, labs)
            for fname, mt, lab, score in zip(
                fnames, mts, labs.cpu().tolist(), scores.cpu().tolist()
            ):
                sec, dom = fname.split("_")[1], fname.split("_")[2]
                records.append((mt, sec, dom, lab, score))
    df = pd.DataFrame(
        records, columns=["machine_type", "section", "domain", "label", "score"]
    )
    results = []
    for (mt, sec), grp in df.groupby(["machine_type", "section"]):
        entry = {
            "machine_type": mt,
            "section": sec,
            "pAUC": roc_auc_score(grp["label"], grp["score"], max_fpr=0.1),
        }
        for dom in ("source", "target"):
            sub = grp[grp["domain"] == dom]
            if not sub.empty:
                y, sc = sub["label"], sub["score"]
                preds = (sc >= cfg.get("threshold", 0.5)).astype(int)
                entry[f"AUC ({dom})"] = roc_auc_score(y, sc)
                entry[f"pAUC ({dom})"] = roc_auc_score(y, sc, max_fpr=0.1)
                entry[f"precision ({dom})"] = precision_score(y, preds, zero_division=0)
                entry[f"recall ({dom})"] = recall_score(y, preds, zero_division=0)
                entry[f"F1 score ({dom})"] = f1_score(y, preds, zero_division=0)
            else:
                for k in ("AUC", "pAUC", "precision", "recall", "F1 score"):
                    entry[f"{k} ({dom})"] = 0.0
        results.append(entry)
    return results


class WrappedSpecDS(Dataset):
    def __init__(self, ds, is_train: bool, machine_type: str, root: str, section: str):
        self.ds = ds
        self.train = is_train
        self.machine_type = machine_type
        self.section = section
        self.attr_map = load_attributes(root, self.machine_type, self.section)
        self.attr_len = len(next(iter(self.attr_map.values())))

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        spec, fname, *rest = self.ds[idx]
        lbl = 0 if self.train else int("_anomaly_" in fname)
        attrs = self.attr_map.get(fname, [0.0] * self.attr_len)
        return (
            spec,
            lbl,
            fname,
            self.machine_type,
            torch.tensor(attrs, dtype=torch.float32),
        )


def pad_collate(batch):
    specs, labels, fnames, mts, attrs = zip(*batch)
    max_W = max(s.shape[-1] for s in specs)
    padded = [F.pad(s, (0, max_W - s.shape[-1])) for s in specs]
    return (
        torch.stack(padded),
        torch.tensor(labels),
        list(fnames),
        list(mts),
        torch.stack(attrs),
    )


def main():
    parser = argparse.ArgumentParser("Train and evaluate ASD model")
    parser.add_argument("--mode", choices=["dev", "test"], default="dev")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--baseline-config", type=str, default="baseline.yaml")
    parser.add_argument(
        "--eval-type",
        choices=["single_domain", "source_target"],
        default="single_domain",
    )
    args = parser.parse_args()

    mode, eval_type = args.mode, args.eval_type
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    global cfg
    cfg = yaml.safe_load(open(args.config))
    param = yaml.safe_load(open(args.baseline_config))

    train_root = cfg["dev_data_root"]
    eval_root = cfg["dev_data_root"] if mode == "dev" else cfg["eval_data_root"]

    name = "DCASE2025T2"
    param["dev_directory"] = train_root

    # Build list of per-machine-section datasets
    train_dsets, eval_dsets = [], []
    machine_type_dict = get_machine_type_dict(name, mode=(mode == "dev"))['machine_type']
    
    for mt, data_type in get_machine_type_dict(name, mode=(mode == "dev"))["machine_type"].items():
        section_list = machine_type_dict[mt][data_type]
        for sec in section_list:
            train_dsets.append(
                WrappedSpecDS(
                    SpectrogramDataset(train_root, mt, mode="train", config=cfg, section=sec),
                    True,
                    mt,
                    train_root,
                    sec,
                )
            )
            eval_dsets.append(
                WrappedSpecDS(
                    SpectrogramDataset(eval_root, mt, mode="eval", config=cfg, section=sec),
                    False,
                    mt,
                    eval_root,
                    sec,
                )
            )

    # Combine datasets and split train/validation
    full_train_ds = ConcatDataset(train_dsets)
    shared_attr_len = train_dsets[0].attr_len
    val_ratio = cfg.get("val_ratio", 0.2)
    num_train = int((1 - val_ratio) * len(full_train_ds))
    num_val = len(full_train_ds) - num_train
    train_ds, val_ds = random_split(
        full_train_ds,
        [num_train, num_val],
        generator=torch.Generator().manual_seed(cfg.get("seed", 42)),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg.get("num_workers", 4),
        collate_fn=pad_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg.get("num_workers", 4),
        collate_fn=pad_collate,
    )
    # Final evaluation loader (dev/test)
    eval_loader = DataLoader(
        ConcatDataset(eval_dsets),
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg.get("num_workers", 4),
        collate_fn=pad_collate,
    )

    # Instantiate models
    global b1, b2, b3, b5, b_attr, fusion
    b1 = BranchPretrained(cfg["ast_model"], cfg).to(device)
    b2 = BranchTransformerAE(cfg["latent_dim"], cfg).to(device)
    b3 = BranchContrastive(cfg["latent_dim"], cfg).to(device)
    b5 = BranchFlow(cfg["flow_dim"]).to(device)
    b_attr = BranchAttrs(
        input_dim=shared_attr_len,
        hidden_dim=cfg["attr_hidden"],
        latent_dim=cfg["attr_latent"],
    ).to(device)
    fusion = FusionAttention(num_branches=3).to(device)

    optimizer = optim.Adam(
        list(b1.parameters())
        + list(b2.parameters())
        + list(b3.parameters())
        + list(b5.parameters())
        + list(b_attr.parameters())
        + list(fusion.parameters()),
        lr=float(cfg["lr"]),
    )

    os.makedirs(cfg["save_dir"], exist_ok=True)
    best_val_auc = 0.0
    metrics_csv = os.path.join(cfg["save_dir"], f"metrics_val_epochs_{eval_type}.csv")

    for epoch in range(1, cfg["epochs"] + 1):
        # Training
        b1.train()
        b2.train()
        b3.train()
        b5.train()
        b_attr.train()
        fusion.train()
        for feats, labs, fnames, mts, attrs in train_loader:
            optimizer.zero_grad()
            x = feats.to(device).unsqueeze(1)
            labs = labs.to(device)
            attrs = attrs.to(device)
            loss2, loss3, loss5, _ = _compute_branch_scores(x, labs, attrs)
            total_loss = loss2.mean() + loss3.mean() + loss5.mean()
            total_loss.backward()
            optimizer.step()

        # Validation
        if eval_type == "single_domain":
            val_results = evaluate_single(fusion, val_loader, device)
        else:
            val_results = evaluate_source_target(fusion, val_loader, device)

        cols = RESULT_COLUMNS[eval_type]
        val_auc = (
            sum(r["AUC"] for r in val_results) / len(val_results)
            if eval_type == "single_domain"
            else sum((r["AUC (source)"] + r["AUC (target)"]) * 0.5 for r in val_results)
            / len(val_results)
        )

        # Checkpointing on best validation AUC
        save_checkpoint(
            fusion,
            optimizer,
            epoch,
            os.path.join(cfg["save_dir"], "checkpoint_last.pth"),
        )
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            save_checkpoint(
                fusion,
                optimizer,
                epoch,
                os.path.join(cfg["save_dir"], "checkpoint_best.pth"),
            )

        # Log validation metrics
        df_metrics = pd.DataFrame(val_results)[cols]
        df_metrics["epoch"] = epoch
        df_metrics.to_csv(
            metrics_csv, mode="a" if epoch > 1 else "w", header=epoch == 1, index=False
        )

        print(f"Epoch {epoch}/{cfg['epochs']} — Val AUC: {val_auc:.4f}")

    print(f"Training complete. Best Val AUC: {best_val_auc:.4f}")

    # Final Evaluation uses the data from eval_loader
    print("Starting final evaluation...")
    if eval_type == "single_domain":
        eval_results = evaluate_single(fusion, eval_loader, device)
    else:
        eval_results = evaluate_source_target(fusion, eval_loader, device)
    eval_df = pd.DataFrame(eval_results)[RESULT_COLUMNS[eval_type]]
    eval_df.to_csv(
        os.path.join(cfg["save_dir"], f"eval_results_{eval_type}.csv"),
        index=False,
    )
    print(f"Final evaluation results saved to {cfg['save_dir']}.")
    print(eval_df)


if __name__ == "__main__":
    main()
    gc.collect()
