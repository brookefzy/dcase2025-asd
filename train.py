#!/usr/bin/env python3
# %%
import argparse
import os
import yaml
import torch
import pandas as pd
import gc
from torch.utils.data import DataLoader, ConcatDataset, Dataset, random_split
from torch.nn.functional import adaptive_avg_pool2d
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from collections import defaultdict

from datasets.dataset_spec import SpectrogramDataset
from datasets.loader_common import select_dirs, get_machine_type_dict
from models.branch_pretrained import BranchPretrained
from models.branch_transformer_ae import BranchTransformerAE
from models.branch_contrastive import BranchContrastive
from models.branch_attr import BranchAttrs

# from models.branch_diffusion     import BranchDiffusion  # unused
from models.branch_flow import BranchFlow
from models.fusion_attention import FusionAttention
import pandas as pd
import csv


def load_attributes(root: str, machine_type: str, section: str):
    """
    Returns a dict fname → [d1v, d2v, …] for this section.
    If noAttributes, returns zeros.
    """
    csv_path = os.path.join(
        root,
        machine_type,
        "attributes_{}.csv".format(section),
    )
    mapping = {}
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            fname = row[0]
            # d1p=row[1], d1v=row[2], d2p=row[3], d2v=row[4], ...
            # skip param names, just take values at even indices
            vals = []
            for i in range(2, len(row), 2):
                v = row[i]
                vals.append(float(v) if v not in ("", "noAttributes") else 0.0)
            mapping[fname] = vals
    # figure out fixed length, pad shorter lists
    max_len = max(len(v) for v in mapping.values())
    for k, v in mapping.items():
        if len(v) < max_len:
            mapping[k] = v + [0.0] * (max_len - len(v))
    return mapping


# result columns for metrics CSV
result_column_dict = {
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


def evaluate(fusion_model, loader, device):
    """
    Single-domain evaluation: unchanged from before.
    """
    fusion_model.eval()
    mts_list, secs, scores, labels = [], [], [], []

    with torch.no_grad():
        for feats, labs, fnames, mts, attrs in loader:
            x = feats.to(device).squeeze().unsqueeze(1)  # [B,1,H,W]
            labs = labs.to(device)
            attrs = attrs.to(device)

            # ── Branch 2 per-sample MSE ────────────────────────────
            recon2, z2 = b2(x)
            feats_ds = adaptive_avg_pool2d(x, (cfg["n_mels"], recon2.shape[-1]))
            # get a [B] vector of MSE per sample
            per_pixel2 = (recon2 - feats_ds).pow(2)
            loss2 = per_pixel2.reshape(per_pixel2.size(0), -1).mean(dim=1)

            # ── Branch 3: assume this now returns per-sample anomaly score as a [B] tensor
            z3, loss3 = b3(x, labs)  # make sure loss3 is shape [B]

            # ── Branch 5: flow gives you a [B] tensor already
            z1 = b1(x)
            z5 = b5(torch.cat([z1, z2, z3], 1))
            z_attr = b_attr(attrs)
            z_cat = torch.cat([z1, z2, z3, z5, z_attr], dim=1)
            loss5 = b5(z_cat)  # [B]

            # ── Stack into [B×3] ───────────────────────────────────
            anomaly_vector = torch.stack([loss2, loss3, loss5], dim=1)

            scores_batch = fusion(anomaly_vector)  # [B] or [B×1], depending on fusion

            # collect
            scores.extend(scores_batch.cpu().tolist())
            labels.extend(labs.cpu().tolist())
            secs.extend([fname.split("_")[1] for fname in fnames])
            mts_list.extend(mts)

    # per-section metrics
    df = pd.DataFrame(
        {"machine_type": mts_list, "section": secs, "score": scores, "label": labels}
    )
    results = []
    for sec, grp in df.groupby("section"):
        y_true = grp["label"].values
        y_score = grp["score"].values

        auc_val = roc_auc_score(y_true, y_score)
        p_auc_val = roc_auc_score(y_true, y_score, max_fpr=0.1)
        preds = (y_score >= 0.5).astype(int)

        results.append(
            {
                "section": sec,
                "AUC": auc_val,
                "pAUC": p_auc_val,
                "precision": precision_score(y_true, preds, zero_division=0),
                "recall": recall_score(y_true, preds, zero_division=0),
                "F1 score": f1_score(y_true, preds, zero_division=0),
            }
        )

    return results


def evaluate_source_target(fusion_model, loader, device):
    """
    Evaluate on mixed source/target loader. Returns a list of dicts,
    one per section, with separate source/target metrics.
    """
    fusion_model.eval()
    records = []

    # collect everything into a flat list
    with torch.no_grad():
        for feats, labs, fnames, mts in loader:
            x = feats.to(device).squeeze().unsqueeze(1)
            labs = labs.to(device)

            # get per-branch losses exactly as in evaluate()
            recon2, z2 = b2(x)
            feats_ds = adaptive_avg_pool2d(x, (cfg["n_mels"], recon2.shape[-1]))
            loss2 = (recon2 - feats_ds).pow(2).flatten(1).mean(1)
            z3, loss3 = b3(x, labs)
            z1 = b1(x)
            zcat = torch.cat([z1, z2, z3], dim=1)
            loss5 = b5(zcat)

            scores = fusion(torch.stack([loss2, loss3, loss5], dim=1))
            # expand back to Python lists
            for fname, l, s, mt in zip(
                fnames, labs.cpu().tolist(), scores.cpu().tolist(), mts
            ):
                # domain detection: assume supplemental filenames had "_anomaly_" only in test,
                # and supplemental training data were from target normal; so here we look at loader.dataset
                # instead we'll assume your filenames embed "target" when appropriate:
                sec = fname.split("_")[1]  # e.g. "00"
                dom = fname.split("_")[2]
                records.append((mt, sec, dom, l, s))

    # build a DataFrame
    df = pd.DataFrame(
        records, columns=["machine_type", "section", "domain", "label", "score"]
    )
    results = []
    for (mt, sec), grp in df.groupby(["machine_type", "section"]):
        out = {"machine_type": mt, "section": sec}
        # overall pAUC
        y, sc = grp["label"], grp["score"]
        out["pAUC"] = roc_auc_score(y, sc, max_fpr=0.1)

        for dom in ("source", "target"):
            sub = grp[grp["domain"] == dom]
            if len(sub):
                y_s, sc_s = sub["label"], sub["score"]
                out[f"AUC ({dom})"] = roc_auc_score(y_s, sc_s)
                out[f"pAUC ({dom})"] = roc_auc_score(y_s, sc_s, max_fpr=0.1)
                preds = (sc_s >= 0.5).astype(int)
                out[f"precision ({dom})"] = precision_score(y_s, preds, zero_division=0)
                out[f"recall ({dom})"] = recall_score(y_s, preds, zero_division=0)
                out[f"F1 score ({dom})"] = f1_score(y_s, preds, zero_division=0)
            else:
                # no target / source in this section: fill zeros
                for k in ("AUC", "pAUC", "precision", "recall", "F1 score"):
                    out[f"{k} ({dom})"] = 0.0

        results.append(out)
    return results


class WrappedSpecDS(Dataset):
    """Wrap SpectrogramDataset to attach binary label and section"""

    def __init__(self, ds, is_train: bool, machine_type: str, root: str, section: str):
        self.ds = ds
        self.train = is_train
        self.machine_type = machine_type
        self.attr_map = load_attributes(root, machine_type, section)
        self.attr_len = len(next(iter(self.attr_map.values())))

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        # keep the raw filename so we can parse section & domain later
        spec, fname, *rest = self.ds[idx]
        lbl = 0 if self.train else int("_anomaly_" in fname)
        # lookup attribute vals (or zeros)
        attrs = self.attr_map.get(fname, [0.0] * self.attr_len)
        # convert to torch tensor
        attr_tensor = torch.tensor(attrs, dtype=torch.float32)
        return spec, lbl, fname, self.machine_type, attr_tensor


def pad_collate(batch):
    specs, labels, fnames, mts, attrs = zip(*batch)
    max_W = max(s.shape[-1] for s in specs)
    padded = [F.pad(s, (0, max_W - s.shape[-1])) for s in specs]
    specs_tensor = torch.stack(padded, dim=0)
    labels_tensor = torch.tensor(labels)
    attrs_tensor = torch.stack(attrs, dim=0)  # [B, attr_len]
    return specs_tensor, labels_tensor, list(fnames), list(mts), attrs_tensor


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate ASD model")
    parser.add_argument(
        "--mode",
        choices=["dev", "test"],
        default="dev",
        help="'dev' for development (split/train only), 'test' for full evaluation",
    )
    parser.add_argument(
        "--train_mode",
        choices=["all", "per_machine"],
        default="all",
        help="'all' to train on all machine types together; 'per_machine' to train separately per machine type",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="Fraction of the dataset to reserve for validation when splitting",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="path to experiment config YAML",
    )
    parser.add_argument(
        "--baseline-config",
        type=str,
        default="baseline.yaml",
        help="path to baseline config YAML",
    )
    parser.add_argument(
        "--eval-type",
        choices=["single_domain", "source_target"],
        default="single_domain",
        help="metric type for evaluation",
    )
    args = parser.parse_args()

    # Load configs
    cfg = yaml.safe_load(open(args.config))
    param = yaml.safe_load(open(args.baseline_config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare directories and metadata
    train_root = cfg["dev_data_root"]
    eval_root = cfg["eval_data_root"]
    base_dirs = select_dirs(param, mode=(args.mode == "dev"))
    mt_dict = get_machine_type_dict("DCASE2025T2", mode=(args.mode == "dev"))

    # Main training routine
    if args.train_mode == "all":
        # Build full dataset for all machines
        train_dsets, eval_dsets = [], []
        for mt, sect_info in mt_dict["machine_type"].items():
            for sec in sect_info["dev"]:
                # Training and supplemental datasets
                ds_train_raw = SpectrogramDataset(
                    train_root, mt, sec, mode="train", config=cfg
                )
                ds_sup_raw = (
                    SpectrogramDataset(
                        train_root, mt, sec, mode="supplemental", config=cfg
                    )
                    if args.mode == "dev"
                    else None
                )
                if len(ds_train_raw):
                    train_dsets.append(
                        WrappedSpecDS(ds_train_raw, True, mt, train_root, sec)
                    )
                if ds_sup_raw and len(ds_sup_raw):
                    train_dsets.append(
                        WrappedSpecDS(ds_sup_raw, True, mt, train_root, sec)
                    )
                # Official test dataset only for --mode test
                if args.mode == "test":
                    ds_test_raw = SpectrogramDataset(
                        eval_root, mt, sec, mode="test", config=cfg
                    )
                    if len(ds_test_raw):
                        eval_dsets.append(
                            WrappedSpecDS(ds_test_raw, False, mt, eval_root, sec)
                        )

        full_train_ds = ConcatDataset(train_dsets)
        print(f"Total examples (all machines): {len(full_train_ds)}")

        # Split training data into train & validation
        n_total = len(full_train_ds)
        n_val = int(n_total * args.val_ratio)
        n_train = n_total - n_val
        train_ds, val_ds = random_split(full_train_ds, [n_train, n_val])

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

        # Evaluation loader for official test set (mode=test)
        if args.mode == "test":
            full_eval_ds = ConcatDataset(eval_dsets)
            print(f"Total test examples: {len(full_eval_ds)}")
            test_loader = DataLoader(
                full_eval_ds,
                batch_size=cfg["batch_size"],
                shuffle=False,
                num_workers=cfg.get("num_workers", 4),
                collate_fn=pad_collate,
            )

        # Instantiate model and optimizer
        b1 = BranchPretrained(cfg["ast_model"], cfg).to(device)
        b2 = BranchTransformerAE(cfg["latent_dim"], cfg).to(device)
        b3 = BranchContrastive(cfg["latent_dim"], cfg).to(device)
        b5 = BranchFlow(cfg["flow_dim"]).to(device)
        b_attr = BranchAttrs(
            input_dim=val_ds.dataset.attr_len,
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

        # Training loop (no evaluation inside)
        for epoch in range(1, cfg["epochs"] + 1):
            b1.train()
            b2.train()
            b3.train()
            b5.train()
            fusion.train()
            b_attr.train()
            total_loss = 0.0
            for feats, labels, _fn, _mt, attrs in train_loader:
                feats = feats.squeeze(1).unsqueeze(1).to(device)
                labels = labels.to(device)
                attrs = attrs.to(device)

                z1 = b1(feats)
                recon2, z2 = b2(feats)
                feats_ds = adaptive_avg_pool2d(feats, (cfg["n_mels"], recon2.shape[-1]))
                loss2 = F.mse_loss(recon2, feats_ds)
                z3, loss3 = b3(feats, labels)
                z_cat = torch.cat([z1, z2, z3], dim=1)
                z5 = b5(z_cat)
                z_attr = b_attr(attrs)
                merged = torch.cat([z1, z2, z3, z5, z_attr], dim=1)
                loss5 = b5(merged)

                loss = (
                    cfg["w2"] * loss2 + cfg["w3"] * loss3 + cfg["w5"] * loss5
                ).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(
                f"Epoch {epoch}/{cfg['epochs']} — Train Loss: {total_loss/len(train_loader):.4f}"
            )
            # Save checkpoint each epoch
            save_checkpoint(
                fusion,
                optimizer,
                epoch,
                os.path.join(cfg["save_dir"], f"checkpoint_epoch_{epoch}.pth"),
            )

        # After training: evaluation on validation and/or test
        print("Training complete.")
        print("Evaluating on validation set...")
        val_results = evaluate(fusion, val_loader, device)
        df_val = pd.DataFrame(val_results)
        df_val.to_csv(
            os.path.join(cfg["save_dir"], "metrics_validation.csv"), index=False
        )

        if args.mode == "test":
            print("Evaluating on official test set...")
            test_results = evaluate(fusion, test_loader, device)
            df_test = pd.DataFrame(test_results)
            df_test.to_csv(
                os.path.join(cfg["save_dir"], "metrics_test.csv"), index=False
            )

    else:  # per_machine
        # Loop over each machine type separately
        for mt, sect_info in mt_dict["machine_type"].items():
            print(f"\n=== Training for machine type: {mt} ===")
            train_dsets, eval_dsets = [], []
            for sec in sect_info["dev"]:
                ds_train_raw = SpectrogramDataset(
                    train_root, mt, sec, mode="train", config=cfg
                )
                ds_sup_raw = (
                    SpectrogramDataset(
                        train_root, mt, sec, mode="supplemental", config=cfg
                    )
                    if args.mode == "dev"
                    else None
                )
                if len(ds_train_raw):
                    train_dsets.append(
                        WrappedSpecDS(ds_train_raw, True, mt, train_root, sec)
                    )
                if ds_sup_raw and len(ds_sup_raw):
                    train_dsets.append(
                        WrappedSpecDS(ds_sup_raw, True, mt, train_root, sec)
                    )

                if args.mode == "test":
                    ds_test_raw = SpectrogramDataset(
                        eval_root, mt, sec, mode="test", config=cfg
                    )
                    if len(ds_test_raw):
                        eval_dsets.append(
                            WrappedSpecDS(ds_test_raw, False, mt, eval_root, sec)
                        )

            full_train_ds = ConcatDataset(train_dsets)
            n_total = len(full_train_ds)
            n_val = int(n_total * args.val_ratio)
            n_train = n_total - n_val
            train_ds, val_ds = random_split(full_train_ds, [n_train, n_val])

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

            if args.mode == "test":
                full_eval_ds = ConcatDataset(eval_dsets)
                test_loader = DataLoader(
                    full_eval_ds,
                    batch_size=cfg["batch_size"],
                    shuffle=False,
                    num_workers=cfg.get("num_workers", 4),
                    collate_fn=pad_collate,
                )

            # Instantiate new model for each machine type
            b1 = BranchPretrained(cfg["ast_model"], cfg).to(device)
            b2 = BranchTransformerAE(cfg["latent_dim"], cfg).to(device)
            b3 = BranchContrastive(cfg["latent_dim"], cfg).to(device)
            b5 = BranchFlow(cfg["flow_dim"]).to(device)
            b_attr = BranchAttrs(
                input_dim=val_ds.dataset.attr_len,
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

            # Training loop for this machine type
            for epoch in range(1, cfg["epochs"] + 1):
                b1.train()
                b2.train()
                b3.train()
                b5.train()
                fusion.train()
                b_attr.train()
                total_loss = 0.0
                for feats, labels, _fn, _mt, attrs in train_loader:
                    feats = feats.squeeze(1).unsqueeze(1).to(device)
                    labels = labels.to(device)
                    attrs = attrs.to(device)

                    z1 = b1(feats)
                    recon2, z2 = b2(feats)
                    feats_ds = adaptive_avg_pool2d(
                        feats, (cfg["n_mels"], recon2.shape[-1])
                    )
                    loss2 = F.mse_loss(recon2, feats_ds)
                    z3, loss3 = b3(feats, labels)
                    z_cat = torch.cat([z1, z2, z3], dim=1)
                    z5 = b5(z_cat)
                    z_attr = b_attr(attrs)
                    merged = torch.cat([z1, z2, z3, z5, z_attr], dim=1)
                    loss5 = b5(merged)

                    loss = (
                        cfg["w2"] * loss2 + cfg["w3"] * loss3 + cfg["w5"] * loss5
                    ).mean()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                print(
                    f"[Machine {mt}] Epoch {epoch}/{cfg['epochs']} — Train Loss: {total_loss/len(train_loader):.4f}"
                )
                save_path = os.path.join(
                    cfg["save_dir"], f"checkpoint_{mt}_epoch_{epoch}.pth"
                )
                save_checkpoint(fusion, optimizer, epoch, save_path)

            # After training for this machine type
            print(f"Training complete for machine {mt}.")
            print(f"Evaluating on validation for machine {mt}...")
            val_results = evaluate(fusion, val_loader, device)
            pd.DataFrame(val_results).to_csv(
                os.path.join(cfg["save_dir"], f"metrics_{mt}_validation.csv"),
                index=False,
            )

            if args.mode == "test":
                print(f"Evaluating on official test for machine {mt}...")
                test_results = evaluate(fusion, test_loader, device)
                pd.DataFrame(test_results).to_csv(
                    os.path.join(cfg["save_dir"], f"metrics_{mt}_test.csv"), index=False
                )


if __name__ == "__main__":
    main()
