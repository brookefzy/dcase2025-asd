import os
import torch
import argparse
import yaml
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from datasets.loader_common import file_list_generator, file_to_vectors
from models.resnet_ae import ResNetAutoEncoder
from models.flow import NormalizingFlow
from models.student import StudentModel
from models.ensemble import EnsembleModel
from utils.metrics import compute_auc


def load_config(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def prepare_dataloader(cfg, machine, section, mode):
    """Prepare DataLoader of feature vectors using baseline utilities."""
    # Determine mode flags
    is_dev = (mode == 'dev')
    # Directory containing train/dev wav files
    target_dir = os.path.join(cfg['data_root'], machine)
    dir_name = 'train' if not is_dev else 'dev'
    # Unique section names for this machine
    # Expect cfg['sections'][machine] contains list of section strings
    unique_sections = cfg['sections'][machine]
    # Generate file list
    files, labels, _ = file_list_generator(
        target_dir, section, unique_sections, dir_name,
        mode=is_dev, train=not is_dev
    )
    # Convert each wav to feature vectors and collect
    all_vectors = []
    for wav in files:
        vecs = file_to_vectors(
            wav,
            n_mels=cfg['n_mels'],
            n_frames=cfg['n_frames'],
            n_fft=cfg['n_fft'],
            hop_length=cfg['hop_length']
        )
        if vecs.size:  # non-empty
            all_vectors.append(vecs)
    if not all_vectors:
        raise RuntimeError(f"No feature vectors for {machine}-{section} in {mode} mode.")
    data = np.vstack(all_vectors)
    tensor = torch.from_numpy(data).float()
    dataset = TensorDataset(tensor)
    return DataLoader(
        dataset,
        batch_size=cfg['batch_size'],
        shuffle=not is_dev,
        num_workers=cfg.get('num_workers', 0)
    )


def train_for_section(cfg, machine, section):
    save_dir = os.path.join(cfg['model_dir'], machine, section)
    os.makedirs(save_dir, exist_ok=True)

    # DataLoaders
    train_loader = prepare_dataloader(cfg, machine, section, mode='train')
    dev_loader   = prepare_dataloader(cfg, machine, section, mode='dev')

    # Models to device
    device = torch.device(cfg.get('device', 'cpu'))
    ae = ResNetAutoEncoder(latent_dim=cfg['latent_dim']).to(device)
    flow = NormalizingFlow(dim=cfg['latent_dim']).to(device)
    student = StudentModel(latent_dim=cfg['student_latent']).to(device)
    ensemble = EnsembleModel(ae, flow, student, w_ae=cfg['w_ae'], w_flow=cfg['w_flow'])

    # Optimizers
    opt_ae_flow = optim.Adam(list(ae.parameters()) + list(flow.parameters()), lr=cfg['lr'])
    opt_student = optim.Adam(student.parameters(), lr=cfg['lr_student'])

    best_auc = 0.0
    for epoch in range(1, cfg['epochs'] + 1):
        ae.train(); flow.train(); student.train()
        for batch in train_loader:
            x = batch[0].to(device)

            # AE forward + loss
            x_rec, z = ae(x.unsqueeze(1))  # add channel dim
            loss_ae = nn.MSELoss()(x_rec, x.unsqueeze(1))

            # Flow forward + loss
            logp = flow(z)
            loss_flow = -torch.mean(logp)

            # Joint distillation if enabled
            if cfg.get('joint_distill', False):
                with torch.no_grad():
                    scores = ensemble.ae_forward(x.unsqueeze(1))
                    target_score = cfg['w_ae'] * scores['ae'] + cfg['w_flow'] * scores['flow']
                student_score, _ = student(x.unsqueeze(1))
                loss_kd = nn.MSELoss()(student_score, target_score)
                loss = loss_ae + cfg['flow_weight'] * loss_flow + cfg['kd_lambda'] * loss_kd
            else:
                loss = loss_ae + cfg['flow_weight'] * loss_flow

            opt_ae_flow.zero_grad()
            loss.backward()
            opt_ae_flow.step()

            if cfg.get('joint_distill', False):
                opt_student.zero_grad()
                loss_kd.backward()
                opt_student.step()

        # Validation
        ae.eval(); flow.eval(); student.eval()
        all_scores, all_labels = [], []
        with torch.no_grad():
            for batch in dev_loader:
                x = batch[0].to(device)
                rec, z = ae(x.unsqueeze(1))
                ae_score = torch.mean((x.unsqueeze(1) - rec) ** 2, dim=[1,2,3]).cpu().numpy()
                flow_score = -flow(z).cpu().numpy()
                ens_score = cfg['w_ae'] * ae_score + cfg['w_flow'] * flow_score
                all_scores.extend(list(ens_score))
                all_labels.extend([0] * len(ens_score))
        auc = compute_auc(all_scores, all_labels)
        print(f"[{machine}-{section}] Epoch {epoch}: Dev AUC = {auc:.4f}")

        # Save best
        if auc > best_auc:
            best_auc = auc
            torch.save(ae.state_dict(), os.path.join(save_dir, 'ae.pth'))
            torch.save(flow.state_dict(), os.path.join(save_dir, 'flow.pth'))
            torch.save(student.state_dict(), os.path.join(save_dir, 'student.pth'))

    print(f"Finished {machine}-{section}, best Dev AUC = {best_auc:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--machines', nargs='+', required=True)
    parser.add_argument('--sections', nargs='+', required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    for m in args.machines:
        for s in args.sections:
            train_for_section(cfg, m, s)

if __name__ == '__main__':
    main()
