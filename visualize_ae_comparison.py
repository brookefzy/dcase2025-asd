import torch
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import pickle
import os
from models.branch_transformer_ae import BranchTransformerAE

def load_feat(path):
    if path.endswith('.npz'):
        return np.load(path)['feat']
    elif path.endswith('.pkl') or path.endswith('.pickle'):
        with open(path, 'rb') as f:
            obj = pickle.load(f)
            if isinstance(obj, dict) and 'feat' in obj:
                return obj['feat']
            raise KeyError("pickle missing 'feat'")
    else:
        raise ValueError("unsupported ext")

def load_b2(model_ckpt, latent_dim, cfg):
    model = BranchTransformerAE(latent_dim, cfg)
    ckpt = torch.load(model_ckpt, map_location='cpu')
    model.load_state_dict(ckpt.get('b2', {}))
    model.eval()
    return model

def run_recon(model, feat):
    with torch.no_grad():
        tensor = torch.tensor(feat).unsqueeze(0).unsqueeze(0).float()  # (1,1,n_mels,frames)
        recon, _ = model(tensor)
    return recon.squeeze().numpy()

def plot_grid(inputs, recons, titles, cfg, save='ae_comparison.png'):
    n_cols = len(inputs)
    fig, axes = plt.subplots(3, n_cols, figsize=(4*n_cols, 8))

    for i in range(n_cols):
        in_spec = inputs[i]
        rec_spec = recons[i]
        residual = in_spec - rec_spec

        librosa.display.specshow(in_spec, sr=cfg['sr'], hop_length=cfg['hop'], ax=axes[0,i], y_axis='mel', x_axis='time')
        axes[0,i].set_title(titles[i] + ' - Input')
        librosa.display.specshow(rec_spec, sr=cfg['sr'], hop_length=cfg['hop'], ax=axes[1,i], y_axis='mel', x_axis='time')
        axes[1,i].set_title('Reconstruction')
        librosa.display.specshow(residual, sr=cfg['sr'], hop_length=cfg['hop'], ax=axes[2,i], y_axis='mel', x_axis='time', cmap='coolwarm')
        axes[2,i].set_title('Residual')

    plt.tight_layout()
    plt.savefig(save, dpi=150)
    print('Saved grid to', save)

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--model_ckpt', required=True, help='full model checkpoint containing b2')
    p.add_argument('--source_normal', required=True)
    p.add_argument('--source_anomaly', required=True)
    p.add_argument('--target_normal', required=True)
    p.add_argument('--target_anomaly', required=True)
    p.add_argument('--latent_dim', type=int, default=128)
    p.add_argument('--save', default='ae_comparison.png')
    p.add_argument('--sr', type=int, default=22050)
    p.add_argument('--hop', type=int, default=512)
    args = p.parse_args()

    cfg = {'latent_dim': args.latent_dim, 'n_mels': None, 'sr': args.sr, 'hop_length': args.hop}

    # Load features
    paths = [args.source_normal, args.source_anomaly, args.target_normal, args.target_anomaly]
    titles = ['Src-Norm', 'Src-Anom', 'Tgt-Norm', 'Tgt-Anom']
    feats = [load_feat(pth) for pth in paths]
    cfg['n_mels'] = feats[0].shape[0]

    model = load_b2(args.model_ckpt, args.latent_dim, cfg)

    recons = [run_recon(model, f) for f in feats]

    plot_grid(feats, recons, titles, cfg, args.save)


"""
# AGENT: all paths are here: /lustre1/g/geog_pyloo/11_octa/dcase2025-asd/data/dcase2025t2/dev_data/processed/ToyCar/test/mels128_fft1024_hop512/section_00_test_mix_TF64-8_mel1024-512.pickle
# we need to get the normal, target normal, source anomaly, target anomaly features from this pickle file

# bash
python visualize_ae_comparison.py \
    --model_ckpt /lustre1/g/geog_pyloo/11_octa/dcase2025-asd/models/checkpoint/multi_branch/DCASE2025MultiBranch_DCASE2025T2ToyCar_id(0_)_seed13711/checkpoint.tar \
    --source_normal /path/to/source/normal/feature.npz \
    --source_anomaly /path/to/source/anomaly/feature.npz \
    --target_normal /path/to/target/normal/feature.npz \
    --target_anomaly /path/to/target/anomaly/feature.npz \
    --save ae_comparison.png
"""