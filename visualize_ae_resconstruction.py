
import torch
import numpy as np
import matplotlib.pyplot as plt
from models.branch_transformer_ae import BranchTransformerAE
import librosa.display
import os
import pickle

def load_feature(file_path):
    if file_path.endswith('.npz'):
        data = np.load(file_path)
        return data['feat']
    elif file_path.endswith('.pickle') or file_path.endswith('.pkl'):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            if isinstance(data, dict) and 'feat' in data:
                return data['feat']
            else:
                raise KeyError("Expected a dict with 'feat' key in the pickle file.")
    else:
        raise ValueError("Unsupported file format. Use .npz or .pkl.")

def visualize_ae_reconstruction(model_ckpt, audio_tensor, cfg, save_path="ae_recon_vis.png"):
    model = BranchTransformerAE(cfg["latent_dim"], cfg)

    if os.path.exists(model_ckpt):
        checkpoint = torch.load(model_ckpt, map_location="cpu")
        model.load_state_dict(checkpoint.get("b2", {}))
    else:
        raise FileNotFoundError(f"Checkpoint file not found: {model_ckpt}")

    model.eval()

    with torch.no_grad():
        recon, _ = model(audio_tensor.unsqueeze(0))  # shape: (1, 1, n_mels, frames)

    input_np = audio_tensor.squeeze().cpu().numpy()
    recon_np = recon.squeeze().cpu().numpy()
    residual_np = input_np - recon_np

    fig, axes = plt.subplots(3, 1, figsize=(10, 8))

    librosa.display.specshow(input_np, sr=cfg["sr"], hop_length=cfg["hop_length"],
                             x_axis="time", y_axis="mel", ax=axes[0])
    axes[0].set_title("Input Log-Mel Spectrogram")

    librosa.display.specshow(recon_np, sr=cfg["sr"], hop_length=cfg["hop_length"],
                             x_axis="time", y_axis="mel", ax=axes[1])
    axes[1].set_title("Reconstructed Spectrogram")

    librosa.display.specshow(residual_np, sr=cfg["sr"], hop_length=cfg["hop_length"],
                             x_axis="time", y_axis="mel", ax=axes[2], cmap='coolwarm')
    axes[2].set_title("Residual (Input - Reconstructed)")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved visualization to {save_path}")

# Example usage:
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_ckpt", 
                        default = "/lustre1/g/geog_pyloo/11_octa/dcase2025-asd/checkpoints/checkpoint_last.pth",
                        # required=True, 
                        help="Path to full model checkpoint (.pth)")
    parser.add_argument("--input_file", 
                        default = "./data/dcase2025t2/dev_data/processed/ToyCar/train/mels128_fft1024_hop512/section_00_train+supplemental_mix_TF16-4_mel1024-512.pickle",
                        help="Path to .npz with 'feat' key (log-mel tensor)")
    parser.add_argument("--save", default="ae_recon_vis.png", help="Output image file path")
    args = parser.parse_args()

    # Load feature
    feat = load_feature(args.input_file)
    tensor_input = torch.tensor(feat).unsqueeze(0).float()  # shape: (1, n_mels, frames)

    # Minimal config
    cfg = {
        "latent_dim": 128,
        "n_mels": feat.shape[0],
        "sr": 22050,
        "hop_length": 512
    }

    visualize_ae_reconstruction(args.model_ckpt, tensor_input, cfg, args.save)


#python visualize_ae_reconstruction.py --model_ckpt your_model.pth --npz_file your_feat.npz
