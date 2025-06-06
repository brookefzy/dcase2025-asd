
import torch
import torch.nn.functional as F
import numpy as np
import common as com
import matplotlib.pyplot as plt
from networks.models import Models
from models.branch_transformer_ae import BranchTransformerAE
import librosa.display
import os
import pickle
import re
from datasets.dcase_dcase202x_t2_loader import DCASE202XT2Loader

def load_feature(file_path):
    if file_path.endswith('.npz'):
        data = np.load(file_path)
        return data['feat']
    elif file_path.endswith('.pickle') or file_path.endswith('.pkl'):
        print(f"Loading pickle file: {file_path}")
        loader = DCASE202XT2Loader.__new__(DCASE202XT2Loader)
        loader.load_pickle(file_path)
        data = loader.data
        m = re.search(r"mels(\d+)", file_path)
        if m:
            n_mels = int(m.group(1))
        else:
            raise ValueError("Cannot parse n_mels from file name")
        m = re.search(r"TF(\d+)-", os.path.basename(file_path))
        if m:
            frames = int(m.group(1))
        else:
            frames = data.shape[1] // n_mels
        return data[0].reshape(n_mels, frames)
        
    else:
        raise ValueError("Unsupported file format. Use .npz or .pkl.")

def visualize_ae_reconstruction(model_ckpt, audio_tensor, cfg, save_path="ae_recon_vis.png"):
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    model = BranchTransformerAE(cfg["latent_dim"], cfg)

    if os.path.exists(model_ckpt):
        checkpoint = torch.load(model_ckpt, map_location=device)
        model.load_state_dict(checkpoint.get("b2", {}))
    else:
        raise FileNotFoundError(f"Checkpoint file not found: {model_ckpt}")

    model.eval()

    with torch.no_grad():
        recon, _ = model(audio_tensor.unsqueeze(0))  # → [1, 1, n_mels, T']

    # ``audio_tensor`` is [1, n_mels, T]
    input_t = audio_tensor.squeeze(0)
    recon_t = recon.squeeze(0).squeeze(0)
    if recon_t.shape[-1] != input_t.shape[-1]:
        input_t = F.adaptive_avg_pool2d(input_t.unsqueeze(0), (cfg["n_mels"], recon_t.shape[-1])).squeeze(0)

    input_np = input_t.cpu().numpy()
    recon_np = recon_t.cpu().numpy()
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
    param = com.yaml_load()
    parser = com.get_argparse()
    parser.add_argument("--model_ckpt", 
                        default = "/lustre1/g/geog_pyloo/11_octa/dcase2025-asd/models/checkpoint/multi_branch/DCASE2025MultiBranch_DCASE2025T2ToyCar_id(0_)_seed13711/checkpoint.tar",
                        # required=True, 
                        help="Path to full model checkpoint (.tar)")
    parser.add_argument("--input_file", 
                        default = "/lustre1/g/geog_pyloo/11_octa/dcase2025-asd/data/dcase2025t2/dev_data/processed/ToyCar/test/mels128_fft1024_hop512/section_00_test_mix_TF64-8_mel1024-512.pickle",
                        help="Path to .npz with 'feat' key (log-mel tensor)")
    parser.add_argument("--save", default="ae_recon_vis.png", help="Output image file path")
    args = parser.parse_args(args=com.param_to_args_list(param))
    args = parser.parse_args(namespace=args)
    args.train_only = False
    args.dev = True
    args.epochs = 20
    
    args.cuda = args.use_cuda and torch.cuda.is_available()
    args.dataset = 'DCASE2025T2ToyCar'

    # Load feature
    feat = load_feature(args.input_file)
    tensor_input = torch.tensor(feat).unsqueeze(0).float()  # shape: (1, n_mels, frames)
    
    print("MODEL: ", args.model)
    print("FRAMES: ", args.frames)
    net = Models(args.model).net(args=args, train=False, test=True)

    # Minimal config
    cfg = {
        "latent_dim": 128,
        "n_mels": feat.shape[0],
        "sr": 22050,
        "hop_length": 512
    }
    # Merge any model-specific configuration if available. ``DCASE2025MultiBranch``
    # exposes a ``cfg`` attribute while ``DCASE2023T2AE`` only keeps the parsed
    # arguments in ``args``.  To support both cases we try to update ``cfg``
    # from ``net.cfg`` and fall back to ``net.args`` when ``cfg`` does not exist.
    if hasattr(net, "cfg"):
        cfg.update(net.cfg)
    elif hasattr(net, "args"):
        cfg.update(vars(net.args))

    visualize_ae_reconstruction(args.model_ckpt, tensor_input, cfg, args.save)


#python visualize_ae_reconstruction.py --model_ckpt your_model.pth --npz_file your_feat.npz
