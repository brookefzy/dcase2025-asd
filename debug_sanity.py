import argparse
import numpy as np
import librosa
import matplotlib.pyplot as plt
import torch

import common as com
from datasets.datasets import Datasets
from networks.models import Models


def check_raw_db_range(wav_path: str, cfg: dict) -> np.ndarray:
    """Compute mel spectrogram of ``wav_path`` and print dB range."""
    signal, sr = librosa.load(wav_path, sr=cfg.get("sr", 16000))
    mel = librosa.feature.melspectrogram(
        y=signal,
        sr=sr,
        n_fft=cfg.get("n_fft", 1024),
        hop_length=cfg.get("hop_length", 512),
        win_length=cfg.get("win_length", cfg.get("n_fft", 1024)),
        n_mels=cfg.get("n_mels", 128),
        power=cfg.get("power", 2.0),
        fmin=cfg.get("fmin", 0.0),
        fmax=cfg.get("fmax", None),
    )
    mel_db = librosa.power_to_db(mel, ref=1.0)
    print(f"raw dB range: min={mel_db.min():.2f}, max={mel_db.max():.2f}")
    return mel_db


def mse_distribution(model: torch.nn.Module, loader, device: torch.device) -> np.ndarray:
    """Return frame-level MSE for all clips in ``loader``."""
    errs = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device).float()
            attr = batch[1].to(device) if len(batch) > 1 else None
            recon, _, _ = model(x, attr_vec=attr)
            diff = (x - recon[..., : x.size(-1)]) ** 2
            frame_err = diff.mean(dim=[1, 2])  # [B, T]
            errs.append(frame_err.cpu().reshape(-1))
    return torch.cat(errs).numpy()


def main() -> None:
    param = com.yaml_load()
    parser = com.get_argparse()
    parser.add_argument("--audio", required=True, help="Path to a training wav file")
    parser.add_argument("--model_ckpt", type=str, default=None, help="Checkpoint to load")
    parser.add_argument("--hist_out", type=str, default="debug_mse_hist.png", help="Histogram output path")
    args = parser.parse_args(com.param_to_args_list(param))
    args = parser.parse_args(namespace=args)

    cfg = vars(args)

    # 1. raw dB range
    mel_db = check_raw_db_range(args.audio, cfg)

    if args.model_ckpt:
        device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
        net = Models(args.model).net(args=args, train=False, test=True)
        state = torch.load(args.model_ckpt, map_location=device)
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        net.model.load_state_dict(state, strict=False)
        net.model.to(device)

        ds = Datasets(args.dataset).data(args)
        train_loader = ds.train_loader

        train_err = mse_distribution(net.model, train_loader, device)
        plt.hist(train_err, bins=50)
        plt.xlabel("frame MSE")
        plt.savefig(args.hist_out)
        plt.close()
        print(f"mean training frame MSE: {train_err.mean():.4f}")

        # Divide vs. sum check on first clip
        F, T = mel_db.shape
        mse_div = ((mel_db - mel_db.mean()) ** 2).mean()
        mse_sum = ((mel_db - mel_db.mean()) ** 2).sum() / (F * T)
        print(f"divide mean={mse_div:.6f}  sum/(F*T)={mse_sum:.6f}")

        # Standardisation
        mu, sigma = train_err.mean(), train_err.std()
        test_z = (train_err - mu) / (sigma + 1e-12)
        print(f"z-score mean: {test_z.mean():.6f}, std: {test_z.std():.6f}")

        # Section aggregation
        section_score = test_z.mean()
        print(f"section score (mean frame z): {section_score:.6f}")
    else:
        print("No model checkpoint provided; skipping MSE checks")


if __name__ == "__main__":
    main()

