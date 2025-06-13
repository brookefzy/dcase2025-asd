#!/usr/bin/env python
"""Create small validation splits for debugging purposes.

This script extracts log-mel spectrograms from a directory of WAV files and
stores two numpy arrays:
    val-OK.npy  - normal clips only (never used for training)
    val-NG.npy  - defective / anomalous clips

Each array will contain roughly ``--max_frames`` total frames (default 256).
The script searches recursively for ``*.wav`` files under ``--dataset-dir`` and
splits them based on the presence of the words ``anomaly`` or ``ng`` in the
filename.  Adjust the keyword detection if your dataset uses different tags.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm


def extract_log_mel(path: Path, cfg: argparse.Namespace) -> np.ndarray:
    """Load ``path`` and return a normalised log-mel spectrogram."""
    y, sr = sf.read(path)
    if sr != cfg.sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=cfg.sr)
    spec = librosa.feature.melspectrogram(
        y=y,
        sr=cfg.sr,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        n_mels=cfg.n_mels,
        power=cfg.power,
        fmin=cfg.fmin,
        fmax=cfg.fmax,
    )
    log_spec = librosa.power_to_db(spec, ref=np.max).astype(np.float32)
    log_spec = (log_spec - log_spec.mean()) / (log_spec.std() + 1e-8)
    if log_spec.shape[1] < cfg.time_steps:
        pad = cfg.time_steps - log_spec.shape[1]
        log_spec = np.pad(log_spec, ((0, 0), (0, pad)), mode="constant")
    else:
        log_spec = log_spec[:, : cfg.time_steps]
    return log_spec


def gather_split(files: list[Path], normal: bool, cfg: argparse.Namespace) -> np.ndarray:
    """Collect features from ``files`` until ``cfg.max_frames`` is reached."""
    feats = []
    frames = 0
    for wav in tqdm(files, desc="val-OK" if normal else "val-NG"):
        is_anomaly = "anomaly" in wav.name.lower() or "ng" in wav.name.lower()
        if normal and is_anomaly:
            continue
        if not normal and not is_anomaly:
            continue
        feat = extract_log_mel(wav, cfg)
        feats.append(feat)
        frames += feat.shape[1]
        if frames >= cfg.max_frames:
            break
    if feats:
        return np.stack(feats)
    return np.empty((0, cfg.n_mels, cfg.time_steps), dtype=np.float32)


def main() -> None:
    p = argparse.ArgumentParser(description="Create debug validation splits")
    p.add_argument("--dataset-dir", type=Path, default=Path("data"), help="Root directory containing WAV files")
    p.add_argument("--output-dir", type=Path, default=Path("debug_splits"), help="Where to store output .npy files")
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--n-mels", type=int, default=128)
    p.add_argument("--n-fft", type=int, default=1024)
    p.add_argument("--hop-length", type=int, default=512)
    p.add_argument("--win-length", type=int, default=1024)
    p.add_argument("--fmin", type=float, default=0.0)
    p.add_argument("--fmax", type=float, default=None)
    p.add_argument("--power", type=float, default=2.0)
    p.add_argument("--time-steps", type=int, default=256, help="Number of frames per example")
    p.add_argument("--max-frames", type=int, default=256, help="Approximate total frames per split")
    cfg = p.parse_args()

    wavs = sorted(cfg.dataset_dir.rglob("*.wav"))
    if not wavs:
        raise FileNotFoundError(f"No WAV files found under {cfg.dataset_dir}")

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    ok = gather_split(wavs, normal=True, cfg=cfg)
    ng = gather_split(wavs, normal=False, cfg=cfg)

    np.save(cfg.output_dir / "val-OK.npy", ok)
    np.save(cfg.output_dir / "val-NG.npy", ng)
    print(f"Saved val-OK: {ok.shape} → {cfg.output_dir/'val-OK.npy'}")
    print(f"Saved val-NG: {ng.shape} → {cfg.output_dir/'val-NG.npy'}")


if __name__ == "__main__":
    main()
