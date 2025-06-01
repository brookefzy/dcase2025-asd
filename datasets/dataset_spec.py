import os
import glob
import soundfile as sf
import numpy as np
import torch
from torch.utils.data import Dataset
from datasets.augmentations import AugmentationPipeline

class SpectrogramDataset(Dataset):
    def __init__(self, base_dir, machine_type, mode, config, section=None):
        # mode: 'train', 'test'
        if section is None:
            pattern = os.path.join(
                base_dir, machine_type, mode, "*.wav"
            )
        else:
            pattern = os.path.join(
                base_dir, machine_type, mode, f"*_{section}_*.wav"
            )

        self.files = glob.glob(pattern)
        self.config = config
        self.augment = AugmentationPipeline(config) if mode in ('train', 'supplemental') else None

        # optional: load attributes CSV for this machine_type if it exists
        csv_path = os.path.join(base_dir, machine_type, "attributes_00.csv")
        if os.path.exists(csv_path):
            import pandas as pd
            self.attrs = pd.read_csv(csv_path, index_col=0).to_dict('index')
        else:
            self.attrs = {}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        wav, sr = sf.read(path)
        # compute log-mel spectrogram
        import librosa
        spec = librosa.feature.melspectrogram(
            y=wav, sr=sr,
            n_mels=self.config['n_mels'],
            n_fft=self.config['n_fft'],
            hop_length=self.config['hop_length']
        )
        log_spec = librosa.power_to_db(spec)
        # normalize
        log_spec = (log_spec - log_spec.mean()) / (log_spec.std() + 1e-6)
        # augmentation
        if self.augment:
            log_spec = self.augment(log_spec)
            
        # log_spec: np.ndarray of shape [n_mels, T]
        T_req = self.config['time_steps']
        _, T = log_spec.shape
        if T < T_req:
            pad_amt = T_req - T
            log_spec = np.pad(log_spec, ((0,0),(0,pad_amt)), mode='constant')
        else:
            log_spec = log_spec[:, :T_req]
        # convert to tensor
        tensor = torch.from_numpy(log_spec).float()             # [n_mels, T]
        tensor = tensor.unsqueeze(0)                           # [1, n_mels, T]
        fname = os.path.basename(path)
        attr = self.attrs.get(fname, {})
        return tensor, fname, attr