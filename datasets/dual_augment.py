import numpy as np
from torch.utils.data import Dataset
from scipy.ndimage import zoom
import random


def _time_stretch(spec, rate):
    """Simple time stretch on mel-spectrogram."""
    stretched = zoom(spec, (1, rate), order=1)
    T = spec.shape[1]
    if stretched.shape[1] > T:
        start = random.randint(0, stretched.shape[1] - T)
        stretched = stretched[:, start:start+T]
    else:
        pad = T - stretched.shape[1]
        stretched = np.pad(stretched, ((0,0),(0,pad)), mode='edge')
    return stretched


def _pitch_shift(spec, shift):
    shifted = np.roll(spec, shift, axis=0)
    if shift > 0:
        shifted[:shift] = spec[0]
    elif shift < 0:
        shifted[shift:] = spec[-1]
    return shifted


def _add_noise(spec, scale=0.02):
    noise = np.random.randn(*spec.shape) * scale
    return spec + noise


def random_aug(spec):
    op = random.choice(['time', 'pitch', 'noise'])
    if op == 'time':
        rate = random.uniform(0.8, 1.2)
        return _time_stretch(spec, rate)
    if op == 'pitch':
        shift = random.randint(-4, 4)
        return _pitch_shift(spec, shift)
    return _add_noise(spec)


class DualAugDataset(Dataset):
    """Wrap a dataset returning mel vectors to produce two augmented views."""
    def __init__(self, base, n_mels, frames):
        self.base = base
        self.n_mels = n_mels
        self.frames = frames

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        data, y_true, cond, basename, index = self.base[idx]
        spec = data.reshape(self.n_mels, self.frames)
        a1 = random_aug(spec)
        a2 = random_aug(spec)
        pair = np.stack([a1, a2], axis=0)  # [2, n_mels, frames]
        pair = pair[:, np.newaxis, :, :]     # [2, 1, n_mels, frames]
        return pair, y_true, cond, basename, index
