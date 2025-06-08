import numpy as np
import random
from scipy.ndimage import zoom

def time_stretch(spec, rate):
    stretched = zoom(spec, (1, rate), order=1)
    T = spec.shape[1]
    if stretched.shape[1] > T:
        start = random.randint(0, stretched.shape[1] - T)
        stretched = stretched[:, start:start+T]
    else:
        pad = T - stretched.shape[1]
        stretched = np.pad(stretched, ((0,0),(0,pad)), mode='edge')
    return stretched

def pitch_shift(spec, shift):
    shifted = np.roll(spec, shift, axis=0)
    if shift > 0:
        shifted[:shift] = spec[0]
    elif shift < 0:
        shifted[shift:] = spec[-1]
    return shifted

def add_noise(spec, scale):
    noise = np.random.randn(*spec.shape) * scale
    return spec + noise

def spec_augment(spec, num_mask=2, freq_masking_max_percentage=0.15, time_masking_max_percentage=0.15):
    spec = spec.copy()
    num_mels, num_frames = spec.shape
    for _ in range(num_mask):
        # freq mask
        f = random.randint(0, int(num_mels * freq_masking_max_percentage))
        f0 = random.randint(0, num_mels - f)
        spec[f0:f0+f, :] = 0
        # time mask
        t = random.randint(0, int(num_frames * time_masking_max_percentage))
        t0 = random.randint(0, num_frames - t)
        spec[:, t0:t0+t] = 0
    return spec

class AugmentationPipeline:
    def __init__(self, config):
        self.config = config

    def __call__(self, spec):
        # Spectrogram augment (frequency & time masks)
        if random.random() < self.config['specaug_p']:
            spec = spec_augment(
                spec,
                num_mask         = self.config['specaug_num'],
                freq_masking_max_percentage = self.config['specaug_freq'],
                time_masking_max_percentage = self.config['specaug_time']
            )
        # time stretch
        if random.random() < self.config.get('time_stretch_p', 0.0):
            r_min, r_max = self.config.get('time_stretch_range', [0.9, 1.1])
            rate = random.uniform(r_min, r_max)
            spec = time_stretch(spec, rate)

        # pitch shift
        if random.random() < self.config.get('pitch_shift_p', 0.0):
            s_min, s_max = self.config.get('pitch_shift_range', [-2, 2])
            shift = random.randint(int(s_min), int(s_max))
            spec = pitch_shift(spec, shift)

        # add noise
        if random.random() < self.config.get('noise_p', 0.0):
            scale = self.config.get('noise_std', 0.02)
            spec = add_noise(spec, scale)

        return spec
