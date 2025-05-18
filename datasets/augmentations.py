import numpy as np
import random
import librosa

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
        return spec