# >>> tools/check_dataset.py  (run:  python tools/check_dataset.py)
import os, json, glob, librosa, tqdm
from collections import Counter
import sys
ROOT_DIR = "/lustre1/g/geog_pyloo/11_octa/dcase2025-asd"
sys.path.append(ROOT_DIR)  # add root to path
from utils.audio import file_to_vectors

CFG_PATH = f"{ROOT_DIR}/config.yaml"           # adjust if needed
DATA_ROOT = f"{ROOT_DIR}/data"               # root of DCASE wav files
TARGET_KEYWORD = "target"          # how you tag target clips in filenames

# load n_mels / n_frames from your yaml so the script stays in sync
import yaml
with open(CFG_PATH) as f:
    cfg = yaml.safe_load(f)
n_mels   = cfg["n_mels"]
n_frames = cfg["frames"]

stats = Counter()
for wav in tqdm.tqdm(glob.glob(os.path.join(DATA_ROOT, "**/*.wav"), recursive=True)):
    signal, sr = librosa.load(wav, sr=None, mono=True)
    vecs = file_to_vectors(signal, sr, n_mels=n_mels, n_frames=n_frames)
    tag  = "target" if TARGET_KEYWORD in wav.lower() else "source"
    if len(vecs) == 0:
        stats[(tag, "skipped")] += 1
    elif vecs.shape[0] == 1:
        stats[(tag, "padded")]  += 1
    else:
        stats[(tag, "ok")]      += 1

print("\n=== Dataset length check ===")
for k, v in sorted(stats.items()):
    print(f"{k[0]:>7} | {k[1]:>7} : {v}")
