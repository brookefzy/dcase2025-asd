import yaml
from types import SimpleNamespace

def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return SimpleNamespace(**cfg)