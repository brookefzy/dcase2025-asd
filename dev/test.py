import argparse
import torch
from data.dataset import SpectrogramDataset
from models.resnet_ae import ResNetAutoEncoder
from models.flow import NormalizingFlow
from models.ensemble import EnsembleModel
from utils.config import load_config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--machine', type=str, required=True)
    parser.add_argument('--section', type=str, required=True)
    parser.add_argument('--mode', type=str, choices=['mse','mahal'], default='mse')
    args = parser.parse_args()
    cfg = load_config(args.config)

    # load models
    ae = ResNetAutoEncoder(latent_dim=cfg.latent_dim)
    ae.load_state_dict(torch.load(f"models/{args.machine}_{args.section}_ae.pth"))
    flow = NormalizingFlow(dim=cfg.latent_dim)
    flow.load_state_dict(torch.load(f"models/{args.machine}_{args.section}_flow.pth"))
    ens = EnsembleModel(ae, flow, w_ae=cfg.w_ae, w_flow=cfg.w_flow)

    ds = SpectrogramDataset(cfg.data_dir, args.machine, args.section, 'test', cfg)
    loader = torch.utils.data.DataLoader(ds, batch_size=1)

    out = []
    for x, fname in loader:
        scores = ens(x)
        val = scores['ae'].item() if args.mode=='mse' else scores['ensemble'].item()
        out.append((fname, val))
    # save CSV
    import csv
    out_file = f"results/{args.machine}_{args.section}_scores_{args.mode}.csv"
    os.makedirs('results', exist_ok=True)
    with open(out_file,'w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename','score'])
        writer.writerows(out)
    print(f"Saved scores to {out_file}")