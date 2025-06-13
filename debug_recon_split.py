import torch
import common as com
from datasets.datasets import Datasets
from networks.models import Models


def compute_mse(model: torch.nn.Module, loader) -> torch.Tensor:
    """Return concatenated per-sample MSE for ``loader``.

    ``loader`` may be a single DataLoader or a list thereof.
    """
    if isinstance(loader, list):
        losses = [compute_mse(model, ld) for ld in loader]
        return torch.cat(losses) if losses else torch.empty(0)

    errs = []
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for batch in loader:
            feats = batch[0].to(device).float()
            attr = None
            for t in batch[1:]:
                if isinstance(t, torch.Tensor) and t.ndim == 2:
                    attr = t.to(feats.device)
                    break
            recon, _, _ = model(feats, attr_vec=attr)
            mse = ((feats - recon[..., : feats.size(-1)]) ** 2).mean(dim=[1, 2, 3])
            errs.append(mse.cpu())
    return torch.cat(errs) if errs else torch.empty(0)


def main() -> None:
    param = com.yaml_load()
    parser = com.get_argparse()
    parser.add_argument("--model_ckpt", required=True, help="Path to trained model checkpoint")
    parser.add_argument("--ok_split", choices=["train", "valid"], default="valid", help="Normal data split")
    parser.add_argument("--ng_split", choices=["train", "valid", "test"], default="test", help="Anomaly data split")
    args = parser.parse_args(com.param_to_args_list(param))
    args = parser.parse_args(namespace=args)

    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    net = Models(args.model).net(args=args, train=False, test=True)
    state = torch.load(args.model_ckpt, map_location=device)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    net.model.load_state_dict(state, strict=False)
    net.model.to(device)

    ds = Datasets(args.dataset).data(args)
    loader_ok = ds.train_loader if args.ok_split == "train" else ds.valid_loader
    if args.ng_split == "train":
        loader_ng = ds.train_loader
    elif args.ng_split == "valid":
        loader_ng = ds.valid_loader
    else:
        loader_ng = ds.test_loader

    rec_ok = compute_mse(net.model, loader_ok)
    rec_ng = compute_mse(net.model, loader_ng)

    print(f"\u03bc OK={rec_ok.mean():.4f}\t\u03c3={rec_ok.std():.4f}")
    print(f"\u03bc NG={rec_ng.mean():.4f}\t\u03c3={rec_ng.std():.4f}")


if __name__ == "__main__":
    main()
