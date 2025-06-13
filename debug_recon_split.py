import torch
import common as com
from datasets.datasets import Datasets
from networks.models import Models

def get_loader(ds, split: str):
    if split == "train":
        return ds.train_loader
    if split == "valid":
        return ds.valid_loader
    if split == "test":
        return ds.test_loader
    raise ValueError(f"unknown split {split}")

def mse_by_label(model: torch.nn.Module, loader) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-sample MSE and return two 1-D tensors:

        (errs_ok, errs_ng)

    for all OK (label==0) and NG (label==1) samples in `loader`.
    """
    errs_ok, errs_ng = [], []
    model.eval()

    device = next(model.parameters()).device
    with torch.no_grad():
        for batch in loader:
            # --- unpack batch -------------------------------------------------
            feats = batch[0].to(device).float()                 # B×C×F×T
            label = next(t for t in batch if t.ndim == 1)        # B
            attr  = next((t for t in batch if t.ndim == 2), None)  # B×A or None

            # --- forward & MSE ----------------------------------------------
            recon, _, _ = model(feats, attr_vec=attr)
            mse = ((feats - recon[..., : feats.size(-1)])**2).mean(dim=[1,2,3])  # B

            # --- split by label ---------------------------------------------
            label = label.to(torch.bool)
            errs_ok.append(mse[~label].cpu())   # 0 → OK
            errs_ng.append(mse[label].cpu())    # 1 → NG

    cat = lambda lst: torch.cat(lst) if lst else torch.empty(0)
    return cat(errs_ok), cat(errs_ng)


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
    parser.add_argument("--model_ckpt", 
                        default="models/checkpoint/ast_singlebranch/ASTAutoencoderASD_DCASE2025T2ToyCar+DCASE2025T2ToyTrain+DCASE2025T2bearing+DCASE2025T2fan+DCASE2025T2gearbox+DCASE2025T2slider+DCASE2025T2valve_id(0_)_seed13711/checkpoint.tar",
                        type=str,
                        help="Path to trained model checkpoint")
    parser.add_argument("--ok_split", choices=["train", "valid", "test"], default="test", help="Normal data split")
    parser.add_argument("--ng_split", choices=["train", "valid", "test"], default="test", help="Anomaly data split")
    args = parser.parse_args(com.param_to_args_list(param))
    args = parser.parse_args(namespace=args)
    args.dataset = "DCASE2025T2ToyTrain"
    args.test_only = True
    args.dev = True

    # -----------------------------------------------------------------------
    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    net    = Models(args.model).net(args=args, train=False, test=True)
    state  = torch.load(args.model_ckpt, map_location=device)
    net.model.load_state_dict(state.get("model_state_dict", state), strict=False)
    net.model.to(device)

    # -----------------------------------------------------------------------
    ds = Datasets(args.dataset).data(args)
    loader_ok_split = get_loader(ds, args.ok_split)
    loader_ng_split = get_loader(ds, args.ng_split)

    # -----------------------------------------------------------------------
    #   • If both splits are identical, a single pass is enough
    #   • Otherwise we evaluate each split separately and merge results
    # -----------------------------------------------------------------------
    errs_ok, errs_ng = torch.empty(0), torch.empty(0)

    if args.ok_split == args.ng_split:
        errs_ok, errs_ng = mse_by_label(net.model, loader_ok_split)
    else:
        errs_ok, _       = mse_by_label(net.model, loader_ok_split)
        _,       errs_ng = mse_by_label(net.model, loader_ng_split)

    # -----------------------------------------------------------------------
    print(f"OK clips : n={len(errs_ok):4d}  μ={errs_ok.mean():.4f}  σ={errs_ok.std():.4f}")
    print(f"NG clips : n={len(errs_ng):4d}  μ={errs_ng.mean():.4f}  σ={errs_ng.std():.4f}")


if __name__ == "__main__":
    main()
