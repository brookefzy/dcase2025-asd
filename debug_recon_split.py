import torch
import common as com
from networks.models import Models


def load_split_tensor(path: str) -> torch.Tensor:
    """Load a tensor saved by ``tools/make_debug_splits.py``.

    The file is expected to be created with ``torch.save`` and contain either a
    single ``Tensor`` or a dictionary with a ``"data"`` key.
    """
    block = torch.load(path, map_location="cpu")
    if isinstance(block, dict) and "data" in block:
        block = block["data"]
    if isinstance(block, (list, tuple)):
        block = torch.stack([torch.as_tensor(x) for x in block])
    elif not isinstance(block, torch.Tensor):
        block = torch.as_tensor(block)
    return block.float()


def compute_mse(
    model: torch.nn.Module,
    data,
    batch_size: int = 256,
) -> torch.Tensor:
    """Return concatenated per-sample MSE for ``data``.

    ``data`` can be a :class:`DataLoader`, :class:`Dataset` or ``Tensor``.
    """
    if isinstance(data, torch.Tensor):
        dataset = torch.utils.data.TensorDataset(data)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    elif isinstance(data, torch.utils.data.Dataset):
        loader = torch.utils.data.DataLoader(data, batch_size=batch_size)
    else:
        loader = data

    errs = []
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                feats = batch[0]
            else:
                feats = batch
            feats = feats.to(device).float()
            recon, _, _ = model(feats)
            mse = ((feats - recon[..., : feats.size(-1)]) ** 2).mean(dim=[1, 2, 3])
            errs.append(mse.cpu())
    return torch.cat(errs) if errs else torch.empty(0)


def main() -> None:
    param = com.yaml_load()
    parser = com.get_argparse()
    parser.add_argument("--model_ckpt", required=True, help="Path to trained model checkpoint")
    parser.add_argument("--ok_file", default="debug_split_ok.pth", help="Tensor file for normal data")
    parser.add_argument("--ng_file", default="debug_split_ng.pth", help="Tensor file for anomaly data")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for evaluation")
    args = parser.parse_args(com.param_to_args_list(param))
    args = parser.parse_args(namespace=args)

    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    net = Models(args.model).net(args=args, train=False, test=True)
    state = torch.load(args.model_ckpt, map_location=device)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    net.model.load_state_dict(state, strict=False)
    net.model.to(device)

    feats_ok = load_split_tensor(args.ok_file)
    feats_ng = load_split_tensor(args.ng_file)

    rec_ok = compute_mse(net.model, feats_ok, batch_size=args.batch_size)
    rec_ng = compute_mse(net.model, feats_ng, batch_size=args.batch_size)

    print(f"\u03bc OK={rec_ok.mean():.4f}\t\u03c3={rec_ok.std():.4f}")
    print(f"\u03bc NG={rec_ng.mean():.4f}\t\u03c3={rec_ng.std():.4f}")


if __name__ == "__main__":
    main()
