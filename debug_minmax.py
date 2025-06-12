import numpy as np
import torch
import common as com
from datasets.datasets import Datasets


def compute_minmax(dataset) -> tuple[float, float]:
    """Return global min and max values from a torch Dataset."""
    base_ds = dataset.dataset if isinstance(dataset, torch.utils.data.Subset) else dataset

    if hasattr(base_ds, "data"):
        mins = [np.min(x) for x in base_ds.data]
        maxs = [np.max(x) for x in base_ds.data]
        return float(np.min(mins)), float(np.max(maxs))

    min_val = float("inf")
    max_val = -float("inf")
    for i in range(len(base_ds)):
        feat = base_ds[i][0]
        arr = feat.numpy() if isinstance(feat, torch.Tensor) else feat
        min_val = min(min_val, float(arr.min()))
        max_val = max(max_val, float(arr.max()))
    return min_val, max_val


def main() -> None:
    param = com.yaml_load()
    parser = com.get_argparse()
    parser.add_argument("--split", choices=["train", "valid"], default="train",
                        help="Dataset split to inspect")
    args = parser.parse_args(com.param_to_args_list(param))
    args = parser.parse_args(namespace=args)

    ds_obj = Datasets(args.dataset).data(args)
    if args.split == "train":
        ds_subset = ds_obj.train_dataset
    else:
        ds_subset = ds_obj.valid_dataset

    mn, mx = compute_minmax(ds_subset)
    print(f"data min: {mn:.4f}, max: {mx:.4f}")


if __name__ == "__main__":
    main()
