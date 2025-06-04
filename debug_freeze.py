import csv
import torch
import common as com
from networks.models import Models
from tools.plot_loss_curve import csv_to_figdata


def set_branch_requires_grad(net, freeze_ls = None):
    branches = {
        "b1": net.b1,
        "b2": net.b2,
        "b3": net.b3,
        "b5": net.b5,
        "b_attr": net.b_attr,
        "fusion": net.fusion,
    }
    for name, module in branches.items():
        req = (name not in freeze_ls)
        for p in module.parameters():
            p.requires_grad = req


def grad_norm(module):
    return sum(
        p.grad.norm().item() for p in module.parameters() if p.grad is not None
    )


def main():
    param = com.yaml_load()
    parser = com.get_argparse()
    parser.add_argument(
        "--freeze",
        type=str,
        default=None,
        choices=["b1", "b2", "b3", "b5", "b_attr", "fusion", None],
        help="Branch to freeze during training",
    )
    parser.add_argument(
        "--keep",
        type=str,
        default=None,
        choices=["b1", "b2", "b3", "b5", "b_attr", "fusion", None],
        help="Only branch to keep during training (if freeze is set, this will be ignored)",
    )

    args = parser.parse_args(args=com.param_to_args_list(param))
    args = parser.parse_args(namespace=args)

    args.train_only = True
    args.dev = True
    args.epochs = 10
    if args.train_only:
        train = True
        test = False
    elif args.test_only:
        train = False
        test = True
    else:
        train = True
        test = True
    
    args.cuda = args.use_cuda and torch.cuda.is_available()
    args.dataset = 'DCASE2025T2ToyCar'

    net = Models(args.model).net(args=args, train=train, test=test)
    
    if args.keep is not None:
        # If a branch is specified to keep, freeze all others
        freeze_ls = ["b1", "b2", "b3", "b5", "b_attr", "fusion"]
        freeze_ls.remove(args.keep)
    else:
        # If no branch is specified to keep, check the freeze argument
        freeze_ls = [args.freeze] if args.freeze else []

    set_branch_requires_grad(net, args.freeze)

    device = net.cfg.get("device", "cpu")
    hist = []

    for epoch in range(1, args.epochs + 1):
        for batch in net.train_loader:
            feats = batch[0].to(device).float()
            labels = torch.argmax(batch[2], dim=1).long().to(device)

            net.optimizer.zero_grad()
            loss2, score3, loss5, scores, loss3_ce = net.forward(feats, labels)

            if epoch == 1 and len(hist) == 0:
                net.mu2 = loss2.mean().item()
                net.mu5 = loss5.mean().item()
            else:
                net.mu2 = 0.99 * net.mu2 + 0.01 * loss2.mean().item()
                net.mu5 = 0.99 * net.mu5 + 0.01 * loss5.mean().item()

            loss2_norm = loss2 / (net.mu2 + 1e-6)
            loss5_norm = loss5 / (net.mu5 + 1e-6)

            fusion_loss = scores.var(unbiased=False)
            loss = (
                net.cfg.get("w2", 1.0) * loss2_norm.mean()
                + net.cfg.get("w3", 1.0) * loss3_ce.mean()
                + net.cfg.get("w5", 1.0) * loss5_norm.mean()
                + net.w_fusion * fusion_loss
            )

            loss.backward()

            grad = {
                "grad_b1": grad_norm(net.b1),
                "grad_b2": grad_norm(net.b2),
                "grad_b3": grad_norm(net.b3),
                "grad_b5": grad_norm(net.b5),
                "grad_b_attr": grad_norm(net.b_attr),
                "grad_fusion": grad_norm(net.fusion),
            }

            net.optimizer.step()

            hist.append({"loss": loss.item(), **grad})

    log_path = net.logs_dir / f"debug_freeze_{args.freeze or 'none'}.csv"
    with open(log_path, "w", newline="") as f:
        fieldnames = ["loss"] + list(hist[0].keys())[1:]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in hist:
            writer.writerow(row)

    csv_to_figdata(
        file_path=log_path,
        column_heading_list=[
            ["loss"],
            ["grad_b1", "grad_b2", "grad_b3", "grad_b5", "grad_b_attr", "grad_fusion"],
        ],
        ylabel="value",
        fig_count=2,
    )


if __name__ == "__main__":
    main()
