import csv
import torch
import common as com
from networks.models import Models
from tools.plot_loss_curve import csv_to_figdata
from torch.nn.utils import clip_grad_norm_

def set_branch_requires_grad(net, train_b2=False, train_b5=False):
    net.b1.requires_grad_(False)
    net.b3.requires_grad_(False)
    net.b_attr.requires_grad_(False)
    net.fusion.requires_grad_(False)
    net.b2.requires_grad_(train_b2)
    net.b5.requires_grad_(train_b5)

def grad_norm(module):
    return sum(p.grad.norm().item() for p in module.parameters() if p.grad is not None)

def main():
    param = com.yaml_load()
    parser = com.get_argparse()
    args = parser.parse_args(args=com.param_to_args_list(param))
    args = parser.parse_args(namespace=args)
    args.train_only = True
    args.dev = True
    args.epochs = 20
    args.cuda = args.use_cuda and torch.cuda.is_available()
    args.dataset = 'DCASE2025T2ToyCar'

    net = Models(args.model).net(args=args, train=True, test=False)
    device = net.cfg.get("device", "cpu")
    hist = []

    for epoch in range(1, args.epochs + 1):
        train_b5 = epoch > 10
        set_branch_requires_grad(net, train_b2=True, train_b5=train_b5)

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

            # Clamp loss5_norm to prevent overflow
            loss5_norm = torch.clamp(loss5_norm, max=50)

            fusion_loss = scores.var(unbiased=False)

            # Limit the contribution of flow loss even when it's active
            w5 = 0.01 if train_b5 else 0.0
            loss = (
                net.cfg.get("w2", 1.0) * loss2_norm.mean() +
                net.cfg.get("w3", 1.0) * loss3_ce.mean() +
                w5 * loss5_norm.mean() +
                net.w_fusion * fusion_loss
            )

            # Gradient clipping
            clip_grad_norm_(net.b2.parameters(), max_norm=5.0)
            if train_b5:
                clip_grad_norm_(net.b5.parameters(), max_norm=1.0)

            loss.backward()
            net.optimizer.step()

            grad = {
                "grad_b2": grad_norm(net.b2),
                "grad_b5": grad_norm(net.b5),
                "loss": loss.item(),
            }
            hist.append(grad)

    log_path = net.logs_dir / "debug_staged_training_stabilized.csv"
    with open(log_path, "w", newline="") as f:
        fieldnames = ["loss", "grad_b2", "grad_b5"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in hist:
            writer.writerow(row)

    csv_to_figdata(
        file_path=log_path,
        column_heading_list=[["loss"], ["grad_b2", "grad_b5"]],
        ylabel="value",
        fig_count=2,
    )

if __name__ == "__main__":
    main()
