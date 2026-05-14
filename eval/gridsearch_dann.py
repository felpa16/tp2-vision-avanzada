"""
Grid search over DANN training hyperparameters.

Same data and architecture as the production DANN run:
  - 3-channel RGB end-to-end (MNIST replicated 1->3, MNIST-M raw RGB; no
    grayscale preprocess).
  - Encoder hidden_dim=128, classifier head_dim=256 (the "big NN").
  - Held-out MNIST-M val split (selection metric); test never touched here.

The axis we sweep is the learning rate (the one with by far the largest
effect on adaptation quality in our prior experiments). Other knobs are
fixed at the defaults that worked best in the production run.

Scoring metric: best target-val accuracy across the full 50-epoch budget
(matches how the production run picks its checkpoint).

Usage
-----
    python -m eval.gridsearch_dann
    python -m eval.gridsearch_dann --epochs 30                    # quicker
    python -m eval.gridsearch_dann --n_source 20000 --n_target 20000   # subset

Saved artefacts
---------------
    checkpoints/gridsearch_dann.csv
"""

import argparse
import csv
import time
from itertools import product
from pathlib import Path

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset

from models.dann import (
    DANNModel,
    MNISTMRGB28,
    load_mnist_rgb_for_dann,
    load_mnistm_raw,
    run_accuracy,
    train_epoch,
)


# ── Grid definition ──────────────────────────────────────────────────────────
# Heavy overnight sweep. LR/schedule already gridsearched in the previous
# pass — we now PIN the top performing (lr, schedule) pairs and sweep the
# architectural axes (encoder width, head width, batch size).
#
# Top 3 (lr, use_lr_schedule) pairs from the previous gridsearch:
#   (1e-3, False)  -> 72.42% BestTgt   (rank 1)
#   (2e-3, False)  -> 72.16%           (rank 2)
#   (5e-3, True )  -> 71.84%           (rank 3)

TOP_LR_SCHED = [
    (1e-3, False),
    (2e-3, False),
    (5e-3, True),
]

DANN_GRID = {
    "lr_sched":   TOP_LR_SCHED,
    "hidden_dim": [128, 192, 256],
    "head_dim":   [256, 512],
    "batch_size": [128, 256],
}
# 3 * 3 * 2 * 2 = 36 configurations. At ~10–18 min each (heavier configs are
# slower), full sweep is ~6–10 hours — fits overnight on a single GPU.


def _grid_configs(grid: dict) -> list[dict]:
    keys = list(grid.keys())
    return [dict(zip(keys, vals)) for vals in product(*grid.values())]


# ── Single configuration trainer ─────────────────────────────────────────────

def train_one_config(
    cfg: dict,
    source_train,
    target_train,
    source_eval_loader: DataLoader,
    target_eval_loader: DataLoader,
    epochs: int,
    weight_decay: float,
    device: torch.device,
    seed: int,
) -> dict:
    """
    Train one DANN configuration. `cfg` is one dict produced by _grid_configs,
    containing per-config lr/schedule/arch/batch axes.

    Returns:
        {"best_tgt_acc": float, "final_tgt_acc": float, "final_src_acc": float,
         "n_params": int}
    """
    torch.manual_seed(seed)

    lr, use_sched = cfg["lr_sched"]
    hidden_dim    = cfg["hidden_dim"]
    head_dim      = cfg["head_dim"]
    batch_size    = cfg["batch_size"]

    source_loader = DataLoader(
        source_train, batch_size=batch_size, shuffle=True, num_workers=2
    )
    target_loader = DataLoader(
        target_train, batch_size=batch_size, shuffle=True, num_workers=2
    )

    # in_channels=3: matches the RGB pipeline used by the production run.
    model = DANNModel(
        hidden_dim=hidden_dim, n_classes=10, head_dim=head_dim, in_channels=3
    ).to(device)
    n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_tgt_acc = 0.0
    final_src_acc = 0.0
    final_tgt_acc = 0.0

    base_lr = lr if use_sched else None

    for epoch in range(1, epochs + 1):
        train_epoch(
            model, source_loader, target_loader,
            optimizer, device, epoch, epochs,
            use_dann=True, base_lr=base_lr,
        )
        src_acc = run_accuracy(model, source_eval_loader, device)
        tgt_acc = run_accuracy(model, target_eval_loader, device)

        final_src_acc = src_acc
        final_tgt_acc = tgt_acc
        if tgt_acc > best_tgt_acc:
            best_tgt_acc = tgt_acc

    return {
        "best_tgt_acc":  best_tgt_acc,
        "final_tgt_acc": final_tgt_acc,
        "final_src_acc": final_src_acc,
    }


# ── Search runner ────────────────────────────────────────────────────────────

def run_gridsearch(
    configs: list[dict],
    source_train,
    target_train,
    source_eval_loader: DataLoader,
    target_eval_loader: DataLoader,
    epochs: int,
    weight_decay: float,
    device: torch.device,
    seed: int,
    save_path: Path,
    top_n: int,
) -> list[tuple[dict, dict]]:
    total = len(configs)
    print(f"\n{'='*70}")
    print(f"  DANN heavy overnight sweep ({total} configurations)")
    print(f"  epochs/config = {epochs}, weight_decay = {weight_decay}")
    print(f"  input         = RGB 3-channel (no preprocess_mnistm)")
    print(f"  axes          = {list(DANN_GRID.keys())}")
    print(f"{'='*70}\n")

    results: list[tuple[dict, dict]] = []
    t_start = time.time()

    def _fmt_cfg(c: dict) -> str:
        lr, sched = c["lr_sched"]
        return (f"lr={lr}, sched={sched}, hidden={c['hidden_dim']}, "
                f"head={c['head_dim']}, bs={c['batch_size']}")

    for i, cfg in enumerate(configs, 1):
        print(f"  [{i:>{len(str(total))}}/{total}]  Training: {_fmt_cfg(cfg)}")
        t0 = time.time()

        try:
            metrics = train_one_config(
                cfg, source_train, target_train,
                source_eval_loader, target_eval_loader,
                epochs=epochs, weight_decay=weight_decay,
                device=device, seed=seed,
            )
        except RuntimeError as e:
            # OOM or numerical failure: log and continue so the overnight
            # sweep doesn't die on one bad config.
            print(f"            FAILED: {type(e).__name__}: {str(e)[:120]}")
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            metrics = {"best_tgt_acc": 0.0, "final_tgt_acc": 0.0,
                       "final_src_acc": 0.0, "n_params": -1}

        dt = time.time() - t0
        results.append((cfg, metrics))
        print(
            f"            best_tgt={metrics['best_tgt_acc']*100:5.2f}%  "
            f"final_tgt={metrics['final_tgt_acc']*100:5.2f}%  "
            f"final_src={metrics['final_src_acc']*100:5.2f}%  "
            f"params={metrics.get('n_params', 0):>9,}  ({dt/60:.1f} min)"
        )

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed/60:.1f} min")

    # Sort by best target accuracy (the metric DANN is optimising for)
    results.sort(key=lambda x: x[1]["best_tgt_acc"], reverse=True)

    # Save CSV (split lr_sched tuple into lr / use_lr_schedule columns)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "lr", "use_lr_schedule", "hidden_dim", "head_dim", "batch_size",
        "n_params", "best_tgt_acc", "final_tgt_acc", "final_src_acc",
    ]
    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for cfg, metrics in results:
            lr, sched = cfg["lr_sched"]
            row = {
                "lr":              lr,
                "use_lr_schedule": sched,
                "hidden_dim":      cfg["hidden_dim"],
                "head_dim":        cfg["head_dim"],
                "batch_size":      cfg["batch_size"],
                "n_params":        metrics.get("n_params", -1),
                "best_tgt_acc":    f"{metrics['best_tgt_acc']:.6f}",
                "final_tgt_acc":   f"{metrics['final_tgt_acc']:.6f}",
                "final_src_acc":   f"{metrics['final_src_acc']:.6f}",
            }
            writer.writerow(row)
    print(f"  Results saved to: {save_path}")

    # Print top-N
    print(f"\n  Top {top_n} configurations (by best target accuracy):")
    print(f"  {'Rank':<6}{'BestTgt':>9}{'FinalTgt':>10}{'FinalSrc':>10}  Configuration")
    print(f"  {'─'*100}")
    for rank, (cfg, m) in enumerate(results[:top_n], 1):
        print(
            f"  {rank:<6}{m['best_tgt_acc']*100:>8.2f}%"
            f"{m['final_tgt_acc']*100:>9.2f}%"
            f"{m['final_src_acc']*100:>9.2f}%  {_fmt_cfg(cfg)}"
        )

    print(f"\n  Worst configuration:")
    worst_cfg, worst_m = results[-1]
    print(
        f"  {worst_m['best_tgt_acc']*100:>8.2f}%"
        f"{worst_m['final_tgt_acc']*100:>9.2f}%"
        f"{worst_m['final_src_acc']*100:>9.2f}%  {_fmt_cfg(worst_cfg)}"
    )

    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Heavy DANN grid search: top LR configs x architecture x batch"
    )
    p.add_argument("--epochs",       type=int,   default=40,
                   help="Training epochs per configuration")
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--n_source",     type=int,   default=20000,
                   help="MNIST source training subset size (speed-up)")
    p.add_argument("--n_target",     type=int,   default=20000,
                   help="MNIST-M target training subset size (speed-up)")
    p.add_argument("--n_val",        type=int,   default=5000,
                   help="MNIST-M validation subset size (model selection set)")
    p.add_argument("--top_n",        type=int,   default=10)
    p.add_argument("--save_dir",     type=str,   default="checkpoints")
    p.add_argument("--seed",         type=int,   default=42)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)

    print(f"Device     : {device}")
    print(f"Search cfg : epochs={args.epochs}, weight_decay={args.weight_decay}")
    print(f"Subsets    : source={args.n_source}, target_train={args.n_target}, "
          f"target_val={args.n_val}")

    # ── Load datasets (RGB end-to-end, matches the production run) ──
    print("\nLoading datasets …")
    mnist_train, mnist_test = load_mnist_rgb_for_dann(augment=False)
    mnistm_train_raw, _ = load_mnistm_raw()
    mnistm_train_pp = MNISTMRGB28(mnistm_train_raw)

    # Deterministic subsets for fairness across configs.
    g = torch.Generator().manual_seed(args.seed)

    src_perm = torch.randperm(len(mnist_train), generator=g).tolist()
    src_subset = Subset(mnist_train, src_perm[: args.n_source])

    tgt_perm = torch.randperm(len(mnistm_train_pp), generator=g).tolist()
    tgt_train_idx = tgt_perm[: args.n_target]
    tgt_val_idx   = tgt_perm[args.n_target : args.n_target + args.n_val]
    tgt_train_subset = Subset(mnistm_train_pp, tgt_train_idx)
    tgt_val_subset   = Subset(mnistm_train_pp, tgt_val_idx)

    # Fixed evaluation loaders (same across all configurations).
    eval_bs = 256
    source_eval_loader = DataLoader(
        mnist_test, batch_size=eval_bs, shuffle=False, num_workers=2
    )
    target_eval_loader = DataLoader(
        tgt_val_subset, batch_size=eval_bs, shuffle=False, num_workers=2
    )

    print(f"  source train  : {len(src_subset):,}")
    print(f"  target train  : {len(tgt_train_subset):,}")
    print(f"  target val    : {len(tgt_val_subset):,}")
    print(f"  source test   : {len(mnist_test):,}")

    # ── Run grid search ──
    configs = _grid_configs(DANN_GRID)
    run_gridsearch(
        configs            = configs,
        source_train       = src_subset,
        target_train       = tgt_train_subset,
        source_eval_loader = source_eval_loader,
        target_eval_loader = target_eval_loader,
        epochs             = args.epochs,
        weight_decay       = args.weight_decay,
        device             = device,
        seed               = args.seed,
        save_path          = save_dir / "gridsearch_dann.csv",
        top_n              = args.top_n,
    )

    print("\nGrid search complete.")


if __name__ == "__main__":
    main()
