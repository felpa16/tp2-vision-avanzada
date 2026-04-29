"""
Grid search over preprocessing hyperparameters for MNIST-M and SVHN.

Searches over axes that have *structural* impact on the output image:
  - Color space for luminance extraction (LAB-L / HSV-V / standard gray)
  - Blur strategy (bilateral / gaussian / none)
  - Whether to apply CLAHE
  - Polarity border margin size
  - Crop size (SVHN only)

Uses the validation splits (not test) so that the test set remains unbiased.
Evaluates at a single K value with fewer episodes to keep runtime manageable.

Usage
-----
    python -m eval.gridsearch_preprocess
    python -m eval.gridsearch_preprocess --dataset mnistm --n_episodes 30
    python -m eval.gridsearch_preprocess --k_eval 4 --top_n 5

Saved artefacts
---------------
    checkpoints/gridsearch_mnistm.csv
    checkpoints/gridsearch_svhn.csv
"""

import argparse
import csv
import random
import time
from functools import partial
from itertools import product
from pathlib import Path

import torch

from models.protonet import ConvNetEncoder, compute_centroids, squared_euclidean_distance
from data.prepare_datasets import load_mnist, load_svhn, load_mnistm
from data.episodic_sampler import _build_class_map
from data.preprocess import preprocess_mnistm, preprocess_svhn


# ── Evaluation (lightweight, single-K) ──────────────────────────────────────

def quick_evaluate(
    encoder: ConvNetEncoder,
    dataset,
    class_map: dict[int, list[int]],
    k_shot: int,
    n_query: int,
    n_way: int,
    n_episodes: int,
    device: torch.device,
    seed: int,
    preprocess=None,
) -> float:
    """Run n_episodes at a single K and return mean accuracy."""
    rng = random.Random(seed)
    accs = []

    for _ in range(n_episodes):
        classes = rng.sample(sorted(class_map.keys()), n_way)
        support_imgs, query_imgs, query_labels = [], [], []

        for local_label, cls in enumerate(classes):
            pool = class_map[cls]
            chosen = rng.sample(pool, k_shot + n_query)

            for idx in chosen[:k_shot]:
                img, _ = dataset[idx]
                if preprocess is not None:
                    img = preprocess(img)
                support_imgs.append(img)

            for idx in chosen[k_shot:]:
                img, _ = dataset[idx]
                if preprocess is not None:
                    img = preprocess(img)
                query_imgs.append(img)
                query_labels.append(local_label)

        support_t = torch.stack(support_imgs).to(device)
        query_t   = torch.stack(query_imgs).to(device)
        q_labels  = torch.tensor(query_labels, device=device)

        with torch.no_grad():
            all_emb     = encoder(torch.cat([support_t, query_t], dim=0))
            support_emb = all_emb[:n_way * k_shot]
            query_emb   = all_emb[n_way * k_shot:]

        centroids = compute_centroids(support_emb, n_way, k_shot)
        dists     = squared_euclidean_distance(query_emb, centroids)
        preds     = (-dists).argmax(dim=1)
        accs.append((preds == q_labels).float().mean().item())

    return sum(accs) / len(accs)


# ── Grid definitions ─────────────────────────────────────────────────────────
# Each axis is chosen to produce *structurally different* output images.
# Continuous parameters (sigma, clip) that had near-zero effect on 32x32
# images are fixed at reasonable values; the search focuses on discrete
# choices and parameters with large impact.

MNISTM_GRID = {
    "color_space":      ["lab", "hsv", "gray"],
    "blur_method":      ["bilateral", "gaussian", "none"],
    "bilateral_d":      [5, 9],           # only used when blur_method=bilateral
    "gaussian_ksize":   [3, 5],           # only used when blur_method=gaussian
    "use_clahe":        [True, False],
    "polarity_margin":  [0.1, 0.2, 0.3],
}
# 3 * 3 * 2 * 2 * 2 * 3 = 216, but many combos are degenerate
# (bilateral_d ignored when blur=gaussian, etc.) — we prune those below.

SVHN_GRID = {
    "crop_size":        [18, 22, 26, 32],  # 32 = no crop
    "color_space":      ["lab", "hsv", "gray"],
    "blur_method":      ["bilateral", "gaussian", "none"],
    "bilateral_d":      [5, 9],
    "gaussian_ksize":   [3, 5],
    "use_clahe":        [True, False],
    "polarity_margin":  [0.1, 0.2, 0.3],
}


def _grid_configs(grid: dict) -> list[dict]:
    """
    Expand a grid dict into a list of parameter combinations,
    pruning irrelevant blur sub-parameters to avoid redundant evaluations.
    """
    keys = list(grid.keys())
    raw  = [dict(zip(keys, vals)) for vals in product(*grid.values())]

    seen    = set()
    pruned  = []
    for cfg in raw:
        # Normalise: zero out params that the blur method won't use
        c = dict(cfg)
        if c["blur_method"] != "bilateral":
            c["bilateral_d"] = grid["bilateral_d"][0]  # canonical value
        if c["blur_method"] != "gaussian":
            c["gaussian_ksize"] = grid["gaussian_ksize"][0]

        key = tuple(sorted(c.items()))
        if key not in seen:
            seen.add(key)
            pruned.append(c)

    return pruned


# ── Search runner ────────────────────────────────────────────────────────────

def run_gridsearch(
    name: str,
    encoder: ConvNetEncoder,
    dataset,
    class_map: dict[int, list[int]],
    base_fn,
    grid: dict,
    k_shot: int,
    n_query: int,
    n_way: int,
    n_episodes: int,
    device: torch.device,
    seed: int,
    save_path: Path,
    top_n: int,
) -> list[tuple[dict, float]]:
    """
    Test every config in the grid and return sorted results.

    Writes a CSV with all results and prints the top-N configurations.
    """
    configs = _grid_configs(grid)
    total   = len(configs)
    print(f"\n{'='*70}")
    print(f"  Grid search: {name}  ({total} unique configurations)")
    print(f"  K={k_shot}, {n_episodes} episodes/config, {n_way}-way")
    print(f"{'='*70}\n")

    results: list[tuple[dict, float]] = []
    t_start = time.time()

    for i, params in enumerate(configs, 1):
        preprocess_fn = partial(base_fn, **params)

        t0  = time.time()
        acc = quick_evaluate(
            encoder, dataset, class_map,
            k_shot=k_shot, n_query=n_query, n_way=n_way,
            n_episodes=n_episodes, device=device, seed=seed,
            preprocess=preprocess_fn,
        )
        dt = time.time() - t0

        results.append((params, acc))

        # Compact progress — show only the axes that matter for this config
        display = {k: v for k, v in params.items()
                   if not (k == "bilateral_d" and params["blur_method"] != "bilateral")
                   and not (k == "gaussian_ksize" and params["blur_method"] != "gaussian")}
        param_str = ", ".join(f"{k}={v}" for k, v in display.items())
        print(f"  [{i:>{len(str(total))}}/{total}]  {acc*100:5.2f}%  ({dt:.1f}s)  {param_str}")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed/60:.1f} min")

    # Sort descending by accuracy
    results.sort(key=lambda x: x[1], reverse=True)

    # Save CSV
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(grid.keys()) + ["accuracy"]
    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for params, acc in results:
            row = {**params, "accuracy": f"{acc:.6f}"}
            for k, v in row.items():
                if isinstance(v, tuple):
                    row[k] = f"{v[0]}x{v[1]}"
            writer.writerow(row)
    print(f"  Full results saved to: {save_path}")

    # Print top-N
    print(f"\n  Top {top_n} configurations:")
    print(f"  {'Rank':<6}{'Accuracy':>10}  Parameters")
    print(f"  {'─'*65}")
    for rank, (params, acc) in enumerate(results[:top_n], 1):
        display = {k: v for k, v in params.items()
                   if not (k == "bilateral_d" and params["blur_method"] != "bilateral")
                   and not (k == "gaussian_ksize" and params["blur_method"] != "gaussian")}
        param_str = ", ".join(f"{k}={v}" for k, v in display.items())
        print(f"  {rank:<6}{acc*100:>9.2f}%  {param_str}")

    # Print worst for comparison
    print(f"\n  Worst configuration:")
    worst_params, worst_acc = results[-1]
    display = {k: v for k, v in worst_params.items()
               if not (k == "bilateral_d" and worst_params["blur_method"] != "bilateral")
               and not (k == "gaussian_ksize" and worst_params["blur_method"] != "gaussian")}
    param_str = ", ".join(f"{k}={v}" for k, v in display.items())
    print(f"  {worst_acc*100:>9.2f}%  {param_str}")

    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Grid search over preprocessing hyperparameters"
    )
    p.add_argument("--checkpoint", type=str, default="checkpoints/best_protonet.pt")
    p.add_argument("--n_way",      type=int, default=10)
    p.add_argument("--n_query",    type=int, default=10)
    p.add_argument("--n_episodes", type=int, default=50,
                   help="Episodes per config (lower = faster, noisier)")
    p.add_argument("--k_eval",     type=int, default=8,
                   help="Single K value to use during search")
    p.add_argument("--top_n",      type=int, default=10,
                   help="Number of top configs to print")
    p.add_argument("--save_dir",   type=str, default="checkpoints")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--dataset",    type=str, default="both",
                   choices=["mnistm", "svhn", "both"],
                   help="Which dataset to search (default: both)")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device     : {device}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Search cfg : {args.n_way}-way, K={args.k_eval}, "
          f"{args.n_query} queries, {args.n_episodes} episodes/config")

    # ── Load encoder ──
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt       = torch.load(ckpt_path, map_location=device)
    hidden_dim = ckpt["args"].get("hidden_dim", 64)
    encoder    = ConvNetEncoder(in_channels=1, hidden_dim=hidden_dim).to(device)
    encoder.load_state_dict(ckpt["model"])
    encoder.eval()

    print(f"Loaded encoder from epoch {ckpt['epoch']}  "
          f"(val query acc: {ckpt['val_q_acc']*100:.2f}%)")

    # ── Load validation splits ──
    print("\nLoading datasets …")
    _, _, _             = load_mnist()
    svhn_val, _         = load_svhn()
    _, mnistm_val, _    = load_mnistm()

    save_dir = Path(args.save_dir)

    # ── MNIST-M grid search ──
    if args.dataset in ("mnistm", "both"):
        mnistm_class_map = _build_class_map(mnistm_val)
        run_gridsearch(
            name       = "MNIST-M",
            encoder    = encoder,
            dataset    = mnistm_val,
            class_map  = mnistm_class_map,
            base_fn    = preprocess_mnistm,
            grid       = MNISTM_GRID,
            k_shot     = args.k_eval,
            n_query    = args.n_query,
            n_way      = args.n_way,
            n_episodes = args.n_episodes,
            device     = device,
            seed       = args.seed,
            save_path  = save_dir / "gridsearch_mnistm.csv",
            top_n      = args.top_n,
        )

    # ── SVHN grid search ──
    if args.dataset in ("svhn", "both"):
        svhn_class_map = _build_class_map(svhn_val)
        run_gridsearch(
            name       = "SVHN",
            encoder    = encoder,
            dataset    = svhn_val,
            class_map  = svhn_class_map,
            base_fn    = preprocess_svhn,
            grid       = SVHN_GRID,
            k_shot     = args.k_eval,
            n_query    = args.n_query,
            n_way      = args.n_way,
            n_episodes = args.n_episodes,
            device     = device,
            seed       = args.seed,
            save_path  = save_dir / "gridsearch_svhn.csv",
            top_n      = args.top_n,
        )

    print("\nGrid search complete.")


if __name__ == "__main__":
    main()
