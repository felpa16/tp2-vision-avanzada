"""
Evaluate the top-N configs from each gridsearch CSV at K=16 (or any K).

Reads the CSV files produced by gridsearch_preprocess.py, takes the top N
rows (already sorted by accuracy), and runs a full evaluation with more
episodes for a reliable comparison.

Usage
-----
    python -m eval.evaluate_top_configs
    python -m eval.evaluate_top_configs --top_n 5 --k_eval 16 --n_episodes 200
    python -m eval.evaluate_top_configs --dataset svhn

Saved artefacts
---------------
    checkpoints/top_configs_mnistm.csv
    checkpoints/top_configs_svhn.csv
"""

import argparse
import csv
import random
import time
from functools import partial
from pathlib import Path

import torch

from models.protonet import ConvNetEncoder, compute_centroids, squared_euclidean_distance
from data.prepare_datasets import load_mnist, load_svhn, load_mnistm
from data.episodic_sampler import _build_class_map
from data.preprocess import preprocess_mnistm, preprocess_svhn


# ── CSV parsing ───────────────��──────────────────────────────────���───────────

_CASTERS = {
    "bilateral_d":           int,
    "bilateral_sigma_color": float,
    "bilateral_sigma_space": float,
    "gaussian_ksize":        int,
    "clahe_clip":            float,
    "polarity_margin":       float,
    "crop_size":             int,
    "accuracy":              float,
}


def _cast(key: str, value: str):
    """Cast a CSV string value to its proper Python type."""
    if key == "use_clahe":
        return value.strip().lower() == "true"
    caster = _CASTERS.get(key)
    return caster(value) if caster else value


def load_top_configs(csv_path: Path, top_n: int) -> list[dict]:
    """
    Read a gridsearch CSV (sorted descending by accuracy) and return
    the top_n rows as dicts with proper types.
    """
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            rows.append({k: _cast(k, v) for k, v in row.items()})
            if len(rows) >= top_n:
                break
    return rows


# ── Evaluation ────────���──────────────────────────────────────────────────────

def evaluate_config(
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


# ── Runner ───────────���───────────────────────────────────────────────────────

def run_top_evaluation(
    name: str,
    encoder: ConvNetEncoder,
    dataset,
    class_map: dict[int, list[int]],
    base_fn,
    configs: list[dict],
    k_shot: int,
    n_query: int,
    n_way: int,
    n_episodes: int,
    device: torch.device,
    seed: int,
    save_path: Path,
) -> list[tuple[dict, float, float]]:
    """
    Evaluate each config and return (params, gridsearch_acc, new_acc) sorted
    by new accuracy descending.
    """
    total = len(configs)
    print(f"\n{'='*70}")
    print(f"  Top-{total} re-evaluation: {name}")
    print(f"  K={k_shot}, {n_episodes} episodes/config, {n_way}-way")
    print(f"{'='*70}\n")

    results = []

    for i, row in enumerate(configs, 1):
        gs_acc = row.pop("accuracy")

        preprocess_fn = partial(base_fn, **row)

        t0  = time.time()
        acc = evaluate_config(
            encoder, dataset, class_map,
            k_shot=k_shot, n_query=n_query, n_way=n_way,
            n_episodes=n_episodes, device=device, seed=seed,
            preprocess=preprocess_fn,
        )
        dt = time.time() - t0

        results.append((row, gs_acc, acc))

        # Display — skip irrelevant blur sub-params
        display = {k: v for k, v in row.items()
                   if not (k == "bilateral_d" and row.get("blur_method") != "bilateral")
                   and not (k == "gaussian_ksize" and row.get("blur_method") != "gaussian")}
        param_str = ", ".join(f"{k}={v}" for k, v in display.items())
        delta = acc - gs_acc
        sign  = "+" if delta >= 0 else ""
        print(f"  [{i:>{len(str(total))}}/{total}]  "
              f"GS={gs_acc*100:5.2f}%  K{k_shot}={acc*100:5.2f}%  "
              f"({sign}{delta*100:.2f}%)  ({dt:.1f}s)")
        print(f"          {param_str}")

    # Sort by new accuracy
    results.sort(key=lambda x: x[2], reverse=True)

    # Save CSV
    save_path.parent.mkdir(parents=True, exist_ok=True)
    sample_keys = list(results[0][0].keys())
    fieldnames  = sample_keys + ["gridsearch_acc", f"k{k_shot}_acc"]
    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for params, gs_acc, new_acc in results:
            csv_row = {**params,
                       "gridsearch_acc": f"{gs_acc:.6f}",
                       f"k{k_shot}_acc":  f"{new_acc:.6f}"}
            writer.writerow(csv_row)
    print(f"\n  Results saved to: {save_path}")

    # Summary table
    print(f"\n  {'Rank':<6}{'GS Acc':>9}{'K='+str(k_shot)+' Acc':>11}{'Delta':>9}  Config")
    print(f"  {'='*75}")
    for rank, (params, gs_acc, new_acc) in enumerate(results, 1):
        delta = new_acc - gs_acc
        sign  = "+" if delta >= 0 else ""
        display = {k: v for k, v in params.items()
                   if not (k == "bilateral_d" and params.get("blur_method") != "bilateral")
                   and not (k == "gaussian_ksize" and params.get("blur_method") != "gaussian")}
        param_str = ", ".join(f"{k}={v}" for k, v in display.items())
        print(f"  {rank:<6}{gs_acc*100:>8.2f}%{new_acc*100:>10.2f}%"
              f"{sign}{delta*100:>8.2f}%  {param_str}")

    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Re-evaluate top gridsearch configs at a target K"
    )
    p.add_argument("--checkpoint",    type=str, default="checkpoints/best_protonet.pt")
    p.add_argument("--mnistm_csv",    type=str, default="checkpoints/gridsearch_mnistm.csv")
    p.add_argument("--svhn_csv",      type=str, default="checkpoints/gridsearch_svhn.csv")
    p.add_argument("--n_way",         type=int, default=10)
    p.add_argument("--n_query",       type=int, default=10)
    p.add_argument("--n_episodes",    type=int, default=200,
                   help="Episodes per config (more than gridsearch for reliability)")
    p.add_argument("--k_eval",        type=int, default=16,
                   help="K value to evaluate at (default: 16)")
    p.add_argument("--top_n",         type=int, default=10)
    p.add_argument("--save_dir",      type=str, default="checkpoints")
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--dataset",       type=str, default="both",
                   choices=["mnistm", "svhn", "both"])
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device     : {device}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Config     : {args.n_way}-way, K={args.k_eval}, "
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
    print("\nLoading datasets ...")
    _, _, _             = load_mnist()
    svhn_val, _         = load_svhn()
    mnistm_val, _       = load_mnistm()

    save_dir = Path(args.save_dir)

    # ── MNIST-M ──
    if args.dataset in ("mnistm", "both"):
        csv_path = Path(args.mnistm_csv)
        if not csv_path.exists():
            print(f"\n  WARNING: {csv_path} not found, skipping MNIST-M."
                  f"\n  Run gridsearch_preprocess.py first.")
        else:
            configs = load_top_configs(csv_path, args.top_n)
            print(f"\nLoaded {len(configs)} configs from {csv_path}")
            mnistm_class_map = _build_class_map(mnistm_val)
            run_top_evaluation(
                name       = "MNIST-M",
                encoder    = encoder,
                dataset    = mnistm_val,
                class_map  = mnistm_class_map,
                base_fn    = preprocess_mnistm,
                configs    = configs,
                k_shot     = args.k_eval,
                n_query    = args.n_query,
                n_way      = args.n_way,
                n_episodes = args.n_episodes,
                device     = device,
                seed       = args.seed,
                save_path  = save_dir / "top_configs_mnistm.csv",
            )

    # ── SVHN ──
    if args.dataset in ("svhn", "both"):
        csv_path = Path(args.svhn_csv)
        if not csv_path.exists():
            print(f"\n  WARNING: {csv_path} not found, skipping SVHN."
                  f"\n  Run gridsearch_preprocess.py first.")
        else:
            configs = load_top_configs(csv_path, args.top_n)
            print(f"\nLoaded {len(configs)} configs from {csv_path}")
            svhn_class_map = _build_class_map(svhn_val)
            run_top_evaluation(
                name       = "SVHN",
                encoder    = encoder,
                dataset    = svhn_val,
                class_map  = svhn_class_map,
                base_fn    = preprocess_svhn,
                configs    = configs,
                k_shot     = args.k_eval,
                n_query    = args.n_query,
                n_way      = args.n_way,
                n_episodes = args.n_episodes,
                device     = device,
                seed       = args.seed,
                save_path  = save_dir / "top_configs_svhn.csv",
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
