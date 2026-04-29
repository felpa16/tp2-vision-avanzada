"""
Cross-domain evaluation of trained ProtoNet models.

Evaluates two models:
  1. Encoder trained on MNIST        (best_protonet.pt)
  2. Encoder trained on MNIST-M      (best_protonet_mnistm.pt)

For each model and each dataset (MNIST, MNIST-M, SVHN) and each K in
{1,2,4,8,16}:
  1. Build a support set of K images per class (10 classes, so N=10-way).
  2. Compute class centroids from the support embeddings.
  3. Sample 10 query images per class, classify by nearest centroid.
  4. Record accuracy.

Results are averaged over a number of episodes (--n_episodes) to reduce
sampling variance, then plotted as accuracy vs K with one line per dataset.

Saved artefacts
---------------
  checkpoints/cross_domain_accuracy_mnist.png
  checkpoints/cross_domain_accuracy_mnistm.png
  checkpoints/embedding_scatter_mnist.png
  checkpoints/embedding_scatter_mnistm.png
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.manifold import TSNE

from models.protonet import ConvNetEncoder, compute_centroids, squared_euclidean_distance
from data.prepare_datasets import load_mnist, load_svhn, load_mnistm
from data.episodic_sampler import _build_class_map
from data.preprocess import preprocess_mnistm, preprocess_svhn

# ── Evaluation helpers ────────────────────────────────────────────────────────

def evaluate_episode(
    encoder: ConvNetEncoder,
    dataset,
    class_map: dict[int, list[int]],
    k_shot: int,
    n_query: int,
    n_way: int,
    device: torch.device,
    rng: random.Random,
    preprocess=None,
) -> float:
    """
    Run a single N-way K-shot episode and return classification accuracy.

    Support and query sets are sampled without replacement so they never
    share images. Query images are drawn separately after the support set.

    preprocess : optional callable applied to each image tensor (C, H, W)
                 before stacking — used to convert RGB/32×32 images from
                 SVHN and MNIST-M to the grayscale 28×28 format the encoder
                 was trained on.
    """
    classes = rng.sample(sorted(class_map.keys()), n_way)

    support_imgs, query_imgs, query_labels = [], [], []

    for local_label, cls in enumerate(classes):
        pool   = class_map[cls]
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

    support_t = torch.stack(support_imgs).to(device)   # (N*K, 1, 28, 28)
    query_t   = torch.stack(query_imgs).to(device)     # (N*Q, 1, 28, 28)
    q_labels  = torch.tensor(query_labels, device=device)

    with torch.no_grad():
        all_emb     = encoder(torch.cat([support_t, query_t], dim=0))
        support_emb = all_emb[:n_way * k_shot]
        query_emb   = all_emb[n_way * k_shot:]

    centroids = compute_centroids(support_emb, n_way, k_shot)
    dists     = squared_euclidean_distance(query_emb, centroids)
    preds     = (-dists).argmax(dim=1)

    return (preds == q_labels).float().mean().item()


def evaluate_dataset(
    encoder: ConvNetEncoder,
    dataset,
    k_values: list[int],
    n_query: int,
    n_episodes: int,
    n_way: int,
    device: torch.device,
    seed: int,
    dataset_name: str,
    preprocess=None,
) -> dict[int, float]:
    """
    Evaluate the encoder on one dataset across all K values.

    For each K, runs `n_episodes` episodes and returns the mean accuracy.

    preprocess : optional transform applied per image before encoding —
                 pass the RGB→grayscale+resize pipeline for SVHN and MNIST-M.

    Returns {k: mean_accuracy} for each k in k_values.
    """
    class_map = _build_class_map(dataset)

    # Verify every class has enough samples for the largest K + n_query
    needed = max(k_values) + n_query
    short  = {c: len(idxs) for c, idxs in class_map.items() if len(idxs) < needed}
    if short:
        raise ValueError(
            f"[{dataset_name}] Some classes have fewer than {needed} samples "
            f"(max_k={max(k_values)} + n_query={n_query}): {short}"
        )
    if n_way > len(class_map):
        raise ValueError(
            f"[{dataset_name}] n_way={n_way} exceeds available classes "
            f"({len(class_map)})."
        )

    results: dict[int, float] = {}

    for k in k_values:
        rng  = random.Random(seed)   # same seed per K for reproducibility
        accs = [
            evaluate_episode(
                encoder, dataset, class_map,
                k_shot=k, n_query=n_query, n_way=n_way,
                device=device, rng=rng, preprocess=preprocess,
            )
            for _ in range(n_episodes)
        ]
        mean_acc = sum(accs) / len(accs)
        results[k] = mean_acc
        print(f"  K={k:>2}  →  {mean_acc*100:.2f}%")

    return results


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_cross_domain(
    results: dict[str, dict[int, float]],
    k_values: list[int],
    save_path: Path,
    n_query: int,
    n_way: int,
    model_name: str = "",
) -> None:
    """
    Line plot of accuracy vs K with one line per dataset.

    X-axis ticks are placed at the exact K values used (1,2,4,8,16) since
    the spacing is non-uniform.
    """
    colours = {"MNIST": "#4C72B0", "MNIST-M": "#DD8452", "SVHN": "#55A868"}
    markers = {"MNIST": "o",       "MNIST-M": "s",        "SVHN": "^"}

    fig, ax = plt.subplots(figsize=(8, 5))

    for dataset_name, acc_by_k in results.items():
        ks   = [k for k in k_values]
        accs = [acc_by_k[k] * 100 for k in k_values]
        ax.plot(
            ks, accs,
            marker    = markers[dataset_name],
            linewidth = 2,
            markersize= 7,
            color     = colours[dataset_name],
            label     = dataset_name,
        )

    ax.set_xscale("log", base=2)
    ax.set_xticks(k_values)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel("K  (support images per class)", fontsize=13)
    ax.set_ylabel("Accuracy (%)", fontsize=13)
    ax.set_ylim(0, 100)
    title = f"Cross-Domain ProtoNet Accuracy"
    if model_name:
        title += f" — trained on {model_name}"
    title += f"\n{n_way}-way,  {n_query} query images per class"
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved to: {save_path}")


def collect_domain_embeddings(
    encoder: ConvNetEncoder,
    dataset,
    n_per_class: int,
    device: torch.device,
    preprocess=None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample `n_per_class` images from every class in `dataset`, encode them,
    and return (embeddings, labels) as numpy arrays.

    preprocess : optional transform applied per image (used for RGB→gray
                 conversion on MNIST-M before encoding).
    """
    rng       = random.Random(seed)
    class_map = _build_class_map(dataset)
    imgs, lbls = [], []

    for cls, indices in sorted(class_map.items()):
        chosen = rng.sample(indices, min(n_per_class, len(indices)))
        for idx in chosen:
            img, _ = dataset[idx]
            if preprocess is not None:
                img = preprocess(img)
            imgs.append(img)
            lbls.append(cls)

    imgs_t = torch.stack(imgs).to(device)
    encoder.eval()
    with torch.no_grad():
        emb = encoder(imgs_t).cpu().numpy()

    return emb, np.array(lbls)


def plot_embedding_scatter(
    mnist_emb:   np.ndarray,
    mnist_lbl:   np.ndarray,
    mnistm_emb:  np.ndarray,
    mnistm_lbl:  np.ndarray,
    save_path:   Path,
    seed:        int = 0,
) -> None:
    """
    t-SNE scatter plot of MNIST and MNIST-M embeddings.

    Colour encodes class (0-9); colour family encodes domain:
      MNIST   → Blues/Purples  (cool tones)
      MNIST-M → Oranges/Reds   (warm tones)

    The legend shows two columns — one per domain — with one coloured
    swatch per class in each column, so every domain-class pair has its
    own explicit colour indicator.
    """
    from matplotlib.lines import Line2D

    n_classes = 10

    # One colour per class from each colormap
    mnist_cmap  = plt.cm.get_cmap("cool",   n_classes)   # cyan → magenta
    mnistm_cmap = plt.cm.get_cmap("autumn", n_classes)   # red  → yellow

    # Fit t-SNE on concatenated embeddings so both domains share one space
    all_emb   = np.concatenate([mnist_emb, mnistm_emb], axis=0)
    xy        = TSNE(n_components=2, random_state=seed, perplexity=30).fit_transform(all_emb)
    xy_mnist  = xy[:len(mnist_emb)]
    xy_mnistm = xy[len(mnist_emb):]

    fig, ax = plt.subplots(figsize=(11, 7))

    for cls in range(n_classes):
        c_mnist  = mnist_cmap (cls / (n_classes - 1))
        c_mnistm = mnistm_cmap(cls / (n_classes - 1))

        mask_m  = mnist_lbl  == cls
        mask_mm = mnistm_lbl == cls

        ax.scatter(xy_mnist [mask_m,  0], xy_mnist [mask_m,  1],
                   color=c_mnist,  marker="o", s=22, alpha=0.75, linewidths=0)
        ax.scatter(xy_mnistm[mask_mm, 0], xy_mnistm[mask_mm, 1],
                   color=c_mnistm, marker="^", s=22, alpha=0.75, linewidths=0)

    # ── Legend: two columns, one per domain ──
    # Column headers (bold domain labels, no marker)
    header_mnist  = Line2D([], [], linestyle="None", marker="None",
                           label="MNIST (●)")
    header_mnistm = Line2D([], [], linestyle="None", marker="None",
                           label="MNIST-M (▲)")

    mnist_handles, mnistm_handles = [header_mnist], [header_mnistm]

    for cls in range(n_classes):
        c_mnist  = mnist_cmap (cls / (n_classes - 1))
        c_mnistm = mnistm_cmap(cls / (n_classes - 1))

        mnist_handles.append(Line2D(
            [], [], linestyle="none", marker="o", markersize=7,
            color=c_mnist,  label=f"Class {cls}",
        ))
        mnistm_handles.append(Line2D(
            [], [], linestyle="none", marker="^", markersize=7,
            color=c_mnistm, label=f"Class {cls}",
        ))

    # Place both columns side by side outside the axes
    leg_mnist = ax.legend(
        handles          = mnist_handles,
        loc              = "upper left",
        bbox_to_anchor   = (1.01, 1.0),
        borderaxespad    = 0,
        fontsize         = 8,
        title_fontsize   = 9,
        framealpha       = 0.8,
    )
    # Manually bold the header entry
    leg_mnist.get_texts()[0].set_fontweight("bold")
    ax.add_artist(leg_mnist)

    leg_mnistm = ax.legend(
        handles          = mnistm_handles,
        loc              = "upper left",
        bbox_to_anchor   = (1.22, 1.0),
        borderaxespad    = 0,
        fontsize         = 8,
        title_fontsize   = 9,
        framealpha       = 0.8,
    )
    leg_mnistm.get_texts()[0].set_fontweight("bold")

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        "t-SNE of Encoder Embeddings — MNIST vs MNIST-M\n"
        "(cool tones = MNIST, warm tones = MNIST-M)",
        fontsize=13,
    )

    fig.tight_layout(rect=[0, 0, 0.78, 1])   # leave room for the two legend columns
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"Embedding scatter plot saved to: {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def load_encoder(ckpt_path: Path, device: torch.device) -> ConvNetEncoder:
    """Load an encoder from a checkpoint file."""
    ckpt       = torch.load(ckpt_path, map_location=device)
    hidden_dim = ckpt["args"].get("hidden_dim", 64)
    encoder    = ConvNetEncoder(in_channels=1, hidden_dim=hidden_dim).to(device)
    encoder.load_state_dict(ckpt["model"])
    encoder.eval()
    print(f"  Loaded from epoch {ckpt['epoch']}  "
          f"(val query acc: {ckpt['val_q_acc']*100:.2f}%)")
    return encoder


def run_evaluation(
    model_name: str,
    encoder: ConvNetEncoder,
    test_datasets: dict,
    args,
    device: torch.device,
    save_dir: Path,
    mnist_test,
    mnistm_test,
) -> dict[str, dict[int, float]]:
    """Run cross-domain evaluation for a single encoder."""
    print(f"\n{'='*60}")
    print(f"  Evaluating encoder trained on {model_name}")
    print(f"{'='*60}\n")

    all_results: dict[str, dict[int, float]] = {}

    for name, (dataset, preprocess) in test_datasets.items():
        print(f"[{name}]")
        all_results[name] = evaluate_dataset(
            encoder      = encoder,
            dataset      = dataset,
            k_values     = args.k_values,
            n_query      = args.n_query,
            n_episodes   = args.n_episodes,
            n_way        = args.n_way,
            device       = device,
            seed         = args.seed,
            dataset_name = name,
            preprocess   = preprocess,
        )
        print()

    # ── Summary table ──
    col_w = 10
    header = f"{'Dataset':<10}" + "".join(f"  K={k:<{col_w-3}}" for k in args.k_values)
    print(header)
    print("-" * len(header))
    for name, acc_by_k in all_results.items():
        row = f"{name:<10}" + "".join(
            f"  {acc_by_k[k]*100:>{col_w-2}.2f}%" for k in args.k_values
        )
        print(row)

    # ── Plot: accuracy vs K ──
    suffix = model_name.lower().replace("-", "")
    plot_cross_domain(
        results    = all_results,
        k_values   = args.k_values,
        save_path  = save_dir / f"cross_domain_accuracy_{suffix}.png",
        n_query    = args.n_query,
        n_way      = args.n_way,
        model_name = model_name,
    )

    # ── Plot: embedding scatter (MNIST vs MNIST-M) ──
    print(f"\nCollecting embeddings for scatter plot ({args.n_embed} images/class) ...")
    mnist_emb,  mnist_lbl  = collect_domain_embeddings(
        encoder, mnist_test,  args.n_embed, device,
        preprocess=None, seed=args.seed,
    )
    mnistm_emb, mnistm_lbl = collect_domain_embeddings(
        encoder, mnistm_test, args.n_embed, device,
        preprocess=preprocess_mnistm, seed=args.seed,
    )
    plot_embedding_scatter(
        mnist_emb, mnist_lbl, mnistm_emb, mnistm_lbl,
        save_path = save_dir / f"embedding_scatter_{suffix}.png",
        seed      = args.seed,
    )

    return all_results


def parse_args():
    p = argparse.ArgumentParser(
        description="Cross-domain ProtoNet evaluation (MNIST / MNIST-M / SVHN)"
    )
    p.add_argument("--checkpoint_mnist",  type=str,
                   default="checkpoints/best_protonet.pt",
                   help="Checkpoint for the model trained on MNIST")
    p.add_argument("--checkpoint_mnistm", type=str,
                   default="checkpoints/best_protonet_mnistm.pt",
                   help="Checkpoint for the model trained on MNIST-M")
    p.add_argument("--n_way",       type=int, default=10,
                   help="Number of classes per episode (default: 10 -- all MNIST classes)")
    p.add_argument("--n_query",     type=int, default=10,
                   help="Query images per class per episode")
    p.add_argument("--n_episodes",  type=int, default=100,
                   help="Episodes per (dataset, K) pair -- more = lower variance")
    p.add_argument("--k_values",    type=int, nargs="+", default=[1, 2, 4, 8, 16])
    p.add_argument("--n_embed",     type=int, default=20,
                   help="Images per class used for the embedding scatter plot")
    p.add_argument("--save_dir",    type=str, default="checkpoints")
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)

    print(f"Device : {device}")
    print(f"Config : {args.n_way}-way  |  K = {args.k_values}  |  "
          f"{args.n_query} queries/class  |  {args.n_episodes} episodes\n")

    # ── Load datasets ──
    print("Loading datasets ...")
    _, mnist_val, mnist_test     = load_mnist()
    svhn_val,     svhn_test      = load_svhn()
    _, mnistm_val, mnistm_test   = load_mnistm()
    print()

    # SVHN and MNIST-M images are RGB 32x32, preprocessed to grayscale 28x28.
    test_datasets = {
        "MNIST":   (mnist_test,  None),
        "MNIST-M": (mnistm_test, preprocess_mnistm),
        "SVHN":    (svhn_test,   preprocess_svhn),
    }

    # ── Evaluate each model ──
    models = [
        ("MNIST",   Path(args.checkpoint_mnist)),
        ("MNIST-M", Path(args.checkpoint_mnistm)),
    ]

    for model_name, ckpt_path in models:
        if not ckpt_path.exists():
            ds_flag = "mnist" if model_name == "MNIST" else "mnistm"
            print(f"\nWARNING: {ckpt_path} not found -- skipping {model_name} model.")
            print(f"  Train with: python -m models.protonet --dataset {ds_flag}")
            continue

        print(f"\nLoading {model_name} encoder from {ckpt_path}")
        encoder = load_encoder(ckpt_path, device)

        run_evaluation(
            model_name     = model_name,
            encoder        = encoder,
            test_datasets  = test_datasets,
            args           = args,
            device         = device,
            save_dir       = save_dir,
            mnist_test     = mnist_test,
            mnistm_test    = mnistm_test,
        )

    print("\nAll evaluations complete.")


if __name__ == "__main__":
    main()