"""
Final comparison: ProtoNet vs MAML vs ProtoMAML.

Tests all three models on test episodes across three domains
(MNIST, MNIST-M, SVHN) with K in {1,2,4,8,16} and Q=10.

Generates:
  - One accuracy-vs-K plot per domain (3 panels), all 3 models on same plot
  - t-SNE scatter plots comparing embeddings from each model

PDF requirement (section 4.2.4):
  "Para los tres modelos, en test episodico sobre los tres dominios
   (MNIST, MNIST-M, SVHN) con K in {1,2,4,8,16} y Q = 10."
"""

import argparse
import random
from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.manifold import TSNE

from models.protonet import ConvNetEncoder, compute_centroids, squared_euclidean_distance
from models.maml import MAMLModel, inner_loop, init_head_from_prototypes
from data.prepare_datasets import load_mnist, load_svhn, load_mnistm
from data.episodic_sampler import _build_class_map
from data.preprocess import preprocess_mnistm, preprocess_svhn


# -- ProtoNet assessment (distance-based) ------------------------------------

def assess_protonet_episode(
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
    """Single ProtoNet episode: classify queries by nearest centroid."""
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
    return (preds == q_labels).float().mean().item()


# -- MAML / ProtoMAML assessment (adapt then classify) -----------------------

def assess_maml_episode(
    model: MAMLModel,
    dataset,
    class_map: dict[int, list[int]],
    k_shot: int,
    n_query: int,
    n_way: int,
    device: torch.device,
    rng: random.Random,
    inner_lr: float,
    inner_steps: int,
    use_proto_init: bool,
    preprocess=None,
) -> float:
    """
    Single MAML/ProtoMAML episode: adapt on support, classify queries.

    The linear head is re-initialised to n_way outputs for each episode
    since n_way at test may differ from training n_way.
    """
    classes = rng.sample(sorted(class_map.keys()), n_way)
    support_imgs, support_labels = [], []
    query_imgs, query_labels = [], []

    for local_label, cls in enumerate(classes):
        pool = class_map[cls]
        chosen = rng.sample(pool, k_shot + n_query)
        for idx in chosen[:k_shot]:
            img, _ = dataset[idx]
            if preprocess is not None:
                img = preprocess(img)
            support_imgs.append(img)
            support_labels.append(local_label)
        for idx in chosen[k_shot:]:
            img, _ = dataset[idx]
            if preprocess is not None:
                img = preprocess(img)
            query_imgs.append(img)
            query_labels.append(local_label)

    support_t = torch.stack(support_imgs).to(device)
    s_labels  = torch.tensor(support_labels, device=device)
    query_t   = torch.stack(query_imgs).to(device)
    q_labels  = torch.tensor(query_labels, device=device)

    # Re-initialise head to match episode n_way (may differ from training)
    if model.head.out_features != n_way:
        model.head = torch.nn.Linear(model.hidden_dim, n_way).to(device)

    params = model.get_params()

    if use_proto_init:
        params = init_head_from_prototypes(
            model, support_t, s_labels, params, n_way, k_shot
        )

    # Inner loop adaptation
    adapted = inner_loop(model, support_t, s_labels, params, inner_lr, inner_steps)

    # Classify queries with adapted parameters
    with torch.no_grad():
        q_logits = model.forward_with_params(query_t, adapted)
        preds = q_logits.argmax(dim=1)

    return (preds == q_labels).float().mean().item()


# -- Generic runner across K values -----------------------------------------

def run_model_on_dataset(
    run_fn,
    dataset,
    k_values: list[int],
    n_query: int,
    n_episodes: int,
    n_way: int,
    device: torch.device,
    seed: int,
    dataset_name: str,
    **run_kwargs,
) -> dict[int, float]:
    """Run a model on one dataset across all K values."""
    class_map = _build_class_map(dataset)

    needed = max(k_values) + n_query
    short = {c: len(idxs) for c, idxs in class_map.items() if len(idxs) < needed}
    if short:
        raise ValueError(
            f"[{dataset_name}] Some classes have fewer than {needed} samples: {short}"
        )

    results: dict[int, float] = {}
    for k in k_values:
        rng = random.Random(seed)
        accs = [
            run_fn(
                dataset=dataset, class_map=class_map,
                k_shot=k, n_query=n_query, n_way=n_way,
                device=device, rng=rng, **run_kwargs,
            )
            for _ in range(n_episodes)
        ]
        mean_acc = sum(accs) / len(accs)
        results[k] = mean_acc
        print(f"    K={k:>2}  ->  {mean_acc*100:.2f}%")

    return results


# -- Plotting ----------------------------------------------------------------

MODEL_COLOURS = {"ProtoNet": "#4C72B0", "MAML": "#DD8452", "ProtoMAML": "#55A868"}
MODEL_MARKERS = {"ProtoNet": "o",       "MAML": "s",        "ProtoMAML": "^"}


def plot_comparison(
    all_results: dict[str, dict[str, dict[int, float]]],
    k_values: list[int],
    save_dir: Path,
    n_query: int,
    n_way: int,
) -> None:
    """
    One panel per domain, all 3 models on same plot.

    all_results[model_name][domain_name][k] = accuracy
    """
    domains = ["MNIST", "MNIST-M", "SVHN"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for ax, domain in zip(axes, domains):
        for model_name in all_results:
            if domain not in all_results[model_name]:
                continue
            acc_by_k = all_results[model_name][domain]
            ks = list(k_values)
            accs = [acc_by_k[k] * 100 for k in ks]
            ax.plot(
                ks, accs,
                marker=MODEL_MARKERS.get(model_name, "o"),
                linewidth=2, markersize=7,
                color=MODEL_COLOURS.get(model_name, "gray"),
                label=model_name,
            )

        ax.set_xscale("log", base=2)
        ax.set_xticks(k_values)
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.set_xlabel("K (shots per class)", fontsize=12)
        ax.set_title(domain, fontsize=14)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=10)

    axes[0].set_ylabel("Accuracy (%)", fontsize=12)
    axes[0].set_ylim(0, 105)

    fig.suptitle(
        f"ProtoNet vs MAML vs ProtoMAML -- {n_way}-way, Q={n_query}",
        fontsize=15, y=1.02,
    )
    fig.tight_layout()
    path = save_dir / "comparison_protonet_maml_protomaml.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nComparison plot saved to: {path}")


def plot_tsne_comparison(
    encoders: dict[str, ConvNetEncoder],
    mnist_test,
    mnistm_test,
    n_per_class: int,
    device: torch.device,
    save_dir: Path,
    seed: int = 42,
) -> None:
    """
    t-SNE scatter: one column per model, rows = MNIST and MNIST-M overlaid.
    """
    from matplotlib.lines import Line2D

    n_models = len(encoders)
    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 6))
    if n_models == 1:
        axes = [axes]

    n_classes = 10
    mnist_cmap = plt.cm.get_cmap("tab10", n_classes)

    for ax, (model_name, encoder) in zip(axes, encoders.items()):
        encoder.eval()  # type: ignore[reportAttributeAccessIssue]
        rng = random.Random(seed)
        class_map_mnist  = _build_class_map(mnist_test)
        class_map_mnistm = _build_class_map(mnistm_test)

        imgs_m, lbls_m = [], []
        for cls, indices in sorted(class_map_mnist.items()):
            chosen = rng.sample(indices, min(n_per_class, len(indices)))
            for idx in chosen:
                img, _ = mnist_test[idx]
                imgs_m.append(img)
                lbls_m.append(cls)

        imgs_mm, lbls_mm = [], []
        rng2 = random.Random(seed)
        for cls, indices in sorted(class_map_mnistm.items()):
            chosen = rng2.sample(indices, min(n_per_class, len(indices)))
            for idx in chosen:
                img, _ = mnistm_test[idx]
                img = preprocess_mnistm(img)
                imgs_mm.append(img)
                lbls_mm.append(cls)

        with torch.no_grad():
            emb_m  = encoder(torch.stack(imgs_m).to(device)).cpu().numpy()
            emb_mm = encoder(torch.stack(imgs_mm).to(device)).cpu().numpy()

        all_emb = np.concatenate([emb_m, emb_mm])
        xy = TSNE(n_components=2, random_state=seed, perplexity=30).fit_transform(all_emb)
        xy_m  = xy[:len(emb_m)]
        xy_mm = xy[len(emb_m):]

        lbls_m  = np.array(lbls_m)
        lbls_mm = np.array(lbls_mm)

        for cls in range(n_classes):
            c = mnist_cmap(cls)
            mask_m  = lbls_m == cls
            mask_mm = lbls_mm == cls
            ax.scatter(xy_m[mask_m, 0], xy_m[mask_m, 1],
                       color=c, marker="o", s=18, alpha=0.6, linewidths=0)
            ax.scatter(xy_mm[mask_mm, 0], xy_mm[mask_mm, 1],
                       color=c, marker="^", s=18, alpha=0.6, linewidths=0)

        ax.set_title(model_name, fontsize=13)
        ax.set_xticks([])
        ax.set_yticks([])

    # Shared legend
    handles = [
        Line2D([], [], linestyle="none", marker="o", color="gray",
               markersize=7, label="MNIST"),
        Line2D([], [], linestyle="none", marker="^", color="gray",
               markersize=7, label="MNIST-M"),
    ]
    axes[-1].legend(handles=handles, fontsize=10, loc="lower right")

    fig.suptitle("t-SNE Embeddings: MNIST vs MNIST-M", fontsize=14, y=1.02)
    fig.tight_layout()
    path = save_dir / "tsne_comparison_all_models.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"t-SNE comparison saved to: {path}")


# -- Model loaders -----------------------------------------------------------

def load_protonet_encoder(ckpt_path: Path, device: torch.device) -> ConvNetEncoder:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    hidden_dim = ckpt["args"].get("hidden_dim", 64)
    encoder = ConvNetEncoder(in_channels=1, hidden_dim=hidden_dim).to(device)
    encoder.load_state_dict(ckpt["model"])
    encoder.eval()  # type: ignore[reportAttributeAccessIssue]
    print(f"  Loaded ProtoNet from epoch {ckpt['epoch']}  "
          f"(val acc: {ckpt['val_q_acc']*100:.2f}%)")
    return encoder


def load_maml_model(ckpt_path: Path, device: torch.device, n_way: int = 5) -> MAMLModel:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    hidden_dim = ckpt["args"].get("hidden_dim", 64)
    train_n_way = ckpt["args"].get("n_way", 5)
    model = MAMLModel(hidden_dim=hidden_dim, n_way=train_n_way).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()  # type: ignore[reportAttributeAccessIssue]
    print(f"  Loaded {ckpt.get('model_type', 'MAML')} from epoch {ckpt['epoch']}  "
          f"(val acc: {ckpt['val_q_acc']*100:.2f}%)")
    return model


# -- Main --------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Final comparison: ProtoNet vs MAML vs ProtoMAML"
    )
    p.add_argument("--ckpt_protonet",  type=str, default="checkpoints/best_protonet.pt")
    p.add_argument("--ckpt_maml",      type=str, default="checkpoints/best_maml.pt")
    p.add_argument("--ckpt_protomaml", type=str, default="checkpoints/best_protomaml.pt")
    p.add_argument("--n_way",          type=int, default=10)
    p.add_argument("--n_query",        type=int, default=10,
                   help="Q=10 for final comparison (PDF requirement)")
    p.add_argument("--n_episodes",     type=int, default=200)
    p.add_argument("--k_values",       type=int, nargs="+", default=[1, 2, 4, 8, 16])
    p.add_argument("--inner_lr",       type=float, default=0.01)
    p.add_argument("--inner_steps",    type=int, default=5)
    p.add_argument("--n_embed",        type=int, default=20,
                   help="Images per class for t-SNE")
    p.add_argument("--save_dir",       type=str, default="checkpoints")
    p.add_argument("--seed",           type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device : {device}")
    print(f"Config : {args.n_way}-way | K = {args.k_values} | Q = {args.n_query} | "
          f"{args.n_episodes} episodes\n")

    # -- Load datasets --
    print("Loading datasets ...")
    _, _, mnist_test       = load_mnist()
    _, svhn_test           = load_svhn()
    _, _, mnistm_test      = load_mnistm()
    print()

    test_datasets_info = {
        "MNIST":   (mnist_test,  None),
        "MNIST-M": (mnistm_test, preprocess_mnistm),
        "SVHN":    (svhn_test,   preprocess_svhn),
    }

    # -- Collect results for all models --
    all_results: dict[str, dict[str, dict[int, float]]] = {}
    encoders_for_tsne: dict[str, ConvNetEncoder] = {}

    # --- ProtoNet ---
    ckpt_path = Path(args.ckpt_protonet)
    if ckpt_path.exists():
        print(f"Loading ProtoNet from {ckpt_path}")
        encoder = load_protonet_encoder(ckpt_path, device)
        encoders_for_tsne["ProtoNet"] = encoder

        all_results["ProtoNet"] = {}
        for domain, (dataset, preprocess) in test_datasets_info.items():
            print(f"  [{domain}]")
            all_results["ProtoNet"][domain] = run_model_on_dataset(
                run_fn=assess_protonet_episode,
                dataset=dataset, k_values=args.k_values,
                n_query=args.n_query, n_episodes=args.n_episodes,
                n_way=args.n_way, device=device, seed=args.seed,
                dataset_name=domain, encoder=encoder, preprocess=preprocess,
            )
        print()
    else:
        print(f"WARNING: {ckpt_path} not found, skipping ProtoNet.\n")

    # --- MAML ---
    ckpt_path = Path(args.ckpt_maml)
    if ckpt_path.exists():
        print(f"Loading MAML from {ckpt_path}")
        maml_model = load_maml_model(ckpt_path, device, n_way=args.n_way)
        encoders_for_tsne["MAML"] = maml_model.encoder

        all_results["MAML"] = {}
        for domain, (dataset, preprocess) in test_datasets_info.items():
            print(f"  [{domain}]")
            all_results["MAML"][domain] = run_model_on_dataset(
                run_fn=assess_maml_episode,
                dataset=dataset, k_values=args.k_values,
                n_query=args.n_query, n_episodes=args.n_episodes,
                n_way=args.n_way, device=device, seed=args.seed,
                dataset_name=domain, model=maml_model,
                inner_lr=args.inner_lr, inner_steps=args.inner_steps,
                use_proto_init=False, preprocess=preprocess,
            )
        print()
    else:
        print(f"WARNING: {ckpt_path} not found, skipping MAML.\n")

    # --- ProtoMAML ---
    ckpt_path = Path(args.ckpt_protomaml)
    if ckpt_path.exists():
        print(f"Loading ProtoMAML from {ckpt_path}")
        protomaml_model = load_maml_model(ckpt_path, device, n_way=args.n_way)
        encoders_for_tsne["ProtoMAML"] = protomaml_model.encoder

        all_results["ProtoMAML"] = {}
        for domain, (dataset, preprocess) in test_datasets_info.items():
            print(f"  [{domain}]")
            all_results["ProtoMAML"][domain] = run_model_on_dataset(
                run_fn=assess_maml_episode,
                dataset=dataset, k_values=args.k_values,
                n_query=args.n_query, n_episodes=args.n_episodes,
                n_way=args.n_way, device=device, seed=args.seed,
                dataset_name=domain, model=protomaml_model,
                inner_lr=args.inner_lr, inner_steps=args.inner_steps,
                use_proto_init=True, preprocess=preprocess,
            )
        print()
    else:
        print(f"WARNING: {ckpt_path} not found, skipping ProtoMAML.\n")

    if not all_results:
        print("No models found. Train them first:")
        print("  python -m models.protonet --dataset mnist")
        print("  python -m models.maml --model both")
        return

    # -- Summary table --
    print("\n" + "=" * 70)
    print("  FINAL COMPARISON")
    print("=" * 70)
    for domain in ["MNIST", "MNIST-M", "SVHN"]:
        print(f"\n  {domain}")
        header = f"  {'Model':<12}" + "".join(f"  K={k:<5}" for k in args.k_values)
        print(header)
        print("  " + "-" * (len(header) - 2))
        for model_name, domain_results in all_results.items():
            if domain in domain_results:
                row = f"  {model_name:<12}" + "".join(
                    f"  {domain_results[domain][k]*100:>5.1f}%" for k in args.k_values
                )
                print(row)

    # -- Plots --
    plot_comparison(all_results, args.k_values, save_dir, args.n_query, args.n_way)

    # -- t-SNE --
    if len(encoders_for_tsne) > 0:
        print("\nGenerating t-SNE comparison ...")
        plot_tsne_comparison(
            encoders_for_tsne, mnist_test, mnistm_test,
            n_per_class=args.n_embed, device=device,
            save_dir=save_dir, seed=args.seed,
        )

    print("\nAll done.")


if __name__ == "__main__":
    main()
