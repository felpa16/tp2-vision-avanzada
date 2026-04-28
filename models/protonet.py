"""
ProtoNet training on MNIST with a 4-block ConvNet encoder.

Episode config : 5-way 5-shot 15-query (train & val)

Encoder        : 4 × (Conv2d → ReLU → MaxPool2d)
                 Input  1×28×28  →  embedding  64-d

Distance metric: negative squared Euclidean distance
Loss           : cross-entropy over query logits (backprop on query only)

Saved artefacts
---------------
  checkpoints/best_protonet.pt       — best val-query-acc checkpoint
  checkpoints/val_query_accuracy.png — val query acc vs epoch
  checkpoints/tsne_embeddings.png    — 1×3 t-SNE grid (pre / mid / final)
"""

import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.manifold import TSNE

from data.prepare_datasets import load_mnist
from data.episodic_sampler import build_episode_loader, _build_class_map


# ── Encoder ───────────────────────────────────────────────────────────────────

def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """Conv → ReLU → MaxPool  (no BatchNorm — keeps train/eval behaviour identical)."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class ConvNetEncoder(nn.Module):
    """
    4-block convolutional encoder.

    MNIST input (1×28×28) passes through four Conv blocks, each halving
    the spatial dimensions via MaxPool:
        28 → 14 → 7 → 3 → 1   (floor division at each pool)
    Output: a 64-dimensional embedding vector per image.
    """

    def __init__(self, in_channels: int = 1, hidden_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(in_channels, hidden_dim),   # (B,  1, 28, 28) → (B, 64, 14, 14)
            conv_block(hidden_dim,  hidden_dim),   # (B, 64, 14, 14) → (B, 64,  7,  7)
            conv_block(hidden_dim,  hidden_dim),   # (B, 64,  7,  7) → (B, 64,  3,  3)
            conv_block(hidden_dim,  hidden_dim),   # (B, 64,  3,  3) → (B, 64,  1,  1)
        )
        self.embed_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W)  →  embeddings: (B, embed_dim)"""
        return self.encoder(x).view(x.size(0), -1)


# ── ProtoNet logic ─────────────────────────────────────────────────────────────

def compute_centroids(support_emb: torch.Tensor, n_way: int, k_shot: int) -> torch.Tensor:
    """
    Average the support embeddings per class to get prototypes.

    support_emb : (N*K, D)  ordered [class0×K, class1×K, …]
    returns     : (N, D)
    """
    return support_emb.view(n_way, k_shot, -1).mean(dim=1)


def squared_euclidean_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Pairwise squared Euclidean distances.

    a : (M, D),  b : (N, D)  →  returns (M, N)
    """
    diff = a.unsqueeze(1) - b.unsqueeze(0)
    return (diff ** 2).sum(dim=-1)


def proto_loss(
    support_emb: torch.Tensor,
    query_emb: torch.Tensor,
    query_labels: torch.Tensor,
    n_way: int,
    k_shot: int,
) -> tuple[torch.Tensor, float, torch.Tensor, float]:
    """
    ProtoNet forward pass for one episode.

    Returns (query_loss, query_acc, support_loss, support_acc).

    Note: support loss is optimistic — each point contributes to its own
    centroid. Use it to diagnose overfitting, not as a primary metric.
    """
    centroids = compute_centroids(support_emb, n_way, k_shot)

    # Query
    q_logits = -squared_euclidean_distance(query_emb, centroids)
    q_loss   = F.cross_entropy(q_logits, query_labels)

    # Support — labels are [0]*K + [1]*K + … + [N-1]*K
    s_labels = torch.arange(n_way, device=support_emb.device).repeat_interleave(k_shot)
    s_logits = -squared_euclidean_distance(support_emb, centroids)
    s_loss   = F.cross_entropy(s_logits, s_labels)

    with torch.no_grad():
        q_acc = (q_logits.argmax(1) == query_labels).float().mean().item()
        s_acc = (s_logits.argmax(1) == s_labels).float().mean().item()

    return q_loss, q_acc, s_loss, s_acc


# ── Training & validation ─────────────────────────────────────────────────────

def run_epoch(
    encoder: ConvNetEncoder,
    loader,
    optimizer,
    device: torch.device,
    n_way: int,
    k_shot: int,
    is_train: bool,
) -> tuple[float, float, float, float]:
    """
    One full training or validation pass.

    Returns (query_loss, query_acc, support_loss, support_acc) averaged over
    episodes. Backprop runs on query loss only.
    """
    encoder.train() if is_train else encoder.eval()

    total_q_loss = total_q_acc = 0.0
    total_s_loss = total_s_acc = 0.0
    n_episodes   = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for support, _, query, q_labels in loader:
            support  = support.squeeze(0).to(device)
            query    = query.squeeze(0).to(device)
            q_labels = q_labels.squeeze(0).to(device)

            all_emb     = encoder(torch.cat([support, query], dim=0))
            support_emb = all_emb[:n_way * k_shot]
            query_emb   = all_emb[n_way * k_shot:]

            q_loss, q_acc, s_loss, s_acc = proto_loss(
                support_emb, query_emb, q_labels, n_way, k_shot
            )

            if is_train:
                optimizer.zero_grad()
                q_loss.backward()
                optimizer.step()

            total_q_loss += q_loss.item()
            total_q_acc  += q_acc
            total_s_loss += s_loss.item()
            total_s_acc  += s_acc
            n_episodes   += 1

    return (
        total_q_loss / n_episodes,
        total_q_acc  / n_episodes,
        total_s_loss / n_episodes,
        total_s_acc  / n_episodes,
    )


# ── t-SNE helpers ─────────────────────────────────────────────────────────────

def sample_tsne_images(
    dataset,
    n_per_class: int = 100,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Draw a fixed, balanced sample of images from `dataset`.

    Uses the same indices every call (controlled by `seed`) so all three
    t-SNE snapshots visualise the exact same images for fair comparison.

    Returns (images, labels) as plain tensors.
    """
    rng       = random.Random(seed)
    class_map = _build_class_map(dataset)
    imgs, lbls = [], []

    for cls, indices in sorted(class_map.items()):
        chosen = rng.sample(indices, min(n_per_class, len(indices)))
        for idx in chosen:
            img, _ = dataset[idx]
            imgs.append(img)
            lbls.append(cls)

    return torch.stack(imgs), torch.tensor(lbls)


def collect_embeddings(
    encoder: ConvNetEncoder,
    images: torch.Tensor,
    device: torch.device,
    batch_size: int = 256,
) -> torch.Tensor:
    """
    Encode a fixed image tensor in batches without affecting training state.

    Temporarily switches to eval + no_grad, then restores the previous mode.

    images  : (N, C, H, W)
    returns : (N, D) CPU tensor
    """
    was_training = encoder.training
    encoder.eval()
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size].to(device)
            embeddings.append(encoder(batch).cpu())
    encoder.train(was_training)
    return torch.cat(embeddings, dim=0)


def run_tsne(embeddings: torch.Tensor, seed: int = 0):
    """Fit t-SNE on (N, D) embeddings; return (N, 2) 2-D projections."""
    return TSNE(n_components=2, random_state=seed, perplexity=30).fit_transform(
        embeddings.numpy()
    )


def plot_tsne_grid(
    snapshots: list[tuple[str, any, torch.Tensor]],
    save_path: Path,
) -> None:
    """
    Draw a 1×3 grid of t-SNE scatter plots, one per training checkpoint.

    snapshots : list of (title, xy, labels)
                xy     : (N, 2) numpy array of 2-D t-SNE projections
                labels : (N,)   integer class labels
    """
    n_classes = int(max(lbl.max().item() for _, _, lbl in snapshots)) + 1
    palette   = list(mcolors.TABLEAU_COLORS.values())[:n_classes]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, (title, xy, labels) in zip(axes, snapshots):
        for cls in range(n_classes):
            mask = (labels == cls).numpy()
            ax.scatter(
                xy[mask, 0], xy[mask, 1],
                s=8, alpha=0.6,
                color=palette[cls],
                label=str(cls),
            )
        ax.set_title(title, fontsize=13)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(title="Class", fontsize=7, markerscale=2,
                  loc="upper right", framealpha=0.6)

    fig.suptitle("t-SNE Projections of Validation Embeddings", fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"t-SNE plot saved to: {save_path}")


# ── Accuracy plot ─────────────────────────────────────────────────────────────

def plot_val_accuracy(
    history: list[float],
    save_path: Path,
    n_way: int,
    k_shot: int,
    n_train_episodes: int,
    worst_acc: float = 0.0
) -> None:
    epochs     = list(range(1, len(history) + 1))
    best_epoch = history.index(max(history)) + 1

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, history, marker="o", linewidth=2, markersize=4,
            color="#4C72B0", label="Val query acc")
    ax.axvline(best_epoch, linestyle="--", linewidth=1.2, color="#C44E52",
               label=f"Best epoch {best_epoch}: {history[best_epoch-1]:.2f}%")

    # ── Primary x-axis: epochs ──
    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_xlim(1, len(epochs))
    ax.set_ylim(0.9*(worst_acc * 100), 100)

    # ── Secondary x-axis: cumulative episodes ──
    # episodes = epoch × n_train_episodes  →  linear transform, no offset
    secax = ax.secondary_xaxis(
        "top",
        functions=(
            lambda epoch: epoch * n_train_episodes,
            lambda ep:    ep    / n_train_episodes,
        ),
    )
    secax.set_xlabel("Episodes", fontsize=13, labelpad=8)

    # Align secondary ticks with the primary epoch ticks
    secax.set_ticks([e * n_train_episodes for e in epochs])
    secax.set_xticklabels(
        [str(e * n_train_episodes) for e in epochs],
        fontsize=8,
        rotation=45,
        ha="left",
    )

    ax.set_ylabel("Accuracy (%)", fontsize=13)
    ax.set_title(
        f"Validation Query Accuracy — {n_way}-way {k_shot}-shot ProtoNet (MNIST)",
        fontsize=13, pad=30,   # pad leaves room for the secondary axis label
    )
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Accuracy plot saved to: {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="ProtoNet training on MNIST")
    p.add_argument("--n_way",            type=int,   default=5)
    p.add_argument("--k_shot",           type=int,   default=5)
    p.add_argument("--q_query",          type=int,   default=15)
    p.add_argument("--n_train_episodes", type=int,   default=10)
    p.add_argument("--n_val_episodes",   type=int,   default=10)
    p.add_argument("--epochs",           type=int,   default=20)
    p.add_argument("--lr",               type=float, default=1e-3)
    p.add_argument("--lr_step",          type=int,   default=10)
    p.add_argument("--lr_gamma",         type=float, default=0.5)
    p.add_argument("--hidden_dim",       type=int,   default=64)
    p.add_argument("--tsne_per_class",   type=int,   default=100,
                   help="Validation images per class used for t-SNE snapshots")
    p.add_argument("--save_dir",         type=str,   default="./checkpoints")
    p.add_argument("--seed",             type=int,   default=42)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Config : {args.n_way}-way  {args.k_shot}-shot  {args.q_query}-query\n")

    torch.manual_seed(args.seed)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ──
    mnist_train, mnist_val, _ = load_mnist()

    train_loader = build_episode_loader(
        dataset    = mnist_train,
        n_episodes = args.n_train_episodes,
        n_way      = args.n_way,
        k_shot     = args.k_shot,
        q_query    = args.q_query,
        seed       = args.seed,
    )
    val_loader = build_episode_loader(
        dataset    = mnist_val,
        n_episodes = args.n_val_episodes,
        n_way      = args.n_way,
        k_shot     = args.k_shot,
        q_query    = args.q_query,
        seed       = args.seed + 1,
    )

    # Fixed validation images for t-SNE — sampled once, reused at all checkpoints
    # so the three panels are directly comparable (same images, different encoder)
    tsne_images, tsne_labels = sample_tsne_images(
        mnist_val,
        n_per_class = args.tsne_per_class,
        seed        = args.seed,
    )
    mid_epoch      = args.epochs // 2
    tsne_snapshots = []   # (title, xy, labels) filled at epochs 0, mid, final

    # ── Model, optimiser, scheduler ──
    encoder   = ConvNetEncoder(in_channels=1, hidden_dim=args.hidden_dim).to(device)
    optimizer = Adam(encoder.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)

    total_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"Encoder parameters: {total_params:,}\n")

    # ── t-SNE snapshot: before training ──
    print("Computing t-SNE snapshot — before training …")
    emb = collect_embeddings(encoder, tsne_images, device)
    tsne_snapshots.append(("Before training (epoch 0)", run_tsne(emb, args.seed), tsne_labels))

    # ── Training loop ──
    best_val_acc      = 0.0
    val_q_acc_history = []

    header = (
        f"{'Epoch':>5}  "
        f"{'Tr Q-Loss':>9}  {'Tr Q-Acc':>8}  {'Tr S-Loss':>9}  {'Tr S-Acc':>8}  "
        f"{'Va Q-Loss':>9}  {'Va Q-Acc':>8}  {'Va S-Loss':>9}  {'Va S-Acc':>8}  "
        f"{'LR':>8}"
    )
    print(header)
    print("─" * len(header))

    for epoch in range(1, args.epochs + 1):
        tr_q_loss, tr_q_acc, tr_s_loss, tr_s_acc = run_epoch(
            encoder, train_loader, optimizer, device,
            args.n_way, args.k_shot, is_train=True,
        )
        va_q_loss, va_q_acc, va_s_loss, va_s_acc = run_epoch(
            encoder, val_loader, optimizer, device,
            args.n_way, args.k_shot, is_train=False,
        )
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        val_q_acc_history.append(va_q_acc * 100)

        print(
            f"{epoch:>5}  "
            f"{tr_q_loss:>9.4f}  {tr_q_acc*100:>7.2f}%  "
            f"{tr_s_loss:>9.4f}  {tr_s_acc*100:>7.2f}%  "
            f"{va_q_loss:>9.4f}  {va_q_acc*100:>7.2f}%  "
            f"{va_s_loss:>9.4f}  {va_s_acc*100:>7.2f}%  "
            f"{current_lr:>8.2e}"
        )

        # ── t-SNE snapshot: midpoint ──
        if epoch == mid_epoch:
            print(f"Computing t-SNE snapshot — epoch {epoch} (mid) …")
            emb = collect_embeddings(encoder, tsne_images, device)
            tsne_snapshots.append(
                (f"Epoch {epoch} (mid)", run_tsne(emb, args.seed), tsne_labels)
            )

        # ── Best checkpoint ──
        if va_q_acc > best_val_acc:
            best_val_acc = va_q_acc
            ckpt = {
                "epoch":      epoch,
                "val_q_acc":  va_q_acc,
                "val_q_loss": va_q_loss,
                "val_s_acc":  va_s_acc,
                "val_s_loss": va_s_loss,
                "model":      encoder.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "args":       vars(args),
            }
            torch.save(ckpt, save_dir / "best_protonet.pt")

    print("─" * len(header))
    print(f"Training complete. Best val query acc: {best_val_acc*100:.2f}%")
    print(f"Best checkpoint saved to: {save_dir / 'best_protonet.pt'}")

    # # ── t-SNE snapshot: final epoch ──
    # print(f"Computing t-SNE snapshot — epoch {args.epochs} (final) …")
    # emb = collect_embeddings(encoder, tsne_images, device)
    # tsne_snapshots.append(
    #     (f"Epoch {args.epochs} (final)", run_tsne(emb, args.seed), tsne_labels)
    # )

    # # ── Save plots ──
    # print(f"Worst accuracy: {worst_acc}")
    # plot_val_accuracy(
    #     val_q_acc_history,
    #     save_dir / "val_query_accuracy.png",
    #     args.n_way,
    #     args.k_shot,
    #     args.n_train_episodes,
    #     worst_acc
    # )
    # plot_tsne_grid(tsne_snapshots, save_dir / "tsne_embeddings.png")


if __name__ == "__main__":
    main()