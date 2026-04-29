"""
Domain-Adversarial Neural Network (DANN) for MNIST -> MNIST-M adaptation.

Architecture (Ganin et al. 2016):
  - Feature extractor G_f : same 4-block ConvNet encoder (64-d embeddings)
  - Label predictor  G_y : FC(64,100) -> ReLU -> FC(100,10)
  - Domain classifier G_d : FC(64,100) -> ReLU -> FC(100,2), connected via GRL

The Gradient Reversal Layer (GRL) acts as identity in the forward pass but
multiplies gradients by -lambda during backpropagation, creating an adversarial
game: G_d tries to distinguish domains while G_f is penalised if its
representations allow G_d to succeed.

Combined loss:
    L = L_cls(MNIST) - lambda * L_dom(MNIST + MNIST-M)

Lambda schedule (from instructions):
    lambda_p = 2 / (1 + exp(-10*p)) - 1,   p in [0, 1]

Two training modes:
  --mode baseline : only L_cls (no GRL, no domain loss)
  --mode dann     : full DANN with GRL and domain loss
  --mode both     : train baseline then DANN sequentially

Saved artefacts
---------------
  checkpoints/best_dann_baseline.pt
  checkpoints/best_dann.pt
  checkpoints/dann_training_curves.png
  checkpoints/dann_accuracy_table.txt
  checkpoints/dann_tsne_comparison.png
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from models.protonet import ConvNetEncoder
from data.prepare_datasets import MNISTMDataset, DATA_DIR
from data.preprocess import preprocess_mnistm


# -- Gradient Reversal Layer --------------------------------------------------

class GradientReversalFunction(torch.autograd.Function):
    """Identity forward, multiply gradient by -lambda backward."""

    @staticmethod
    def forward(ctx, x, lambda_val):
        ctx.lambda_val = lambda_val
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_val * grad_output, None


class GradientReversalLayer(nn.Module):
    """Wraps GradientReversalFunction as a module."""

    def __init__(self):
        super().__init__()
        self.lambda_val = 0.0

    def set_lambda(self, val: float):
        self.lambda_val = val

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_val)


# -- Lambda schedule ----------------------------------------------------------

def lambda_schedule(p: float) -> float:
    """lambda_p = 2 / (1 + exp(-10*p)) - 1,  p in [0, 1]."""
    return 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0


def lr_schedule(p: float, mu_0: float, alpha: float = 10.0, beta: float = 0.75) -> float:
    """Decaying LR from the DANN paper: mu_p = mu_0 / (1 + alpha*p)^beta."""
    return mu_0 / (1.0 + alpha * p) ** beta


# -- DANN model ---------------------------------------------------------------

class DANNModel(nn.Module):
    """
    Feature extractor (shared encoder) + label predictor + domain classifier.

    The domain classifier is connected to the encoder via a GRL so that
    backpropagation through it reverses gradient sign for the encoder.
    """

    def __init__(self, hidden_dim: int = 64, n_classes: int = 10):
        super().__init__()

        # G_f: feature extractor (same ConvNet encoder as ProtoNet)
        self.encoder = ConvNetEncoder(in_channels=1, hidden_dim=hidden_dim)

        # G_y: label predictor (task classifier)
        self.label_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, n_classes),
        )

        # G_d: domain classifier (binary: source=0 / target=1)
        self.grl = GradientReversalLayer()
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 2),
        )

    def forward(self, x):
        features = self.encoder(x)
        class_logits = self.label_predictor(features)
        domain_logits = self.domain_classifier(self.grl(features))
        return class_logits, domain_logits, features

    def set_lambda(self, val: float):
        self.grl.set_lambda(val)


# -- Preprocessed dataset wrapper ---------------------------------------------

class PreprocessedMNISTM(Dataset):
    """
    Wraps MNIST-M dataset, applying preprocess_mnistm on the fly,
    then normalising with MNIST statistics so the encoder sees
    comparable input distributions for both domains.

    Expects the underlying dataset to return (RGB tensor [0,1], label).
    Returns (normalised grayscale 1x28x28 tensor, label).
    """

    MNIST_MEAN = 0.1307
    MNIST_STD  = 0.3081

    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        gray = preprocess_mnistm(img)          # (1, 28, 28) in [0, 1]
        normalized = (gray - self.MNIST_MEAN) / self.MNIST_STD
        return normalized, label


def load_mnistm_raw(data_dir: str = DATA_DIR):
    """
    Load MNIST-M with ToTensor only (no normalisation).

    preprocess_mnistm expects [0,1] RGB tensors.
    """
    from torchvision import transforms
    from pathlib import Path as P

    raw_transform = transforms.ToTensor()
    mnistm_dir = P(data_dir) / "mnist_m"

    train_img_dir  = mnistm_dir / "mnist_m_train"
    train_lbl_file = mnistm_dir / "mnist_m_train_labels.txt"
    test_img_dir   = mnistm_dir / "mnist_m_test"
    test_lbl_file  = mnistm_dir / "mnist_m_test_labels.txt"

    train_set = MNISTMDataset(train_img_dir, train_lbl_file, transform=raw_transform)
    test_set  = MNISTMDataset(test_img_dir,  test_lbl_file,  transform=raw_transform)

    return train_set, test_set


def load_mnist_for_dann(data_dir: str = DATA_DIR):
    """Load MNIST with standard normalisation for DANN training."""
    from torchvision import datasets, transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_set = datasets.MNIST(root=data_dir, train=True,  download=True, transform=transform)
    test_set  = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    return train_set, test_set


# -- Training loop ------------------------------------------------------------

def train_epoch(
    model: DANNModel,
    source_loader: DataLoader,
    target_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    n_epochs: int,
    use_dann: bool,
    base_lr: float | None = None,
) -> dict[str, float]:
    """
    One training epoch.

    Each batch:
      1. Source batch -> L_cls (always) + L_dom on source (if DANN)
      2. Target batch -> L_dom on target (if DANN)

    Lambda grows according to the schedule.
    """
    model.train()

    totals = {"cls_loss": 0.0, "dom_loss": 0.0, "src_acc": 0.0, "dom_acc": 0.0}
    n_batches = 0

    target_iter = iter(target_loader)

    for source_imgs, source_labels in source_loader:
        # Get target batch (cycle if target is shorter)
        try:
            target_imgs, _ = next(target_iter)
        except StopIteration:
            target_iter = iter(target_loader)
            target_imgs, _ = next(target_iter)

        source_imgs   = source_imgs.to(device)
        source_labels = source_labels.to(device)
        target_imgs   = target_imgs.to(device)

        # Progress in [0, 1]
        p = (epoch - 1 + n_batches / len(source_loader)) / n_epochs

        # Lambda schedule (DANN only)
        lam = lambda_schedule(p) if use_dann else 0.0
        model.set_lambda(lam)

        # Learning rate schedule (DANN paper: decaying LR)
        if use_dann and base_lr is not None:
            new_lr = lr_schedule(p, base_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = new_lr

        optimizer.zero_grad()

        # Source: classification + domain
        cls_logits, dom_logits_src, _ = model(source_imgs)
        cls_loss = F.cross_entropy(cls_logits, source_labels)

        total_loss = cls_loss

        if use_dann:
            # Domain labels: source=0, target=1
            src_dom_labels = torch.zeros(source_imgs.size(0), dtype=torch.long, device=device)
            tgt_dom_labels = torch.ones(target_imgs.size(0), dtype=torch.long, device=device)

            _, dom_logits_tgt, _ = model(target_imgs)

            dom_loss = (
                F.cross_entropy(dom_logits_src, src_dom_labels) +
                F.cross_entropy(dom_logits_tgt, tgt_dom_labels)
            ) / 2.0

            # The GRL handles the sign flip for encoder gradients.
            # We ADD dom_loss: domain classifier minimises it normally,
            # while encoder receives reversed gradients (maximises it).
            total_loss = cls_loss + lam * dom_loss

            totals["dom_loss"] += dom_loss.item()

            # Domain accuracy
            with torch.no_grad():
                all_dom_preds = torch.cat([dom_logits_src.argmax(1), dom_logits_tgt.argmax(1)])
                all_dom_labels = torch.cat([src_dom_labels, tgt_dom_labels])
                totals["dom_acc"] += (all_dom_preds == all_dom_labels).float().mean().item()

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        totals["cls_loss"] += cls_loss.item()
        with torch.no_grad():
            totals["src_acc"] += (cls_logits.argmax(1) == source_labels).float().mean().item()
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


@torch.no_grad()
def run_accuracy(
    model: DANNModel,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Classification accuracy on a dataset."""
    model.eval()
    correct, total = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits, _, _ = model(imgs)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return correct / total if total > 0 else 0.0


# -- Plotting -----------------------------------------------------------------

def plot_training_curves(
    histories: dict[str, dict[str, list[float]]],
    save_dir: Path,
) -> None:
    """
    Plot L_cls, L_dom, and target accuracy over training.

    histories[mode] = {"cls_loss": [...], "dom_loss": [...], "tgt_acc": [...]}
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for mode, hist in histories.items():
        epochs = list(range(1, len(hist["cls_loss"]) + 1))
        style = "-" if mode == "DANN" else "--"

        axes[0].plot(epochs, hist["cls_loss"], style, label=mode, linewidth=2)
        axes[1].plot(epochs, hist["dom_loss"], style, label=mode, linewidth=2)
        axes[2].plot(epochs, [a * 100 for a in hist["tgt_acc"]], style, label=mode, linewidth=2)

    axes[0].set_title("Classification Loss (source)", fontsize=13)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")

    axes[1].set_title("Domain Loss", fontsize=13)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")

    axes[2].set_title("Target Accuracy (MNIST-M)", fontsize=13)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Accuracy (%)")

    for ax in axes:
        ax.legend(fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle("DANN vs Baseline Training Curves", fontsize=15, y=1.02)
    fig.tight_layout()
    path = save_dir / "dann_training_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Training curves saved to: {path}")


def plot_tsne_visualization(
    encoders: dict[str, ConvNetEncoder],
    source_dataset,
    target_dataset,
    n_per_class: int,
    device: torch.device,
    save_dir: Path,
    seed: int = 42,
) -> None:
    """
    t-SNE of MNIST + MNIST-M embeddings for each encoder state.

    Two rows: top = colored by domain, bottom = colored by class.
    One column per encoder state (pretrained, baseline, DANN).
    """
    import random
    from data.episodic_sampler import _build_class_map
    from matplotlib.lines import Line2D

    n_classes = 10
    class_cmap = plt.cm.get_cmap("tab10", n_classes)
    n_states = len(encoders)

    fig, axes = plt.subplots(2, n_states, figsize=(6 * n_states, 10))
    if n_states == 1:
        axes = axes.reshape(2, 1)

    for col, (state_name, encoder) in enumerate(encoders.items()):
        encoder.eval()

        # Sample source images
        rng = random.Random(seed)
        src_map = _build_class_map(source_dataset)
        src_imgs, src_lbls = [], []
        for cls in sorted(src_map.keys()):
            chosen = rng.sample(src_map[cls], min(n_per_class, len(src_map[cls])))
            for idx in chosen:
                img, _ = source_dataset[idx]
                src_imgs.append(img)
                src_lbls.append(cls)

        # Sample target images
        rng2 = random.Random(seed)
        tgt_map = _build_class_map(target_dataset)
        tgt_imgs, tgt_lbls = [], []
        for cls in sorted(tgt_map.keys()):
            chosen = rng2.sample(tgt_map[cls], min(n_per_class, len(tgt_map[cls])))
            for idx in chosen:
                img, _ = target_dataset[idx]
                tgt_imgs.append(img)
                tgt_lbls.append(cls)

        # Compute embeddings
        with torch.no_grad():
            src_emb = encoder(torch.stack(src_imgs).to(device)).cpu().numpy()
            tgt_emb = encoder(torch.stack(tgt_imgs).to(device)).cpu().numpy()

        all_emb = np.concatenate([src_emb, tgt_emb])
        xy = TSNE(n_components=2, random_state=seed, perplexity=30).fit_transform(all_emb)
        xy_src = xy[:len(src_emb)]
        xy_tgt = xy[len(src_emb):]

        src_lbls = np.array(src_lbls)
        tgt_lbls = np.array(tgt_lbls)

        # Row 0: colored by domain
        ax_dom = axes[0, col]
        ax_dom.scatter(xy_src[:, 0], xy_src[:, 1], c="tab:blue", marker="o",
                       s=15, alpha=0.5, linewidths=0, label="MNIST")
        ax_dom.scatter(xy_tgt[:, 0], xy_tgt[:, 1], c="tab:red", marker="^",
                       s=15, alpha=0.5, linewidths=0, label="MNIST-M")
        ax_dom.set_title(f"{state_name}\n(by domain)", fontsize=12)
        ax_dom.set_xticks([])
        ax_dom.set_yticks([])
        ax_dom.legend(fontsize=8, loc="lower right")

        # Row 1: colored by class
        ax_cls = axes[1, col]
        for cls in range(n_classes):
            c = class_cmap(cls)
            mask_s = src_lbls == cls
            mask_t = tgt_lbls == cls
            ax_cls.scatter(xy_src[mask_s, 0], xy_src[mask_s, 1],
                          color=c, marker="o", s=15, alpha=0.5, linewidths=0)
            ax_cls.scatter(xy_tgt[mask_t, 0], xy_tgt[mask_t, 1],
                          color=c, marker="^", s=15, alpha=0.5, linewidths=0)
        ax_cls.set_title(f"{state_name}\n(by class)", fontsize=12)
        ax_cls.set_xticks([])
        ax_cls.set_yticks([])

    # Shared legend for class plot
    handles = [Line2D([], [], linestyle="none", marker="o", color=class_cmap(i),
                      markersize=6, label=str(i)) for i in range(n_classes)]
    axes[1, -1].legend(handles=handles, fontsize=7, loc="lower right",
                       ncol=2, title="Digit")

    fig.suptitle("t-SNE: MNIST vs MNIST-M Embeddings", fontsize=14, y=1.02)
    fig.tight_layout()
    path = save_dir / "dann_tsne_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"t-SNE visualization saved to: {path}")


# -- Training function --------------------------------------------------------

def train_model(
    mode: str,
    args,
    device: torch.device,
    save_dir: Path,
    source_train_loader: DataLoader,
    target_train_loader: DataLoader,
    source_test_loader: DataLoader,
    target_test_loader: DataLoader,
) -> tuple[DANNModel, dict[str, list[float]]]:
    """Train a single DANN or baseline model."""
    use_dann = (mode == "dann")
    mode_label = "DANN" if use_dann else "Baseline"
    ckpt_name = "best_dann.pt" if use_dann else "best_dann_baseline.pt"

    print(f"\n{'='*60}")
    print(f"  Training {mode_label}")
    print(f"  {'with' if use_dann else 'without'} Gradient Reversal Layer")
    print(f"{'='*60}\n")

    torch.manual_seed(args.seed)
    model = DANNModel(hidden_dim=args.hidden_dim, n_classes=10).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}\n")

    history = {"cls_loss": [], "dom_loss": [], "src_acc": [], "tgt_acc": [], "dom_acc": []}
    best_tgt_acc = 0.0

    header = (
        f"{'Ep':>3}  {'ClsLoss':>8}  {'DomLoss':>8}  "
        f"{'SrcAcc':>7}  {'TgtAcc':>7}  {'DomAcc':>7}  {'Lambda':>7}"
    )
    print(header)
    print("-" * len(header))

    for epoch in range(1, args.epochs + 1):
        tr = train_epoch(
            model, source_train_loader, target_train_loader,
            optimizer, device, epoch, args.epochs, use_dann,
            base_lr=args.lr,
        )

        src_acc = run_accuracy(model, source_test_loader, device)
        tgt_acc = run_accuracy(model, target_test_loader, device)

        p = epoch / args.epochs
        lam = lambda_schedule(p) if use_dann else 0.0

        history["cls_loss"].append(tr["cls_loss"])
        history["dom_loss"].append(tr["dom_loss"])
        history["src_acc"].append(src_acc)
        history["tgt_acc"].append(tgt_acc)
        history["dom_acc"].append(tr["dom_acc"])

        print(
            f"{epoch:>3}  "
            f"{tr['cls_loss']:>8.4f}  "
            f"{tr['dom_loss']:>8.4f}  "
            f"{src_acc*100:>6.2f}%  "
            f"{tgt_acc*100:>6.2f}%  "
            f"{tr['dom_acc']*100:>6.2f}%  "
            f"{lam:>7.4f}"
        )

        if tgt_acc > best_tgt_acc:
            best_tgt_acc = tgt_acc
            ckpt = {
                "epoch": epoch,
                "src_acc": src_acc,
                "tgt_acc": tgt_acc,
                "model": model.state_dict(),
                "args": vars(args),
                "mode": mode,
            }
            torch.save(ckpt, save_dir / ckpt_name)

    print("-" * len(header))
    print(f"Best target acc: {best_tgt_acc*100:.2f}%")
    print(f"Checkpoint: {save_dir / ckpt_name}")

    return model, history


# -- Main ---------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="DANN: Domain-Adversarial Neural Network")
    p.add_argument("--mode",       type=str, default="both",
                   choices=["baseline", "dann", "both"])
    p.add_argument("--epochs",     type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--n_tsne",     type=int, default=30,
                   help="Images per class for t-SNE")
    p.add_argument("--save_dir",   type=str, default="./checkpoints")
    p.add_argument("--seed",       type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Mode:   {args.mode}\n")

    # -- Load datasets --
    print("Loading datasets ...")

    # Source: MNIST (with normalisation)
    mnist_train, mnist_test = load_mnist_for_dann()
    source_train_loader = DataLoader(mnist_train, batch_size=args.batch_size,
                                     shuffle=True, num_workers=2)
    source_test_loader  = DataLoader(mnist_test,  batch_size=args.batch_size,
                                     shuffle=False, num_workers=2)

    # Target: MNIST-M (raw RGB -> preprocessed to grayscale 28x28)
    mnistm_train_raw, mnistm_test_raw = load_mnistm_raw()
    mnistm_train_pp = PreprocessedMNISTM(mnistm_train_raw)
    mnistm_test_pp  = PreprocessedMNISTM(mnistm_test_raw)
    target_train_loader = DataLoader(mnistm_train_pp, batch_size=args.batch_size,
                                     shuffle=True, num_workers=2)
    target_test_loader  = DataLoader(mnistm_test_pp,  batch_size=args.batch_size,
                                     shuffle=False, num_workers=2)

    print(f"Source: MNIST train={len(mnist_train):,}  test={len(mnist_test):,}")
    print(f"Target: MNIST-M train={len(mnistm_train_raw):,}  test={len(mnistm_test_raw):,}")
    print()

    # -- Train models --
    modes = ["baseline", "dann"] if args.mode == "both" else [args.mode]
    all_histories: dict[str, dict[str, list[float]]] = {}
    trained_models: dict[str, DANNModel] = {}

    for m in modes:
        model, history = train_model(
            mode=m, args=args, device=device, save_dir=save_dir,
            source_train_loader=source_train_loader,
            target_train_loader=target_train_loader,
            source_test_loader=source_test_loader,
            target_test_loader=target_test_loader,
        )
        label = "DANN" if m == "dann" else "Baseline"
        all_histories[label] = history
        trained_models[label] = model

    # -- Training curves plot --
    if len(all_histories) > 0:
        plot_training_curves(all_histories, save_dir)

    # -- Accuracy table --
    print(f"\n{'='*50}")
    print("  FINAL ACCURACY TABLE")
    print(f"{'='*50}")
    print(f"  {'Model':<12}  {'Source (MNIST)':>14}  {'Target (MNIST-M)':>16}")
    print(f"  {'-'*46}")
    for label, hist in all_histories.items():
        src = hist["src_acc"][-1] * 100
        tgt = hist["tgt_acc"][-1] * 100
        print(f"  {label:<12}  {src:>13.2f}%  {tgt:>15.2f}%")

    # Save accuracy table
    with open(save_dir / "dann_accuracy_table.txt", "w") as f:
        f.write(f"{'Model':<12}  {'Source (MNIST)':>14}  {'Target (MNIST-M)':>16}\n")
        f.write(f"{'-'*46}\n")
        for label, hist in all_histories.items():
            src = hist["src_acc"][-1] * 100
            tgt = hist["tgt_acc"][-1] * 100
            f.write(f"{label:<12}  {src:>13.2f}%  {tgt:>15.2f}%\n")
    print(f"\nAccuracy table saved to: {save_dir / 'dann_accuracy_table.txt'}")

    # -- t-SNE visualization (section 4.3.3) --
    # Three moments: (1) pretrained on MNIST, (2) baseline, (3) DANN
    print("\nPreparing t-SNE visualization ...")

    encoders_for_tsne: dict[str, ConvNetEncoder] = {}

    # 1. Pretrained (ProtoNet encoder if available)
    protonet_ckpt = save_dir / "best_protonet.pt"
    if protonet_ckpt.exists():
        ckpt = torch.load(protonet_ckpt, map_location=device, weights_only=False)
        proto_hdim = ckpt["args"].get("hidden_dim", 64)
        pre_encoder = ConvNetEncoder(in_channels=1, hidden_dim=proto_hdim).to(device)
        pre_encoder.load_state_dict(ckpt["model"])
        pre_encoder.eval()
        encoders_for_tsne["Pre-trained\n(ProtoNet)"] = pre_encoder
        print("  Loaded pretrained encoder from best_protonet.pt")
    else:
        print("  WARNING: best_protonet.pt not found, skipping pretrained t-SNE")

    # 2. Baseline encoder (from memory or checkpoint)
    if "Baseline" in trained_models:
        trained_models["Baseline"].eval()
        encoders_for_tsne["Baseline\n(no GRL)"] = trained_models["Baseline"].encoder
    else:
        baseline_ckpt = save_dir / "best_dann_baseline.pt"
        if baseline_ckpt.exists():
            ckpt = torch.load(baseline_ckpt, map_location=device, weights_only=False)
            bl_hdim = ckpt["args"].get("hidden_dim", 64)
            bl_model = DANNModel(hidden_dim=bl_hdim, n_classes=10).to(device)
            bl_model.load_state_dict(ckpt["model"])
            bl_model.eval()
            encoders_for_tsne["Baseline\n(no GRL)"] = bl_model.encoder
            print("  Loaded baseline encoder from best_dann_baseline.pt")

    # 3. DANN encoder (from memory or checkpoint)
    if "DANN" in trained_models:
        trained_models["DANN"].eval()
        encoders_for_tsne["DANN\n(with GRL)"] = trained_models["DANN"].encoder
    else:
        dann_ckpt = save_dir / "best_dann.pt"
        if dann_ckpt.exists():
            ckpt = torch.load(dann_ckpt, map_location=device, weights_only=False)
            d_hdim = ckpt["args"].get("hidden_dim", 64)
            d_model = DANNModel(hidden_dim=d_hdim, n_classes=10).to(device)
            d_model.load_state_dict(ckpt["model"])
            d_model.eval()
            encoders_for_tsne["DANN\n(with GRL)"] = d_model.encoder
            print("  Loaded DANN encoder from best_dann.pt")

    if encoders_for_tsne:
        # For t-SNE: use test sets (source=MNIST test, target=MNIST-M test preprocessed)
        plot_tsne_visualization(
            encoders=encoders_for_tsne,
            source_dataset=mnist_test,
            target_dataset=mnistm_test_pp,
            n_per_class=args.n_tsne,
            device=device,
            save_dir=save_dir,
            seed=args.seed,
        )

    print("\nAll done.")


if __name__ == "__main__":
    main()
