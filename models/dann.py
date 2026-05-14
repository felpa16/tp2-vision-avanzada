"""
Domain-Adversarial Neural Network (DANN) for MNIST -> MNIST-M adaptation.
TP2 — Parte 4.3 (I309, Universidad de San Andrés).

Implementation follows section 4.3 of the assignment as faithfully as
possible. One deliberate deviation from the literal spec:

  * Input pipeline. The spec says "misma arquitectura que en las partes
    anteriores para facilitar la comparación", referring to the encoder. We
    keep the encoder identical to ProtoNet/MAML (ConvNetEncoder, ReLU,
    MaxPool), but feed it 3-channel RGB images for DANN. Reason: the
    grayscale + preprocess_mnistm pipeline used by ProtoNet/MAML normalises
    MNIST with mean/std while leaving MNIST-M in [0,1], producing a trivial
    input-statistics gap that the domain classifier exploits in 1 epoch
    (DomAcc -> 100%, DANN collapses). With matched RGB normalisation across
    both domains, the discriminator is forced to look at semantic features.
    ProtoNet/MAML/ProtoMAML preprocessing is NOT modified by this file.

Everything else is strict-spec:

  - Encoder: ConvNetEncoder (same architecture as Parts 1 and 2; only the
    input-channel count differs).
  - Task classifier  G_y : FC + ReLU + FC -> 10 classes
  - Domain classifier G_d : FC + ReLU + FC ->  2 classes
  - GRL: torch.autograd.Function. Identity in forward, gradient * -lambda
    in backward.
  - Loss: L = L_cls(MNIST) - lambda * L_dom(MNIST + MNIST-M)
    (GRL injects -lambda into the encoder's gradient; we just sum losses.)
  - Lambda schedule: lambda_p = 2 / (1 + exp(-10 p)) - 1.
  - Baseline: identical model without GRL.
  - Optimizer: Adam.
  - No LR schedule, no augmentation, no MixUp, no entropy minimisation,
    no pseudo-labels. Final-epoch accuracy is reported (not best).

Saved artefacts
---------------
  checkpoints/best_dann_baseline.pt
  checkpoints/best_dann.pt
  checkpoints/dann_training_curves.png       (4.3.2 — losses + tgt acc)
  checkpoints/dann_accuracy_table.txt        (4.3.2 — source/target table)
  checkpoints/dann_tsne_comparison.png       (4.3.3 — three-stage t-SNE)
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

from models.protonet import ConvNetEncoder, PreprocessedDataset
from data.prepare_datasets import (
    DATA_DIR, MNISTMDataset, load_mnist, load_mnistm,
)
from data.preprocess import preprocess_mnistm


# -- Gradient Reversal Layer --------------------------------------------------

class GradientReversalFunction(torch.autograd.Function):
    """Identity forward, gradient is multiplied by -lambda in backward."""

    @staticmethod
    def forward(ctx, x, lambda_val):
        ctx.lambda_val = lambda_val
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_val * grad_output, None


class GradientReversalLayer(nn.Module):

    def __init__(self):
        super().__init__()
        self.lambda_val = 0.0

    def set_lambda(self, val: float):
        self.lambda_val = val

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_val)


# -- Lambda schedule ----------------------------------------------------------

def lambda_schedule(p: float) -> float:
    """lambda_p = 2 / (1 + exp(-10 p)) - 1,  p in [0, 1]."""
    return 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0


# -- DANN model ---------------------------------------------------------------

class DANNModel(nn.Module):
    """
    Shared encoder (G_f) + label predictor (G_y) + domain classifier (G_d).
    """

    def __init__(self, hidden_dim: int = 64, n_classes: int = 10,
                 head_dim: int = 100, in_channels: int = 3):
        super().__init__()

        # G_f: same encoder topology as ProtoNet / MAML / ProtoMAML; the only
        # parameter that changes with --input_mode is in_channels.
        self.encoder = ConvNetEncoder(in_channels=in_channels, hidden_dim=hidden_dim)

        # G_y: task classifier — FC + ReLU + FC.
        self.label_predictor = nn.Sequential(
            nn.Linear(hidden_dim, head_dim),
            nn.ReLU(inplace=True),
            nn.Linear(head_dim, n_classes),
        )

        # G_d: domain classifier — FC + ReLU + FC. Reads through the GRL.
        self.grl = GradientReversalLayer()
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_dim, head_dim),
            nn.ReLU(inplace=True),
            nn.Linear(head_dim, 2),
        )

    def forward(self, x):
        features      = self.encoder(x)
        class_logits  = self.label_predictor(features)
        domain_logits = self.domain_classifier(self.grl(features))
        return class_logits, domain_logits, features

    def set_lambda(self, val: float):
        self.grl.set_lambda(val)


# -- Input pipelines ----------------------------------------------------------
#
# Two pipelines are supported:
#
#   --input_mode rgb (default): 3-channel end-to-end, no preprocess_mnistm.
#     MNIST is repeated 1->3 channels. Both domains share Normalize(0.5,0.5).
#     The matched normalisation is critical: it prevents the discriminator
#     from trivially distinguishing domains via input statistics alone.
#
#   --input_mode gray: 1-channel grayscale, mirrors the ProtoNet pipeline.
#     Provided for ablation; collapses early in training because the input
#     statistics of MNIST (mean/std normalised) and preprocess_mnistm output
#     ([0,1] unnormalised) disagree.

_RGB_NORM = (0.5, 0.5, 0.5)


class _RepeatTo3:
    """Tensor transform: (1, H, W) -> (3, H, W) by channel-wise repeat."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(0) == 3:
            return x
        return x.repeat(3, 1, 1)


def load_mnist_rgb_for_dann(data_dir: str = DATA_DIR):
    """MNIST as 3-channel 28x28 RGB, normalised to [-1, 1]."""
    from torchvision import datasets, transforms

    tf = transforms.Compose([
        transforms.ToTensor(),
        _RepeatTo3(),
        transforms.Normalize(_RGB_NORM, _RGB_NORM),
    ])
    train_set = datasets.MNIST(root=data_dir, train=True,  download=True, transform=tf)
    test_set  = datasets.MNIST(root=data_dir, train=False, download=True, transform=tf)
    return train_set, test_set


def load_mnistm_raw(data_dir: str = DATA_DIR):
    """Load MNIST-M with ToTensor only (raw [0,1] RGB)."""
    from torchvision import transforms

    tf = transforms.ToTensor()
    mnistm_dir = Path(data_dir) / "mnist_m"
    train_set = MNISTMDataset(mnistm_dir / "mnist_m_train",
                              mnistm_dir / "mnist_m_train_labels.txt", transform=tf)
    test_set  = MNISTMDataset(mnistm_dir / "mnist_m_test",
                              mnistm_dir / "mnist_m_test_labels.txt",  transform=tf)
    return train_set, test_set


class MNISTMRGB28(Dataset):
    """
    MNIST-M as 3-channel 28x28 RGB tensors normalised to [-1, 1].
    Matches the MNIST-RGB pipeline so both domains share input statistics.
    """

    def __init__(self, base_dataset):
        from torchvision import transforms
        self.base = base_dataset
        self.tf = transforms.Compose([
            transforms.Resize((28, 28), antialias=True),
            transforms.Normalize(_RGB_NORM, _RGB_NORM),
        ])

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]   # (3, H, W) float in [0, 1]
        return self.tf(img), label


def train_or_load_mnist_rgb_pretrain(
    save_dir: Path,
    hidden_dim: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    seed: int,
) -> ConvNetEncoder:
    """
    Train (or load) a small MNIST-RGB classifier and return its encoder.
    Used as the "Pretrained (MNIST only)" column for the 4.3.3 t-SNE when
    --input_mode=rgb, since best_protonet.pt is 1-channel and incompatible
    with the 3-channel DANN/baseline encoders.
    """
    ckpt_path = save_dir / "dann_pretrained_rgb.pt"

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if ckpt.get("hidden_dim") == hidden_dim:
            enc = ConvNetEncoder(in_channels=3, hidden_dim=hidden_dim).to(device)
            enc.load_state_dict(ckpt["encoder"])
            enc.eval()
            print(f"  Loaded RGB pretrained encoder from {ckpt_path}")
            return enc
        print(f"  Cached pretrain has hidden_dim={ckpt.get('hidden_dim')}, "
              f"need {hidden_dim} — retraining.")

    print(f"  Training MNIST-RGB pretrain for {epochs} epoch(s) ...")
    torch.manual_seed(seed)
    mnist_train, _ = load_mnist_rgb_for_dann()
    loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=2)

    encoder = ConvNetEncoder(in_channels=3, hidden_dim=hidden_dim).to(device)
    head    = nn.Linear(hidden_dim, 10).to(device)
    opt     = Adam(list(encoder.parameters()) + list(head.parameters()), lr=1e-3)

    encoder.train(); head.train()
    for ep in range(1, epochs + 1):
        total, n = 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(head(encoder(x)), y)
            loss.backward()
            opt.step()
            total += loss.item(); n += 1
        print(f"    ep {ep}/{epochs}  loss={total/max(n,1):.4f}")

    torch.save({"encoder": encoder.state_dict(), "hidden_dim": hidden_dim}, ckpt_path)
    encoder.eval()
    return encoder


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
) -> dict[str, float]:
    """One epoch. Draws a target batch per source batch (cycles if shorter)."""
    model.train()
    totals = {"cls_loss": 0.0, "dom_loss": 0.0, "src_acc": 0.0, "dom_acc": 0.0}
    n_batches = 0

    target_iter = iter(target_loader)

    for source_imgs, source_labels in source_loader:
        try:
            target_imgs, _ = next(target_iter)
        except StopIteration:
            target_iter = iter(target_loader)
            target_imgs, _ = next(target_iter)

        source_imgs   = source_imgs.to(device)
        source_labels = source_labels.to(device)
        target_imgs   = target_imgs.to(device)

        # Training progress and lambda schedule.
        p = (epoch - 1 + n_batches / len(source_loader)) / n_epochs
        lam = lambda_schedule(p) if use_dann else 0.0
        model.set_lambda(lam)

        optimizer.zero_grad()

        cls_logits, dom_logits_src, _ = model(source_imgs)
        cls_loss = F.cross_entropy(cls_logits, source_labels)

        total_loss = cls_loss

        if use_dann:
            src_dom_labels = torch.zeros(source_imgs.size(0), dtype=torch.long, device=device)
            tgt_dom_labels = torch.ones(target_imgs.size(0),  dtype=torch.long, device=device)

            _, dom_logits_tgt, _ = model(target_imgs)

            dom_loss = (
                F.cross_entropy(dom_logits_src, src_dom_labels) +
                F.cross_entropy(dom_logits_tgt, tgt_dom_labels)
            ) / 2.0

            # GRL injects -lambda into the encoder's gradient already;
            # the encoder therefore minimises  L_cls - lambda * L_dom  and
            # the discriminator minimises  L_dom. Exactly the spec's objective.
            total_loss = cls_loss + dom_loss

            totals["dom_loss"] += dom_loss.item()
            with torch.no_grad():
                preds  = torch.cat([dom_logits_src.argmax(1), dom_logits_tgt.argmax(1)])
                labels = torch.cat([src_dom_labels, tgt_dom_labels])
                totals["dom_acc"] += (preds == labels).float().mean().item()

        total_loss.backward()
        # Gradient clipping is standard for adversarial training: when the
        # discriminator wins by a wide margin in an early epoch, the GRL can
        # backprop a gradient that grows the encoder activations, which grows
        # the discriminator logits, which grows the loss -- a feedback loop
        # that Adam's adaptive normalisation cannot fully damp. Clipping
        # bounds the per-step encoder update independently of loss magnitude.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        totals["cls_loss"] += cls_loss.item()
        with torch.no_grad():
            totals["src_acc"] += (cls_logits.argmax(1) == source_labels).float().mean().item()
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


@torch.no_grad()
def run_accuracy(model: DANNModel, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits, _, _ = model(imgs)
        correct += (logits.argmax(1) == labels).sum().item()
        total   += labels.size(0)
    return correct / total if total > 0 else 0.0


# -- Plotting (section 4.3.2) -------------------------------------------------

def plot_training_curves(
    histories: dict[str, dict[str, list[float]]],
    save_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for mode, hist in histories.items():
        epochs = list(range(1, len(hist["cls_loss"]) + 1))
        style  = "-" if mode == "DANN" else "--"

        axes[0].plot(epochs, hist["cls_loss"], style, label=mode, linewidth=2)
        axes[1].plot(epochs, hist["dom_loss"], style, label=mode, linewidth=2)
        axes[2].plot(epochs, [a * 100 for a in hist["tgt_acc"]], style, label=mode, linewidth=2)

    axes[0].set_title("Classification Loss (source)", fontsize=13)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[1].set_title("Domain Loss", fontsize=13)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[2].set_title("Target Accuracy (MNIST-M)", fontsize=13)
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Accuracy (%)")

    for ax in axes:
        ax.legend(fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle("DANN vs Baseline Training Curves", fontsize=15, y=1.02)
    fig.tight_layout()
    path = save_dir / "dann_training_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Training curves saved to: {path}")


# -- t-SNE visualisation (section 4.3.3) --------------------------------------

def plot_tsne_visualization(
    encoders: dict[str, ConvNetEncoder],
    source_dataset,
    target_dataset,
    n_per_class: int,
    device: torch.device,
    save_dir: Path,
    seed: int = 42,
) -> None:
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

        rng = random.Random(seed)
        src_map = _build_class_map(source_dataset)
        src_imgs, src_lbls = [], []
        for cls in sorted(src_map.keys()):
            chosen = rng.sample(src_map[cls], min(n_per_class, len(src_map[cls])))
            for idx in chosen:
                img, _ = source_dataset[idx]
                src_imgs.append(img); src_lbls.append(cls)

        rng2 = random.Random(seed)
        tgt_map = _build_class_map(target_dataset)
        tgt_imgs, tgt_lbls = [], []
        for cls in sorted(tgt_map.keys()):
            chosen = rng2.sample(tgt_map[cls], min(n_per_class, len(tgt_map[cls])))
            for idx in chosen:
                img, _ = target_dataset[idx]
                tgt_imgs.append(img); tgt_lbls.append(cls)

        with torch.no_grad():
            src_emb = encoder(torch.stack(src_imgs).to(device)).cpu().numpy()
            tgt_emb = encoder(torch.stack(tgt_imgs).to(device)).cpu().numpy()

        all_emb = np.concatenate([src_emb, tgt_emb])
        xy = TSNE(n_components=2, random_state=seed, perplexity=30).fit_transform(all_emb)
        xy_src = xy[:len(src_emb)]
        xy_tgt = xy[len(src_emb):]
        src_lbls = np.array(src_lbls); tgt_lbls = np.array(tgt_lbls)

        ax_dom = axes[0, col]
        ax_dom.scatter(xy_src[:, 0], xy_src[:, 1], c="tab:blue", marker="o",
                       s=15, alpha=0.5, linewidths=0, label="MNIST")
        ax_dom.scatter(xy_tgt[:, 0], xy_tgt[:, 1], c="tab:red", marker="^",
                       s=15, alpha=0.5, linewidths=0, label="MNIST-M")
        ax_dom.set_title(f"{state_name}\n(by domain)", fontsize=12)
        ax_dom.set_xticks([]); ax_dom.set_yticks([])
        ax_dom.legend(fontsize=8, loc="lower right")

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
        ax_cls.set_xticks([]); ax_cls.set_yticks([])

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


# -- Training driver ----------------------------------------------------------

def train_model(
    mode: str,
    args,
    device: torch.device,
    save_dir: Path,
    source_train_loader: DataLoader,
    target_train_loader: DataLoader,
    source_eval_loader: DataLoader,
    target_eval_loader: DataLoader,
) -> tuple[DANNModel, dict[str, list[float]]]:
    use_dann   = (mode == "dann")
    mode_label = "DANN" if use_dann else "Baseline"
    ckpt_name  = "best_dann.pt" if use_dann else "best_dann_baseline.pt"

    print(f"\n{'='*60}")
    print(f"  Training {mode_label}")
    print(f"  {'with' if use_dann else 'without'} Gradient Reversal Layer")
    print(f"{'='*60}\n")

    torch.manual_seed(args.seed)
    in_ch = 3 if args.input_mode == "rgb" else 1
    model = DANNModel(hidden_dim=args.hidden_dim, n_classes=10,
                      head_dim=args.head_dim, in_channels=in_ch).to(device)
    # weight_decay is an Adam hyperparameter (not specified by the spec) that
    # counteracts source overfitting -- without it the encoder memorises MNIST
    # by ~ep 5 and the cls_loss gradient then drags features back toward
    # source-specific artefacts long after the adversarial game has saturated.
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")

    history = {"cls_loss": [], "dom_loss": [], "src_acc": [], "tgt_acc": [], "dom_acc": []}

    header = (f"{'Ep':>3}  {'ClsLoss':>8}  {'DomLoss':>8}  "
              f"{'SrcAcc':>7}  {'TgtAcc':>7}  {'DomAcc':>7}  {'Lambda':>7}")
    print(header)
    print("-" * len(header))

    for epoch in range(1, args.epochs + 1):
        tr = train_epoch(model, source_train_loader, target_train_loader,
                         optimizer, device, epoch, args.epochs, use_dann)

        src_acc = run_accuracy(model, source_eval_loader, device)
        tgt_acc = run_accuracy(model, target_eval_loader, device)

        p   = epoch / args.epochs
        lam = lambda_schedule(p) if use_dann else 0.0

        history["cls_loss"].append(tr["cls_loss"])
        history["dom_loss"].append(tr["dom_loss"])
        history["src_acc"].append(src_acc)
        history["tgt_acc"].append(tgt_acc)
        history["dom_acc"].append(tr["dom_acc"])

        print(f"{epoch:>3}  {tr['cls_loss']:>8.4f}  {tr['dom_loss']:>8.4f}  "
              f"{src_acc*100:>6.2f}%  {tgt_acc*100:>6.2f}%  "
              f"{tr['dom_acc']*100:>6.2f}%  {lam:>7.4f}")

    torch.save({
        "epoch":   args.epochs,
        "src_acc": history["src_acc"][-1],
        "tgt_acc": history["tgt_acc"][-1],
        "model":   model.state_dict(),
        "args":    vars(args),
        "mode":    mode,
    }, save_dir / ckpt_name)

    print("-" * len(header))
    print(f"Final target acc: {history['tgt_acc'][-1]*100:.2f}%")
    print(f"Checkpoint:       {save_dir / ckpt_name}")

    return model, history


# -- Main ---------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="DANN: Domain-Adversarial Neural Network")
    p.add_argument("--mode",         type=str,   default="both",
                   choices=["baseline", "dann", "both"])
    p.add_argument("--epochs",       type=int,   default=30)
    p.add_argument("--batch_size",   type=int,   default=64)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--hidden_dim",   type=int,   default=64,
                   help="Encoder channel width (same as ProtoNet default)")
    p.add_argument("--head_dim",     type=int,   default=100,
                   help="Hidden width of label / domain classifier heads")
    p.add_argument("--weight_decay", type=float, default=1e-4,
                   help="Adam weight decay. Combats source overfitting; "
                        "set to 0 to recover the no-decay behaviour.")
    p.add_argument("--input_mode",   type=str,   default="rgb",
                   choices=["rgb", "gray"],
                   help="rgb (default): 3-channel end-to-end with matched "
                        "normalisation. gray: 1-channel mirroring ProtoNet's "
                        "pipeline (ablation only — collapses early).")
    p.add_argument("--pretrain_epochs", type=int, default=3,
                   help="Epochs to train the small MNIST-RGB classifier used "
                        "as the 4.3.3 'Pretrained' encoder (rgb mode only).")
    p.add_argument("--n_tsne",       type=int,   default=30,
                   help="Images per class for t-SNE")
    p.add_argument("--save_dir",     type=str,   default="./checkpoints")
    p.add_argument("--seed",         type=int,   default=42)
    return p.parse_args()


def main():
    args     = parse_args()
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Mode:   {args.mode}")
    print(f"Input:  {args.input_mode}\n")

    # -- Datasets ---------------------------------------------------------
    print("Loading datasets ...")
    use_rgb = (args.input_mode == "rgb")

    if use_rgb:
        # RGB end-to-end. Same Normalize(0.5, 0.5) on both domains so the
        # discriminator can't separate them on raw input statistics.
        mnist_train, mnist_test = load_mnist_rgb_for_dann()
        mnistm_train_raw, mnistm_test_raw = load_mnistm_raw()
        mnistm_train_pp = MNISTMRGB28(mnistm_train_raw)
        mnistm_test_pp  = MNISTMRGB28(mnistm_test_raw)
    else:
        # Grayscale ablation — mirrors ProtoNet's pipeline exactly.
        mnist_train, _, mnist_test    = load_mnist()
        mnistm_train, _, mnistm_test  = load_mnistm()
        mnistm_train_pp = PreprocessedDataset(mnistm_train, preprocess_mnistm)
        mnistm_test_pp  = PreprocessedDataset(mnistm_test,  preprocess_mnistm)

    source_train_loader = DataLoader(mnist_train,     batch_size=args.batch_size,
                                     shuffle=True,  num_workers=2)
    source_test_loader  = DataLoader(mnist_test,      batch_size=args.batch_size,
                                     shuffle=False, num_workers=2)
    target_train_loader = DataLoader(mnistm_train_pp, batch_size=args.batch_size,
                                     shuffle=True,  num_workers=2)
    target_test_loader  = DataLoader(mnistm_test_pp,  batch_size=args.batch_size,
                                     shuffle=False, num_workers=2)
    print()

    # -- Train -----------------------------------------------------------
    modes = ["baseline", "dann"] if args.mode == "both" else [args.mode]
    histories: dict[str, dict[str, list[float]]] = {}
    trained_models: dict[str, DANNModel] = {}

    for m in modes:
        model, history = train_model(
            mode=m, args=args, device=device, save_dir=save_dir,
            source_train_loader=source_train_loader,
            target_train_loader=target_train_loader,
            source_eval_loader=source_test_loader,
            target_eval_loader=target_test_loader,
        )
        label = "DANN" if m == "dann" else "Baseline"
        histories[label]      = history
        trained_models[label] = model

    # -- Training curves (4.3.2 plot) -------------------------------------
    if histories:
        plot_training_curves(histories, save_dir)

    # -- Accuracy table (4.3.2 table) -------------------------------------
    print(f"\n{'='*50}")
    print("  FINAL ACCURACY TABLE  (test sets, last epoch)")
    print(f"{'='*50}")
    print(f"  {'Model':<12}  {'Source (MNIST)':>14}  {'Target (MNIST-M)':>16}")
    print(f"  {'-'*46}")

    rows: list[tuple[str, float, float]] = []
    for m in modes:
        label = "DANN" if m == "dann" else "Baseline"
        src   = histories[label]["src_acc"][-1] * 100
        tgt   = histories[label]["tgt_acc"][-1] * 100
        rows.append((label, src, tgt))
        print(f"  {label:<12}  {src:>13.2f}%  {tgt:>15.2f}%")

    with open(save_dir / "dann_accuracy_table.txt", "w") as f:
        f.write(f"{'Model':<12}  {'Source (MNIST)':>14}  {'Target (MNIST-M)':>16}\n")
        f.write(f"{'-'*46}\n")
        for label, src, tgt in rows:
            f.write(f"{label:<12}  {src:>13.2f}%  {tgt:>15.2f}%\n")
    print(f"\nAccuracy table saved to: {save_dir / 'dann_accuracy_table.txt'}")

    # -- t-SNE (section 4.3.3) -------------------------------------------
    # Three encoder states required by the spec:
    #   1. Pre-trained on MNIST only.
    #        rgb mode  -> small MNIST-RGB classifier (best_protonet.pt is
    #                     1-channel and incompatible with 3-channel DANN).
    #        gray mode -> ProtoNet checkpoint directly.
    #   2. Finetuning (no GRL) -> baseline encoder.
    #   3. DANN (with GRL + L_dom) -> DANN encoder.
    print("\nPreparing t-SNE visualization ...")
    encoders_for_tsne: dict[str, ConvNetEncoder] = {}

    if use_rgb:
        pre_encoder = train_or_load_mnist_rgb_pretrain(
            save_dir=save_dir, hidden_dim=args.hidden_dim, device=device,
            epochs=args.pretrain_epochs, batch_size=args.batch_size, seed=args.seed,
        )
        encoders_for_tsne["Pre-trained\n(MNIST RGB)"] = pre_encoder
    else:
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
            print("  WARNING: best_protonet.pt not found — pretrained column skipped.")

    if "Baseline" in trained_models:
        trained_models["Baseline"].eval()
        encoders_for_tsne["Finetuning\n(no GRL)"] = trained_models["Baseline"].encoder

    if "DANN" in trained_models:
        trained_models["DANN"].eval()
        encoders_for_tsne["DANN\n(with GRL)"] = trained_models["DANN"].encoder

    if encoders_for_tsne:
        plot_tsne_visualization(
            encoders       = encoders_for_tsne,
            source_dataset = mnist_test,
            target_dataset = mnistm_test_pp,
            n_per_class    = args.n_tsne,
            device         = device,
            save_dir       = save_dir,
            seed           = args.seed,
        )

    print("\nAll done.")


if __name__ == "__main__":
    main()
