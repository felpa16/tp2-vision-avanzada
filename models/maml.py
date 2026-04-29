"""
MAML and ProtoMAML training on MNIST.

MAML (Finn et al. 2017):
  Inner loop -- L steps of SGD on the support set, using torch.autograd.grad
               to compute second-order gradients through the adaptation.
  Outer loop -- Adam on the meta-parameters theta, minimising the query loss
               with the adapted parameters phi.

ProtoMAML (Triantafillou et al. 2020):
  Identical to MAML, except the linear classification head is initialised
  from prototypes computed on the support set before the inner loop:
      W_c = 2 * v_c^T       b_c = -||v_c||^2
  This makes the network equivalent to a ProtoNet at step 0 of adaptation.

Episode config : 5-way 1-shot 15-query (train & val)
Encoder        : same 4-block ConvNet as ProtoNet (64-d embeddings)

Saved artefacts
---------------
  checkpoints/best_maml.pt
  checkpoints/best_protomaml.pt
"""

import argparse
from pathlib import Path
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

import matplotlib.pyplot as plt

from models.protonet import ConvNetEncoder
from data.prepare_datasets import load_mnist
from data.episodic_sampler import build_episode_loader


# ── MAML / ProtoMAML model ──────────────────────────────────────────────────

class MAMLModel(nn.Module):
    """
    Encoder + linear head for MAML / ProtoMAML.

    Uses torch.func.functional_call for differentiable forward passes
    with external parameter tensors (needed for the inner loop).
    """

    def __init__(self, hidden_dim: int = 64, n_way: int = 5):
        super().__init__()
        self.encoder = ConvNetEncoder(in_channels=1, hidden_dim=hidden_dim)
        self.head    = nn.Linear(hidden_dim, n_way)
        self.hidden_dim = hidden_dim
        self.n_way      = n_way

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))

    def get_params(self) -> OrderedDict:
        """Return an OrderedDict of all parameters (keeps grad graph)."""
        return OrderedDict((n, p) for n, p in self.named_parameters())

    def forward_with_params(self, x: torch.Tensor, params: OrderedDict) -> torch.Tensor:
        """
        Functional forward using external params (for inner loop).

        Uses torch.func.functional_call to properly thread all params
        through the computation graph for second-order differentiation.
        """
        return torch.func.functional_call(self, params, (x,))


# ── Inner loop (adaptation) ─────────────────────────────────────────────────

def inner_loop(
    model: MAMLModel,
    support: torch.Tensor,
    s_labels: torch.Tensor,
    params: OrderedDict,
    inner_lr: float,
    inner_steps: int,
) -> OrderedDict:
    """
    L steps of SGD on the support set.

    Uses torch.autograd.grad so gradients are NOT accumulated into
    model.parameters(), preserving the computation graph for
    second-order differentiation in the outer loop.
    """
    adapted = OrderedDict((k, v) for k, v in params.items())

    for _ in range(inner_steps):
        logits = model.forward_with_params(support, adapted)
        loss   = F.cross_entropy(logits, s_labels)

        grads = torch.autograd.grad(
            loss,
            adapted.values(),
            create_graph=True,   # second-order gradients
        )

        # Clip inner-loop gradients to prevent explosion (especially ProtoMAML)
        grads = tuple(g.clamp(-10.0, 10.0) for g in grads)

        adapted = OrderedDict(
            (name, param - inner_lr * grad)
            for (name, param), grad in zip(adapted.items(), grads)
        )

    return adapted


# ── ProtoMAML head initialisation ───────────────────────────────────────────

def init_head_from_prototypes(
    model: MAMLModel,
    support: torch.Tensor,
    s_labels: torch.Tensor,
    params: OrderedDict,
    n_way: int,
    k_shot: int,
) -> OrderedDict:
    """
    Initialise the linear head W and b from prototypes:
        W_c = 2 * v_c^T       b_c = -||v_c||^2

    This makes the model equivalent to a ProtoNet before any adaptation steps.
    The encoder params keep their grad graph; the head params are detached.
    """
    # Compute embeddings using current encoder params (functional, no grad)
    # Strip "encoder." prefix so keys match model.encoder's named_parameters
    encoder_params = {
        n[len("encoder."):]: p
        for n, p in params.items()
        if n.startswith("encoder.")
    }
    with torch.no_grad():
        emb = torch.func.functional_call(model.encoder, encoder_params, (support,))

    # Compute prototypes (class centroids)
    # s_labels are [0]*K + [1]*K + ... + [N-1]*K (ordered by class)
    prototypes = emb.view(n_way, k_shot, -1).mean(dim=1)  # (N, D)

    # W_c = 2 * v_c,  b_c = -||v_c||^2, scaled by temperature to prevent
    # logit explosion when embeddings have large magnitudes
    proto_norms_sq = (prototypes * prototypes).sum(dim=1)    # (N,)
    tau = proto_norms_sq.mean().clamp(min=1.0)               # scalar temperature
    W = (2.0 / tau) * prototypes                             # (N, D)
    b = -(1.0 / tau) * proto_norms_sq                        # (N,)

    # Replace head params; these are new leaf tensors that require grad
    # so the inner loop can differentiate through them
    new_params = OrderedDict()
    for name, p in params.items():
        if name == "head.weight":
            new_params[name] = W.requires_grad_(True)
        elif name == "head.bias":
            new_params[name] = b.requires_grad_(True)
        else:
            new_params[name] = p

    return new_params


# ── Episode step ─────────────────────────────────────────────────────────────

def maml_episode(
    model: MAMLModel,
    support: torch.Tensor,
    s_labels: torch.Tensor,
    query: torch.Tensor,
    q_labels: torch.Tensor,
    inner_lr: float,
    inner_steps: int,
    n_way: int,
    k_shot: int,
    use_proto_init: bool = False,
) -> dict:
    """
    Run one MAML (or ProtoMAML) episode.

    Returns:
      query_loss            -- for outer-loop backprop
      pre_support_acc       -- before adaptation
      post_support_acc      -- after adaptation
      post_query_acc        -- after adaptation, on query set
    """
    params = model.get_params()

    # ProtoMAML: initialise head from prototypes
    if use_proto_init:
        params = init_head_from_prototypes(
            model, support, s_labels, params, n_way, k_shot
        )

    # Pre-adaptation accuracy on support
    with torch.no_grad():
        pre_logits = model.forward_with_params(support, params)
        pre_support_acc = (pre_logits.argmax(1) == s_labels).float().mean().item()

    # Inner loop: adapt on support
    adapted = inner_loop(model, support, s_labels, params, inner_lr, inner_steps)

    # Post-adaptation accuracy on support
    with torch.no_grad():
        post_s_logits = model.forward_with_params(support, adapted)
        post_support_acc = (post_s_logits.argmax(1) == s_labels).float().mean().item()

    # Query loss and accuracy with adapted params
    q_logits   = model.forward_with_params(query, adapted)
    query_loss = F.cross_entropy(q_logits, q_labels)

    with torch.no_grad():
        post_query_acc = (q_logits.argmax(1) == q_labels).float().mean().item()

    return {
        "query_loss":      query_loss,
        "pre_support_acc": pre_support_acc,
        "post_support_acc": post_support_acc,
        "post_query_acc":  post_query_acc,
    }


# ── Training / validation epoch ──────────────────────────────────────────────

def run_epoch(
    model: MAMLModel,
    loader,
    optimizer,
    device: torch.device,
    n_way: int,
    k_shot: int,
    inner_lr: float,
    inner_steps: int,
    is_train: bool,
    use_proto_init: bool = False,
) -> dict[str, float]:
    """One full training or validation pass. Returns averaged metrics."""
    model.train() if is_train else model.eval()

    totals = {
        "query_loss": 0.0,
        "pre_support_acc": 0.0,
        "post_support_acc": 0.0,
        "post_query_acc": 0.0,
    }
    n_episodes = 0

    for support, s_lbls, query, q_lbls in loader:
        support = support.squeeze(0).to(device)
        s_lbls  = s_lbls.squeeze(0).to(device)
        query   = query.squeeze(0).to(device)
        q_lbls  = q_lbls.squeeze(0).to(device)

        result = maml_episode(
            model, support, s_lbls, query, q_lbls,
            inner_lr=inner_lr, inner_steps=inner_steps,
            n_way=n_way, k_shot=k_shot,
            use_proto_init=use_proto_init,
        )

        if is_train:
            optimizer.zero_grad()
            result["query_loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

        totals["query_loss"]       += result["query_loss"].item()
        totals["pre_support_acc"]  += result["pre_support_acc"]
        totals["post_support_acc"] += result["post_support_acc"]
        totals["post_query_acc"]   += result["post_query_acc"]
        n_episodes += 1

    return {k: v / n_episodes for k, v in totals.items()}


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_val_accuracy(
    history: list[float],
    save_path: Path,
    model_name: str,
    n_way: int,
    k_shot: int,
) -> None:
    """Plot post-adaptation query accuracy on validation over epochs."""
    epochs = list(range(1, len(history) + 1))
    best_epoch = history.index(max(history)) + 1

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, history, marker="o", linewidth=2, markersize=4,
            color="#4C72B0", label="Val post-adapt query acc")
    ax.axvline(best_epoch, linestyle="--", linewidth=1.2, color="#C44E52",
               label=f"Best epoch {best_epoch}: {history[best_epoch-1]:.2f}%")
    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Accuracy (%)", fontsize=13)
    ax.set_title(
        f"{model_name} -- Validation Post-Adaptation Query Accuracy\n"
        f"{n_way}-way {k_shot}-shot",
        fontsize=13,
    )
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Accuracy plot saved to: {save_path}")


# ── Training function ────────────────────────────────────────────────────────

def train_one(
    model_name: str,
    use_proto_init: bool,
    ckpt_name: str,
    args,
    device: torch.device,
    save_dir: Path,
    train_loader,
    val_loader,
):
    """Train a single MAML or ProtoMAML model."""
    print(f"\n{'='*60}")
    print(f"  Training {model_name}")
    print(f"  {args.n_way}-way {args.k_shot}-shot {args.q_query}-query")
    print(f"  Inner: {args.inner_steps} steps, lr={args.inner_lr}")
    print(f"  Outer: lr={args.outer_lr}")
    print(f"{'='*60}\n")

    torch.manual_seed(args.seed)

    model = MAMLModel(hidden_dim=args.hidden_dim, n_way=args.n_way).to(device)
    optimizer = Adam(model.parameters(), lr=args.outer_lr)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}\n")

    best_val_acc      = 0.0
    val_q_acc_history = []

    header = (
        f"{'Ep':>3}  "
        f"{'TrPreS':>7}  {'TrPosS':>7}  {'TrPosQ':>7}  {'TrLoss':>7}  "
        f"{'VaPreS':>7}  {'VaPosS':>7}  {'VaPosQ':>7}  {'VaLoss':>7}"
    )
    print(header)
    print("-" * len(header))

    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(
            model, train_loader, optimizer, device,
            args.n_way, args.k_shot, args.inner_lr, args.inner_steps,
            is_train=True, use_proto_init=use_proto_init,
        )
        va = run_epoch(
            model, val_loader, None, device,
            args.n_way, args.k_shot, args.inner_lr, args.inner_steps,
            is_train=False, use_proto_init=use_proto_init,
        )

        val_q_acc_history.append(va["post_query_acc"] * 100)

        print(
            f"{epoch:>3}  "
            f"{tr['pre_support_acc']*100:>6.2f}%  "
            f"{tr['post_support_acc']*100:>6.2f}%  "
            f"{tr['post_query_acc']*100:>6.2f}%  "
            f"{tr['query_loss']:>7.4f}  "
            f"{va['pre_support_acc']*100:>6.2f}%  "
            f"{va['post_support_acc']*100:>6.2f}%  "
            f"{va['post_query_acc']*100:>6.2f}%  "
            f"{va['query_loss']:>7.4f}"
        )

        if va["post_query_acc"] > best_val_acc:
            best_val_acc = va["post_query_acc"]
            ckpt = {
                "epoch":        epoch,
                "val_q_acc":    va["post_query_acc"],
                "val_q_loss":   va["query_loss"],
                "model":        model.state_dict(),
                "optimizer":    optimizer.state_dict(),
                "args":         vars(args),
                "model_type":   model_name,
            }
            torch.save(ckpt, save_dir / ckpt_name)

    print("-" * len(header))
    print(f"Training complete. Best val post-adapt query acc: {best_val_acc*100:.2f}%")
    print(f"Checkpoint saved to: {save_dir / ckpt_name}")

    plot_val_accuracy(
        val_q_acc_history,
        save_dir / f"val_query_acc_{ckpt_name.replace('.pt', '.png')}",
        model_name, args.n_way, args.k_shot,
    )


# ── Main ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="MAML / ProtoMAML training on MNIST")
    p.add_argument("--model",             type=str, default="both",
                   choices=["maml", "protomaml", "both"])
    p.add_argument("--n_way",             type=int, default=5)
    p.add_argument("--k_shot",            type=int, default=1)
    p.add_argument("--q_query",           type=int, default=15)
    p.add_argument("--n_train_episodes",  type=int, default=50)
    p.add_argument("--n_val_episodes",    type=int, default=50)
    p.add_argument("--epochs",            type=int, default=50)
    p.add_argument("--inner_lr",          type=float, default=0.01)
    p.add_argument("--inner_steps",       type=int, default=5)
    p.add_argument("--outer_lr",          type=float, default=1e-3)
    p.add_argument("--hidden_dim",        type=int, default=64)
    p.add_argument("--save_dir",          type=str, default="./checkpoints")
    p.add_argument("--seed",              type=int, default=42)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Model  : {args.model}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

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

    models_to_train = (
        ["maml", "protomaml"] if args.model == "both"
        else [args.model]
    )

    for m in models_to_train:
        if m == "maml":
            train_one(
                model_name="MAML", use_proto_init=False,
                ckpt_name="best_maml.pt",
                args=args, device=device, save_dir=save_dir,
                train_loader=train_loader, val_loader=val_loader,
            )
        else:
            train_one(
                model_name="ProtoMAML", use_proto_init=True,
                ckpt_name="best_protomaml.pt",
                args=args, device=device, save_dir=save_dir,
                train_loader=train_loader, val_loader=val_loader,
            )

    print("\nAll training complete.")


if __name__ == "__main__":
    main()
