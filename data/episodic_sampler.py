"""
Episodic sampler for Prototypical Networks (ProtoNets).

Each episode contains:
  - Support set : N classes × K images  →  (N*K, C, H, W)
  - Query set   : N classes × Q images  →  (N*Q, C, H, W)

Support and query sets never share images within an episode.
Labels are remapped to [0, N-1] (relative / episode-local).

Usage
-----
    from episodic_sampler import EpisodicDataset, build_episode_loader

    # Wrap any torch Dataset (MNIST, SVHN, MNIST-M, etc.)
    ep_dataset = EpisodicDataset(
        dataset    = mnist_train,
        n_episodes = 1000,   # how many episodes per "epoch"
        n_way      = 5,
        k_shot     = 5,
        q_query    = 15,
    )
    loader = DataLoader(ep_dataset, batch_size=1, collate_fn=ep_dataset.collate)

    for support, s_labels, query, q_labels in loader:
        # support : (1, N*K, C, H, W)  →  squeeze → (N*K, C, H, W)
        # query   : (1, N*Q, C, H, W)  →  squeeze → (N*Q, C, H, W)
        ...
"""

import random
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from data.prepare_datasets import load_mnist, load_svhn, load_mnistm


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_targets(dataset: Dataset) -> list[int]:
    """
    Extract integer class labels from a dataset.

    Works with:
      • torchvision datasets  (have a .targets attribute)
      • Subset wrapping any of the above
      • MNISTMDataset         (stores labels in .samples)
      • Any dataset whose underlying base has .targets or .samples
    """
    # Unwrap nested Subsets to reach the base dataset
    base    = dataset
    indices = None

    while isinstance(base, Subset):
        if indices is None:
            indices = list(base.indices)
        else:
            indices = [base.indices[i] for i in indices]
        base = base.dataset

    # Try .targets (MNIST, SVHN via torchvision)
    if hasattr(base, "targets"):
        targets = base.targets
        if isinstance(targets, torch.Tensor):
            targets = targets.tolist()
        if indices is not None:
            targets = [targets[i] for i in indices]
        return targets

    # Try .samples list of (filename, label) — MNISTMDataset
    if hasattr(base, "samples"):
        targets = [label for _, label in base.samples]
        if indices is not None:
            targets = [targets[i] for i in indices]
        return targets

    # Last resort: iterate (slow for large datasets)
    if indices is not None:
        return [int(base[i][1]) for i in indices]
    return [int(base[i][1]) for i in range(len(base))]  # type: ignore[arg-type]


def _build_class_map(dataset: Dataset) -> dict[int, list[int]]:
    """
    Returns {class_id: [idx, idx, ...]} for every class in the dataset.
    Indices are local to `dataset` (not to any underlying base dataset).
    """
    targets = _get_targets(dataset)
    class_map: dict[int, list[int]] = {}
    for idx, label in enumerate(targets):
        class_map.setdefault(label, []).append(idx)
    return class_map


# ── Core episodic dataset ─────────────────────────────────────────────────────

class EpisodicDataset(Dataset):
    """
    Wraps any image-classification Dataset and produces episodes on the fly.

    Parameters
    ----------
    dataset    : source Dataset (train / val / test split)
    n_episodes : number of episodes that constitute one "epoch"
    n_way      : N — number of classes per episode
    k_shot     : K — support images per class
    q_query    : Q — query images per class
    seed       : optional fixed seed for reproducibility (None = random)
    """

    def __init__(
        self,
        dataset: Dataset,
        n_episodes: int,
        n_way: int,
        k_shot: int,
        q_query: int,
        seed: int | None = None,
    ):
        self.dataset    = dataset
        self.n_episodes = n_episodes
        self.n_way      = n_way
        self.k_shot     = k_shot
        self.q_query    = q_query

        self.class_map  = _build_class_map(dataset)
        self.classes    = sorted(self.class_map.keys())

        # Validate that every class has enough samples
        needed = k_shot + q_query
        short  = {c: len(idxs) for c, idxs in self.class_map.items()
                  if len(idxs) < needed}
        if short:
            raise ValueError(
                f"k_shot + q_query = {needed}, but these classes have fewer "
                f"samples: {short}"
            )
        if n_way > len(self.classes):
            raise ValueError(
                f"n_way={n_way} exceeds the number of available classes "
                f"({len(self.classes)})."
            )

        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self.n_episodes

    # ------------------------------------------------------------------
    def __getitem__(self, _: Any):
        """Sample one episode. The index argument is ignored."""
        rng = self._rng

        # 1. Pick N classes at random
        episode_classes = rng.sample(self.classes, self.n_way)

        support_imgs, support_lbls = [], []
        query_imgs,   query_lbls   = [], []

        for local_label, cls in enumerate(episode_classes):
            # 2. Sample K + Q indices without replacement
            pool    = self.class_map[cls]
            chosen  = rng.sample(pool, self.k_shot + self.q_query)

            support_indices = chosen[:self.k_shot]
            query_indices   = chosen[self.k_shot:]

            for idx in support_indices:
                img, _ = self.dataset[idx]
                support_imgs.append(img)
                support_lbls.append(local_label)

            for idx in query_indices:
                img, _ = self.dataset[idx]
                query_imgs.append(img)
                query_lbls.append(local_label)

        support = torch.stack(support_imgs)            # (N*K, C, H, W)
        query   = torch.stack(query_imgs)              # (N*Q, C, H, W)
        s_lbls  = torch.tensor(support_lbls)           # (N*K,)
        q_lbls  = torch.tensor(query_lbls)             # (N*Q,)

        return support, s_lbls, query, q_lbls

    # ------------------------------------------------------------------
    @staticmethod
    def collate(batch):
        """
        Collate fn that keeps the (support, s_lbls, query, q_lbls) structure
        when DataLoader wraps individual episodes into a batch.

        Pass this as collate_fn= to DataLoader.
        """
        supports, s_lbls, queries, q_lbls = zip(*batch)
        return (
            torch.stack(supports),   # (B, N*K, C, H, W)
            torch.stack(s_lbls),     # (B, N*K)
            torch.stack(queries),    # (B, N*Q, C, H, W)
            torch.stack(q_lbls),     # (B, N*Q)
        )


# ── Convenience builder ───────────────────────────────────────────────────────

def build_episode_loader(
    dataset: Dataset,
    n_episodes: int,
    n_way: int,
    k_shot: int,
    q_query: int,
    batch_size: int = 1,
    num_workers: int = 0,
    seed: int | None = None,
) -> DataLoader:
    """
    One-liner that creates an EpisodicDataset and wraps it in a DataLoader.

    Note: num_workers > 0 requires the dataset to be picklable.
    With a fixed seed, parallel workers each get the same RNG state —
    keep num_workers=0 for fully deterministic episode generation.
    """
    ep_dataset = EpisodicDataset(
        dataset    = dataset,
        n_episodes = n_episodes,
        n_way      = n_way,
        k_shot     = k_shot,
        q_query    = q_query,
        seed       = seed,
    )
    return DataLoader(
        ep_dataset,
        batch_size  = batch_size,
        shuffle     = False,   # ordering is already random inside each episode
        num_workers = num_workers,
        collate_fn  = EpisodicDataset.collate,
    )


# ── Quick smoke-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":

    N_WAY  = 5
    K_SHOT = 5
    Q      = 15

    print(f"Episode config: {N_WAY}-way  {K_SHOT}-shot  {Q}-query\n")

    # ── MNIST (train split) ──
    mnist_train, mnist_val, mnist_test = load_mnist()

    mnist_loader = build_episode_loader(
        dataset    = mnist_train,
        n_episodes = 100,
        n_way      = N_WAY,
        k_shot     = K_SHOT,
        q_query    = Q,
        seed       = 42,
    )

    support, s_lbls, query, q_lbls = next(iter(mnist_loader))
    print(f"[MNIST train]")
    print(f"  support : {tuple(support.shape)}   labels: {tuple(s_lbls.shape)}")
    print(f"  query   : {tuple(query.shape)}   labels: {tuple(q_lbls.shape)}")
    print(f"  classes in episode : {q_lbls[0].unique().tolist()}\n")

    # ── SVHN (val split) ──
    svhn_val, svhn_test = load_svhn()

    svhn_loader = build_episode_loader(
        dataset    = svhn_val,
        n_episodes = 100,
        n_way      = N_WAY,
        k_shot     = K_SHOT,
        q_query    = Q,
        seed       = 42,
    )

    support, s_lbls, query, q_lbls = next(iter(svhn_loader))
    print(f"[SVHN val]")
    print(f"  support : {tuple(support.shape)}   labels: {tuple(s_lbls.shape)}")
    print(f"  query   : {tuple(query.shape)}   labels: {tuple(q_lbls.shape)}")
    print(f"  classes in episode : {q_lbls[0].unique().tolist()}\n")

    # ── MNIST-M (train split) ──
    mnistm_val, mnistm_test = load_mnistm()

    mnistm_loader = build_episode_loader(
        dataset    = mnistm_test,
        n_episodes = 100,
        n_way      = N_WAY,
        k_shot     = K_SHOT,
        q_query    = Q,
        seed       = 42,
    )

    support, s_lbls, query, q_lbls = next(iter(mnistm_loader))
    print(f"[MNIST-M train]")
    print(f"  support : {tuple(support.shape)}   labels: {tuple(s_lbls.shape)}")
    print(f"  query   : {tuple(query.shape)}   labels: {tuple(q_lbls.shape)}")
    print(f"  classes in episode : {q_lbls[0].unique().tolist()}")