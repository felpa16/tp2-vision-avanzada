"""
Dataset preparation script for MNIST, SVHN, and MNIST-M using torchvision.

MNIST   → train / validation / test
SVHN    → validation / test
MNIST-M → train / validation / test
"""

import os
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from torchvision import datasets, transforms


# ── Constants ────────────────────────────────────────────────────────────────

DATA_DIR          = "./data"
SEED              = 42

# MNIST split sizes (out of 60 000 training samples)
MNIST_TRAIN_SIZE  = 50_000
MNIST_VAL_SIZE    = 10_000

# SVHN split sizes (out of 26 032 test samples)
SVHN_VAL_SIZE     = 10_000
SVHN_TEST_SIZE    = 16_032   # remainder

# MNIST-M split sizes (out of 60 000 training samples)
MNISTM_TRAIN_SIZE = 50_000
MNISTM_VAL_SIZE   = 10_000

BATCH_SIZE        = 64

# ── Transforms ───────────────────────────────────────────────────────────────

mnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),   # MNIST mean / std
])

# SVHN is RGB 32×32; normalise per-channel
svhn_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4377, 0.4438, 0.4728),
                         (0.1980, 0.2010, 0.1970)),
])

# MNIST-M is RGB 32×32 with coloured backgrounds
mnistm_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4570, 0.4570, 0.4570),
                         (0.2570, 0.2570, 0.2570)),
])


# ── MNIST-M Custom Dataset ────────────────────────────────────────────────────

class MNISTMDataset(Dataset):
    """
    MNIST-M dataset loader for the flat-folder + label-file layout.

    Expected structure:
        <root>/
            mnist_m_train/
                00001.png
                00002.png
                ...
            mnist_m_train_labels.txt      # "00001.png 3"
            mnist_m_test/
                00001.png
                ...
            mnist_m_test_labels.txt       # "00001.png 7"

    The label file must have one entry per line in the format:
        <filename> <integer_label>
    """

    def __init__(self, img_dir: str, label_file: str, transform=None):
        self.img_dir   = Path(img_dir)
        self.transform = transform
        self.samples   = []   # list of (filename, label) tuples

        with open(label_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 2:
                    raise ValueError(
                        f"Expected 'filename label' per line, got: '{line}'"
                    )
                filename, label = parts
                self.samples.append((filename, int(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, label = self.samples[idx]
        img_path = self.img_dir / filename
        image    = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# ── MNIST ────────────────────────────────────────────────────────────────────

def load_mnist(data_dir: str = DATA_DIR):
    """
    Downloads MNIST and returns (train_set, val_set, test_set).

    The official 60 000-sample training split is further divided into
    50 000 training and 10 000 validation samples.
    The official 10 000-sample test split is kept as-is.
    """
    full_train = datasets.MNIST(
        root=data_dir, train=True, download=True, transform=mnist_transform
    )
    test_set = datasets.MNIST(
        root=data_dir, train=False, download=True, transform=mnist_transform
    )

    generator = torch.Generator().manual_seed(SEED)
    train_set, val_set = random_split(
        full_train,
        [MNIST_TRAIN_SIZE, MNIST_VAL_SIZE],
        generator=generator,
    )

    print(
        f"[MNIST]  train={len(train_set):>6,} | "
        f"val={len(val_set):>6,} | "
        f"test={len(test_set):>6,}"
    )
    return train_set, val_set, test_set


# ── SVHN ─────────────────────────────────────────────────────────────────────

def load_svhn(data_dir: str = DATA_DIR):
    """
    Downloads SVHN and returns (val_set, test_set).

    SVHN's official test split (26 032 samples) is divided into
    10 000 validation and 16 032 test samples.

    Note: SVHN also has an 'extra' split (~531k samples) which is not
    loaded here. Set split='extra' if you need it.
    """
    full_test = datasets.SVHN(
        root=data_dir, split="test", download=True, transform=svhn_transform
    )

    total = len(full_test)                          # 26 032
    val_size  = SVHN_VAL_SIZE
    test_size = total - val_size                    # 16 032

    generator = torch.Generator().manual_seed(SEED)
    indices   = torch.randperm(total, generator=generator).tolist()

    val_set  = Subset(full_test, indices[:val_size])
    test_set = Subset(full_test, indices[val_size:])

    print(
        f"[SVHN ]  train=     N/A | "
        f"val={len(val_set):>6,} | "
        f"test={len(test_set):>6,}"
    )
    return val_set, test_set



# ── MNIST-M ───────────────────────────────────────────────────────────────────

def load_mnistm(data_dir: str = DATA_DIR):
    """
    Loads MNIST-M from a flat-folder + label-file layout and returns
    (train_set, val_set, test_set).

    The 60 000-sample training split is divided into 50 000 train and
    10 000 validation samples.  The test split is kept as-is.
    """
    mnistm_dir = Path(data_dir) / "mnist_m"

    # ── adjust these two pairs if your filenames differ ──
    train_img_dir  = mnistm_dir / "mnist_m_train"
    train_lbl_file = mnistm_dir / "mnist_m_train_labels.txt"
    test_img_dir   = mnistm_dir / "mnist_m_test"
    test_lbl_file  = mnistm_dir / "mnist_m_test_labels.txt"

    full_train = MNISTMDataset(train_img_dir, train_lbl_file, transform=mnistm_transform)
    test_set   = MNISTMDataset(test_img_dir,  test_lbl_file,  transform=mnistm_transform)

    total     = len(full_train)
    val_size  = min(MNISTM_VAL_SIZE, total // 5)   # at most 20% for val
    train_size = total - val_size

    generator  = torch.Generator().manual_seed(SEED)
    train_set, val_set = random_split(
        full_train,
        [train_size, val_size],
        generator=generator,
    )

    print(
        f"[MNIST-M] train={len(train_set):>6,} | "
        f"val={len(val_set):>6,} | "
        f"test={len(test_set):>6,}"
    )
    return train_set, val_set, test_set


# ── DataLoaders ───────────────────────────────────────────────────────────────

def make_loaders(
    mnist_splits: tuple,
    svhn_splits: tuple,
    mnistm_splits: tuple,
    batch_size: int = BATCH_SIZE,
) -> dict:
    """Wraps every split in a DataLoader and returns them in a dict."""
    mnist_train,  mnist_val,  mnist_test   = mnist_splits
    svhn_val,     svhn_test                = svhn_splits
    mnistm_train, mnistm_val, mnistm_test  = mnistm_splits

    loaders = {
        "mnist_train":  DataLoader(mnist_train,  batch_size=batch_size,
                                   shuffle=True,  num_workers=2),
        "mnist_val":    DataLoader(mnist_val,    batch_size=batch_size,
                                   shuffle=False, num_workers=2),
        "mnist_test":   DataLoader(mnist_test,   batch_size=batch_size,
                                   shuffle=False, num_workers=2),
        "svhn_val":     DataLoader(svhn_val,     batch_size=batch_size,
                                   shuffle=False, num_workers=2),
        "svhn_test":    DataLoader(svhn_test,    batch_size=batch_size,
                                   shuffle=False, num_workers=2),
        "mnistm_train": DataLoader(mnistm_train, batch_size=batch_size,
                                   shuffle=True,  num_workers=2),
        "mnistm_val":   DataLoader(mnistm_val,   batch_size=batch_size,
                                   shuffle=False, num_workers=2),
        "mnistm_test":  DataLoader(mnistm_test,  batch_size=batch_size,
                                   shuffle=False, num_workers=2),
    }
    return loaders


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Downloading / loading datasets …\n")

    mnist_splits  = load_mnist()
    svhn_splits   = load_svhn()
    mnistm_splits = load_mnistm()

    loaders = make_loaders(mnist_splits, svhn_splits, mnistm_splits)

    print("\nDataLoaders ready:")
    for name, loader in loaders.items():
        print(f"  {name:<15} → {len(loader):>4} batches  (batch_size={BATCH_SIZE})")

    # Quick sanity check — peek at one batch from each loader
    print("\nSanity check (shape of first batch):")
    for name, loader in loaders.items():
        imgs, labels = next(iter(loader))
        print(f"  {name:<16} images={tuple(imgs.shape)}  labels={tuple(labels.shape)}")