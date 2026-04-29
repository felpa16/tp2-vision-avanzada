"""
Domain-specific preprocessing pipelines for MNIST-M and SVHN.

Both pipelines convert RGB 32x32 images into grayscale 28x28 images that
approximate the MNIST appearance (smooth anti-aliased white digit on black
background), so they can be fed into an encoder trained exclusively on
original MNIST.

Design principle: preserve grayscale gradients — the encoder was trained on
anti-aliased MNIST, NOT binary images.  Hard thresholding destroys the
gradient information the encoder relies on, so we avoid it.

Each function takes a torch tensor (C, H, W) in [0, 1] and returns a
tensor (1, 28, 28) in [0, 1].
"""

import cv2
import numpy as np
import torch


# ── Tuneable defaults ────────────────────────────────────────────────────────

# MNIST-M
MNISTM_COLOR_SPACE         = "hsv"       # "lab", "hsv", or "gray"
MNISTM_BLUR_METHOD         = "gaussian" # "bilateral", "gaussian", or "none"
MNISTM_BILATERAL_D         = 5
MNISTM_BILATERAL_SIGMA_CLR = 75
MNISTM_BILATERAL_SIGMA_SPC = 75
MNISTM_GAUSSIAN_KSIZE      = 3
MNISTM_USE_CLAHE           = False
MNISTM_CLAHE_CLIP          = 2.0
MNISTM_CLAHE_GRID          = (8, 8)
MNISTM_POLARITY_MARGIN     = 0.3        # fraction of image size for border

# SVHN
SVHN_CROP_SIZE             = 32           # center-crop to reduce neighbouring digit clutter
SVHN_COLOR_SPACE           = "hsv"      # "lab", "hsv", or "gray"
SVHN_BLUR_METHOD           = "gaussian" # "bilateral", "gaussian", or "none"
SVHN_BILATERAL_D           = 9
SVHN_BILATERAL_SIGMA_CLR   = 75
SVHN_BILATERAL_SIGMA_SPC   = 75
SVHN_GAUSSIAN_KSIZE        = 5
SVHN_USE_CLAHE             = True
SVHN_CLAHE_CLIP            = 2.0
SVHN_CLAHE_GRID            = (8, 8)
SVHN_POLARITY_MARGIN       = 0.3


# ── Helpers ──────────────────────────────────────────────────────────────────

def _tensor_to_uint8(img: torch.Tensor) -> np.ndarray:
    """Convert a (C, H, W) float tensor in [0,1] to a (H, W, C) uint8 numpy array."""
    arr = img.permute(1, 2, 0).numpy()
    return np.clip(arr * 255, 0, 255).astype(np.uint8)


def _uint8_gray_to_tensor(gray: np.ndarray) -> torch.Tensor:
    """Convert an (H, W) uint8 grayscale image to a (1, 28, 28) float tensor."""
    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(resized).unsqueeze(0).float() / 255.0


def _extract_luminance(bgr: np.ndarray, color_space: str) -> np.ndarray:
    """Extract a single-channel luminance image from a BGR input."""
    if color_space == "lab":
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[:, :, 0]
    elif color_space == "hsv":
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 2]
    else:  # "gray"
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def _apply_blur(bgr: np.ndarray, method: str, bilateral_d: int,
                bilateral_sigma_color: float, bilateral_sigma_space: float,
                gaussian_ksize: int) -> np.ndarray:
    """Apply the selected blur method to a BGR image."""
    if method == "bilateral":
        return cv2.bilateralFilter(bgr, bilateral_d,
                                   bilateral_sigma_color, bilateral_sigma_space)
    elif method == "gaussian":
        return cv2.GaussianBlur(bgr, (gaussian_ksize, gaussian_ksize), 0)
    else:  # "none"
        return bgr


def _ensure_white_on_black(gray: np.ndarray, margin_frac: float = 0.2) -> np.ndarray:
    """
    Dynamic polarity heuristic: compare border brightness vs centre brightness.
    If the border is lighter than the centre, invert so the background is black.

    margin_frac: fraction of the smaller dimension used as border width.
    """
    h, w = gray.shape
    margin = max(1, int(min(h, w) * margin_frac))

    border_mean = np.mean(np.concatenate([
        gray[:margin, :].ravel(),
        gray[-margin:, :].ravel(),
        gray[:, :margin].ravel(),
        gray[:, -margin:].ravel(),
    ]))
    centre = gray[margin:-margin, margin:-margin]
    centre_mean = np.mean(centre) if centre.size > 0 else border_mean

    if border_mean > centre_mean:
        return cv2.bitwise_not(gray)
    return gray


# ── Pipeline 1: MNIST-M ─────────────────────────────────────────────────────

def preprocess_mnistm(
    img: torch.Tensor,
    color_space: str = MNISTM_COLOR_SPACE,
    blur_method: str = MNISTM_BLUR_METHOD,
    bilateral_d: int = MNISTM_BILATERAL_D,
    bilateral_sigma_color: float = MNISTM_BILATERAL_SIGMA_CLR,
    bilateral_sigma_space: float = MNISTM_BILATERAL_SIGMA_SPC,
    gaussian_ksize: int = MNISTM_GAUSSIAN_KSIZE,
    use_clahe: bool = MNISTM_USE_CLAHE,
    clahe_clip: float = MNISTM_CLAHE_CLIP,
    clahe_grid: tuple[int, int] = MNISTM_CLAHE_GRID,
    polarity_margin: float = MNISTM_POLARITY_MARGIN,
) -> torch.Tensor:
    """
    MNIST-M preprocessing: remove coloured textures while keeping gradients.

    Steps:
      1. Blur (bilateral / gaussian / none) — smooth texture noise.
      2. Luminance extraction (LAB-L / HSV-V / standard gray).
      3. Optional CLAHE — homogenise local contrast.
      4. Dynamic polarity — ensure white-on-black via border heuristic.
      5. Resize to 28x28.

    Parameters
    ----------
    img : torch.Tensor, shape (3, H, W), float [0, 1]

    Returns
    -------
    torch.Tensor, shape (1, 28, 28), float [0, 1]
    """
    rgb = _tensor_to_uint8(img)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # 1. Blur
    bgr = _apply_blur(bgr, blur_method, bilateral_d,
                       bilateral_sigma_color, bilateral_sigma_space, gaussian_ksize)

    # 2. Luminance extraction
    gray = _extract_luminance(bgr, color_space)

    # 3. Optional CLAHE
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
        gray = clahe.apply(gray)

    # 4. Dynamic polarity
    gray = _ensure_white_on_black(gray, polarity_margin)

    # 5. Resize to 28x28 and return
    return _uint8_gray_to_tensor(gray)


# ── Pipeline 2: SVHN ────────────────────────────────────────────────────────

def preprocess_svhn(
    img: torch.Tensor,
    crop_size: int = SVHN_CROP_SIZE,
    color_space: str = SVHN_COLOR_SPACE,
    blur_method: str = SVHN_BLUR_METHOD,
    bilateral_d: int = SVHN_BILATERAL_D,
    bilateral_sigma_color: float = SVHN_BILATERAL_SIGMA_CLR,
    bilateral_sigma_space: float = SVHN_BILATERAL_SIGMA_SPC,
    gaussian_ksize: int = SVHN_GAUSSIAN_KSIZE,
    use_clahe: bool = SVHN_USE_CLAHE,
    clahe_clip: float = SVHN_CLAHE_CLIP,
    clahe_grid: tuple[int, int] = SVHN_CLAHE_GRID,
    polarity_margin: float = SVHN_POLARITY_MARGIN,
) -> torch.Tensor:
    """
    SVHN preprocessing: isolate the central digit from real-world photos.

    Steps:
      1. Center crop — reduce distracting neighbouring digits.
      2. Blur (bilateral / gaussian / none) — smooth background textures.
      3. Luminance extraction (LAB-L / HSV-V / standard gray).
      4. Optional CLAHE — flatten harsh shadows and lighting gradients.
      5. Dynamic polarity — ensure white-on-black via border heuristic.
      6. Resize to 28x28.

    Parameters
    ----------
    img : torch.Tensor, shape (3, H, W), float [0, 1]

    Returns
    -------
    torch.Tensor, shape (1, 28, 28), float [0, 1]
    """
    rgb = _tensor_to_uint8(img)
    h, w = rgb.shape[:2]

    # 1. Center crop
    y0 = (h - crop_size) // 2
    x0 = (w - crop_size) // 2
    rgb = rgb[y0:y0 + crop_size, x0:x0 + crop_size]

    # 2. Blur
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    bgr = _apply_blur(bgr, blur_method, bilateral_d,
                       bilateral_sigma_color, bilateral_sigma_space, gaussian_ksize)

    # 3. Luminance extraction
    gray = _extract_luminance(bgr, color_space)

    # 4. Optional CLAHE
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
        gray = clahe.apply(gray)

    # 5. Dynamic polarity
    gray = _ensure_white_on_black(gray, polarity_margin)

    # 6. Resize to 28x28 and return
    return _uint8_gray_to_tensor(gray)


# ── Pipeline visualization ──────────────────────────────────────────────────

def _collect_mnistm_steps(img: torch.Tensor) -> list[tuple[str, np.ndarray]]:
    """Return (title, uint8_image) for each step of the MNIST-M pipeline."""
    rgb = _tensor_to_uint8(img)
    steps = [("Original RGB", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))]

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    bgr = _apply_blur(bgr, MNISTM_BLUR_METHOD, MNISTM_BILATERAL_D,
                       MNISTM_BILATERAL_SIGMA_CLR, MNISTM_BILATERAL_SIGMA_SPC,
                       MNISTM_GAUSSIAN_KSIZE)
    steps.append((f"Blur ({MNISTM_BLUR_METHOD})", bgr.copy()))

    gray = _extract_luminance(bgr, MNISTM_COLOR_SPACE)
    steps.append((f"Luminance ({MNISTM_COLOR_SPACE.upper()})", gray.copy()))

    if MNISTM_USE_CLAHE:
        clahe = cv2.createCLAHE(clipLimit=MNISTM_CLAHE_CLIP, tileGridSize=MNISTM_CLAHE_GRID)
        gray = clahe.apply(gray)
        steps.append(("CLAHE", gray.copy()))

    gray = _ensure_white_on_black(gray, MNISTM_POLARITY_MARGIN)
    steps.append(("Polarity fix", gray.copy()))

    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    steps.append(("Resize 28x28", resized))

    return steps


def _collect_svhn_steps(img: torch.Tensor) -> list[tuple[str, np.ndarray]]:
    """Return (title, uint8_image) for each step of the SVHN pipeline."""
    rgb = _tensor_to_uint8(img)
    steps = [("Original RGB", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))]

    h, w = rgb.shape[:2]
    y0 = (h - SVHN_CROP_SIZE) // 2
    x0 = (w - SVHN_CROP_SIZE) // 2
    rgb = rgb[y0:y0 + SVHN_CROP_SIZE, x0:x0 + SVHN_CROP_SIZE]
    steps.append((f"Center crop {SVHN_CROP_SIZE}x{SVHN_CROP_SIZE}",
                  cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)))

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    bgr = _apply_blur(bgr, SVHN_BLUR_METHOD, SVHN_BILATERAL_D,
                       SVHN_BILATERAL_SIGMA_CLR, SVHN_BILATERAL_SIGMA_SPC,
                       SVHN_GAUSSIAN_KSIZE)
    steps.append((f"Blur ({SVHN_BLUR_METHOD})", bgr.copy()))

    gray = _extract_luminance(bgr, SVHN_COLOR_SPACE)
    steps.append((f"Luminance ({SVHN_COLOR_SPACE.upper()})", gray.copy()))

    if SVHN_USE_CLAHE:
        clahe = cv2.createCLAHE(clipLimit=SVHN_CLAHE_CLIP, tileGridSize=SVHN_CLAHE_GRID)
        gray = clahe.apply(gray)
        steps.append(("CLAHE", gray.copy()))

    gray = _ensure_white_on_black(gray, SVHN_POLARITY_MARGIN)
    steps.append(("Polarity fix", gray.copy()))

    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    steps.append(("Resize 28x28", resized))

    return steps


def save_pipeline_examples(
    mnistm_dataset,
    svhn_dataset,
    n_examples: int = 10,
    save_dir: str = "checkpoints",
    seed: int = 42,
) -> None:
    """
    Save side-by-side images showing each preprocessing step for random
    samples from MNIST-M and SVHN.

    Generates two PNG files:
      - <save_dir>/pipeline_mnistm.png
      - <save_dir>/pipeline_svhn.png

    Each image is a grid: rows = samples, columns = pipeline steps.
    """
    import matplotlib.pyplot as plt
    import random as _random
    from pathlib import Path

    rng = _random.Random(seed)
    out = Path(save_dir)
    out.mkdir(parents=True, exist_ok=True)

    for ds_name, dataset, step_fn in [
        ("mnistm", mnistm_dataset, _collect_mnistm_steps),
        ("svhn",   svhn_dataset,   _collect_svhn_steps),
    ]:
        indices = rng.sample(range(len(dataset)), n_examples)

        # Collect all steps for all samples to find column count
        all_rows = []
        for idx in indices:
            img, label = dataset[idx]
            steps = step_fn(img)
            all_rows.append((label, steps))

        n_cols = max(len(steps) for _, steps in all_rows)

        fig, axes = plt.subplots(n_examples, n_cols,
                                 figsize=(2.5 * n_cols, 2.5 * n_examples))
        if n_examples == 1:
            axes = [axes]

        for row_idx, (label, steps) in enumerate(all_rows):
            for col_idx in range(n_cols):
                ax = axes[row_idx][col_idx]
                if col_idx < len(steps):
                    title, im = steps[col_idx]
                    if im.ndim == 3:
                        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
                    else:
                        ax.imshow(im, cmap="gray", vmin=0, vmax=255)
                    if row_idx == 0:
                        ax.set_title(title, fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])
                if col_idx == 0:
                    ax.set_ylabel(f"Label {label}", fontsize=9)

        fig.suptitle(f"Preprocessing pipeline: {ds_name.upper()}", fontsize=13, y=1.01)
        fig.tight_layout()
        path = out / f"pipeline_{ds_name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Pipeline visualization saved to: {path}")


# ── CLI entry point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    from data.prepare_datasets import load_mnistm, load_svhn

    print("Loading datasets ...")
    _, mnistm_val, _ = load_mnistm()
    svhn_val, _      = load_svhn()

    save_pipeline_examples(mnistm_val, svhn_val)
