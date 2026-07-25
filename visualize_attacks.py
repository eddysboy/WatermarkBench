"""Visualize all attack transforms on the small_test image.

Reads the single image in datasets/small_test, applies every transform
defined in evaluation/attacks._SHARED_ATTACKS, and lays out the results
on a grid with 5 columns and the transform name in the top-left corner.
"""

import sys
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import functional as TVF

# Ensure the project root is on sys.path
PROJECT_ROOT = Path(r"D:\document\Codex\WatermarkBench")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.attacks import _SHARED_ATTACKS, COMMON_ATTACK_KEYS


def pil_to_tensor(pil_image: Image.Image) -> torch.Tensor:
    """Convert PIL RGB image to [1, C, H, W] tensor in [0, 1]."""
    t = TVF.to_tensor(pil_image)  # [C, H, W]
    return t.unsqueeze(0)  # [1, C, H, W]


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """Convert [1, C, H, W] tensor in [0, 1] back to PIL RGB."""
    t = t.squeeze(0).clamp(0, 1)
    return TVF.to_pil_image(t)


def main():
    # Locate the image
    image_path = PROJECT_ROOT / "datasets" / "small_test" / "000000014439.jpg"
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        sys.exit(1)

    pil_img = Image.open(image_path).convert("RGB")
    x = pil_to_tensor(pil_img)  # [1, C, H, W]

    # Collect results: list of (label, PIL image)
    results = []
    results.append(("Original", pil_img))

    # Only show the strongest variant per attack type.
    # Categories: JPEG (Q=20), GaussianBlur (k=9), Brightness (x1.50),
    # Contrast (x1.50), Saturation (x1.50), GaussianNoise (s=0.05),
    # SaltPepper (p=0.02), Rotate (30 deg), Resize (0.50x), Crop (0.50x),
    # HFlip, TextOverlay (a=0.50), and all combo attacks.
    STRONGEST_KEYS = [
        "jpeg_20",
        "gaussian_blur_k9",
        "brightness_1.50",
        "contrast_1.50",
        "saturation_1.50",
        "gaussian_noise_0.05",
        "salt_pepper_0.02",
        "rotate_30",
        "resize_0.50",
        "crop_0.50",
        "hflip",
        "text_overlay_1.00",
        
    ] + [k for k in COMMON_ATTACK_KEYS if k.startswith("combo_")]

    for key in STRONGEST_KEYS:
        if key not in _SHARED_ATTACKS:
            continue
        label, fn = _SHARED_ATTACKS[key]
        with torch.no_grad():
            transformed = fn(x.clone())
        results.append((label, tensor_to_pil(transformed)))

    # Layout: 5 columns
    ncols = 5
    nrows = (len(results) + ncols - 1) // ncols

    # Size each cell to match the original image resolution
    img_w, img_h = pil_img.size       # 640 x 404
    dpi = 200
    cell_w = img_w / dpi              # inches per cell = 3.2
    cell_h = img_h / dpi              # inches per cell = 2.02

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * cell_w, nrows * cell_h),
        dpi=dpi,
        gridspec_kw={"wspace": -0.05, "hspace": 0.10},
    )
    axes = axes.flatten()

    for idx, (label, pil_out) in enumerate(results):
        ax = axes[idx]
        ax.imshow(pil_out)
        ax.text(
            0.01, 1.01, label,
            transform=ax.transAxes, fontsize=6, color="black", fontweight="bold",
            va="bottom", ha="left", clip_on=False,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.55, lw=0),
        )
        ax.axis("off")

    for idx in range(len(results), len(axes)):
        axes[idx].set_visible(False)

    out_path = PROJECT_ROOT / "attack_visualization.png"
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved {len(results)} images (1 original + {len(results)-1} attacks) to {out_path}")


if __name__ == "__main__":
    main()