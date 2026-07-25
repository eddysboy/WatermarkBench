#!/usr/bin/env python3
"""Prepare datasets and checkpoints for WatermarkBench first-time setup.

Downloads missing checkpoint files and sets up test image datasets.
All operations are idempotent -- re-running is safe.

Usage:
    python prepare.py                  # setup everything (default)
    python prepare.py --check-only     # only verify, don't download
    python prepare.py --no-coco        # skip COCO dataset download

What this does:
    1. Copies 5 built-in images from WAM assets to datasets/images_5/
    2. Downloads InvisMark paper.ckpt checkpoint
    3. Downloads watermark-anything checkpoint + config
    4. Pre-downloads FID/CMMD backbone checkpoints (LPIPS AlexNet, InceptionV3,
       OpenCLIP) for offline use
    5. Optionally downloads COCO val2017 for large-scale evaluation
"""

import argparse
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False
from urllib.request import urlretrieve
from urllib.error import URLError

ROOT = Path(__file__).resolve().parent
CHECKPOINTS_DIR = ROOT / "checkpoints"
DATASETS_DIR = ROOT / "datasets"
MODELS_DIR = ROOT / "models"

# ---------------------------------------------------------------------------
# Download URLs (verified as of 2025)
# ---------------------------------------------------------------------------
URLS = {
    # watermark-anything checkpoint + config from Meta FAIR
    "wam_checkpoint": "https://dl.fbaipublicfiles.com/watermark_anything/wam_mit.pth",
    # params.json is in the repo already, but we'll ensure it exists

    # LPIPS AlexNet backbone (torch hub auto-download URL)
    "lpips_alexnet": "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth",

    # OpenCLIP ViT-B-32 for CMMD
    "open_clip": "https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K/resolve/main/open_clip_pytorch_model.bin",
}


def parse_args():
    p = argparse.ArgumentParser(description="WatermarkBench first-time setup")
    p.add_argument(
        "--check-only", action="store_true",
        help="Only verify existing files, do not download anything"
    )
    p.add_argument(
        "--no-coco", action="store_true",
        help="Skip COCO val2017 dataset download"
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _status(msg: str):
    print(f"  {msg}")


def _ok(msg: str = "OK"):
    print(f"    \u2713 {msg}")


def _skip(msg: str):
    print(f"    \u25CB {msg}")


def _fail(msg: str):
    print(f"    \u2717 {msg}")


def _download(url: str, dest: Path, desc: str) -> bool:
    """Download a file with progress, returns True on success."""
    if dest.exists():
        _skip(f"{desc} already exists ({_human_size(dest.stat().st_size)})")
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"    Downloading {desc} ...", end=" ", flush=True)
    try:
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        urlretrieve(url, str(tmp))
        tmp.rename(dest)
        print(f"{_human_size(dest.stat().st_size)}")
        return True
    except (URLError, OSError) as e:
        print(f"FAILED: {e}")
        if tmp.exists():
            tmp.unlink()
        return False


def _human_size(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


# ---------------------------------------------------------------------------
# Setup steps
# ---------------------------------------------------------------------------

def setup_images_5():
    """Copy 5 built-in WAM test images to datasets/images_5/."""
    print("\n[1/5] Setting up test images (images_5) ...")
    src_dir = MODELS_DIR / "watermark-anything" / "assets" / "images"
    dst_dir = DATASETS_DIR / "images_5"
    dst_dir.mkdir(parents=True, exist_ok=True)

    expected = ["alpaca.jpg", "ducks.jpg", "gauguin_256.jpg",
                "seabackground.jpg", "trex_bike.jpg"]
    for name in expected:
        src = src_dir / name
        dst = dst_dir / name
        if not src.exists():
            _fail(f"Source image missing: {src}")
            continue
        if dst.exists():
            _skip(f"{name} already present")
        else:
            shutil.copy2(src, dst)
            _ok(f"Copied {name}")
    _status(f"Test images: {len(list(dst_dir.glob('*.jpg')))} files in {dst_dir}")


def setup_invis_mark():
    """Ensure InvisMark checkpoint exists. Downloads from HuggingFace."""
    print("\n[2/5] InvisMark checkpoint ...")
    ckpt_dir = MODELS_DIR / "InvisMark" / "ckpts"
    ckpt_path = ckpt_dir / "paper.ckpt"

    if ckpt_path.exists():
        _ok(f"paper.ckpt present ({_human_size(ckpt_path.stat().st_size)})")
        return

    if not _HF_AVAILABLE:
        _fail("huggingface_hub not installed. Run: pip install huggingface_hub")
        return

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    _status("Downloading from huggingface.co/shelock/watermark-invismark-model ...")
    try:
        downloaded = hf_hub_download(
            repo_id="shelock/watermark-invismark-model",
            filename="paper.ckpt",
            local_dir=ckpt_dir,
            local_dir_use_symlinks=False,
        )
        _ok(f"Downloaded paper.ckpt ({_human_size(Path(downloaded).stat().st_size)})")
    except Exception as e:
        _fail(f"Download failed: {e}")

def setup_wam():
    """Ensure watermark-anything checkpoint exists."""
    print("\n[3/5] Watermark-Anything checkpoint ...")
    ckpt_dir = MODELS_DIR / "watermark-anything" / "checkpoints"
    ckpt_path = ckpt_dir / "checkpoint.pth"
    params_path = ckpt_dir / "params.json"

    # Checkpoint
    if ckpt_path.exists():
        _ok(f"checkpoint.pth present ({_human_size(ckpt_path.stat().st_size)})")
    else:
        _download(URLS["wam_checkpoint"], ckpt_path, "WAM checkpoint.pth")

    # params.json -- should be in repo
    if params_path.exists():
        _ok("params.json present")
    else:
        # Try to copy from the repo notebooks directory as fallback
        alt = MODELS_DIR / "watermark-anything" / "notebooks" / "params.json"
        if alt.exists():
            shutil.copy2(alt, params_path)
            _ok("params.json copied from notebooks/")
        else:
            _fail("params.json NOT found -- this file should be in the repository.")


def setup_fid_checkpoints():
    """Pre-download FID/CMMD backbone models for offline use."""
    print("\n[4/5] LPIPS / CMMD backbone checkpoints ...")
    CHECKPOINTS_DIR.mkdir(exist_ok=True)

    # LPIPS AlexNet
    _download(
        URLS["lpips_alexnet"],
        CHECKPOINTS_DIR / "alexnet-owt-7be5be79.pth",
        "LPIPS AlexNet backbone"
    )

    # OpenCLIP for CMMD
    _download(
        URLS["open_clip"],
        CHECKPOINTS_DIR / "open_clip_pytorch_model.bin",
        "CMMD OpenCLIP backbone"
    )


def setup_coco(skip: bool = False):
    """Optionally download COCO val2017."""
    print("\n[5/5] COCO val2017 dataset ...")
    if skip:
        _skip("Skipped (--no-coco)")
        return

    coco_dir = DATASETS_DIR / "COCO_val2017" / "val2017"
    if coco_dir.is_dir() and len(list(coco_dir.glob("*.jpg"))) >= 100:
        _ok(f"COCO val2017 present ({len(list(coco_dir.glob('*.jpg')))} images)")
        return

    _status("COCO val2017 (~1 GB) not found.")
    _status("To download manually:")
    _status("  wget http://images.cocodataset.org/zips/val2017.zip")
    _status(f"  unzip val2017.zip -d {coco_dir.parent}")
    _status("Or use --no-coco to skip this step.")
    _skip("COCO download skipped (manual step required).")


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_all() -> dict:
    """Check all required files and return status dict."""
    checks = {}

    # Test images
    img_dir = DATASETS_DIR / "images_5"
    checks["images_5"] = img_dir.is_dir() and len(list(img_dir.glob("*.jpg"))) >= 5

    # InvisMark
    checks["invis_mark"] = (MODELS_DIR / "InvisMark" / "ckpts" / "paper.ckpt").exists()

    # WAM
    wam_ckpt = MODELS_DIR / "watermark-anything" / "checkpoints" / "checkpoint.pth"
    wam_params = MODELS_DIR / "watermark-anything" / "checkpoints" / "params.json"
    checks["wam"] = wam_ckpt.exists() and wam_params.exists()

    # MiniWatermarkDemo
    checks["miniwm"] = (MODELS_DIR / "MiniWatermarkDemo" / "checkpoints"
                        / "frequency_dct_v4_keyed32_patch.pt").exists()

    # FID/CMMD checkpoints
    checks["lpips"] = (CHECKPOINTS_DIR / "alexnet-owt-7be5be79.pth").exists()
    checks["cmmd"] = (CHECKPOINTS_DIR / "open_clip_pytorch_model.bin").exists()

    return checks


def print_verdict(checks: dict):
    """Print a summary table of what's ready and what's missing."""
    print("\n" + "=" * 60)
    print("  VERIFICATION SUMMARY")
    print("=" * 60)

    labels = {
        "images_5":     "Test images (images_5)",
        "invis_mark":   "InvisMark checkpoint",
        "miniwm":       "MiniWatermarkDemo checkpoint",
        "wam":          "Watermark-Anything checkpoint",
        "lpips":        "LPIPS backbone (AlexNet)",
        "cmmd":         "CMMD backbone (OpenCLIP)",
    }

    all_ok = True
    for key, label in labels.items():
        ok = checks.get(key, False)
        mark = "\u2713" if ok else "\u2717"
        print(f"  {mark} {label}")
        if not ok:
            all_ok = False

    print("-" * 60)
    if all_ok:
        print("  All checks passed. Ready to run evaluation.")
        print(f"\n  python run_evaluation.py --config configs/default.yaml")
    else:
        print("  Some items are missing. Re-run without --check-only to download.")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print("=" * 60)
    print("  WatermarkBench -- First-Time Setup")
    print(f"  Project root: {ROOT}")
    if args.check_only:
        print("  Mode: CHECK ONLY (no downloads)")
    print("=" * 60)

    if args.check_only:
        checks = verify_all()
        print_verdict(checks)
        return

    setup_images_5()
    setup_invis_mark()
    setup_wam()
    setup_fid_checkpoints()
    setup_coco(skip=args.no_coco)

    # Final verification
    checks = verify_all()
    print_verdict(checks)


if __name__ == "__main__":
    main()