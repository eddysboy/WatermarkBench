"""Evaluation configuration."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Test images to use for evaluation
TEST_IMAGES = [
    MODELS_DIR / "watermark-anything" / "assets" / "images" / "gauguin_256.jpg",
    MODELS_DIR / "watermark-anything" / "assets" / "images" / "alpaca.jpg",
    MODELS_DIR / "watermark-anything" / "assets" / "images" / "ducks.jpg",
    MODELS_DIR / "watermark-anything" / "assets" / "images" / "seabackground.jpg",
    MODELS_DIR / "watermark-anything" / "assets" / "images" / "trex_bike.jpg",
]

# Detection thresholds
PRESENCE_THRESHOLD = 0.5  # for MiniWatermarkDemo presence score
WAM_MASK_THRESHOLD = 0.5  # for watermark-anything mask detection

# Number of random messages to test per image for FPR calculation
FPR_SAMPLES = 10

# Device
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
