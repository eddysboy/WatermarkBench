# WatermarkBench

A unified evaluation framework for benchmarking image watermarking models across image quality, bit accuracy, detection robustness, and false-positive rates.

## Supported Models

| Model | Payload | Resolution | Source |
|---|---|---|---|
| InvisMark | 100 bits | 256x256 | [microsoft/InvisMark](https://github.com/microsoft/InvisMark) |
| MiniWatermarkDemo | 32 bits | 256x256 | [Phnt0mW/MiniWatermarkDemo](https://github.com/Phnt0mW/MiniWatermarkDemo) |
| TrustMark-P | 61 bits | 256x256 | [adobe/trustmark](https://github.com/adobe/trustmark) |
| TrustMark-Q | 61 bits | 256x256 | [adobe/trustmark](https://github.com/adobe/trustmark) |
| watermark-anything | 32 bits | 256x256 | [facebookresearch/watermark-anything](https://github.com/facebookresearch/watermark-anything) |

## Metrics

**Image Quality**: PSNR, SSIM, LPIPS, FID, CMMD

**Watermark Effectiveness**: Bit Accuracy (clean), Bit Accuracy (under attack), TPR, FPR

## Robustness Attacks (28 attacks)

JPEG compression (4 levels), Gaussian blur (3), Gaussian noise (3), brightness/contrast/saturation adjustment (3 each), salt & pepper noise (2), rotation (3), resize (2), crop (2), horizontal flip, text overlay (2), and 5 composite attacks.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare datasets and checkpoints
python prepare.py

# 3. Run evaluation with default config
python run_evaluation.py --config configs/default.yaml

# 4. Visualize attack transforms (generates attack_visualization.png)
python visualize_attacks.py
```

## Usage

### CLI mode

```bash
# Evaluate specific models
python run_evaluation.py --models InvisMark TrustMark-Q --images path/to/images

# Skip slow steps
python run_evaluation.py --skip-attacks --skip-fpr

# Custom output
python run_evaluation.py --output my_results.csv --json-output my_results.json
```

### YAML config mode

```bash
python run_evaluation.py --config configs/default.yaml
```

CLI arguments override YAML values. See `configs/default.yaml` for all options.

## Project Structure

```
WatermarkBench/
├── run_evaluation.py       # Main evaluation entry point
├── prepare.py              # First-time setup (datasets + checkpoints)
├── visualize_attacks.py    # Attack transform visualization
├── requirements.txt        # Python dependencies
├── configs/
│   └── default.yaml        # Default evaluation config
├── evaluation/
│   ├── metrics.py          # Metric implementations (PSNR, SSIM, LPIPS, FID, CMMD)
│   ├── attacks.py          # Attack transform library (28 attacks)
│   ├── wrappers.py         # Unified model wrappers
│   └── config.py           # Default paths and constants
├── models/                 # Model source code (checkpoints excluded via .gitignore)
├── datasets/               # Test images (images_5/ kept; COCO excluded)
├── checkpoints/            # Backbone models for FID/CMMD (excluded)
└── results/                # Generated evaluation output (excluded)
```

## Adding a New Model

1. Create a model directory under `models/` with source code and checkpoint
2. Implement a wrapper class in `evaluation/wrappers.py` inheriting from `ModelWrapper`
3. Register the wrapper in the `_WRAPPERS` dict
4. Add the model name to `REGISTERED_MODELS` in `evaluation/attacks.py`
5. Optionally add model-specific config to `configs/default.yaml`

## Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA (or CPU)
- ~3 GB disk for checkpoints