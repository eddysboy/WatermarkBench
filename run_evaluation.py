#!/usr/bin/env python3
"""Watermark Evaluation Framework.

Evaluates watermarking models on image quality and robustness metrics:
- Effective encoding bits (payload size)
- PSNR, SSIM, LPIPS (image quality)
- Bit accuracy (message recovery)
- TPR (true positive rate: watermark detected when present)
- FPR (false positive rate: watermark detected when absent)
- Robustness under attacks

Usage:
    python run_evaluation.py                          # all available models
    python run_evaluation.py --config configs/default.yaml  # YAML config
    python run_evaluation.py --models InvisMark       # single model
    python run_evaluation.py --images path/to/dir     # custom images
    python run_evaluation.py --output results.csv     # CSV output
"""

import argparse
import yaml
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np

# Add project root to path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from evaluation.metrics import (
    MetricsCalculator, compute_psnr, compute_ssim, compute_lpips, compute_bit_accuracy,
    compute_fid, compute_cmmd
)
from evaluation.attacks import get_attacks, list_models as list_registered_models
from evaluation.wrappers import get_wrapper, list_available_models
from evaluation.config import TEST_IMAGES, RESULTS_DIR, DEVICE, FPR_SAMPLES


def parse_args():
    parser = argparse.ArgumentParser(
        description="Watermark Evaluation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file (e.g. configs/default.yaml). "
             "CLI arguments override YAML values."
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Models to evaluate (default: all available)"
    )
    parser.add_argument(
        "--images", nargs="+", default=None,
        help="Paths to test images (default: built-in test set)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for CSV results (default: results/results_TIMESTAMP.csv)"
    )
    parser.add_argument(
        "--json-output", type=str, default=None,
        help="Output path for detailed JSON results"
    )
    parser.add_argument(
        "--device", type=str, default=DEVICE,
        help=f"Device to use (default: {DEVICE})"
    )
    parser.add_argument(
        "--skip-attacks", action="store_true",
        help="Skip attack evaluation (only image quality and clean bit accuracy)"
    )
    parser.add_argument(
        "--skip-fpr", action="store_true",
        help="Skip FPR calculation (can be slow)"
    )
    return parser.parse_args()


def load_yaml_config(path: str) -> dict:
    """Load and parse a YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _merge_yaml_config(args, yaml_cfg: dict):
    """Merge YAML config into args. CLI arguments take precedence over YAML values."""
    cli_flags = set()
    for a in sys.argv[1:]:
        if a.startswith("--"):
            name = a.lstrip("-").replace("-", "_")
            if "=" in name:
                name = name.split("=")[0]
            cli_flags.add(name)

    if "models" not in cli_flags and yaml_cfg.get("models"):
        if isinstance(yaml_cfg["models"], dict):
            args.models = list(yaml_cfg["models"].keys())

    if "images" not in cli_flags and yaml_cfg.get("images", {}).get("path"):
        args.images = [yaml_cfg["images"]["path"]]

    if "device" not in cli_flags and yaml_cfg.get("device"):
        args.device = yaml_cfg["device"]

    if "output" not in cli_flags and yaml_cfg.get("output", {}).get("path"):
        args._yaml_output_dir = yaml_cfg["output"]["path"]

    if "skip_attacks" not in cli_flags and yaml_cfg.get("skip_attacks"):
        args.skip_attacks = True

    if "skip_fpr" not in cli_flags and yaml_cfg.get("skip_fpr"):
        args.skip_fpr = True

    if yaml_cfg.get("fpr_samples"):
        import evaluation.config as eval_config
        eval_config.FPR_SAMPLES = int(yaml_cfg["fpr_samples"])

    # Per-model thresholds from YAML
    if yaml_cfg.get("models"):
        thresholds = {}
        for mname, mcfg in yaml_cfg["models"].items():
            if isinstance(mcfg, dict) and "threshold" in mcfg:
                thresholds[mname] = float(mcfg["threshold"])
        if thresholds:
            args._model_thresholds = thresholds

    return args


def get_test_images(image_paths: Optional[List[str]] = None) -> List[Path]:
    """Get list of test image paths."""
    if image_paths == None:
        image_paths = TEST_IMAGES

    for path in image_paths:
        # if path is a directory, get all image files in it
        p = Path(path)
        if p.is_dir():
            return [f for f in p.glob("*") if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]]
        else:
            # if path is a file, return it if it exists
            if p.exists() and p.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
                return [p]
    # if image_paths:
        # return [Path(p) for p in image_paths if Path(p).exists()]
    # return [p for p in TEST_IMAGES if p.exists()]


def evaluate_model(
    wrapper,
    model_name: str,
    images: List[Path],
    attacks: Dict,
    device: str,
    skip_attacks: bool = False,
    skip_fpr: bool = False,
) -> Dict:
    """Run full evaluation for one model."""
    print(f"\n{'=' * 70}")
    print(f"  Evaluating: {model_name}")
    print(f"  Payload: {wrapper.payload_bits} bits | Image size: {wrapper.image_size}")
    print(f"  Attacks: {len(attacks)} | Test images: {len(images)}")
    print(f"{'=' * 70}")

    # Build decode kwargs (optional threshold override from YAML)
    decode_kwargs = {}
    if hasattr(wrapper, "threshold"):
        decode_kwargs["threshold"] = wrapper.threshold

    metrics = MetricsCalculator(device=device)
    metrics.set_payload_bits(wrapper.payload_bits)
    attack_results = {}

    # Load and preprocess all images
    preprocessed_images = []
    for img_path in images:
        try:
            tensor = wrapper.load_image(str(img_path))
            preprocessed_images.append((img_path.name, tensor))
        except Exception as e:
            print(f"  [SKIP] Failed to load {img_path.name}: {e}")

    if not preprocessed_images:
        print(f"  [SKIP] No valid images for {model_name}")
        return {"model": model_name, "error": "No valid images"}

    try:
        wrapper._ensure_loaded()
    except Exception as e:
        print(f"  [SKIP] Failed to load model {model_name}: {e}")
        return {"model": model_name, "error": str(e)}

    # ================================================================
    # Phase 1: Image quality metrics (PSNR, SSIM, LPIPS)
    # ================================================================
    print(f"\n  --- Phase 1: Image Quality ---")
    originals_for_fid = []
    watermarked_for_fid = []

    use_progress = len(preprocessed_images) > 10
    if use_progress:
        from tqdm import tqdm
        pbar = tqdm(preprocessed_images, desc="  Image Quality", unit="img",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
    else:
        pbar = None

    for name, image in preprocessed_images:
        try:
            msg = wrapper.random_message(wrapper.payload_bits)
            watermarked = wrapper.encode(image, msg)

            psnr_val = compute_psnr(image, watermarked)
            ssim_val = compute_ssim(image, watermarked)
            lpips_val = compute_lpips(image, watermarked, device=device)

            metrics.record_quality(psnr_val, ssim_val, lpips_val)
            originals_for_fid.append(image)
            watermarked_for_fid.append(watermarked)

            if use_progress:
                pbar.update(1)
                pbar.set_postfix(
                    PSNR=f"{np.mean(metrics.psnr_values):.2f}",
                    SSIM=f"{np.mean(metrics.ssim_values):.4f}",
                    LPIPS=f"{np.mean(metrics.lpips_values):.4f}"
                )
            else:
                print(f"    {name:30s}  PSNR={psnr_val:.2f}  SSIM={ssim_val:.4f}  LPIPS={lpips_val:.4f}")
        except Exception as e:
            if use_progress:
                pbar.write(f"    {name:30s}  [ERROR] {e}")
            else:
                print(f"    {name:30s}  [ERROR] {e}")
    # FID and CMMD (distribution-level metrics across all images)
    if originals_for_fid:
        fid_val = compute_fid(originals_for_fid, watermarked_for_fid, device=device)
        cmmd_val = compute_cmmd(originals_for_fid, watermarked_for_fid, device=device)
        metrics.record_fid(fid_val)
        metrics.record_cmmd(cmmd_val)
        print(f"    {'FID':30s}  = {fid_val:.4f}  (across {len(originals_for_fid)} images)")
        print(f"    {'CMMD':30s}  = {cmmd_val:.4f}  (across {len(originals_for_fid)} images)")

    # ================================================================
    # Phase 2: Clean bit accuracy (no attack)
    # ================================================================
    print(f"\n  --- Phase 2: Clean Bit Accuracy ---")

    bit_acc = 0.0
    status = "N/A"

    for name, image in tqdm(preprocessed_images, desc=f"{name:30s}  Bit Acc={bit_acc:.4f}  Detected={status}") if len(preprocessed_images) > 10 else preprocessed_images:
        try:
            msg = wrapper.random_message(wrapper.payload_bits)
            watermarked = wrapper.encode(image, msg)
            decoded_bits, detected = wrapper.decode(watermarked, **decode_kwargs)

            bit_acc = compute_bit_accuracy(decoded_bits[:wrapper.payload_bits], msg)
            metrics.record_bit_acc(bit_acc)
            status = "OK" if detected else "MISS"

            # print(f"    {name:30s}  Bit Acc={bit_acc:.4f}  Detected={status}")
        except Exception as e:
            print(f"    {name:30s}  [ERROR] {e}")

    if len(preprocessed_images) > 10:
        print(f"    ... and {len(preprocessed_images) - 10} more images evaluated for clean bit accuracy.")

    # ================================================================
    # Phase 3: Robustness under attacks (TPR per attack)
    # ================================================================
    if not skip_attacks and attacks:
        print(f"\n  --- Phase 3: Attack Robustness ---")
        for attack_key, (attack_name, attack_fn) in attacks.items():
            detections = 0
            total = 0
            bit_accs = []
            for name, image in preprocessed_images:
                try:
                    msg = wrapper.random_message(wrapper.payload_bits)
                    watermarked = wrapper.encode(image, msg)
                    attacked = attack_fn(watermarked)
                    decoded_bits, detected = wrapper.decode(attacked, **decode_kwargs)

                    bit_acc = compute_bit_accuracy(decoded_bits[:wrapper.payload_bits], msg)
                    bit_accs.append(bit_acc)
                    if detected:
                        detections += 1
                    total += 1
                except Exception as e:
                    total += 1

            tpr = detections / total if total > 0 else 0.0
            avg_bit_acc = np.mean(bit_accs) if bit_accs else float('nan')
            attack_results[attack_key] = {
                "attack_name": attack_name,
                "tpr": tpr,
                "bit_acc": avg_bit_acc,
                "detections": detections,
                "total": total,
            }
            metrics.record_tpr(tpr)
            bar = "#" * int(tpr * 20)
            print(f"    {attack_name:30s}  TPR={tpr:.3f}  BitAcc={avg_bit_acc:.4f}  [{detections}/{total}] {bar}")

    # ================================================================
    # Phase 4: FPR (decode non-watermarked images)
    # ================================================================
    if not skip_fpr:
        print(f"\n  --- Phase 4: False Positive Rate ---")
        fp_detections = 0
        fp_total = 0
        for name, image in preprocessed_images:
            for _ in range(FPR_SAMPLES):
                try:
                    # Decode original (un-watermarked) image
                    _, detected = wrapper.decode(image, **decode_kwargs)
                    if detected:
                        fp_detections += 1
                    fp_total += 1
                except Exception:
                    fp_total += 1

        fpr = fp_detections / fp_total if fp_total > 0 else 0.0
        metrics.record_fpr(fpr)
        print(f"    FPR = {fpr:.4f}  [{fp_detections}/{fp_total}]")

    # ================================================================
    # Build summary
    # ================================================================
    summary = metrics.summary()
    result = {
        "model": model_name,
        "available": True,
        "payload_bits": wrapper.payload_bits,
        "image_size": f"{wrapper.image_size[0]}x{wrapper.image_size[1]}",
        "num_test_images": len(images),
        **summary,
        "attacks": attack_results,
    }

    # Print summary
    print(f"\n  --- {model_name} Summary ---")
    print(f"  Payload:        {wrapper.payload_bits} bits")
    print(f"  PSNR:           {summary['psnr_mean']:.2f} +/- {summary['psnr_std']:.2f} dB")
    print(f"                   95% CI: [{summary['psnr_ci_low']:.2f}, {summary['psnr_ci_high']:.2f}]")
    print(f"  SSIM:           {summary['ssim_mean']:.4f} +/- {summary['ssim_std']:.4f}")
    print(f"                   95% CI: [{summary['ssim_ci_low']:.4f}, {summary['ssim_ci_high']:.4f}]")
    print(f"  LPIPS:          {summary['lpips_mean']:.4f} +/- {summary['lpips_std']:.4f}")
    print(f"                   95% CI: [{summary['lpips_ci_low']:.4f}, {summary['lpips_ci_high']:.4f}]")
    _fid_s = f"{summary['fid']:.4f}" if isinstance(summary.get('fid'), (int, float)) and summary.get('fid') == summary.get('fid') else "N/A"
    print(f"  FID:            {_fid_s}")
    _cmmd_s = f"{summary['cmmd']:.4f}" if isinstance(summary.get('cmmd'), (int, float)) and summary.get('cmmd') == summary.get('cmmd') else "N/A"
    print(f"  CMMD:           {_cmmd_s}")
    print(f"  Bit Acc (clean):{summary['bit_acc_clean_mean']:.4f} +/- {summary['bit_acc_clean_std']:.4f}")
    print(f"  TPR (mean):     {summary['tpr_mean']:.4f}")
    print(f"  FPR:            {summary['fpr_mean']:.4f}")

    return result


def save_csv(results: List[Dict], output_path: Path):
    """Save results to CSV file with summary, cross-model attack table, and per-model details."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)

        # Section 1: Summary
        writer.writerow(["=== SUMMARY ==="])
        headers = [
            "Model", "Payload (bits)",
            "PSNR_mean", "PSNR_std", "SSIM_mean", "SSIM_std", "LPIPS_mean", "LPIPS_std",
            "FID", "CMMD",
            "Bit Acc (clean)", "TPR (mean)", "FPR", "Available"
        ]
        writer.writerow(headers)
        for r in results:
            def _fmt(val, prec):
                return f"{val:.{prec}f}" if isinstance(val, (int, float)) and val == val else "N/A"
            psnr_m = _fmt(r.get('psnr_mean'), 2)
            psnr_s = _fmt(r.get('psnr_std'), 2)
            ssim_m = _fmt(r.get('ssim_mean'), 4)
            ssim_s = _fmt(r.get('ssim_std'), 4)
            lpips_m = _fmt(r.get('lpips_mean'), 4)
            lpips_s = _fmt(r.get('lpips_std'), 4)
            fid = _fmt(r.get('fid'), 2)
            cmmd = _fmt(r.get('cmmd'), 4)
            bitacc = f"{r.get('bit_acc_clean_mean', 'N/A'):.4f}" if isinstance(r.get('bit_acc_clean_mean'), (int, float)) and r.get('bit_acc_clean_mean') == r.get('bit_acc_clean_mean') else "N/A"
            tpr = f"{r.get('tpr_mean', 'N/A'):.4f}" if isinstance(r.get('tpr_mean'), (int, float)) and r.get('tpr_mean') == r.get('tpr_mean') else "N/A"
            fpr = f"{r.get('fpr_mean', 'N/A'):.4f}" if isinstance(r.get('fpr_mean'), (int, float)) and r.get('fpr_mean') == r.get('fpr_mean') else "N/A"
            writer.writerow([
                r.get("model", ""), r.get("payload_bits", ""),
                psnr_m, psnr_s, ssim_m, ssim_s, lpips_m, lpips_s, fid, cmmd,
                bitacc, tpr, fpr,
                str(r.get("available", False))
            ])

        # Section 2: Cross-model attack table (BitAcc / TPR)
        writer.writerow([])
        writer.writerow(["=== ATTACK DETAILS (BitAcc / TPR) ==="])
        all_attack_keys = {}
        for r in results:
            if "attacks" in r and r["attacks"]:
                for key, ainfo in sorted(r["attacks"].items()):
                    all_attack_keys[key] = ainfo.get("attack_name", key)
        sorted_attacks = sorted(all_attack_keys.items(), key=lambda x: x[1])
        models_with_attacks = [r for r in results if "attacks" in r and r["attacks"]]
        # Sort: MiniWatermarkDemo first, then alphabetical
        def _model_sort_key(r):
            name = r.get("model", "")
            return (0, name) if name == "MiniWatermarkDemo" else (1, name)
        models_with_attacks.sort(key=_model_sort_key)
        header = ["Attack"]
        for r in models_with_attacks:
            header.append(f"{r['model']} BitAcc")
            header.append(f"{r['model']} TPR")
        writer.writerow(header)
        for attack_key, attack_name in sorted_attacks:
            row = [attack_name]
            for r in models_with_attacks:
                ainfo = r["attacks"].get(attack_key)
                if ainfo:
                    ba = ainfo.get("bit_acc", float('nan'))
                    row.append(f"{ba:.4f}" if not (isinstance(ba, float) and ba != ba) else "--")
                else:
                    row.append("--")
                ainfo = r["attacks"].get(attack_key)
                row.append(f"{ainfo.get('tpr', 0):.4f}" if ainfo else "--")
            writer.writerow(row)

        # Section 3: Per-model detailed view
        for r in results:
            writer.writerow([])
            if "error" in r:
                writer.writerow([f"=== {r['model']}: ERROR === {r.get('error', '')} ==="])
                continue
            writer.writerow([f"=== {r['model']} ==="])
            writer.writerow(["Attack", "BitAcc", "TPR", "Detections/Total"])
            if "attacks" in r and r["attacks"]:
                for key, ainfo in sorted(r["attacks"].items(), key=lambda x: x[1].get("attack_name", x[0])):
                    ba = ainfo.get("bit_acc", float('nan'))
                    ba_s = f"{ba:.4f}" if not (isinstance(ba, float) and ba != ba) else "--"
                    writer.writerow([
                        ainfo.get("attack_name", key), ba_s,
                        f"{ainfo.get('tpr', 0):.4f}",
                        f"{ainfo.get('detections', 0)}/{ainfo.get('total', 0)}"
                    ])

    print(f"\nResults saved to {output_path}")


def print_final_table(results: List[Dict]):
    """Print a formatted summary table."""
    print(f"\n{'=' * 100}")
    print(f"  WATERMARK EVALUATION RESULTS")
    print(f"{'=' * 100}")
    header = f"  {'Model':<22s} {'Payload':>8s} {'PSNR':>8s} {'SSIM':>8s} {'LPIPS':>8s} {'FID':>8s} {'CMMD':>8s} {'BitAcc':>8s} {'TPR':>8s} {'FPR':>8s}"
    print(header)
    print(f"  {'-' * 90}")

    for r in results:
        if "error" in r:
            print(f"  {r['model']:<25s} {'ERROR: ' + r['error'][:50]}")
            continue
        psnr_s = f"{r.get('psnr_mean', 0):.2f}" if isinstance(r.get('psnr_mean'), (int, float)) else "N/A"
        psnr_std_s = f"/{r.get('psnr_std', 0):.2f}" if isinstance(r.get('psnr_std'), (int, float)) and r.get('psnr_std') > 0 else ""
        ssim_s = f"{r.get('ssim_mean', 0):.4f}" if isinstance(r.get('ssim_mean'), (int, float)) else "N/A"
        ssim_std_s = f"/{r.get('ssim_std', 0):.4f}" if isinstance(r.get('ssim_std'), (int, float)) and r.get('ssim_std') > 0 else ""
        lpips_s = f"{r.get('lpips_mean', 0):.4f}" if isinstance(r.get('lpips_mean'), (int, float)) else "N/A"
        lpips_std_s = f"/{r.get('lpips_std', 0):.4f}" if isinstance(r.get('lpips_std'), (int, float)) and r.get('lpips_std') > 0 else ""
        fid_s = f"{r.get('fid', 0):.4f}" if isinstance(r.get('fid'), (int, float)) else "N/A"
        cmmd_s = f"{r.get('cmmd', 0):.4f}" if isinstance(r.get('cmmd'), (int, float)) else "N/A"
        bitacc_s = f"{r.get('bit_acc_clean_mean', 0):.4f}" if isinstance(r.get('bit_acc_clean_mean'), (int, float)) else "N/A"
        tpr_s = f"{r.get('tpr_mean', 0):.4f}" if isinstance(r.get('tpr_mean'), (int, float)) else "N/A"
        fpr_s = f"{r.get('fpr_mean', 0):.4f}" if isinstance(r.get('fpr_mean'), (int, float)) else "N/A"
        payload_s = str(r.get('payload_bits', 'N/A'))
        
        print(f"  {r['model']:<22s} {payload_s:>8s} {psnr_s+psnr_std_s:>8s} {ssim_s+ssim_std_s:>8s} {lpips_s+lpips_std_s:>8s} {fid_s:>8s} {cmmd_s:>8s} {bitacc_s:>8s} {tpr_s:>8s} {fpr_s:>8s}")
    
    print(f"{'=' * 100}")
    
    # Print cross-model attack comparison table
    print(f"\n  Attack Robustness Details (BitAcc / TPR per attack):")
    print(f"  {'-' * 100}")

    # Collect all attack keys across all models
    all_attack_keys = {}
    for r in results:
        if "attacks" in r and r["attacks"]:
            for key, ainfo in sorted(r["attacks"].items()):
                name = ainfo.get("attack_name", key)
                all_attack_keys[key] = name

    # Build model column widths
    model_names_sorted = [r["model"] for r in results if "attacks" in r and r["attacks"]]
    # MiniWatermarkDemo first, then alphabetical
    model_names_sorted.sort(key=lambda n: (0, n) if n == "MiniWatermarkDemo" else (1, n))
    if model_names_sorted:
        col_w = 12
        # Header
        cols = "  ".join(f"{m:{col_w}s}" for m in model_names_sorted)
        print(f"  {'Attack':<22s} {cols}")
        print(f"  {'-' * 22} {'-' * (len(model_names_sorted) * (col_w + 2))}")

        for attack_key, attack_name in sorted(all_attack_keys.items(), key=lambda x: x[1]):
            row_parts = []
            # Iterate in same sorted order as header
            result_by_name = {r["model"]: r for r in results}
            for mname in model_names_sorted:
                r = result_by_name.get(mname)
                if "attacks" not in r or not r["attacks"]:
                    row_parts.append(f"{'N/A':>{col_w}s}")
                    continue
                ainfo = r["attacks"].get(attack_key)
                if ainfo:
                    ba = ainfo.get("bit_acc", float('nan'))
                    tp = ainfo.get("tpr", 0)
                    ba_s = f"{ba:.2f}" if not (isinstance(ba, float) and (ba != ba)) else " --"
                    tp_s = f"{tp:.2f}" if not (isinstance(tp, float) and (tp != tp)) else " --"
                    row_parts.append(f"{ba_s}/{tp_s:{4}s}".rjust(col_w) if tp_s != " --" and ba_s != " --" else f"{'--/--':>{col_w}s}")
                else:
                    row_parts.append(f"{'--/--':>{col_w}s}")
            print(f"  {attack_name:<22s} {'  '.join(row_parts)}")


def main():
    args = parse_args()

    # Load and merge YAML config if provided
    if args.config:
        yaml_cfg = load_yaml_config(args.config)
        args = _merge_yaml_config(args, yaml_cfg)

    # Determine which models to evaluate
    if args.models:
        model_names = args.models
    else:
        model_names = list_registered_models()

    # Get test images
    images = get_test_images(args.images)
    if not images:
        print("ERROR: No test images found.")
        print(f"  Checked: {TEST_IMAGES if not args.images else args.images}")
        sys.exit(1)
    print(f"Test images ({len(images)}):")
    # 输出前10张图：
    for img in images[:10]:
        print(f"  {img.name}")
    if len(images) > 10:
        print(f"  ... and {len(images) - 10} more images.")

    # 原来的代码：
    # for img in images:
    #     print(f"  {img.name}")

    # Output paths
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(getattr(args, "_yaml_output_dir", None) or RESULTS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output or str(output_dir / f"results_{timestamp}.csv")
    json_path = args.json_output or str(output_dir / f"results_{timestamp}.json")

    all_results = []

    for model_name in model_names:
        wrapper = get_wrapper(model_name, device=args.device)
        if wrapper is None:
            all_results.append({
                "model": model_name,
                "available": False,
                "error": "Checkpoint not found or dependencies missing"
            })
            print(f"\n[SKIP] {model_name}: not available (missing checkpoint or dependencies)")
            continue

        ready = wrapper.is_available()
        if not ready:
            all_results.append({
                "model": model_name,
                "available": False,
                "error": "Checkpoint not found"
            })
            print(f"\n[SKIP] {model_name}: checkpoint not found")
            continue

        # Apply per-model threshold from YAML config
        thresholds = getattr(args, "_model_thresholds", {})
        if model_name in thresholds:
            wrapper.threshold = thresholds[model_name]

        attacks = get_attacks(model_name)
        result = evaluate_model(
            wrapper, model_name, images, attacks, args.device,
            skip_attacks=args.skip_attacks,
            skip_fpr=args.skip_fpr,
        )
        all_results.append(result)

    # Print final table
    print_final_table(all_results)

    # Save outputs
    save_csv(all_results, Path(csv_path))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str, ensure_ascii=False)
    print(f"JSON results saved to {json_path}")


if __name__ == "__main__":
    main()
import yaml
