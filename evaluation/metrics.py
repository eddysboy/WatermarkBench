"""Evaluation metrics for watermark benchmarking.

Metrics: PSNR, SSIM, LPIPS, Bit Accuracy, TPR, FPR, effective payload bits.
"""

import torch
import torch.nn.functional as F
import numpy as np
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from skimage.metrics import peak_signal_noise_ratio as sk_psnr

# Lazy import for lpips (heavy dependency)
_lpips_fn = None


def _get_lpips_fn(device: str = "cpu") -> object:
    global _lpips_fn
    if _lpips_fn is None:
        try:
            import lpips
            import os
            import torch.hub

            # Point torch hub to project checkpoints/ for AlexNet backbone
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            alexnet_path = os.path.join(project_root, 'checkpoints', 'alexnet-owt-7be5be79.pth')
            if not os.path.exists(alexnet_path):
                print("[WARN] AlexNet backbone not found in checkpoints/. LPIPS skipped.")
                return None

            old_hub_dir = torch.hub.get_dir()
            torch.hub.set_dir(project_root)
            try:
                _lpips_fn = lpips.LPIPS(net='alex', verbose=False).to(device)
            finally:
                torch.hub.set_dir(old_hub_dir)
        except ImportError:
            print("[WARN] lpips not installed. LPIPS metric will return NaN.")
            return None
    return _lpips_fn


_fid_obj = None
_clipped_model = None
_clipped_preprocess = None
_clipped_device = None


def _get_fid_obj(device: str = "cpu") -> object:
    """Lazy-init FrechetInceptionDistance for FID computation."""
    global _fid_obj
    if _fid_obj is None:
        import os
        import torch.hub
        from torchmetrics.image.fid import FrechetInceptionDistance
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        inception_path = os.path.join(project_root, 'checkpoints', 'weights-inception-2015-12-05-6726825d.pth')
        if not os.path.exists(inception_path):
            print("[WARN] InceptionV3 weights not found in checkpoints/. FID skipped.")
            return None
        old_hub_dir = torch.hub.get_dir()
        torch.hub.set_dir(project_root)
        try:
            _fid_obj = FrechetInceptionDistance(normalize=True).to(device)
        finally:
            torch.hub.set_dir(old_hub_dir)
    return _fid_obj


def _get_clip_fn(device: str = "cpu"):
    """Lazy-init open_clip ViT-B-32 for CMMD feature extraction."""
    global _clipped_model, _clipped_preprocess, _clipped_device
    if _clipped_model is None:
        import os
        import open_clip
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        ckpt_path = os.path.join(project_root, 'checkpoints', 'open_clip_pytorch_model.bin')
        if not os.path.exists(ckpt_path):
            print("[WARN] open_clip weights not found in checkpoints/. CMMD skipped.")
            return None, None
        _clipped_model, _, _clipped_preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained=ckpt_path, cache_dir=os.path.join(project_root, 'checkpoints')
        )
        _clipped_model = _clipped_model.to(device).eval()
        _clipped_device = device
    return _clipped_model, _clipped_preprocess


def compute_psnr(original: torch.Tensor, watermarked: torch.Tensor, data_range: float = 1.0) -> float:
    """Compute PSNR between two images [C, H, W] or [1, C, H, W] in [0, data_range]."""
    if original.dim() == 4:
        original = original.squeeze(0)
        watermarked = watermarked.squeeze(0)
    original_np = original.detach().cpu().permute(1, 2, 0).numpy()
    watermarked_np = watermarked.detach().cpu().permute(1, 2, 0).numpy()
    return float(sk_psnr(original_np, watermarked_np, data_range=data_range))


def compute_ssim(original: torch.Tensor, watermarked: torch.Tensor, data_range: float = 1.0) -> float:
    """Compute SSIM between two images [1, C, H, W] in [0, data_range]."""
    ssim = StructuralSimilarityIndexMeasure(data_range=data_range)
    val = ssim(watermarked, original)
    return float(val.item())


def compute_lpips(original: torch.Tensor, watermarked: torch.Tensor, device: str = "cpu") -> float:
    """Compute LPIPS between two images [1, C, H, W] in [0, 1]."""
    try:
        fn = _get_lpips_fn(device)
        if fn is None:
            return float('nan')
    except Exception:
        return float('nan')
    # lpips expects [-1, 1]
    a = (original * 2 - 1).to(device)
    b = (watermarked * 2 - 1).to(device)
    try:
        val = fn(a, b)
        return float(val.item())
    except Exception as e:
        return float('nan')


def compute_fid(originals: list, watermarked: list, device: str = "cpu") -> float:
    """Compute Frechet Inception Distance between two sets of images.

    originals/watermarked: lists of torch.Tensor [1, C, H, W] float [0, 1]
    """
    fid_obj = _get_fid_obj(device)
    if fid_obj is None:
        return float('nan')
    try:
        import torch.nn.functional as F
        fid_obj.reset()  # ensure clean state per call
        for img in originals:
            img_4d = img.float() if img.dim() == 4 else img.float().unsqueeze(0)
            img_resized = F.interpolate(img_4d, size=(299, 299), mode='bilinear', align_corners=False)
            img_uint8 = (img_resized * 255).clamp(0, 255).to(torch.uint8).to(device)
            fid_obj.update(img_uint8, real=True)
        for img in watermarked:
            img_4d = img.float() if img.dim() == 4 else img.float().unsqueeze(0)
            img_resized = F.interpolate(img_4d, size=(299, 299), mode='bilinear', align_corners=False)
            img_uint8 = (img_resized * 255).clamp(0, 255).to(torch.uint8).to(device)
            fid_obj.update(img_uint8, real=False)
        return float(fid_obj.compute().item())
    except Exception as e:
        print(f"[WARN] FID computation failed: {e}")
        return float('nan')


def compute_cmmd(originals: list, watermarked: list, device: str = "cpu") -> float:
    """Compute CLIP Maximum Mean Discrepancy (CMMD) between two sets of images.

    Uses open_clip ViT-B-32 features + Gaussian RBF kernel MMD² with σ=10.
    """
    model, preprocess = _get_clip_fn(device)
    if model is None:
        return float('nan')
    try:
        from PIL import Image
        import torch.nn.functional as F

        def extract_features(images):
            feats = []
            for img in images:
                # img: [1, C, H, W] float [0, 1]
                img_np = (img.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                pil_img = Image.fromarray(img_np).convert('RGB')
                img_t = preprocess(pil_img).unsqueeze(0).to(device)
                with torch.no_grad():
                    feat = model.encode_image(img_t)
                feats.append(feat.squeeze(0))
            return torch.stack(feats, dim=0)  # [N, 512]

        feats_orig = extract_features(originals)
        feats_wm = extract_features(watermarked)

        # MMD² with Gaussian RBF kernel, σ² = 10 (standard CMMD)
        def mmd2(x, y, sigma_sq=10.0):
            m, n = x.shape[0], y.shape[0]
            # Pairwise squared distances
            xx = torch.mm(x, x.t())
            yy = torch.mm(y, y.t())
            xy = torch.mm(x, y.t())
            x_sqn = xx.diag().unsqueeze(1)
            y_sqn = yy.diag().unsqueeze(1)
            K_xx = torch.exp(-(x_sqn + x_sqn.t() - 2 * xx) / (2 * sigma_sq))
            K_yy = torch.exp(-(y_sqn + y_sqn.t() - 2 * yy) / (2 * sigma_sq))
            K_xy = torch.exp(-(x_sqn + y_sqn.t() - 2 * xy) / (2 * sigma_sq))
            mmd = K_xx.sum() / (m * m) + K_yy.sum() / (n * n) - 2 * K_xy.sum() / (m * n)
            return mmd

        return float(mmd2(feats_orig, feats_wm).item())
    except Exception as e:
        print(f"[WARN] CMMD computation failed: {e}")
        return float('nan')


def _ci95(values: list) -> tuple:
    """95% CI (mean, low, high) via normal approx. Returns nan on <2 samples."""
    if len(values) < 2:
        m = float(np.nanmean(values)) if values else float('nan')
        return m, float('nan'), float('nan')
    mean = float(np.nanmean(values))
    std = float(np.nanstd(values))
    half = 1.96 * std / np.sqrt(len(values))
    return mean, mean - half, mean + half


def compute_bit_accuracy(decoded: torch.Tensor, original: torch.Tensor) -> float:
    """Compute per-bit accuracy between decoded bits and original bits.

    Both tensors are 1D [num_bits] of float 0/1.
    Returns fraction of bits that match.
    """
    if decoded.numel() == 0 or original.numel() == 0:
        return float('nan')
    decoded_bits = (decoded >= 0.5).float()
    original_bits = (original >= 0.5).float()
    correct = (decoded_bits == original_bits).float().sum().item()
    return correct / max(decoded.numel(), original.numel())


def compute_detection_rate(
    detected_list: list[bool],
) -> dict:
    """Compute TPR and FPR from lists of detection results.

    Args:
        detected_list: list of booleans where True = watermark "detected"

    Returns dict with 'tpr', 'fpr', 'total' keys.
    TPR = fraction of watermarked images correctly detected.
    FPR = fraction of non-watermarked images incorrectly detected.
    """
    if not detected_list:
        return {"tpr": float('nan'), "fpr": float('nan'), "total": 0}
    positive = sum(1 for d in detected_list if d)
    return {
        "detection_rate": positive / len(detected_list),
        "positives": positive,
        "total": len(detected_list),
    }


class MetricsCalculator:
    """Collects and aggregates evaluation metrics."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._reset()

    def _reset(self):
        self.psnr_values = []
        self.ssim_values = []
        self.lpips_values = []
        self.bit_acc_values = []
        self.fid_value = float('nan')
        self.cmmd_value = float('nan')
        self.tpr_values = []  # per-attack TPR
        self.fpr_values = []  # overall FPR
        self.payload_bits = None

    def record_quality(self, psnr: float, ssim: float, lpips: float):
        self.psnr_values.append(psnr)
        self.ssim_values.append(ssim)
        self.lpips_values.append(lpips)

    def record_fid(self, fid: float):
        self.fid_value = fid

    def record_cmmd(self, cmmd: float):
        self.cmmd_value = cmmd

    def record_bit_acc(self, bit_acc: float):
        self.bit_acc_values.append(bit_acc)

    def record_tpr(self, tpr: float):
        self.tpr_values.append(tpr)

    def record_fpr(self, fpr: float):
        self.fpr_values.append(fpr)

    def set_payload_bits(self, bits: int):
        self.payload_bits = bits

    def summary(self) -> dict:
        def safe_mean(values):
            return float(np.nanmean(values)) if values else float('nan')

        psnr_ci = _ci95(self.psnr_values)
        ssim_ci = _ci95(self.ssim_values)
        lpips_ci = _ci95(self.lpips_values)

        return {
            "payload_bits": self.payload_bits,
            "psnr_mean": safe_mean(self.psnr_values),
            "psnr_std": float(np.nanstd(self.psnr_values)) if len(self.psnr_values) > 1 else 0.0,
            "psnr_ci_low": psnr_ci[1],
            "psnr_ci_high": psnr_ci[2],
            "ssim_mean": safe_mean(self.ssim_values),
            "ssim_std": float(np.nanstd(self.ssim_values)) if len(self.ssim_values) > 1 else 0.0,
            "ssim_ci_low": ssim_ci[1],
            "ssim_ci_high": ssim_ci[2],
            "lpips_mean": safe_mean(self.lpips_values),
            "lpips_std": float(np.nanstd(self.lpips_values)) if len(self.lpips_values) > 1 else 0.0,
            "lpips_ci_low": lpips_ci[1],
            "lpips_ci_high": lpips_ci[2],
            "fid": self.fid_value,
            "cmmd": self.cmmd_value,
            "bit_acc_clean_mean": safe_mean(self.bit_acc_values),
            "bit_acc_clean_std": float(np.nanstd(self.bit_acc_values)) if len(self.bit_acc_values) > 1 else 0.0,
            "tpr_mean": safe_mean(self.tpr_values),
            "fpr_mean": safe_mean(self.fpr_values),
            "num_images": len(self.psnr_values),
        }
