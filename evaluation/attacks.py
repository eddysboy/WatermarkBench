"""Attack implementations for watermark robustness evaluation.

All models share the same attack set for fair comparison.
JPEG uses torchvision.io.encode_jpeg/decode_jpeg (tv 0.27+).
"""

import math
import random
from typing import Callable, Dict, List, Tuple

import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TVF
from PIL import Image, ImageDraw, ImageFont
from torchvision.io import encode_jpeg, decode_jpeg


# ============================================================================
# Generic attack helpers (work on [0, 1] tensors of shape [1, C, H, W])
# ============================================================================

def jpeg_compress(x: torch.Tensor, quality: int) -> torch.Tensor:
    """JPEG compress + decompress roundtrip. Uses torchvision.io (works on [1,C,H,W])."""
    if quality < 1 or quality > 100:
        return x
    # encode_jpeg expects 3D uint8 (C, H, W) or 4D (N, C, H, W) in [0, 255]
    x_uint8 = (x.clamp(0, 1) * 255).to(torch.uint8)
    # Process each image in batch individually
    results = []
    for i in range(x_uint8.shape[0]):
        single = x_uint8[i]  # (C, H, W)
        encoded = encode_jpeg(single, quality)
        decoded = decode_jpeg(encoded)  # (C, H, W) uint8
        results.append(decoded)
    return torch.stack(results).float() / 255.0


def gaussian_noise(x: torch.Tensor, std: float) -> torch.Tensor:
    return (x + torch.randn_like(x) * std).clamp(0, 1)


def salt_pepper_noise(x: torch.Tensor, prob: float = 0.01) -> torch.Tensor:
    result = x.clone()
    mask_salt = torch.rand_like(result) < prob / 2
    mask_pepper = torch.rand_like(result) < prob / 2
    result[mask_salt] = 1.0
    result[mask_pepper] = 0.0
    return result


def gaussian_blur(x: torch.Tensor, kernel_size: int, sigma: float = None) -> torch.Tensor:
    if kernel_size % 2 == 0:
        kernel_size += 1
    if sigma is None:
        sigma = kernel_size * 0.15 + 0.35
    return TVF.gaussian_blur(x, kernel_size, sigma)


def resize_attack(x: torch.Tensor, scale: float) -> torch.Tensor:
    h, w = x.shape[-2:]
    new_h, new_w = max(4, int(h * scale)), max(4, int(w * scale))
    return TVF.resize(TVF.resize(x, (new_h, new_w), antialias=True), (h, w), antialias=True)


def crop_attack(x: torch.Tensor, scale: float) -> torch.Tensor:
    h, w = x.shape[-2:]
    new_h, new_w = max(4, int(h * scale)), max(4, int(w * scale))
    return TVF.resize(TVF.center_crop(x, (new_h, new_w)), (h, w), antialias=True)


def rotate_attack(x: torch.Tensor, degrees: float) -> torch.Tensor:
    return TVF.rotate(x, degrees, fill=0.5)


def hflip_attack(x: torch.Tensor) -> torch.Tensor:
    return TVF.hflip(x)


def brightness_attack(x: torch.Tensor, factor: float) -> torch.Tensor:
    return TVF.adjust_brightness(x.clamp(0, 1), factor).clamp(0, 1)


def contrast_attack(x: torch.Tensor, factor: float) -> torch.Tensor:
    return TVF.adjust_contrast(x.clamp(0, 1), factor).clamp(0, 1)


def saturation_attack(x: torch.Tensor, factor: float) -> torch.Tensor:
    return TVF.adjust_saturation(x.clamp(0, 1), factor).clamp(0, 1)

def text_overlay_attack(x: torch.Tensor, alpha: float = 0.3) -> torch.Tensor:
    """Render random text strings at random positions with random colors/sizes."""
    h, w = x.shape[-2:]
    img_pil = TVF.to_pil_image(x.squeeze(0).clamp(0, 1))

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # texts = ["Hello", "World", "Test", "Watermark", "AI", "Image",
    #          "Photo", "Sample", "Text", "Logo", "2024", "OK"]

    for _ in range(random.randint(1, 2)):
        # text = [0-9a-zA-Z]{6}
        text = "".join(random.choices("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ", k=6))
        # text = random.choice(texts)
        font_size = random.randint(max(40, h // 20), max(80, h // 8))
        # r, g, b = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
        # use light colors for better visibility
        r, g, b = random.randint(128, 255), random.randint(128, 255), random.randint(128, 255)
        x_pos = random.randint(0, max(1, w - font_size * len(text) // 2))
        y_pos = random.randint(0, max(1, h - font_size))

        try:
            font = ImageFont.truetype("arial.ttf", size=font_size)
        except OSError:
            font = ImageFont.load_default()

        # draw.text((x_pos, y_pos), text, fill=(r, g, b, int(alpha * 255)), font=font)
        # create a new blank image for the text with alpha channel
        text_img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        # rotate the text image by a random angle
        angle = random.uniform(-30, 30)

        text_draw = ImageDraw.Draw(text_img)
        text_draw.text((x_pos, y_pos), text, fill=(r, g, b, int(alpha * 255)), font=font)
        text_img = text_img.rotate(angle, expand=1, resample=Image.BICUBIC)

        # paste the rotated text image onto the overlay
        # resize the text image to fit within the overlay
        text_img = text_img.resize((w, h), resample=Image.BICUBIC)
        overlay = Image.alpha_composite(overlay, text_img)

    img_rgba = img_pil.convert("RGBA")
    result = Image.alpha_composite(img_rgba, overlay).convert("RGB")
    t = TVF.to_tensor(result).unsqueeze(0)
    return t.clamp(0, 1)



def _combo_jpeg_noise(x: torch.Tensor) -> torch.Tensor:
    return gaussian_noise(jpeg_compress(x, 40), 0.03)


def _combo_blur_resize(x: torch.Tensor) -> torch.Tensor:
    return resize_attack(gaussian_blur(x, 5), 0.50)


def _combo_crop_rotate(x: torch.Tensor) -> torch.Tensor:
    return rotate_attack(crop_attack(x, 0.75), 10.0)


def _combo_jpeg_resize(x: torch.Tensor) -> torch.Tensor:
    return resize_attack(jpeg_compress(x, 60), 0.50)


def _combo_noise_blur_jpeg(x: torch.Tensor) -> torch.Tensor:
    return jpeg_compress(gaussian_blur(gaussian_noise(x, 0.02), 3), 50)



AttackFn = Callable[[torch.Tensor], torch.Tensor]


# ============================================================================
# Shared attack registry -- same attacks for all models
# ============================================================================

_SHARED_ATTACKS: Dict[str, Tuple[str, AttackFn]] = {
    "jpeg_80": ("JPEG Q=80", lambda x: jpeg_compress(x, 80)),
    "jpeg_60": ("JPEG Q=60", lambda x: jpeg_compress(x, 60)),
    "jpeg_40": ("JPEG Q=40", lambda x: jpeg_compress(x, 40)),
    "jpeg_20": ("JPEG Q=20", lambda x: jpeg_compress(x, 20)),
    "gaussian_blur_k3": ("GaussianBlur k=3", lambda x: gaussian_blur(x, 3)),
    "gaussian_blur_k5": ("GaussianBlur k=5", lambda x: gaussian_blur(x, 5)),
    "gaussian_blur_k9": ("GaussianBlur k=9", lambda x: gaussian_blur(x, 9)),
    "gaussian_noise_0.01": ("GaussianNoise s=0.01", lambda x: gaussian_noise(x, 0.01)),
    "gaussian_noise_0.03": ("GaussianNoise s=0.03", lambda x: gaussian_noise(x, 0.03)),
    "gaussian_noise_0.05": ("GaussianNoise s=0.05", lambda x: gaussian_noise(x, 0.05)),
    "brightness_0.75": ("Brightness x0.75", lambda x: brightness_attack(x, 0.75)),
    "brightness_1.25": ("Brightness x1.25", lambda x: brightness_attack(x, 1.25)),
    "brightness_1.50": ("Brightness x1.50", lambda x: brightness_attack(x, 1.50)),
    "contrast_0.75": ("Contrast x0.75", lambda x: contrast_attack(x, 0.75)),
    "contrast_1.25": ("Contrast x1.25", lambda x: contrast_attack(x, 1.25)),
    "contrast_1.50": ("Contrast x1.50", lambda x: contrast_attack(x, 1.50)),
    "saturation_0.75": ("Saturation x0.75", lambda x: saturation_attack(x, 0.75)),
    "saturation_1.25": ("Saturation x1.25", lambda x: saturation_attack(x, 1.25)),
    "saturation_1.50": ("Saturation x1.50", lambda x: saturation_attack(x, 1.50)),
    "rotate_5": ("Rotate 5 deg", lambda x: rotate_attack(x, 5)),
    "rotate_10": ("Rotate 10 deg", lambda x: rotate_attack(x, 10)),
    "rotate_30": ("Rotate 30 deg", lambda x: rotate_attack(x, 30)),
    "hflip": ("Horizontal Flip", lambda x: hflip_attack(x)),
    "resize_0.75": ("Resize 0.75x", lambda x: resize_attack(x, 0.75)),
    "resize_0.50": ("Resize 0.50x", lambda x: resize_attack(x, 0.50)),
    "crop_0.75": ("Crop 0.75x", lambda x: crop_attack(x, 0.75)),
    "crop_0.50": ("Crop 0.50x", lambda x: crop_attack(x, 0.50)),
    "salt_pepper_0.01": ("SaltPepper p=0.01", lambda x: salt_pepper_noise(x, 0.01)),
        "salt_pepper_0.02": ("SaltPepper p=0.02", lambda x: salt_pepper_noise(x, 0.02)),
"text_overlay_0.25": ("TextOverlay a=0.25", lambda x: text_overlay_attack(x, 0.25)),
    "text_overlay_0.50": ("TextOverlay a=0.50", lambda x: text_overlay_attack(x, 0.50)),
    "text_overlay_1.00": ("TextOverlay a=1.00", lambda x: text_overlay_attack(x, 1.00)),
    "combo_jpeg_noise":    ("JPEG Q=40 + Noise", _combo_jpeg_noise),
    "combo_blur_resize":   ("Blur k=5 + Resize 50%", _combo_blur_resize),
    "combo_crop_rotate":   ("Crop 75% + Rotate 10", _combo_crop_rotate),
    "combo_jpeg_resize":   ("JPEG Q=60 + Resize 50%", _combo_jpeg_resize),
    "combo_noise_blur_jpeg": ("Noise+B+JPEG", _combo_noise_blur_jpeg),
}

# Single shared attack key list used by all models
COMMON_ATTACK_KEYS = [
    # JPEG: lightest + heaviest
    "jpeg_80", "jpeg_20",
    # GaussianBlur: lightest + heaviest
    "gaussian_blur_k3", "gaussian_blur_k9",
    # Brightness: darkest + brightest
    "brightness_0.75", "brightness_1.50",
    # Contrast: lowest + highest
    "contrast_0.75", "contrast_1.50",
    # Saturation: lowest + highest
    "saturation_0.75", "saturation_1.50",
    # GaussianNoise: lightest + heaviest
    "gaussian_noise_0.01", "gaussian_noise_0.05",
    # Salt & Pepper: lightest + heaviest
    "salt_pepper_0.01", "salt_pepper_0.02",
    # Rotate: lightest + heaviest
    "rotate_5", "rotate_30",
    # Resize: lightest + heaviest
    "resize_0.75", "resize_0.50",
    # Crop: lightest + heaviest
    "crop_0.75", "crop_0.50",
    # Horizontal Flip
    "hflip",
    # Text Overlay: light + heavy
    "text_overlay_0.25", "text_overlay_0.50",
    # Attack Combinations
    "combo_jpeg_noise",
    "combo_blur_resize",
    "combo_crop_rotate",
    "combo_jpeg_resize",
    "combo_noise_blur_jpeg",
]

# All registered models use the same attack set
REGISTERED_MODELS = ["InvisMark", "MiniWatermarkDemo", "TrustMark-P", "TrustMark-Q", "watermark-anything"]


def get_attacks(model_name: str) -> Dict[str, Tuple[str, AttackFn]]:
    """Get the shared attack set. Identical for all models."""
    result = {}
    for key in COMMON_ATTACK_KEYS:
        if key in _SHARED_ATTACKS:
            result[key] = _SHARED_ATTACKS[key]
    return result


def list_models() -> List[str]:
    return list(REGISTERED_MODELS)
