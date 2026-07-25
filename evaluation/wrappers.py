"""Model wrappers providing a unified interface for all watermark models.

Each wrapper exposes:
- name, payload_bits, image_size
- encode(image, message) -> watermarked_image
- decode(watermarked_image) -> (decoded_bits, is_detected)
- attacks -> Dict of attack functions
- is_available -> bool
"""

import sys
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"


class ModelWrapper(ABC):
    """Abstract base for watermark model wrappers."""

    name: str = ""
    payload_bits: int = 0
    image_size: Tuple[int, int] = (256, 256)

    @abstractmethod
    def is_available(self) -> bool:
        ...

    @abstractmethod
    def encode(self, image: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def decode(self, image: torch.Tensor, threshold: float = 0.5) -> Tuple[torch.Tensor, bool]:
        ...

    def load_image(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        img = img.resize((self.image_size[1], self.image_size[0]), Image.LANCZOS)
        tensor = torch.from_numpy(np.array(img)).float() / 255.0
        return tensor.permute(2, 0, 1).unsqueeze(0)

    @staticmethod
    def random_message(num_bits: int) -> torch.Tensor:
        return torch.randint(0, 2, (num_bits,)).float()


# ============================================================================
# InvisMark wrapper (needs checkpoint)
# ============================================================================

class InvisMarkWrapper(ModelWrapper):
    name = "InvisMark"
    payload_bits = 100  # updated from checkpoint config on load
    image_size = (256, 256)

    def __init__(self, device="cpu"):
        self.device = device
        self._ready = False

    def is_available(self) -> bool:
        ckpt_dir = MODELS_DIR / "InvisMark" / "ckpts"
        if not ckpt_dir.exists():
            return False
        ckpt_files = list(ckpt_dir.glob("paper.ckpt")) or list(ckpt_dir.glob("*.ckpt"))
        return bool(ckpt_files)

    def _ensure_loaded(self):
        if self._ready:
            return

        # Add InvisMark root to sys.path
        invismark_root = str(MODELS_DIR / "InvisMark")
        import sys as _sys
        if invismark_root not in _sys.path:
            _sys.path.insert(0, invismark_root)

        # Load only encoder/decoder from checkpoint (avoid discriminator, lpips, etc.)
        from model import Encoder, Extractor
        import torchvision.transforms as transforms

        ckpt_dir = MODELS_DIR / "InvisMark" / "ckpts"
        ckpt_files = list(ckpt_dir.glob("paper.ckpt")) or list(ckpt_dir.glob("*.ckpt"))
        if not ckpt_files:
            raise RuntimeError(f"No InvisMark checkpoint in {ckpt_dir}")
        ckpt_path = ckpt_files[0]

        state_dict = torch.load(str(ckpt_path), map_location=self.device, weights_only=False)
        config = state_dict["config"]
        self.payload_bits = config.num_encoded_bits
        self.image_size = config.image_shape

        # Build encoder/decoder only (no discriminator, no lpips, no ffl loss)
        self._encoder = Encoder(config).to(self.device)
        self._decoder = Extractor(config).to(self.device)
        self._encoder.load_state_dict(state_dict["encoder_state_dict"])
        self._decoder.load_state_dict(state_dict["decoder_state_dict"])
        self._encoder.eval()
        self._decoder.eval()

        self._transform = transforms.Compose([
            transforms.Resize(self.image_size),
        ])
        self._ready = True
        print(f"[InvisMark] Loaded {ckpt_path.name}  payload={self.payload_bits}  size={self.image_size}")

    def encode(self, image: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        self._ensure_loaded()
        with torch.no_grad():
            img_norm = (image.to(self.device) * 2 - 1).clamp(-1, 1)
            msg = message[:self.payload_bits].unsqueeze(0).to(self.device)

            # Resize and encode (same logic as Watermark._encode without the orig_diff step)
            resize_inputs = self._transform(img_norm).to(self.device)
            residual = self._encoder(resize_inputs, msg)
            encoded_output = residual + resize_inputs

            # Resize residual back to original size
            from torchvision.transforms import Resize
            orig_diff = Resize(img_norm.shape[-2:])(encoded_output - resize_inputs).cpu()
            final_output = torch.clamp(img_norm.cpu() + orig_diff, -1, 1)
            return (final_output + 1) / 2

    def decode(self, image: torch.Tensor, threshold: float = 0.5) -> Tuple[torch.Tensor, bool]:
        self._ensure_loaded()
        with torch.no_grad():
            img_norm = (image.to(self.device) * 2 - 1).clamp(-1, 1)
            trans_images = self._transform(img_norm)
            extracted_secret = self._decoder(trans_images)
            bits = extracted_secret.squeeze(0).cpu()
            detected = bits.mean().item() > threshold  # simple threshold for detection
            return bits, detected

class MiniWatermarkDemoWrapper(ModelWrapper):
    name = "MiniWatermarkDemo"
    payload_bits = 32
    image_size = (256, 256)

    def __init__(self, device="cpu"):
        self.device = device
        self._ready = False
        self._model = None

    def is_available(self) -> bool:
        ckpt = MODELS_DIR / "MiniWatermarkDemo" / "checkpoints" / "frequency_dct_v4_keyed32_patch.pt"
        return ckpt.exists()

    def _ensure_loaded(self):
        if self._ready:
            return
        mwd_root = str(MODELS_DIR / "MiniWatermarkDemo")
        if mwd_root not in sys.path:
            sys.path.insert(0, mwd_root)
        from src.scripts.common import load_model, center_box, encode_center
        from src.ecc.bch64 import ExtendedBCH64_36
        self._ecc = ExtendedBCH64_36()
        self._encode_center = encode_center
        ckpt_path = MODELS_DIR / "MiniWatermarkDemo" / "checkpoints" / "frequency_dct_v4_keyed32_patch.pt"
        self._model, self._metadata = load_model(ckpt_path, torch.device(self.device))
        self._model.eval()
        self._ready = True

    def encode(self, image: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        self._ensure_loaded()
        with torch.no_grad():
            img = image.to(self.device)
            msg_bits = message[:32].round().long()
            msg_id = int(self._model.bits_to_ids(msg_bits.unsqueeze(0)).item())
            watermarked, _ = self._encode_center(self._model, img, msg_id)
            return watermarked.clamp(0, 1)

    def decode(self, image: torch.Tensor, threshold: float = 0.5) -> Tuple[torch.Tensor, bool]:
        self._ensure_loaded()
        with torch.no_grad():
            img = image.to(self.device)
            result = self._model.decode(img)
            code_logits = result["code_logits"]
            presence = torch.sigmoid(result["presence_logits"]).item()
            decoded_id = self._model.decode_ids(code_logits).item()
            bits = self._model.ids_to_bits(torch.tensor([decoded_id])).squeeze(0).cpu()
            return bits, presence > threshold


# ============================================================================
# TrustMark wrappers (P and Q variants)
# ============================================================================

class _BaseTrustMarkWrapper(ModelWrapper):
    """Shared TrustMark implementation; subclasses set model_type."""

    model_type: str = ""
    payload_bits = 61
    image_size = (256, 256)

    def __init__(self, device="cpu"):
        self.device = device
        self._ready = False
        self._tm = None
        self._capacity = None

    def is_available(self) -> bool:
        try:
            import sys
            sys.path.insert(0, str(MODELS_DIR / "trustmark" / "python"))
            from trustmark import TrustMark
            return True
        except ImportError:
            return False

    def _ensure_loaded(self):
        if self._ready:
            return
        import sys
        tm_path = str(MODELS_DIR / "trustmark" / "python")
        if tm_path not in sys.path:
            sys.path.insert(0, tm_path)
        from trustmark import TrustMark
        self._tm = TrustMark(verbose=False, model_type=self.model_type,
                             encoding_type=TrustMark.Encoding.BCH_5,
                             device=self.device)
        self._capacity = self._tm.schemaCapacity()
        self.payload_bits = self._capacity
        self._ready = True
        print(f"[TrustMark-{self.model_type}] Loaded  capacity={self._capacity}bits")

    def encode(self, image: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        self._ensure_loaded()
        import numpy as np
        img_np = (image.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        from PIL import Image
        cover = Image.fromarray(img_np).convert('RGB')
        bits = message[:self._capacity].round().long().tolist()
        bitstring = ''.join(str(int(b)) for b in bits)
        encoded = self._tm.encode(cover, bitstring, MODE='binary')
        encoded_np = np.array(encoded).astype(np.float32) / 255.0
        return torch.from_numpy(encoded_np).permute(2, 0, 1).unsqueeze(0)

    def decode(self, image: torch.Tensor, threshold: float = 1.0) -> Tuple[torch.Tensor, bool]:
        self._ensure_loaded()
        import numpy as np
        img_np = (image.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        from PIL import Image
        stego = Image.fromarray(img_np).convert('RGB')
        wm_secret, wm_present, wm_schema = self._tm.decode(stego, MODE='binary')
        if wm_present and len(wm_secret) >= self._capacity:
            bits = torch.tensor([float(c) for c in wm_secret[:self._capacity]])
            return bits, True
        elif wm_present and len(wm_secret) > 0:
            bits_list = [float(c) for c in wm_secret]
            bits_list += [0.0] * (self._capacity - len(bits_list))
            return torch.tensor(bits_list), True
        return torch.zeros(self._capacity), False


class TrustMarkPWrapper(_BaseTrustMarkWrapper):
    name = "TrustMark-P"
    model_type = "P"


class TrustMarkQWrapper(_BaseTrustMarkWrapper):
    name = "TrustMark-Q"
    model_type = "Q"


TrustMarkWrapper = TrustMarkQWrapper  # backward compat


class WatermarkAnythingWrapper(ModelWrapper):
    name = "watermark-anything"
    payload_bits = 32
    image_size = (256, 256)

    def __init__(self, device="cpu"):
        self.device = device
        self._ready = False

    def is_available(self) -> bool:
        ckpt = MODELS_DIR / "watermark-anything" / "checkpoints" / "checkpoint.pth"
        json_path = MODELS_DIR / "watermark-anything" / "checkpoints" / "params.json"
        return ckpt.exists() and json_path.exists()

    def _ensure_loaded(self):
        if self._ready:
            return
        wam_root = str(MODELS_DIR / "watermark-anything")
        if wam_root not in sys.path:
            sys.path.insert(0, wam_root)
        import os as _os
        _orig_cwd = _os.getcwd()
        _os.chdir(wam_root)
        try:
            from notebooks.inference_utils import load_model_from_checkpoint
            from watermark_anything.data.transforms import default_transform, normalize_img, unnormalize_img

            json_path = MODELS_DIR / "watermark-anything" / "checkpoints" / "params.json"
            ckpt_path = MODELS_DIR / "watermark-anything" / "checkpoints" / "checkpoint.pth"
            self._wam = load_model_from_checkpoint(str(json_path), str(ckpt_path)).to(self.device)
            self._wam.eval()
            self._ready = True
        finally:
            _os.chdir(_orig_cwd)

    def encode(self, image: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        self._ensure_loaded()
        with torch.no_grad():
            from watermark_anything.data.transforms import normalize_img, unnormalize_img
            img = normalize_img(image.clamp(0, 1)).to(self.device)
            msg = message.unsqueeze(0).to(self.device)
            outputs = self._wam.embed(img, msg)
            return unnormalize_img(outputs["imgs_w"]).clamp(0, 1)

    def decode(self, image: torch.Tensor, threshold: float = 0.1) -> Tuple[torch.Tensor, bool]:
        self._ensure_loaded()
        with torch.no_grad():
            from watermark_anything.data.transforms import normalize_img
            from watermark_anything.data.metrics import msg_predict_inference
            img_norm = normalize_img(image.clamp(0, 1)).to(self.device)
            preds = self._wam.detect(img_norm)["preds"]
            mask_preds = torch.sigmoid(preds[:, 0, :, :])
            bit_preds = preds[:, 1:, :, :]
            pred_msg = msg_predict_inference(bit_preds, mask_preds).cpu().float().squeeze(0)
            detected = mask_preds.mean().item() > threshold
            return pred_msg, detected


# ============================================================================
# Wrapper factory
# ============================================================================

_WRAPPERS = {
    "InvisMark": InvisMarkWrapper,
    "MiniWatermarkDemo": MiniWatermarkDemoWrapper,
    "TrustMark": TrustMarkWrapper,
    "TrustMark-P": TrustMarkPWrapper,
    "TrustMark-Q": TrustMarkQWrapper,
    "watermark-anything": WatermarkAnythingWrapper,
}


def get_wrapper(model_name: str, device: str = "cpu") -> Optional[ModelWrapper]:
    if model_name not in _WRAPPERS:
        print(f"[WARN] Unknown model: {model_name}")
        return None
    wrapper = _WRAPPERS[model_name](device=device)
    if not wrapper.is_available():
        print(f"[WARN] {model_name}: checkpoint or dependencies not available, skipping.")
        return None
    return wrapper


def list_available_models() -> List[str]:
    available = []
    for name, cls in _WRAPPERS.items():
        try:
            if cls().is_available():
                available.append(name)
        except Exception:
            pass
    return available
