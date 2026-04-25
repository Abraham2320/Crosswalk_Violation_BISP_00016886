from __future__ import annotations

import os
import re
from pathlib import Path

import cv2
import numpy as np

from config import AppSettings
from schemas import OCRResult

# ---------------------------------------------------------------------------
# Constants — plate quality gates and character-correction table
# ---------------------------------------------------------------------------
MIN_PLATE_WIDTH  = 80
MIN_PLATE_HEIGHT = 30

# Common OCR misreads at digit positions on license plates
CHAR_FIXES: dict[str, str] = {
    'O': '0', 'I': '1', 'Z': '2',
    'S': '5', 'B': '8', 'G': '6',
    'T': '7', 'L': '1',
}

# ---------------------------------------------------------------------------
# Super-resolution — three tiers, first available is used
# ---------------------------------------------------------------------------
# Tier 1: Real-ESRGAN (best quality, needs: pip install realesrgan basicsr
#         + weights/RealESRGAN_x2plus.pth from github.com/xinntao/Real-ESRGAN/releases)
try:
    from realesrgan import RealESRGANer as _RealESRGANer          # type: ignore
    from basicsr.archs.rrdbnet_arch import RRDBNet as _RRDBNet    # type: ignore
    _REALESRGAN_AVAILABLE = True
except ImportError:
    _REALESRGAN_AVAILABLE = False

# Tier 2: PyTorch bicubic (already available via ultralytics/YOLO)
try:
    import torch as _torch
    import torch.nn.functional as _F
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def _torch_upscale(crop: np.ndarray, scale: int = 4) -> np.ndarray:
    """GPU-accelerated bicubic upscale via PyTorch (available if YOLO is installed)."""
    try:
        bgr = crop if len(crop.shape) == 3 else cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        t = _torch.from_numpy(bgr.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        device = "cuda" if _torch.cuda.is_available() else "cpu"
        t = t.to(device)
        up = _F.interpolate(t, scale_factor=scale, mode="bicubic", align_corners=False)
        up = up.squeeze(0).permute(1, 2, 0).cpu().numpy()
        return (np.clip(up, 0, 1) * 255).astype(np.uint8)
    except Exception:
        return crop


def _load_super_resolution():
    """Return (upsampler, mode) — Real-ESRGAN model, torch callable, or None."""
    # Try Real-ESRGAN first
    if _REALESRGAN_AVAILABLE:
        weights = Path("weights/RealESRGAN_x2plus.pth")
        if weights.exists():
            try:
                model = _RRDBNet(
                    num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=23, num_grow_ch=32, scale=2,
                )
                sr = _RealESRGANer(
                    scale=2, model_path=str(weights), model=model,
                    tile=256, tile_pad=10, gpu_id=0,
                )
                return sr, "realesrgan"
            except Exception:
                pass
    # Fall back to PyTorch bicubic
    if _TORCH_AVAILABLE:
        return _torch_upscale, "torch"
    return None, "none"


# ---------------------------------------------------------------------------
# Pure utility functions
# ---------------------------------------------------------------------------

def clean_plate_text(raw: str) -> str:
    """Strip non-alphanumeric characters; reject strings shorter than 4 chars."""
    text = raw.upper().strip()
    text = re.sub(r'[^A-Z0-9]', '', text)
    return text if len(text) >= 4 else ""


def fix_plate_chars(text: str) -> str:
    """
    Correct common letter↔digit OCR misreads at known digit positions.
    Adjust the index set to match the local plate format if needed.
    Digit positions (0, 1, 3, 4, 5) → correct letter-to-digit confusion.
    """
    corrected = []
    for i, ch in enumerate(text):
        if i in (0, 1, 3, 4, 5) and ch in CHAR_FIXES:
            corrected.append(CHAR_FIXES[ch])
        else:
            corrected.append(ch)
    return ''.join(corrected)


def is_crop_viable(crop) -> bool:
    """Return False if the crop is None or below the minimum useful size for OCR."""
    if crop is None:
        return False
    h, w = crop.shape[:2]
    return w >= MIN_PLATE_WIDTH and h >= MIN_PLATE_HEIGHT


_GAMMA_TABLE = (np.array([((i / 255.0) ** 0.45) * 255 for i in range(256)], dtype=np.uint8))


def _correct_gamma(gray: np.ndarray) -> np.ndarray:
    """Brighten underexposed plates (gamma ≈ 0.45 lifts dark pixels significantly)."""
    mean = float(gray.mean())
    if mean < 80:
        return cv2.LUT(gray, _GAMMA_TABLE)
    return gray


def preprocess_plate(crop: np.ndarray) -> np.ndarray:
    """
    Full preprocessing pipeline applied before every readtext call:
    upscale → grayscale → gamma → CLAHE → denoise → bilateral → sharpen → threshold.
    """
    h, w = crop.shape[:2]
    # Step 1: upscale to at least 120 px height (Lanczos4 preserves edges better)
    scale = max(1, 120 // max(h, 1))
    crop = cv2.resize(crop, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4)

    # Step 2: grayscale
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop.copy()

    # Step 3: gamma correction — recovers detail in dark/night plates
    gray = _correct_gamma(gray)

    # Step 4: CLAHE — fixes remaining exposure issues
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    gray = clahe.apply(gray)

    # Step 5: bilateral filter — preserves character edges while smoothing noise
    gray = cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)

    # Step 6: sharpening
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    gray = cv2.filter2D(gray, -1, kernel)

    # Step 7: adaptive threshold — clean black/white output
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, 4,
    )
    return binary


def upscale_plate(crop: np.ndarray, upsampler, mode: str = "realesrgan") -> np.ndarray:
    """Apply super-resolution to the plate crop. Falls back to original on error."""
    try:
        if mode == "realesrgan":
            upscaled, _ = upsampler.enhance(crop, outscale=4)
            return upscaled
        if mode == "torch":
            return upsampler(crop, scale=4)  # _torch_upscale callable
    except Exception:
        pass
    return crop


def read_plate_multi(crop: np.ndarray, reader) -> tuple[str, float]:
    """
    Run OCR on three preprocessed variants of the plate crop and return the
    highest-confidence (text, confidence) pair across all variants.
    Variants: preprocessed binary, inverted binary, raw 4× Lanczos upscale.
    """
    if crop is None or crop.shape[0] < 10 or crop.shape[1] < 10:
        return "", 0.0

    processed = preprocess_plate(crop)
    variants = [
        processed,
        cv2.bitwise_not(processed),                                      # inverted: light plate / dark text
        cv2.resize(crop, None, fx=6, fy=6, interpolation=cv2.INTER_LANCZOS4),  # raw high-res upscale
    ]

    best_text = ""
    best_conf = 0.0

    for variant in variants:
        try:
            results = reader.readtext(
                variant,
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                batch_size=4,
                detail=1,
                paragraph=False,
                min_size=10,
                contrast_ths=0.1,
                adjust_contrast=0.7,
                text_threshold=0.6,
                low_text=0.3,
                link_threshold=0.3,
            )
            for (_, text, conf) in results:
                cleaned = clean_plate_text(text)
                if cleaned and conf > best_conf:
                    best_text = cleaned
                    best_conf = conf
        except Exception:
            continue

    return best_text, best_conf


# ---------------------------------------------------------------------------
# OCR engine class
# ---------------------------------------------------------------------------

class OCREngine:
    def __init__(self, settings: AppSettings):
        self.settings = settings
        self._reader    = None
        self._upsampler = None

        backend = settings.models.ocr_backend.lower()
        if backend == "easyocr":
            try:
                import easyocr  # type: ignore
                use_gpu    = os.getenv("OCR_USE_GPU", "1") != "0"
                models_dir = os.getenv("OCR_MODEL_DIR", "models/")
                try:
                    self._reader = easyocr.Reader(
                        ['en'],
                        gpu=use_gpu,
                        model_storage_directory=models_dir,
                        download_enabled=True,
                        detector=True,
                        recognizer=True,
                        verbose=False,
                    )
                except Exception:
                    self._reader = easyocr.Reader(
                        ['en'],
                        gpu=False,
                        model_storage_directory=models_dir,
                        download_enabled=True,
                        verbose=False,
                    )
            except Exception:
                self._reader = None

        elif backend == "paddleocr":
            try:
                from paddleocr import PaddleOCR  # type: ignore
                self._reader = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
            except Exception:
                self._reader = None

        # SR upsampler — Real-ESRGAN > PyTorch bicubic > none
        self._upsampler, self._sr_mode = _load_super_resolution()
        if self._upsampler is not None:
            print(f"[OCREngine] SR mode: {self._sr_mode}")
        self._plate_pattern = re.compile(settings.models.plate_regex)

    def recognize_array(self, crop: np.ndarray) -> OCRResult:
        """
        Full OCR pipeline on a numpy crop (no disk I/O):
        1. size gate  2. optional SR  3. multi-variant readtext
        4. clean      5. char-fix     6. validate
        """
        if not is_crop_viable(crop):
            return OCRResult(plate_text=None, confidence=0.0, raw_text="", accepted=False)
        if self._reader is None:
            return OCRResult(plate_text=None, confidence=0.0, raw_text="", accepted=False)

        working = crop
        if self._upsampler is not None:
            working = upscale_plate(working, self._upsampler, self._sr_mode)

        raw_text, confidence = read_plate_multi(working, self._reader)
        clean = fix_plate_chars(clean_plate_text(raw_text))
        accepted = (
            bool(clean)
            and confidence >= self.settings.models.ocr_confidence_threshold
            and bool(self._plate_pattern.match(clean))
        )

        return OCRResult(
            plate_text=clean if accepted else None,
            confidence=confidence,
            raw_text=raw_text,
            accepted=accepted,
        )

    def recognize(self, image_path: str) -> OCRResult:
        """Read plate from a saved image file. Delegates to recognize_array."""
        image = cv2.imread(str(image_path))
        if image is None:
            return OCRResult(plate_text=None, confidence=0.0, raw_text="", accepted=False)
        return self.recognize_array(image)
