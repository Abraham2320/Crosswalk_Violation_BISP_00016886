from __future__ import annotations

import os
import re

import cv2

from config import AppSettings
from schemas import OCRResult


class OCREngine:
    def __init__(self, settings: AppSettings):
        self.settings = settings
        self._reader = None
        backend = settings.models.ocr_backend.lower()
        if backend == "easyocr":
            try:
                import easyocr  # type: ignore
                use_gpu = os.getenv("OCR_USE_GPU", "1") != "0"
                try:
                    self._reader = easyocr.Reader(["en"], gpu=use_gpu)
                except Exception:
                    self._reader = easyocr.Reader(["en"], gpu=False)
            except Exception:
                self._reader = None
        elif backend == "paddleocr":
            try:
                from paddleocr import PaddleOCR  # type: ignore

                self._reader = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
            except Exception:
                self._reader = None

        self._plate_pattern = re.compile(settings.models.plate_regex)

    def preprocess(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(gray)
        blurred = cv2.GaussianBlur(equalized, (3, 3), 0)
        _, thresholded = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return thresholded

    def _enhance_plate_variants(self, image):
        """
        Build multiple enhanced plate candidates for OCR:
        upscaling, denoise, local-contrast boost, sharpening, and binarization.
        """
        if image is None or image.size == 0:
            return []

        h, w = image.shape[:2]
        scale = 3 if max(h, w) < 120 else 2
        up = cv2.resize(image, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
        den = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)
        clahe = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(8, 8)).apply(den)

        # Unsharp mask style enhancement for character edges.
        g_blur = cv2.GaussianBlur(clahe, (0, 0), 1.4)
        sharp = cv2.addWeighted(clahe, 1.8, g_blur, -0.8, 0)

        _, otsu = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        adaptive = cv2.adaptiveThreshold(
            sharp,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            8,
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        otsu_clean = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, kernel)
        adaptive_clean = cv2.morphologyEx(adaptive, cv2.MORPH_OPEN, kernel)

        return [
            self.preprocess(up),
            sharp,
            otsu,
            adaptive,
            otsu_clean,
            adaptive_clean,
        ]

    def _extract_text(self, processed) -> tuple[str, float]:
        if self._reader is None:
            return "", 0.0

        backend = self.settings.models.ocr_backend.lower()
        if backend == "easyocr":
            results = self._reader.readtext(processed)
            if not results:
                return "", 0.0
            text = " ".join(item[1] for item in results)
            confidence = float(sum(item[2] for item in results) / len(results))
            return text, confidence

        results = self._reader.ocr(processed, cls=True)
        lines = results[0] if results else []
        if not lines:
            return "", 0.0
        text = " ".join(item[1][0] for item in lines)
        confidence = float(sum(item[1][1] for item in lines) / len(lines))
        return text, confidence

    def clean_text(self, text: str) -> str:
        return re.sub(r"[^A-Z0-9]", "", text.upper())

    def validate(self, text: str, confidence: float) -> bool:
        return bool(text) and confidence >= self.settings.models.ocr_confidence_threshold and bool(
            self._plate_pattern.match(text)
        )

    def recognize(self, image_path: str) -> OCRResult:
        image = cv2.imread(str(image_path))
        if image is None:
            return OCRResult(plate_text=None, confidence=0.0, raw_text="", accepted=False)

        candidates = self._enhance_plate_variants(image)
        if not candidates:
            candidates = [self.preprocess(image)]

        best_raw = ""
        best_clean = ""
        best_conf = 0.0
        best_accepted = False

        for processed in candidates:
            raw_text, confidence = self._extract_text(processed)
            clean_text = self.clean_text(raw_text)
            accepted = self.validate(clean_text, confidence)

            # Prefer valid plate matches; otherwise keep highest-confidence fallback.
            if accepted and (not best_accepted or confidence > best_conf):
                best_raw = raw_text
                best_clean = clean_text
                best_conf = confidence
                best_accepted = True
            elif (not best_accepted) and confidence > best_conf:
                best_raw = raw_text
                best_clean = clean_text
                best_conf = confidence

        return OCRResult(
            plate_text=best_clean if best_accepted else None,
            confidence=best_conf,
            raw_text=best_raw,
            accepted=best_accepted,
        )
