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

        processed = self.preprocess(image)
        raw_text, confidence = self._extract_text(processed)
        plate_text = self.clean_text(raw_text)
        accepted = self.validate(plate_text, confidence)
        return OCRResult(
            plate_text=plate_text if accepted else None,
            confidence=confidence,
            raw_text=raw_text,
            accepted=accepted,
        )
