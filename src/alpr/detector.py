from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np
from config import AppSettings
from schemas import EvidenceBundle, PlateDetectionResult

try:
    from ultralytics import YOLO as _YOLO
except ImportError:
    _YOLO = None  # type: ignore


class _PlateYOLO:
    def __init__(self, model_path: str, classes, conf: float, imgsz: int) -> None:
        self._model = _YOLO(model_path)
        self._classes = classes
        self._conf = conf
        self._imgsz = imgsz

    def detect(self, image):
        return self._model.predict(
            image,
            conf=self._conf,
            imgsz=self._imgsz,
            classes=self._classes,
            verbose=False,
        )


class LicensePlateDetector:
    def __init__(self, settings: AppSettings):
        self.settings = settings
        self._detector: Optional[_PlateYOLO] = None
        model_path = settings.models.license_plate_model_path
        if model_path and Path(model_path).exists() and _YOLO is not None:
            self._detector = _PlateYOLO(
                model_path=model_path,
                classes=settings.models.plate_classes or None,
                conf=settings.models.plate_confidence,
                imgsz=settings.models.image_size,
            )
        self._cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_russian_plate_number.xml"
        )
    def _best_box(self, image) -> Tuple[Optional[Tuple[int, int, int, int]], float]:
        if image is None:
            return None, 0.0
        if self._detector is not None:
            results = self._detector.detect(image)
            if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                best_idx = int(np.argmax(confs))
                x1, y1, x2, y2 = boxes[best_idx].astype(int).tolist()
                return (x1, y1, x2, y2), float(confs[best_idx])
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        candidates = self._cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
        if len(candidates) > 0:
            x, y, w, h = max(candidates, key=lambda item: item[2] * item[3])
            return (int(x), int(y), int(x + w), int(y + h)), 0.5
        return None, 0.0
    def detect(self, evidence: EvidenceBundle) -> PlateDetectionResult:
        vehicle_image = cv2.imread(str(evidence.vehicle_crop_path))
        bbox, confidence = self._best_box(vehicle_image)
        source = "vehicle_crop"
        plate_crop_path = None
        if bbox is not None and vehicle_image is not None:
            x1, y1, x2, y2 = bbox
            crop = vehicle_image[y1:y2, x1:x2].copy()
            plate_crop_path = self.settings.storage.plate_crops_dir / f"{evidence.event.violation_id}.jpg"
            cv2.imwrite(str(plate_crop_path), crop)
        return PlateDetectionResult(
            plate_bbox=bbox,
            plate_crop_path=plate_crop_path,
            source=source,
            confidence=confidence,
        )
