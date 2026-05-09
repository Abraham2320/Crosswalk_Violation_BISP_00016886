from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:  # noqa: F401
    YOLO = None  # type: ignore

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_BYTETRACK_CFG = _PROJECT_ROOT / "bytetrack.yaml"
_TRACKER_ARG = str(_BYTETRACK_CFG) if _BYTETRACK_CFG.exists() else "bytetrack.yaml"

VEHICLE_CLASS_NAMES: Dict[int, str] = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

VEHICLE_CLASS_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (50, 220, 50),
    1: (0, 165, 255),
    2: (255, 80, 0),
    3: (200, 0, 200),
    5: (0, 200, 200),
    7: (0, 0, 230),
}


VEHICLE_MASK_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (30, 180, 30),
    1: (0, 120, 200),
    2: (200, 60, 0),
    3: (160, 0, 160),
    5: (0, 155, 155),
    7: (0, 0, 190),
}


def class_label(cls: int) -> str:
    return VEHICLE_CLASS_NAMES.get(cls, f"cls{cls}")


def class_color(cls: int) -> Tuple[int, int, int]:
    return VEHICLE_CLASS_COLORS.get(cls, (180, 180, 180))


def mask_color(cls: int) -> Tuple[int, int, int]:
    return VEHICLE_MASK_COLORS.get(cls, (100, 100, 100))


def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def mask_intersects_polygon(
    mask: Optional[np.ndarray],
    polygon: np.ndarray,
    min_ratio: float = 0.02,
) -> bool:
    if mask is None or mask.size == 0:
        return False
    h, w = mask.shape[:2]
    canvas = np.zeros((h, w), dtype=np.uint8)
    pts = polygon.astype(np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(canvas, [pts], 1)
    inter = int(np.count_nonzero(mask & canvas))
    mask_area = int(np.count_nonzero(mask))
    if mask_area == 0:
        return False
    return (inter / mask_area) >= min_ratio


def mask_centroid(mask: np.ndarray) -> Optional[Tuple[float, float]]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


class SegmentedYOLODetector:

    def __init__(
        self,
        model_path: str,
        seg_model_path: str,
        classes: List[int],
        conf: float,
        imgsz: int,
        run_every_n_frames: int = 3,
    ) -> None:
        if YOLO is None:
            raise ImportError("ultralytics package is required")

        self.classes = classes
        self.conf = conf
        self.imgsz = imgsz
        self.run_every_n_frames = max(1, run_every_n_frames)
        self.device = os.getenv("YOLO_DEVICE", "0")
        self.half   = os.getenv("YOLO_HALF", "1") != "0"

        self.det_model = YOLO(model_path)
        self.seg_model: Optional[YOLO] = None
        if seg_model_path:
            try:
                self.seg_model = YOLO(seg_model_path)
                print(f"[INFO] Segmentation model loaded: {seg_model_path}")
            except Exception as exc:
                print(f"[WARN] Segmentation model unavailable ({exc}); "
                      "falling back to bbox-only mode.")

        self._frame_count: int = 0
        self._cached_masks: List[Optional[np.ndarray]] = []
        self._cached_boxes: Optional[np.ndarray] = None

        self._warmup()

    def _warmup(self) -> None:
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        for model, name in ((self.det_model, "detection"),
                            (self.seg_model,  "segmentation")):
            if model is None:
                continue
            try:
                model.predict(
                    dummy, imgsz=self.imgsz, verbose=False, device=self.device
                )
                print(f"[INFO] {name.capitalize()} model warm-up complete.")
            except Exception:
                pass

    def detect(self, frame: np.ndarray):
        return self.det_model.track(
            frame,
            persist=True,
            conf=self.conf,
            imgsz=self.imgsz,
            classes=self.classes,
            tracker=_TRACKER_ARG,
            device=self.device,
            half=self.half,
            verbose=False,
        )

    def segment_frame(
        self,
        frame: np.ndarray,
        tracked_boxes: np.ndarray,
    ) -> List[Optional[np.ndarray]]:
        self._frame_count += 1
        n = len(tracked_boxes)

        if n == 0:
            self._cached_masks = []
            self._cached_boxes = tracked_boxes
            return []

        if (
            self._frame_count % self.run_every_n_frames != 1
            and self._cached_boxes is not None
            and len(self._cached_masks) == n
        ):
            return list(self._cached_masks)

        masks_out: List[Optional[np.ndarray]] = [None] * n

        if self.seg_model is None:
            self._cached_masks = masks_out
            self._cached_boxes = tracked_boxes
            return masks_out

        try:
            results = self.seg_model.predict(
                frame,
                conf=self.conf,
                imgsz=self.imgsz,
                classes=self.classes,
                device=self.device,
                half=self.half,
                verbose=False,
            )
        except Exception as exc:
            print(f"[WARN] Segmentation inference failed: {exc}")
            self._cached_masks = masks_out
            self._cached_boxes = tracked_boxes
            return masks_out

        if not results or results[0].masks is None:
            self._cached_masks = masks_out
            self._cached_boxes = tracked_boxes
            return masks_out

        fh, fw = frame.shape[:2]
        seg_boxes = results[0].boxes.xyxy.cpu().numpy()
        raw_masks = results[0].masks.data.cpu().numpy()

        for det_i, det_box in enumerate(tracked_boxes):
            best_iou  = 0.15
            best_mask: Optional[np.ndarray] = None
            for seg_j, seg_box in enumerate(seg_boxes):
                iou = _box_iou(det_box, seg_box)
                if iou > best_iou:
                    best_iou = iou
                    resized = cv2.resize(
                        raw_masks[seg_j], (fw, fh),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    best_mask = (resized > 0.5).astype(np.uint8)
            masks_out[det_i] = best_mask

        self._cached_masks = masks_out
        self._cached_boxes = tracked_boxes
        return masks_out
