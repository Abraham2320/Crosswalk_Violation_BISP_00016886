from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from ultralytics import YOLO

# Custom config sits at the project root (two levels above this file)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_BYTETRACK_CFG = _PROJECT_ROOT / "bytetrack.yaml"
_TRACKER_ARG = str(_BYTETRACK_CFG) if _BYTETRACK_CFG.exists() else "bytetrack.yaml"


class YOLODetector:
    def __init__(self, model_path, classes, conf, imgsz):
        self.model = YOLO(model_path)
        self.classes = classes
        self.conf = conf
        self.imgsz = imgsz
        # Live/device tuning is env-driven so production can force CUDA while
        # local development can stay on CPU without code changes.
        self.device = os.getenv("YOLO_DEVICE", "") or None
        # Half-precision (FP16) only works on CUDA; default to off so CPU runs correctly.
        self.half = os.getenv("YOLO_HALF", "0") != "0"

        # GPU warm-up: one silent inference on a blank frame so the first real
        # frame doesn't pay the JIT/CUDA kernel compilation cost.
        try:
            dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
            self.model.predict(dummy, imgsz=imgsz, verbose=False, device=self.device)
            print("[INFO] YOLO GPU warm-up complete.")
        except Exception:
            pass

    def detect(self, frame):
        return self.model.track(
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
