from __future__ import annotations

import json
from typing import Tuple

import cv2
import numpy as np

from config import AppSettings
from schemas import EvidenceBundle, ViolationEvent


def clamp_bbox(
    bbox: Tuple[int, int, int, int], width: int, height: int
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    return (
        max(0, min(x1, width - 1)),
        max(0, min(y1, height - 1)),
        max(0, min(x2, width)),
        max(0, min(y2, height)),
    )


class EvidenceBuilder:
    def __init__(self, settings: AppSettings):
        self.settings = settings

    def capture_event(self, frame: np.ndarray, event: ViolationEvent) -> EvidenceBundle:
        height, width = frame.shape[:2]
        bbox = clamp_bbox(event.vehicle_bbox, width, height)
        x1, y1, x2, y2 = bbox
        vehicle_crop = frame[y1:y2, x1:x2].copy()

        frame_path = self.settings.storage.frames_dir / f"{event.violation_id}.jpg"
        vehicle_path = self.settings.storage.vehicle_crops_dir / f"{event.violation_id}.jpg"
        metadata_path = self.settings.storage.reports_dir / f"{event.violation_id}.json"

        cv2.imwrite(str(frame_path), frame)
        cv2.imwrite(str(vehicle_path), vehicle_crop)
        metadata_path.write_text(json.dumps(event.to_metadata(), indent=2), encoding="utf-8")

        return EvidenceBundle(
            event=event,
            frame_path=frame_path,
            vehicle_crop_path=vehicle_path,
            vehicle_crop_bbox=bbox,
            frame_shape=frame.shape,
        )
