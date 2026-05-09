
from __future__ import annotations
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
class VehicleState:
    OUTSIDE = "outside"
    ENTER = "enter"
    INSIDE = "inside"
    EXIT = "exit"
class ObjectFSM:
    def __init__(self):
        self.states = defaultdict(lambda: VehicleState.OUTSIDE)
        self.prev_inside = defaultdict(lambda: False)
    def update(self, obj_id, inside_now):
        inside_prev = self.prev_inside[obj_id]
        if not inside_prev and inside_now:
            self.states[obj_id] = VehicleState.ENTER
        elif inside_prev and inside_now:
            self.states[obj_id] = VehicleState.INSIDE
        elif inside_prev and not inside_now:
            self.states[obj_id] = VehicleState.EXIT
        else:
            self.states[obj_id] = VehicleState.OUTSIDE
        self.prev_inside[obj_id] = inside_now
        return self.states[obj_id]
class TrackState:
    def __init__(self, history_len):
        self.positions = deque(maxlen=history_len)
        self.state = VehicleState.OUTSIDE
    def add(self, cx, cy):
        self.positions.append((cx, cy))
    def ready(self):
        return len(self.positions) == self.positions.maxlen
def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
def apply_cross_class_nms(
    boxes: np.ndarray,
    classes: np.ndarray,
    ids: np.ndarray,
    confs: np.ndarray,
    iou_threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    VEHICLE_CLS = {1, 2, 3, 4, 5, 6, 7}
    n = len(boxes)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        if int(classes[i]) not in VEHICLE_CLS:
            continue
        for j in range(i + 1, n):
            if not keep[j]:
                continue
            if int(classes[j]) not in VEHICLE_CLS:
                continue
            if _iou(boxes[i], boxes[j]) > iou_threshold:
                if confs[i] >= confs[j]:
                    keep[j] = False
                else:
                    keep[i] = False
                    break
    return boxes[keep], classes[keep], ids[keep], confs[keep]
class IDMerger:
    def __init__(self, proximity_px: float = 40.0, min_frames: int = 3):
        self.proximity_px = proximity_px
        self.min_frames = min_frames
        self._close_count: Dict[Tuple[int, int], int] = defaultdict(int)
        self._remap: Dict[int, int] = {}
    def update(
        self,
        ids: np.ndarray,
        boxes: np.ndarray,
    ) -> np.ndarray:
        remapped = np.array([self._remap.get(int(i), int(i)) for i in ids])
        centroids: Dict[int, np.ndarray] = {}
        for idx, track_id in enumerate(remapped):
            b = boxes[idx]
            cx = (b[0] + b[2]) / 2.0
            cy = (b[1] + b[3]) / 2.0
            centroids[int(track_id)] = np.array([cx, cy])
        unique_ids = list(centroids.keys())
        seen_pairs: set = set()
        for i, id_a in enumerate(unique_ids):
            for id_b in unique_ids[i + 1:]:
                pair = (min(id_a, id_b), max(id_a, id_b))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                dist = float(np.linalg.norm(centroids[id_a] - centroids[id_b]))
                if dist < self.proximity_px:
                    self._close_count[pair] += 1
                    if self._close_count[pair] >= self.min_frames:
                        lo, hi = pair
                        self._remap[hi] = lo
                else:
                    self._close_count[pair] = 0
        return np.array([self._remap.get(int(i), int(i)) for i in ids])
@dataclass
class PedestrianTrack:
    track_id: int
    state: str = "OUTSIDE"
    entry_frame: Optional[int] = None
    midline_crossed_frame: Optional[int] = None
    exit_frame: Optional[int] = None
    frames_outside_count: int = 0
    state_since_frame: Optional[int] = None
    centroid: Optional[Tuple[float, float]] = None
    prev_centroid: Optional[Tuple[float, float]] = None
    velocity_history: deque = field(default_factory=lambda: deque(maxlen=10))
    bbox: Optional[Tuple[float, float, float, float]] = None
    axis_progress: float = 0.0
    axis_history: deque = field(default_factory=lambda: deque(maxlen=10))
    direction: str = "STATIC"
    frames_inside_polygon: int = 0
    keypoints: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = field(default=None, repr=False)

@dataclass
class VehicleTrack:
    track_id: int
    polygon_entry_frame: Optional[int] = None
    approach_axis: Optional[str] = None
    polygon_midline: Optional[float] = None
    centroid: Optional[Tuple[float, float]] = None
    prev_centroid: Optional[Tuple[float, float]] = None
    velocity_history: deque = field(default_factory=lambda: deque(maxlen=20))
    centroid_history: deque = field(default_factory=lambda: deque(maxlen=30))
    pre_entry_velocity_snapshot: Optional[List[Tuple[float, float]]] = None
    bbox: Optional[Tuple[float, float, float, float]] = None
    axis_progress: float = 0.0
    violation_pending_frames: int = 0
    is_violator: bool = False
    last_violation_frame: Optional[int] = None
    vehicle_class: int = 2
    mask: Optional[np.ndarray] = field(default=None, repr=False)
    frames_outside_count: int = 0
