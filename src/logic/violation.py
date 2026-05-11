from __future__ import annotations
import math
import os
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple
import cv2
import numpy as np
from detector.tracker import PedestrianTrack, VehicleTrack
SAFE_ENTRY_DELAY_FRAMES: int = int(os.getenv("SAFE_ENTRY_DELAY_FRAMES", "20"))
ENTRY_EVAL_WINDOW_FRAMES: int = 30
YIELD_SPEED_THRESHOLD_PX: float = 4.0
TRACK_RESET_FRAMES: int = 90
YIELD_MIN_SAMPLES: int = 4
MIN_PED_ACTIVE_FRAMES: int = int(os.getenv("MIN_PED_ACTIVE_FRAMES", "6"))
VEHICLE_COOLDOWN_FRAMES: int = int(os.getenv("VEHICLE_COOLDOWN_FRAMES", "60"))
VIOLATION_CONFIRM_FRAMES: int = int(os.getenv("VIOLATION_CONFIRM_FRAMES", "2"))
@dataclass
class Violation:
    car_id: int
    ped_id: int
    frame_number: int
    violation_type: str
    severity: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
def compute_approach_axis(
    car_centroid_history: deque,
    polygon: np.ndarray,
) -> str:
    if len(car_centroid_history) < 2:
        return "from_bottom"
    pts = np.array(list(car_centroid_history), dtype=float)
    avg_pos = pts.mean(axis=0)
    poly_pts = np.array(polygon, dtype=float)
    poly_cx = poly_pts[:, 0].mean()
    poly_cy = poly_pts[:, 1].mean()
    dx = avg_pos[0] - poly_cx
    dy = avg_pos[1] - poly_cy
    if abs(dy) >= abs(dx):
        return "from_top" if dy < 0 else "from_bottom"
    return "from_left" if dx < 0 else "from_right"
def get_polygon_midline(polygon: np.ndarray, approach_axis: str) -> float:
    pts = np.array(polygon, dtype=float)
    if approach_axis in ("from_top", "from_bottom"):
        return float((pts[:, 1].min() + pts[:, 1].max()) / 2.0)
    return float((pts[:, 0].min() + pts[:, 0].max()) / 2.0)
def pedestrian_is_in_exit_zone(
    ped_centroid: Tuple[float, float],
    polygon_midline: float,
    approach_axis: str,
) -> bool:
    px, py = ped_centroid
    if approach_axis == "from_bottom":
        return py < polygon_midline
    if approach_axis == "from_top":
        return py > polygon_midline
    if approach_axis == "from_right":
        return px < polygon_midline
    return px > polygon_midline
def _point_in_polygon(point: Tuple[float, float], polygon: np.ndarray) -> bool:
    pt = (float(point[0]), float(point[1]))
    poly = polygon.astype(np.float32).reshape((-1, 1, 2))
    return cv2.pointPolygonTest(poly, pt, False) >= 0
def _box_overlaps_polygon(bbox: Tuple[float, float, float, float], polygon: np.ndarray,
                          min_ratio: float = 0.01) -> bool:
    x1, y1, x2, y2 = map(float, bbox)
    box_pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    poly_pts = polygon.astype(np.float32)
    inter, _ = cv2.intersectConvexConvex(poly_pts, box_pts)
    box_area = (x2 - x1) * (y2 - y1)
    return (inter / box_area) >= min_ratio if box_area > 0 else False
def _mask_bottom_overlaps_polygon(
    mask: np.ndarray,
    polygon: np.ndarray,
    bottom_ratio: float = 0.20,
) -> bool:
    h, w = mask.shape[:2]
    cut = max(0, int(h * (1.0 - bottom_ratio)))
    foot_mask = np.zeros_like(mask)
    foot_mask[cut:, :] = mask[cut:, :]
    canvas = np.zeros((h, w), dtype=np.uint8)
    pts = polygon.astype(np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(canvas, [pts], 1)
    return bool(np.any(foot_mask & canvas))


def _ped_in_polygon(ped_track, polygon: np.ndarray) -> bool:
    mask = getattr(ped_track, "mask", None)
    if mask is not None and mask.size > 0:
        if _mask_bottom_overlaps_polygon(mask, polygon, bottom_ratio=0.20):
            return True
    if ped_track.bbox is not None:
        x1, y1, x2, y2 = map(float, ped_track.bbox)
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        foot_points = [
            (x1 + 0.50 * w, y2 - 1),
            (x1 + 0.20 * w, y2 - 1),
            (x1 + 0.80 * w, y2 - 1),
        ]
        if any(_point_in_polygon(pt, polygon) for pt in foot_points):
            return True
        lower_box = (x1, y1 + 0.55 * h, x2, y2)
        if _box_overlaps_polygon(lower_box, polygon, min_ratio=0.015):
            return True
        return _box_overlaps_polygon((x1, y1, x2, y2), polygon, min_ratio=0.004)
    if ped_track.centroid is not None:
        return _point_in_polygon(ped_track.centroid, polygon)
    return False
def _update_pedestrian_state_orig(
    ped_track: PedestrianTrack,
    polygon: np.ndarray,
    polygon_midline: float,
    approach_axis: str,
    frame_number: int,
) -> None:
    if ped_track.centroid is None:
        return
    in_polygon = _ped_in_polygon(ped_track, polygon)
    def _set_state(new_state: str) -> None:
        ped_track.state = new_state
        ped_track.state_since_frame = frame_number
    if ped_track.state == "OUTSIDE":
        if in_polygon:
            _set_state("ENTERING")
            ped_track.entry_frame = frame_number
            ped_track.frames_outside_count = 0
    elif ped_track.state == "ENTERING":
        if not in_polygon:
            _set_state("OUTSIDE")
        else:
            _set_state("CROSSING")
    elif ped_track.state == "CROSSING":
        if not in_polygon:
            _set_state("EXITED")
            ped_track.exit_frame = frame_number
    elif ped_track.state == "EXITED":
        if in_polygon:
            _set_state("ENTERING")
            ped_track.entry_frame = frame_number
            ped_track.exit_frame = None
            ped_track.frames_outside_count = 0
def compute_speed(velocity_history: deque) -> float:
    if not velocity_history:
        return 0.0
    return float(sum(math.hypot(dx, dy) for dx, dy in velocity_history) / len(velocity_history))
def was_yielding(
    car_track: VehicleTrack,
    yield_threshold_px: float = YIELD_SPEED_THRESHOLD_PX,
) -> bool:
    snap = car_track.pre_entry_velocity_snapshot
    speeds = (
        np.array([math.hypot(dx, dy) for dx, dy in snap], dtype=float)
        if snap else np.empty(0, dtype=float)
    )
    if speeds.size >= YIELD_MIN_SAMPLES:
        p70 = float(np.percentile(speeds, 70))
        tail = speeds[-min(4, speeds.size):]
        tail_mean = float(tail.mean())
        return p70 < yield_threshold_px and tail_mean < (yield_threshold_px * 1.1)
    h = list(car_track.velocity_history)
    if h:
        cur_speed = sum(math.hypot(dx, dy) for dx, dy in h[-4:]) / min(4, len(h))
        if cur_speed >= yield_threshold_px:
            return False
    return True
def _ped_direction_from_velocity(ped_track: PedestrianTrack) -> str:
    if not ped_track.velocity_history:
        return "STATIC"
    total_dy = sum(dy for _, dy in ped_track.velocity_history)
    if total_dy > 5:
        return "DOWN"
    if total_dy < -5:
        return "UP"
    return "STATIC"
def _car_is_ahead_of_ped(car_track: VehicleTrack, ped_track: PedestrianTrack) -> bool:
    ped_dir = _ped_direction_from_velocity(ped_track)
    if ped_dir == "STATIC":
        return True
    if car_track.centroid is None or ped_track.centroid is None:
        return True
    car_y = car_track.centroid[1]
    ped_y = ped_track.centroid[1]
    if ped_dir == "UP":
        return car_y <= ped_y + 20
    return car_y >= ped_y - 20
def check_violation(
    car_track: VehicleTrack,
    ped_track: PedestrianTrack,
    polygon: np.ndarray,
    frame_number: int,
    approach_axis: str,
    polygon_midline: float,
) -> Optional[Violation]:
    if car_track.polygon_entry_frame is None:
        return None
    if (frame_number - car_track.polygon_entry_frame) > ENTRY_EVAL_WINDOW_FRAMES:
        return None
    if (
        car_track.last_violation_frame is not None
        and (frame_number - car_track.last_violation_frame) < VEHICLE_COOLDOWN_FRAMES
    ):
        return None
    ped_state = ped_track.state
    ped_overlaps = _ped_in_polygon(ped_track, polygon)
    car_not_yielding = not was_yielding(car_track)
    ped_frames_active = (
        (frame_number - ped_track.state_since_frame)
        if ped_track.state_since_frame is not None else 0
    )

    in_path = (
        ped_state in ("ENTERING", "CROSSING")
        and ped_overlaps
        and car_not_yielding
        and ped_frames_active >= MIN_PED_ACTIVE_FRAMES
        and _car_is_ahead_of_ped(car_track, ped_track)
    )
    if in_path:
        car_track.violation_pending_frames += 1
        if car_track.violation_pending_frames >= VIOLATION_CONFIRM_FRAMES:
            return Violation(
                car_id=car_track.track_id,
                ped_id=ped_track.track_id,
                frame_number=frame_number,
                violation_type="FAILED_TO_YIELD",
                severity="HIGH",
            )
    else:
        car_track.violation_pending_frames = max(0, car_track.violation_pending_frames - 1)

    if ped_state == "EXITED":
        if (
            ped_track.exit_frame is not None
            and (frame_number - ped_track.exit_frame) <= SAFE_ENTRY_DELAY_FRAMES
            and car_not_yielding
        ):
            return Violation(
                car_id=car_track.track_id,
                ped_id=ped_track.track_id,
                frame_number=frame_number,
                violation_type="UNSAFE_REENTRY",
                severity="LOW",
            )
    return None
class AxisKalman:
    def __init__(self) -> None:
        self.x = np.zeros(2, dtype=float)
        self.P = np.eye(2) * 100.0
        self.F = np.array([[1.0, 1.0], [0.0, 1.0]])
        self.H = np.array([[1.0, 0.0]])
        self.Q = np.diag([0.5, 0.1])
        self.R = np.array([[2.0]])
        self.initialised = False
    def update(self, measurement: float) -> float:
        if not self.initialised:
            self.x[0] = float(measurement)
            self.initialised = True
            return float(measurement)
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        y = float(measurement) - float((self.H @ self.x)[0])
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ np.array([y])).flatten()
        self.P = (np.eye(2) - K @ self.H) @ self.P
        return float(self.x[0])
def update_pedestrian_axis(
    ped_track: PedestrianTrack,
    axis,
    *,
    history_length: int = 6,
    direction_threshold: float = 0.5,
) -> None:
    if axis is None or ped_track.centroid is None:
        return
    from geometry.polygon_axis import project_onto_axis
    raw_progress = project_onto_axis(ped_track.centroid, axis)
    if not hasattr(ped_track, "_axis_kalman"):
        ped_track._axis_kalman = AxisKalman()
    smoothed = ped_track._axis_kalman.update(raw_progress)
    ped_track.axis_progress = smoothed
    ped_track.axis_history.append(smoothed)
    if len(ped_track.axis_history) >= 3:
        arr = list(ped_track.axis_history)
        slope = (arr[-1] - arr[0]) / max(1, len(arr) - 1)
        if slope > direction_threshold:
            ped_track.direction = "UP"
        elif slope < -direction_threshold:
            ped_track.direction = "DOWN"
        else:
            ped_track.direction = "STATIC"
    else:
        ped_track.direction = "STATIC"
def update_pedestrian_state(
    ped_track: PedestrianTrack,
    polygon: np.ndarray,
    polygon_midline: float,
    approach_axis: str,
    frame_number: int,
    *,
    axis=None,
) -> None:
    _update_pedestrian_state_orig(
        ped_track, polygon, polygon_midline, approach_axis, frame_number,
    )
def update_vehicle_axis(
    veh_track: VehicleTrack,
    axis,
    inside: bool,
    frame_number: int,
) -> None:
    if inside and veh_track.polygon_entry_frame is None:
        veh_track.polygon_entry_frame = frame_number
        veh_track.pre_entry_velocity_snapshot = list(veh_track.velocity_history)
    if not inside:
        veh_track.polygon_entry_frame = None
    if axis is not None and veh_track.centroid is not None:
        from geometry.polygon_axis import project_onto_axis
        veh_track.axis_progress = project_onto_axis(veh_track.centroid, axis)
def vehicle_is_reversing(
    veh_track: VehicleTrack,
    forward_sign: int = 1,
    direction_threshold: int = 15,
    min_avg_motion: float = 2.0,
    min_net_displacement: int = 18,
) -> bool:
    if forward_sign == 0:
        return False
    positions = list(veh_track.centroid_history)
    if len(positions) < 8:
        return False
    recent = positions[-8:]
    dx = recent[-1][0] - recent[0][0]
    avg_x_motion = (
        sum(abs(recent[i + 1][0] - recent[i][0]) for i in range(len(recent) - 1))
        / (len(recent) - 1)
    )
    if avg_x_motion < min_avg_motion:
        return False
    net_disp = max(
        abs(positions[-1][0] - positions[-min(16, len(positions))][0]),
        abs(positions[-1][1] - positions[-min(16, len(positions))][1]),
    )
    if net_disp < min_net_displacement:
        return False
    return (dx * forward_sign) < -direction_threshold
def check_violation_axis(
    veh_track: VehicleTrack,
    ped_tracks: Dict[int, PedestrianTrack],
    polygon: np.ndarray,
    axis,
    frame_number: int,
) -> Optional[Violation]:
    try:
        from config import Config
    except Exception:
        class Config:
            VIOLATION_CONFIRM_FRAMES = 4
            PED_MIN_FRAMES_TO_QUALIFY = 3
            VIOLATION_GRACE_FRAMES = 2
    if veh_track.is_violator:
        return None
    vehicle_inside = False
    if veh_track.centroid is not None and polygon is not None:
        box = veh_track.bbox
        if box is not None:
            vehicle_inside = _box_overlaps_polygon(box, polygon, min_ratio=0.02)
        else:
            vehicle_inside = _point_in_polygon(veh_track.centroid, polygon)
    if not vehicle_inside:
        veh_track.violation_pending_frames = max(
            0, veh_track.violation_pending_frames - Config.VIOLATION_GRACE_FRAMES
        )
        return None
    active_peds = [
        p for p in ped_tracks.values()
        if p.direction in ("UP", "DOWN")
        and p.state in ("ENTERING", "CROSSING")
        and p.frames_inside_polygon >= Config.PED_MIN_FRAMES_TO_QUALIFY
    ]
    if not active_peds:
        veh_track.violation_pending_frames = 0
        return None
    in_path = any(_vehicle_conflicts_with_ped(veh_track, p, axis) for p in active_peds)
    if in_path:
        veh_track.violation_pending_frames += 1
        if veh_track.violation_pending_frames >= Config.VIOLATION_CONFIRM_FRAMES:
            veh_track.is_violator = True
            primary_ped = active_peds[0]
            return Violation(
                car_id=veh_track.track_id,
                ped_id=primary_ped.track_id,
                frame_number=frame_number,
                violation_type="FAILED_TO_YIELD",
                severity="HIGH",
            )
    else:
        veh_track.violation_pending_frames = max(
            0, veh_track.violation_pending_frames - Config.VIOLATION_GRACE_FRAMES
        )
    return None
def _vehicle_conflicts_with_ped(
    veh: VehicleTrack,
    ped: PedestrianTrack,
    axis,
) -> bool:
    if axis is None:
        if veh.centroid is None or ped.centroid is None:
            return False
        vx, vy = veh.centroid
        px, py = ped.centroid
        return math.hypot(vx - px, vy - py) < 150.0
    from geometry.polygon_axis import project_onto_axis
    perp_unit = axis["axis_perp_unit"]
    entry_A = axis["entry_A"]
    veh_arr = np.asarray(veh.centroid, dtype=np.float32)
    ped_arr = np.asarray(ped.centroid, dtype=np.float32)
    veh_perp = float(np.dot(veh_arr - entry_A, perp_unit))
    ped_perp = float(np.dot(ped_arr - entry_A, perp_unit))
    half_width = axis.get("axis_perp_length", 200.0) * 0.5 + 50.0
    return abs(veh_perp - ped_perp) < half_width
