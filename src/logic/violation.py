from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Tuple

import cv2
import numpy as np

from detector.tracker import PedestrianTrack, VehicleTrack

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAFE_ENTRY_DELAY_FRAMES: int = 45       # frames after ped exits; car entry within = violation
ENTRY_EVAL_WINDOW_FRAMES: int = 3       # tolerate brief frame drops around vehicle entry
YIELD_SPEED_THRESHOLD_PX: float = 4.0  # px/frame — below this counts as yielding
TRACK_RESET_FRAMES: int = 90            # frames outside before ped track is pruned


# ---------------------------------------------------------------------------
# Violation event
# ---------------------------------------------------------------------------

@dataclass
class Violation:
    car_id: int
    ped_id: int
    frame_number: int
    violation_type: str    # "FAILED_TO_YIELD" | "UNSAFE_REENTRY"
    severity: str          # "HIGH" | "LOW"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Geometric helpers
# ---------------------------------------------------------------------------

def compute_approach_axis(
    car_centroid_history: deque,
    polygon: np.ndarray,
) -> str:
    """
    Determine which side the car approaches from using its pre-entry positions
    relative to the polygon centroid.
    Returns: "from_top" | "from_bottom" | "from_left" | "from_right"
    """
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
    """
    Return the midline coordinate perpendicular to the approach axis.
    Vertical approaches (from_top/from_bottom) → mid-Y.
    Horizontal approaches (from_left/from_right) → mid-X.
    """
    pts = np.array(polygon, dtype=float)
    if approach_axis in ("from_top", "from_bottom"):
        return float((pts[:, 1].min() + pts[:, 1].max()) / 2.0)
    return float((pts[:, 0].min() + pts[:, 0].max()) / 2.0)


def pedestrian_is_in_exit_zone(
    ped_centroid: Tuple[float, float],
    polygon_midline: float,
    approach_axis: str,
) -> bool:
    """
    True if the pedestrian centroid is in the half of the polygon furthest
    from the car's approach side (i.e. the exit zone the pedestrian heads toward).
    """
    px, py = ped_centroid
    if approach_axis == "from_bottom":
        return py < polygon_midline     # exit zone = top half
    if approach_axis == "from_top":
        return py > polygon_midline     # exit zone = bottom half
    if approach_axis == "from_right":
        return px < polygon_midline     # exit zone = left half
    return px > polygon_midline         # from_left → exit zone = right half


def _point_in_polygon(point: Tuple[float, float], polygon: np.ndarray) -> bool:
    """cv2.pointPolygonTest wrapper; True if point is inside or on boundary."""
    pt = (float(point[0]), float(point[1]))
    poly = polygon.astype(np.float32).reshape((-1, 1, 2))
    return cv2.pointPolygonTest(poly, pt, False) >= 0


def _box_overlaps_polygon(bbox: Tuple[float, float, float, float], polygon: np.ndarray,
                          min_ratio: float = 0.01) -> bool:
    """
    True when the pedestrian bounding box overlaps the polygon by at least
    min_ratio of the box area.  This handles angled camera setups where the
    pedestrian centroid can be outside the polygon boundary even though the
    body clearly overlaps the crosswalk zone.
    """
    x1, y1, x2, y2 = map(float, bbox)
    box_pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    poly_pts = polygon.astype(np.float32)
    inter, _ = cv2.intersectConvexConvex(poly_pts, box_pts)
    box_area = (x2 - x1) * (y2 - y1)
    return (inter / box_area) >= min_ratio if box_area > 0 else False


def _ped_in_polygon(ped_track, polygon: np.ndarray) -> bool:
    """Check pedestrian against polygon using bbox when available, centroid otherwise."""
    if ped_track.bbox is not None:
        return _box_overlaps_polygon(ped_track.bbox, polygon)
    if ped_track.centroid is not None:
        return _point_in_polygon(ped_track.centroid, polygon)
    return False


# ---------------------------------------------------------------------------
# Pedestrian FSM update
# ---------------------------------------------------------------------------

def update_pedestrian_state(
    ped_track: PedestrianTrack,
    polygon: np.ndarray,
    polygon_midline: float,
    approach_axis: str,
    frame_number: int,
) -> None:
    """
    Mutate ped_track.state based on the pedestrian's current centroid position.

    OUTSIDE  → ENTERING  : box/centroid enters polygon
    ENTERING → CROSSING  : in near half (not yet past midline)
    ENTERING → CLEARING  : already past midline on entry
    CROSSING → CLEARING  : centroid crosses midline
    CLEARING → EXITED    : box/centroid leaves polygon
    EXITED   → ENTERING  : box/centroid re-enters (reuse track, reset exit_frame)
    Any inside state → OUTSIDE : box/centroid left unexpectedly (short visit)
    """
    if ped_track.centroid is None:
        return

    in_polygon = _ped_in_polygon(ped_track, polygon)

    if ped_track.state == "OUTSIDE":
        if in_polygon:
            ped_track.state = "ENTERING"
            ped_track.entry_frame = frame_number
            ped_track.frames_outside_count = 0

    elif ped_track.state == "ENTERING":
        if not in_polygon:
            ped_track.state = "OUTSIDE"
        elif pedestrian_is_in_exit_zone(ped_track.centroid, polygon_midline, approach_axis):
            ped_track.state = "CLEARING"
            ped_track.midline_crossed_frame = frame_number
        else:
            ped_track.state = "CROSSING"

    elif ped_track.state == "CROSSING":
        if not in_polygon:
            ped_track.state = "OUTSIDE"
        elif pedestrian_is_in_exit_zone(ped_track.centroid, polygon_midline, approach_axis):
            ped_track.state = "CLEARING"
            ped_track.midline_crossed_frame = frame_number

    elif ped_track.state == "CLEARING":
        if not in_polygon:
            ped_track.state = "EXITED"
            ped_track.exit_frame = frame_number

    elif ped_track.state == "EXITED":
        if in_polygon:
            ped_track.state = "ENTERING"
            ped_track.entry_frame = frame_number
            ped_track.exit_frame = None
            ped_track.frames_outside_count = 0


# ---------------------------------------------------------------------------
# Speed / yield helpers
# ---------------------------------------------------------------------------

def compute_speed(velocity_history: deque) -> float:
    """Mean pixel displacement per frame from velocity_history [(dx, dy), ...]."""
    if not velocity_history:
        return 0.0
    return float(sum(math.hypot(dx, dy) for dx, dy in velocity_history) / len(velocity_history))


def was_yielding(
    car_track: VehicleTrack,
    yield_threshold_px: float = YIELD_SPEED_THRESHOLD_PX,
) -> bool:
    """
    True if the car's pre-entry velocity snapshot shows it was moving slowly
    (min speed < yield_threshold_px px/frame) — i.e. it yielded for pedestrians.
    Returns False (not yielding) when no snapshot is available.
    """
    if not car_track.pre_entry_velocity_snapshot:
        return False
    speeds = [math.hypot(dx, dy) for dx, dy in car_track.pre_entry_velocity_snapshot]
    return bool(speeds) and min(speeds) < yield_threshold_px


# ---------------------------------------------------------------------------
# Main violation checker
# ---------------------------------------------------------------------------

def check_violation(
    car_track: VehicleTrack,
    ped_track: PedestrianTrack,
    polygon: np.ndarray,
    frame_number: int,
    approach_axis: str,
    polygon_midline: float,
) -> Optional[Violation]:
    """
    Evaluate a (car, pedestrian) pair for a crosswalk violation.
    Fires within a short window after the car enters the polygon to tolerate
    dropped/skipped frames in live pipelines.

    Scenario A — ped EXITED:
        No direct overlap violation. Exception → UNSAFE_REENTRY (LOW):
        if ped exited within SAFE_ENTRY_DELAY_FRAMES and car was not yielding.

    Scenario B — ped ENTERING/CROSSING/CLEARING and still overlapping polygon:
        FAILED_TO_YIELD (HIGH) if car was not yielding.

    Returns None when no violation applies.
    """
    if car_track.polygon_entry_frame is None:
        return None
    if (frame_number - car_track.polygon_entry_frame) > ENTRY_EVAL_WINDOW_FRAMES:
        return None

    ped_state = ped_track.state
    ped_overlaps = _ped_in_polygon(ped_track, polygon)
    car_not_yielding = not was_yielding(car_track)

    if ped_state in ("ENTERING", "CROSSING", "CLEARING") and ped_overlaps and car_not_yielding:
        return Violation(
            car_id=car_track.track_id,
            ped_id=ped_track.track_id,
            frame_number=frame_number,
            violation_type="FAILED_TO_YIELD",
            severity="HIGH",
        )

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
