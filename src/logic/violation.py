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

    OUTSIDE  → ENTERING  : centroid enters polygon
    ENTERING → CROSSING  : in near half (not yet past midline)
    ENTERING → CLEARING  : already past midline on entry
    CROSSING → CLEARING  : centroid crosses midline
    CLEARING → EXITED    : centroid leaves polygon
    EXITED   → ENTERING  : centroid re-enters (reuse track, reset exit_frame)
    Any inside state → OUTSIDE : centroid left unexpectedly (short visit)
    """
    if ped_track.centroid is None:
        return

    in_polygon = _point_in_polygon(ped_track.centroid, polygon)

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
    Only fires on the frame the car first enters the polygon
    (car_track.polygon_entry_frame == frame_number).

    Scenario A — ped CLEARING/EXITED (already past midline):
        No violation.  Exception → UNSAFE_REENTRY (LOW):
        if ped exited within SAFE_ENTRY_DELAY_FRAMES and car was not yielding.

    Scenario B — ped ENTERING/CROSSING (still in danger zone):
        FAILED_TO_YIELD (HIGH) if car was not yielding.

    Returns None when no violation applies.
    """
    if car_track.polygon_entry_frame != frame_number:
        return None

    ped_state = ped_track.state

    if ped_state in ("CLEARING", "EXITED"):
        if (
            ped_state == "EXITED"
            and ped_track.exit_frame is not None
            and (frame_number - ped_track.exit_frame) <= SAFE_ENTRY_DELAY_FRAMES
            and not was_yielding(car_track)
        ):
            return Violation(
                car_id=car_track.track_id,
                ped_id=ped_track.track_id,
                frame_number=frame_number,
                violation_type="UNSAFE_REENTRY",
                severity="LOW",
            )
        return None

    if ped_state in ("ENTERING", "CROSSING"):
        # Guard: confirm centroid is inside polygon and not yet past the midline.
        # This prevents false positives when the FSM state lags the actual position.
        if (
            ped_track.centroid is not None
            and _point_in_polygon(ped_track.centroid, polygon)
            and not pedestrian_is_in_exit_zone(ped_track.centroid, polygon_midline, approach_axis)
            and not was_yielding(car_track)
        ):
            return Violation(
                car_id=car_track.track_id,
                ped_id=ped_track.track_id,
                frame_number=frame_number,
                violation_type="FAILED_TO_YIELD",
                severity="HIGH",
            )

    return None
