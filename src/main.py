from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Set

import cv2
import numpy as np

from config import (
    CONF_THRESHOLD,
    DETECTION_CLASSES,
    IMG_SIZE,
    MODEL_PATH,
    VIDEO_PATH,
    settings,
)

# ---------------------------------------------------------------------------
# Project root — resolved relative to this file (src/../)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SNAPSHOTS_DIR = _PROJECT_ROOT / "static" / "snapshots"
from detector.tracker import (
    IDMerger,
    PedestrianTrack,
    VehicleTrack,
    apply_cross_class_nms,
)
from detector.yolo_detector import YOLODetector
from geometry.crosswalk import CrosswalkZone
from geometry.polygon_editor import PolygonEditor
from logic.violation import (
    TRACK_RESET_FRAMES,
    check_violation,
    compute_approach_axis,
    get_polygon_midline,
    update_pedestrian_state,
)
from schemas import ViolationEvent
from services.pipeline import EnforcementPipeline
from vision.draw import draw_box
from vision.stabilizer import VideoStabilizer


WINDOW_NAME = "Crosswalk Violation System"

# Amber #F59E0B in BGR
_AMBER_BGR = (11, 158, 245)
_RED_BGR   = (0, 0, 239)
_BLUE_BGR  = (255, 0, 0)


def _save_violation_snapshot(
    frame: np.ndarray,
    violation_box,
    all_boxes,
    all_classes,
    polygon,
    event,
) -> Optional[str]:
    """
    Draw annotation overlay on a frame copy and save to static/snapshots/.
    Returns the relative path "snapshots/filename.jpg", or None on failure.
    """
    try:
        snap   = frame.copy()
        _, w   = snap.shape[:2]

        # ── Red semi-transparent banner (top 40 px, opacity 0.6) ────────────
        overlay = snap.copy()
        cv2.rectangle(overlay, (0, 0), (w, 40), _RED_BGR, -1)
        cv2.addWeighted(overlay, 0.6, snap, 0.4, 0, snap)

        # ── All bounding boxes ───────────────────────────────────────────────
        for b, c in zip(all_boxes, all_classes):
            x1, y1, x2, y2 = [int(v) for v in b]
            color = _BLUE_BGR if int(c) == 0 else (0, 0, 255)
            cv2.rectangle(snap, (x1, y1), (x2, y2), color, 2)

        # ── Offending vehicle bbox (bright red, 3 px) ────────────────────────
        vx1, vy1, vx2, vy2 = [int(v) for v in violation_box]
        cv2.rectangle(snap, (vx1, vy1), (vx2, vy2), (0, 0, 255), 3)

        # ── Polygon outline in amber ─────────────────────────────────────────
        poly_pts = np.array(polygon, dtype=np.int32)
        cv2.polylines(snap, [poly_pts], True, _AMBER_BGR, 2)

        # ── Text overlay (top-left, inside banner area) ──────────────────────
        ts = event.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        lines = [
            "VIOLATION DETECTED",
            f"Type: {event.violation_type}",
            f"Vehicle ID: {event.vehicle_id}",
            "Plate: UNDETECTED",
            f"Time: {ts}",
        ]
        y_off = 14
        for line in lines:
            cv2.putText(
                snap, line, (8, y_off),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA,
            )
            y_off += 16

        # ── Save ─────────────────────────────────────────────────────────────
        _SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
        fname     = f"snapshot_{event.violation_id}_{event.frame_index}.jpg"
        save_path = _SNAPSHOTS_DIR / fname
        cv2.imwrite(str(save_path), snap)
        return f"snapshots/{fname}"
    except Exception as exc:
        print(f"[WARN] Failed to save violation snapshot: {exc}")
        return None


def draw_zone_overlay(frame, polygon, color, alpha=0.2):
    overlay = frame.copy()
    cv2.fillPoly(overlay, [polygon], color)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.polylines(frame, [polygon], True, color, 2)


def _ped_direction(ped_track: PedestrianTrack) -> str:
    if not ped_track.velocity_history:
        return "STATIC"
    total_dy = sum(dy for _, dy in ped_track.velocity_history)
    if total_dy > 5:
        return "DOWN"
    if total_dy < -5:
        return "UP"
    return "STATIC"


def _estimate_speed(vt: VehicleTrack) -> Optional[float]:
    """Return mean speed in px/frame from pre-entry velocity snapshot, or None."""
    history = getattr(vt, "pre_entry_velocity_snapshot", None)
    if not history:
        return None
    magnitudes = [(dx ** 2 + dy ** 2) ** 0.5 for dx, dy in history]
    return round(sum(magnitudes) / len(magnitudes), 2)


def build_event(
    frame_index: int,
    box,
    car_id: int,
    violation,
    polygon,
    ped_track: PedestrianTrack,
    veh_track: Optional[VehicleTrack] = None,
) -> ViolationEvent:
    x1, y1, x2, y2 = [int(v) for v in box]
    return ViolationEvent.create(
        vehicle_id=car_id,
        frame_index=frame_index,
        vehicle_bbox=(x1, y1, x2, y2),
        vehicle_zone=None,
        polygon=[tuple(map(int, pt)) for pt in polygon],
        pedestrian_direction=_ped_direction(ped_track),
        pedestrian_zone=None,
        confidence=1.0,
        location=settings.runtime.location_name,
        violation_type=violation.violation_type,
        severity=violation.severity,
        vehicle_speed_estimate=_estimate_speed(veh_track) if veh_track else None,
    )


def main():
    parser = argparse.ArgumentParser(description="Crosswalk Violation System")
    parser.add_argument(
        "--chatbot", action="store_true",
        help="Launch interactive chatbot instead of processing video",
    )
    parser.add_argument(
        "--no-stabilize", action="store_true",
        help="Disable video stabilisation",
    )
    parser.add_argument(
        "--video", metavar="PATH",
        help="Path to video file (overrides VIDEO_PATH env / config default)",
    )
    args = parser.parse_args()

    if args.chatbot:
        from chatbot import run_chatbot
        run_chatbot()
        return

    enable_stabilization = not args.no_stabilize
    video_source = args.video if args.video else VIDEO_PATH

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    editor = PolygonEditor(WINDOW_NAME)
    polygon_loaded = editor.load()
    cv2.setMouseCallback(WINDOW_NAME, editor.mouse_callback)

    if not polygon_loaded:
        print("Calibration mode:")
        print("LEFT click -> add polygon point")
        print("RIGHT click -> finish polygon")
        while True:
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("Cannot read frame during calibration")

            editor.draw(frame)
            cv2.imshow(WINDOW_NAME, frame)
            cv2.waitKey(1)

            if editor.done:
                editor.save()
                break

    polygon = editor.get_polygon()
    if polygon is None:
        raise RuntimeError("Polygon missing or invalid")

    np_polygon = np.array(polygon, dtype=np.float32)

    crosswalk = CrosswalkZone(polygon)
    upper_poly, lower_poly = crosswalk.get_split_polygons(ratio=settings.runtime.split_ratio)

    # Default approach axis derived from polygon aspect ratio
    pw = float(np_polygon[:, 0].max() - np_polygon[:, 0].min())
    ph = float(np_polygon[:, 1].max() - np_polygon[:, 1].min())
    default_approach_axis = "from_bottom" if ph >= pw else "from_left"
    default_polygon_midline = get_polygon_midline(np_polygon, default_approach_axis)

    detector = YOLODetector(MODEL_PATH, DETECTION_CLASSES, CONF_THRESHOLD, IMG_SIZE)
    id_merger = IDMerger(proximity_px=40.0, min_frames=3)
    enforcement_pipeline = EnforcementPipeline(settings)
    stabilizer = VideoStabilizer() if enable_stabilization else None

    ped_tracks: Dict[int, PedestrianTrack] = {}
    veh_tracks: Dict[int, VehicleTrack] = {}
    vehicles_in_polygon: Set[int] = set()    # IDs currently inside polygon
    active_violation_cars: Set[int] = set()  # cars currently showing violation state
    triggered_pairs: Set[tuple] = set()      # (car_id, ped_id) fired this entry cycle

    frame_index = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_index += 1

            if stabilizer is not None:
                if frame_index == 1:
                    stabilizer.init_reference(frame)
                else:
                    frame = stabilizer.stabilize(frame)

            results = detector.detect(frame)

            crosswalk.draw(frame)
            crosswalk.draw_half_split(frame, ratio=settings.runtime.split_ratio)
            draw_zone_overlay(frame, upper_poly, (255, 0, 0), alpha=0.15)
            draw_zone_overlay(frame, lower_poly, (0, 255, 0), alpha=0.15)

            if stabilizer is not None:
                stab_label = "Stabilised" if stabilizer.is_stable else "Unstable"
                stab_color = (0, 255, 0) if stabilizer.is_stable else (0, 0, 255)
                cv2.putText(frame, stab_label, (frame.shape[1] - 160, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, stab_color, 2)

            cv2.putText(
                frame,
                f"P:{len(ped_tracks)} V:{len(veh_tracks)}",
                (16, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

            if results and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy().astype(int)
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                confs = results[0].boxes.conf.cpu().numpy()

                boxes, classes, ids, confs = apply_cross_class_nms(
                    boxes, classes, ids, confs, iou_threshold=0.5
                )
                ids = id_merger.update(ids, boxes)

                current_ped_ids: Set[int] = set()
                current_veh_ids: Set[int] = set()
                newly_in_polygon: Set[int] = set()

                # ── First pass: update track objects ─────────────────────
                for box, cls, obj_id in zip(boxes, classes, ids):
                    obj_id = int(obj_id)
                    x1, y1, x2, y2 = box
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0

                    if cls == 0:  # pedestrian
                        current_ped_ids.add(obj_id)
                        if obj_id not in ped_tracks:
                            ped_tracks[obj_id] = PedestrianTrack(track_id=obj_id)
                        pt = ped_tracks[obj_id]
                        pt.prev_centroid = pt.centroid
                        pt.centroid = (cx, cy)
                        if pt.prev_centroid is not None:
                            pt.velocity_history.append((
                                cx - pt.prev_centroid[0],
                                cy - pt.prev_centroid[1],
                            ))
                        pt.frames_outside_count = 0
                        update_pedestrian_state(
                            pt, np_polygon,
                            default_polygon_midline,
                            default_approach_axis,
                            frame_index,
                        )

                    else:  # vehicle
                        current_veh_ids.add(obj_id)
                        if obj_id not in veh_tracks:
                            veh_tracks[obj_id] = VehicleTrack(track_id=obj_id)
                        vt = veh_tracks[obj_id]
                        vt.prev_centroid = vt.centroid
                        vt.centroid = (cx, cy)
                        if vt.prev_centroid is not None:
                            vt.velocity_history.append((
                                cx - vt.prev_centroid[0],
                                cy - vt.prev_centroid[1],
                            ))
                        vt.centroid_history.append((cx, cy))

                        inside = crosswalk.intersects_box(box, min_ratio=0.02)
                        was_inside = obj_id in vehicles_in_polygon

                        if inside:
                            newly_in_polygon.add(obj_id)
                            if not was_inside:
                                # First entry this cycle: compute approach, snapshot velocity
                                approach_axis = compute_approach_axis(
                                    vt.centroid_history, np_polygon
                                )
                                vt.approach_axis = approach_axis
                                vt.polygon_midline = get_polygon_midline(
                                    np_polygon, approach_axis
                                )
                                vt.polygon_entry_frame = frame_index
                                vt.pre_entry_velocity_snapshot = list(vt.velocity_history)
                                # Reset triggered pairs and violation state for new entry
                                triggered_pairs -= {p for p in triggered_pairs if p[0] == obj_id}
                                active_violation_cars.discard(obj_id)
                        else:
                            vt.polygon_entry_frame = None
                            active_violation_cars.discard(obj_id)

                vehicles_in_polygon = newly_in_polygon

                # Age out missing pedestrian tracks
                for gone_id in list(ped_tracks.keys()):
                    if gone_id not in current_ped_ids:
                        ped_tracks[gone_id].frames_outside_count += 1
                        if ped_tracks[gone_id].frames_outside_count > TRACK_RESET_FRAMES:
                            del ped_tracks[gone_id]

                # Clean up disappeared vehicle tracks
                for gone_id in list(veh_tracks.keys()):
                    if gone_id not in current_veh_ids:
                        triggered_pairs -= {p for p in triggered_pairs if p[0] == gone_id}
                        active_violation_cars.discard(gone_id)
                        del veh_tracks[gone_id]

                # ── Second pass: violation checks + drawing ───────────────
                for box, cls, obj_id in zip(boxes, classes, ids):
                    obj_id = int(obj_id)
                    x1, y1, x2, y2 = box
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    if cls != 0:  # vehicle
                        vt = veh_tracks.get(obj_id)
                        if vt is not None and vt.polygon_entry_frame == frame_index:
                            for ped_id, pt in ped_tracks.items():
                                pair = (obj_id, ped_id)
                                if pair in triggered_pairs:
                                    continue
                                v = check_violation(
                                    car_track=vt,
                                    ped_track=pt,
                                    polygon=np_polygon,
                                    frame_number=frame_index,
                                    approach_axis=vt.approach_axis or default_approach_axis,
                                    polygon_midline=vt.polygon_midline or default_polygon_midline,
                                )
                                if v is not None:
                                    triggered_pairs.add(pair)
                                    active_violation_cars.add(obj_id)
                                    event = build_event(
                                        frame_index=frame_index,
                                        box=box,
                                        car_id=obj_id,
                                        violation=v,
                                        polygon=polygon,
                                        ped_track=pt,
                                        veh_track=vt,
                                    )
                                    # Save annotated snapshot before submitting
                                    snap_path = _save_violation_snapshot(
                                        frame=frame,
                                        violation_box=box,
                                        all_boxes=boxes,
                                        all_classes=classes,
                                        polygon=polygon,
                                        event=event,
                                    )
                                    event.snapshot_path = snap_path
                                    enforcement_pipeline.submit_violation(frame.copy(), event)

                    obj_class = "person" if cls == 0 else "vehicle"
                    violation_active = cls != 0 and obj_id in active_violation_cars

                    box_color = (
                        (0, 0, 255) if violation_active
                        else (0, 255, 255) if (cls != 0 and obj_id in vehicles_in_polygon)
                        else (0, 255, 0)
                    )
                    draw_box(frame, box, obj_class, box_color)

                    if cls == 0:
                        pt = ped_tracks.get(obj_id)
                        if pt:
                            cv2.putText(
                                frame, pt.state, (cx, cy + 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                            )
                            direction = _ped_direction(pt)
                            if direction != "STATIC":
                                cv2.putText(
                                    frame, direction, (cx, cy - 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2,
                                )
                    else:
                        vt = veh_tracks.get(obj_id)
                        state_label = "INSIDE" if (vt and vt.polygon_entry_frame is not None) else "OUTSIDE"
                        cv2.putText(
                            frame, state_label, (cx, cy + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                        )
                        if violation_active:
                            cv2.putText(
                                frame, "VIOLATION", (cx, cy - 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2,
                            )

            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) == 27:
                break
    finally:
        enforcement_pipeline.shutdown()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
