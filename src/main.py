from __future__ import annotations

import argparse
import queue as _queue
import re
import threading
import time
import types
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set

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
    ENTRY_EVAL_WINDOW_FRAMES,
    TRACK_RESET_FRAMES,
    check_violation,
    compute_approach_axis,
    get_polygon_midline,
    update_pedestrian_state,
)
from schemas import ViolationEvent
from services.pipeline import EnforcementPipeline
from alpr.detector import LicensePlateDetector
from OCR.engine import OCREngine
from vision.draw import draw_box
from vision.stabilizer import VideoStabilizer


WINDOW_NAME = "Crosswalk Violation System"

# Colour constants (BGR)
_AMBER_BGR  = (11, 158, 245)
_RED_BGR    = (0, 0, 239)
_BLUE_BGR   = (255, 0, 0)
_ORANGE_BGR = (0, 165, 255)    # wrong-direction bounding box colour

# ---------------------------------------------------------------------------
# Directional violation constants — calibrate per camera installation
# ---------------------------------------------------------------------------
# Minimum X-pixel displacement across the 8-frame history window to confirm movement.
DIRECTION_THRESHOLD = 15
# Centroid Y band considered active road area; vehicles whose centroid falls
# outside this range are treated as parked / off-road and are excluded from
# wrong-direction detection.  Adjust to match your camera's road region.
ROAD_Y_RANGE = (350, 1080)   # (y_min, y_max)

# ---------------------------------------------------------------------------
# Deferred plate-capture constants
# ---------------------------------------------------------------------------
# Consecutive absent-detection frames before a violation record is finalised.
MISSING_FRAMES_FINALIZE = 15
# Frames between successive deferred OCR attempts for the same vehicle.
OCR_RETRY_INTERVAL = 5
# Immediately persist the plate at this confidence level.
OCR_HIGH_CONF_THRESHOLD = 0.75
# Minimum confidence to store any plate reading (below this → keep trying).
OCR_MIN_ACCEPT_CONF = 0.35

# ---------------------------------------------------------------------------
# Performance feature flags — set to False to disable for speed testing
# ---------------------------------------------------------------------------
# Heaviest: runs a dedicated plate-YOLO on every visible vehicle every frame.
ENABLE_PLATE_DETECTOR = False
# Heavy: EasyOCR with 3-variant inference + fastNlMeansDenoising per attempt.
# Runs every OCR_RETRY_INTERVAL frames per violated vehicle after it exits zone.
ENABLE_DEFERRED_OCR   = False
# Moderate: one-shot OCR for vehicles that have never violated (runs once per track).
ENABLE_REGULAR_OCR    = False
# Light: pure numpy math on centroid history — essentially free, but toggleable.
ENABLE_WRONG_DIR      = True

# Protects concurrent writes to active_violations entries from OCR threads.
_violations_lock = threading.Lock()

# Plate vote accumulation — indexed by track_id
_plate_votes: defaultdict = defaultdict(list)
_plate_votes_lock = threading.Lock()


def submit_plate_reading(track_id: int, text: str, conf: float) -> None:
    """Record a single OCR reading for later majority-vote selection."""
    with _plate_votes_lock:
        _plate_votes[track_id].append((text, conf))


def get_best_plate(track_id: int):
    """
    Return (best_text, vote_count).
    Picks the most-frequent text among readings with ≥2 votes;
    falls back to highest-confidence single reading when no majority exists.
    """
    with _plate_votes_lock:
        readings = list(_plate_votes.get(track_id, []))
    if not readings:
        return None, 0
    counts = Counter(text for text, _ in readings)
    top_text, top_count = counts.most_common(1)[0]
    if top_count >= 2:
        return top_text, top_count
    best_text, _ = max(readings, key=lambda x: x[1])
    return best_text, 1


def clear_plate_votes(track_id: int) -> None:
    """Remove accumulated votes for a track (called when track is cleaned up)."""
    with _plate_votes_lock:
        _plate_votes.pop(track_id, None)


# ---------------------------------------------------------------------------
# Background YOLO worker
# ---------------------------------------------------------------------------

class _InferenceWorker:
    """Run stabilization + YOLO inference in a background daemon thread."""

    def __init__(self, detector: YOLODetector, stabilizer=None) -> None:
        self._detector = detector
        self._stabilizer = stabilizer
        self._in: _queue.Queue = _queue.Queue(maxsize=1)
        self._out: _queue.Queue = _queue.Queue(maxsize=1)
        self._thread = threading.Thread(target=self._run, daemon=True, name="yolo-worker")
        self._thread.start()

    def _run(self) -> None:
        while True:
            item = self._in.get()
            if item is None:
                break
            frame, frame_index = item
            if self._stabilizer is not None:
                if frame_index == 1:
                    self._stabilizer.init_reference(frame)
                    stable_frame = frame
                else:
                    stable_frame = self._stabilizer.stabilize(frame)
            else:
                stable_frame = frame
            result = self._detector.detect(stable_frame)
            try:
                self._out.get_nowait()
            except _queue.Empty:
                pass
            self._out.put((stable_frame, result))

    def submit(self, frame: np.ndarray, frame_index: int) -> None:
        try:
            self._in.put_nowait((frame.copy(), frame_index))
        except _queue.Full:
            pass

    def get_result(self):
        try:
            return self._out.get_nowait()
        except _queue.Empty:
            return None

    def stop(self) -> None:
        try:
            self._in.put_nowait(None)
        except _queue.Full:
            pass
        self._thread.join(timeout=3.0)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def clean_plate_text(raw: str) -> str:
    """Strip non-alphanumeric characters; returns empty string if fewer than 4 chars."""
    text = re.sub(r"[^A-Z0-9]", "", raw.upper().strip())
    return text if len(text) >= 4 else ""


def _compute_vehicle_direction(vt: VehicleTrack) -> str:
    """Return 'FORWARD', 'REVERSE', or 'STATIONARY' from the last 8 centroid positions."""
    positions = list(vt.centroid_history)
    if len(positions) < 8:
        return "STATIONARY"
    recent = positions[-8:]
    dx = recent[-1][0] - recent[0][0]
    # Gate: require meaningful per-frame X motion so parked/facing cars don't trigger.
    # Detection jitter on a stationary car is typically < 3px/frame; real movement > 5px/frame.
    avg_x_motion = sum(
        abs(recent[i + 1][0] - recent[i][0]) for i in range(len(recent) - 1)
    ) / (len(recent) - 1)
    if avg_x_motion < 5.0:
        return "STATIONARY"
    if dx < -DIRECTION_THRESHOLD:
        return "REVERSE"
    if dx > DIRECTION_THRESHOLD:
        return "FORWARD"
    return "STATIONARY"


def _get_plate_region(
    frame: np.ndarray,
    box,
    plate_detector: Optional[LicensePlateDetector],
) -> Optional[np.ndarray]:
    """
    Return the best plate crop for OCR:
    - If a plate sub-detector is available, use its detected bounding box.
    - Otherwise fall back to the bottom 25% of the vehicle bounding box
      (most likely plate location from a rear/front-facing camera).
    Returns None if the region is too small (< 80x40 px minimum size gate).
    """
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    fh, fw = frame.shape[:2]
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(fw - 1, x2), min(fh - 1, y2)
    bw, bh = x2c - x1c, y2c - y1c

    if bw < 80 or bh < 40:
        return None

    if plate_detector is not None and bw >= 32 and bh >= 32:
        try:
            crop = frame[y1c:y2c, x1c:x2c]
            pbbox, _ = plate_detector._best_box(crop)
            if pbbox is not None:
                px1, py1, px2, py2 = pbbox
                region = crop[py1:py2, px1:px2]
                if region.size > 0 and (px2 - px1) > 8 and (py2 - py1) > 8:
                    return region
        except Exception:
            pass

    # Fallback: bottom 25% of vehicle bbox
    plate_top = y2c - max(int(bh * 0.25), 20)
    region = frame[plate_top:y2c, x1c:x2c]
    return region if region.size > 0 else None


# ---------------------------------------------------------------------------
# Snapshot saving
# ---------------------------------------------------------------------------

def _save_violation_snapshot(
    frame: np.ndarray,
    violation_box,
    all_boxes,
    all_classes,
    polygon,
    event,
) -> Optional[str]:
    try:
        snap = frame.copy()
        _, w  = snap.shape[:2]

        overlay = snap.copy()
        cv2.rectangle(overlay, (0, 0), (w, 40), _RED_BGR, -1)
        cv2.addWeighted(overlay, 0.6, snap, 0.4, 0, snap)

        for b, c in zip(all_boxes, all_classes):
            x1, y1, x2, y2 = [int(v) for v in b]
            color = _BLUE_BGR if int(c) == 0 else (0, 0, 255)
            cv2.rectangle(snap, (x1, y1), (x2, y2), color, 2)

        vx1, vy1, vx2, vy2 = [int(v) for v in violation_box]
        cv2.rectangle(snap, (vx1, vy1), (vx2, vy2), (0, 0, 255), 3)

        poly_pts = np.array(polygon, dtype=np.int32)
        cv2.polylines(snap, [poly_pts], True, _AMBER_BGR, 2)

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
            cv2.putText(snap, line, (8, y_off),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            y_off += 16

        _SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
        fname     = f"snapshot_{event.violation_id}_{event.frame_index}.jpg"
        save_path = _SNAPSHOTS_DIR / fname
        cv2.imwrite(str(save_path), snap)
        return f"snapshots/{fname}"
    except Exception as exc:
        print(f"[WARN] Failed to save violation snapshot: {exc}")
        return None


# ---------------------------------------------------------------------------
# Track helpers
# ---------------------------------------------------------------------------

def _ped_direction(ped_track) -> str:
    if not ped_track.velocity_history:
        return "STATIC"
    total_dy = sum(dy for _, dy in ped_track.velocity_history)
    if total_dy > 5:
        return "DOWN"
    if total_dy < -5:
        return "UP"
    return "STATIC"


def _estimate_speed(vt: VehicleTrack) -> Optional[float]:
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
    ped_track,
    veh_track: Optional[VehicleTrack] = None,
    plate_number: Optional[str] = None,
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
        plate_number=plate_number,
    )


# ---------------------------------------------------------------------------
# OCR worker functions
# ---------------------------------------------------------------------------

def _run_ocr_for_vehicle(
    ocr_engine: OCREngine,
    obj_id: int,
    plate_crop: np.ndarray,
    plate_text_cache: Dict[int, str],
    plate_ocr_pending: Set[int],
) -> None:
    """Standard one-shot OCR for non-violated vehicles."""
    try:
        result = ocr_engine.recognize_array(plate_crop)
        if result.plate_text:
            plate_text_cache[obj_id] = result.plate_text
        elif result.raw_text and len(result.raw_text.strip()) >= 3:
            plate_text_cache[obj_id] = f"~{result.raw_text.strip()[:10]}"
    except Exception:
        pass
    finally:
        plate_ocr_pending.discard(obj_id)


def _run_deferred_ocr(
    ocr_engine: OCREngine,
    track_id: int,
    plate_crop: np.ndarray,
    active_violations: Dict[int, dict],
    plate_ocr_pending: Set[int],
    pipeline: EnforcementPipeline,
) -> None:
    """
    Deferred OCR for vehicles in active_violations.  Keeps trying after the
    car has exited the crosswalk zone (when the plate becomes visible).
    On a high-confidence reading the violation record is updated immediately;
    on track loss the best reading is persisted by _finalise_active_violation.
    """
    try:
        result = ocr_engine.recognize_array(plate_crop)

        raw_clean = clean_plate_text(result.raw_text or "")
        best_text = result.plate_text or (raw_clean if len(raw_clean) >= 4 else None)
        conf = result.confidence

        if best_text and conf >= OCR_MIN_ACCEPT_CONF:
            submit_plate_reading(track_id, best_text, conf)
            with _violations_lock:
                if track_id not in active_violations:
                    return
                av = active_violations[track_id]
                if conf > av["best_confidence"]:
                    av["best_plate"]      = best_text
                    av["best_confidence"] = conf

                # Persist immediately when confidence is high enough
                if conf >= OCR_HIGH_CONF_THRESHOLD and not av["plate_saved"]:
                    av["plate_saved"] = True
                    future       = av.get("future")
                    violation_id = av["violation_id"]
                    if future is not None:
                        try:
                            future.result(timeout=3.0)
                        except Exception:
                            pass
                    pipeline.update_violation_plate(violation_id, best_text, conf)
    except Exception:
        pass
    finally:
        plate_ocr_pending.discard(track_id)


def _finalise_active_violation(
    track_id: int,
    av: dict,
    pipeline: EnforcementPipeline,
) -> None:
    """Persist the best plate reading (or UNREAD) when a track is definitively lost."""
    if av["plate_saved"]:
        return
    voted_plate, _ = get_best_plate(track_id)
    plate  = voted_plate or av["best_plate"] or "UNREAD"
    conf   = av["best_confidence"]
    future = av.get("future")
    if future is not None:
        try:
            future.result(timeout=2.0)
        except Exception:
            pass
    pipeline.update_violation_plate(av["violation_id"], plate, conf)
    av["plate_saved"] = True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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

    # Stabilizer disabled until a faster machine is available — it runs optical
    # flow on every frame inside the YOLO worker and is the second-largest bottleneck.
    enable_stabilization = False
    video_source = args.video if args.video else VIDEO_PATH

    print(f"[INFO] Starting pipeline with source: {video_source}")
    print(f"[INFO] Model: {MODEL_PATH} | conf={CONF_THRESHOLD} | imgsz={IMG_SIZE}")

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    # Frame pacing — read the source FPS so display matches real-time speed.
    # Falls back to 30 fps for live cameras that report 0.
    _src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    _frame_delay = 1.0 / _src_fps
    _last_display_t = 0.0

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
    crosswalk  = CrosswalkZone(polygon)

    pw = float(np_polygon[:, 0].max() - np_polygon[:, 0].min())
    ph = float(np_polygon[:, 1].max() - np_polygon[:, 1].min())
    default_approach_axis   = "from_bottom" if ph >= pw else "from_left"
    default_polygon_midline = get_polygon_midline(np_polygon, default_approach_axis)

    # ── Parallel model loading ────────────────────────────────────────────────
    # YOLO (with GPU warm-up) loads on the main thread.
    # EasyOCR and the plate-detector YOLO load concurrently so neither blocks.
    _plate_box: List[Optional[LicensePlateDetector]] = [None]
    _ocr_box:   List[Optional[OCREngine]]            = [None]

    def _load_plate():
        try:
            _plate_box[0] = LicensePlateDetector(settings)
        except Exception:
            pass

    def _load_ocr():
        try:
            _ocr_box[0] = OCREngine(settings)
        except Exception:
            pass

    _plate_thread = threading.Thread(target=_load_plate, daemon=True)
    _ocr_thread   = threading.Thread(target=_load_ocr,   daemon=True)
    _plate_thread.start()
    _ocr_thread.start()

    # YOLO loads here; warm-up happens inside YOLODetector.__init__
    detector   = YOLODetector(MODEL_PATH, DETECTION_CLASSES, CONF_THRESHOLD, IMG_SIZE)
    stabilizer = VideoStabilizer() if enable_stabilization else None
    worker     = _InferenceWorker(detector, stabilizer)
    id_merger  = IDMerger(proximity_px=40.0, min_frames=3)
    enforcement_pipeline = EnforcementPipeline(settings)

    _plate_thread.join()
    _ocr_thread.join()
    plate_detector = _plate_box[0]
    ocr_engine     = _ocr_box[0]

    # ── Per-track state ───────────────────────────────────────────────────────
    ped_tracks: Dict[int, PedestrianTrack] = {}
    veh_tracks: Dict[int, VehicleTrack]    = {}
    vehicles_in_polygon: Set[int]   = set()
    peds_in_polygon: Set[int]       = set()
    active_violation_cars: Set[int] = set()
    triggered_pairs: Set[tuple]     = set()
    _cached_nms_data                = None
    _last_stable_frame: Optional[np.ndarray] = None

    plate_bbox_cache: Dict[int, tuple] = {}
    plate_text_cache: Dict[int, str]   = {}
    plate_ocr_pending: Set[int]        = set()

    # Wrong-direction tracking
    wrong_dir_flagged: Set[int] = set()

    # Deferred plate capture: track_id → violation meta dict
    # Each entry: {violation_id, violation_type, triggered_at_frame,
    #              best_plate, best_confidence, plate_saved,
    #              frames_missing, last_seen_frame, last_ocr_frame, future}
    active_violations: Dict[int, dict] = {}

    frame_index = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if frame_index == 0:
                    raise RuntimeError(
                        f"Video opened but no frames could be decoded: {video_source}"
                    )
                print(f"[INFO] End of video reached after {frame_index} frames.")
                break

            frame_index += 1

            worker.submit(frame, frame_index)
            worker_result = worker.get_result()
            if worker_result is not None:
                frame, results = worker_result
                _last_stable_frame = frame
            else:
                # Reuse the last analyzed frame so cached boxes stay aligned with the image.
                if _last_stable_frame is not None:
                    frame = _last_stable_frame.copy()
                results = None

            # ── Zone overlay — yellow outline ────────────────────────────────
            cv2.polylines(frame, [np.array(polygon, dtype=np.int32)], True, (0, 255, 255), 2)

            if stabilizer is not None:
                stab_label = "Stabilised" if stabilizer.is_stable else "Unstable"
                stab_color = (0, 255, 0) if stabilizer.is_stable else (0, 0, 255)
                cv2.putText(frame, stab_label, (frame.shape[1] - 160, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, stab_color, 2)

            cv2.putText(
                frame,
                f"P:{len(peds_in_polygon)} V:{len(vehicles_in_polygon)}",
                (16, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
            )

            if results and results[0].boxes.id is not None:
                boxes   = results[0].boxes.xyxy.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy().astype(int)
                ids     = results[0].boxes.id.cpu().numpy().astype(int)
                confs   = results[0].boxes.conf.cpu().numpy()

                boxes, classes, ids, confs = apply_cross_class_nms(
                    boxes, classes, ids, confs, iou_threshold=0.35
                )
                ids = id_merger.update(ids, boxes)

                if len(boxes) > 0:
                    areas      = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                    is_vehicle = classes != 0
                    conf_ok    = np.where(is_vehicle, confs >= 0.30, confs >= 0.25)
                    area_ok    = np.where(is_vehicle, areas >= 1400.0, areas >= 600.0)
                    keep       = conf_ok & area_ok
                    boxes      = boxes[keep]
                    classes    = classes[keep]
                    ids        = ids[keep]
                    confs      = confs[keep]

                _cached_nms_data = (boxes, classes, ids)

                current_ped_ids: Set[int] = set()
                current_veh_ids: Set[int] = set()
                newly_in_polygon: Set[int] = set()
                newly_in_ped_polygon: Set[int] = set()

                # ── First pass: update track objects ──────────────────────────
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
                        pt.bbox = (x1, y1, x2, y2)
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
                        if crosswalk.intersects_box(box, min_ratio=0.02):
                            newly_in_ped_polygon.add(obj_id)

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

                        inside     = crosswalk.intersects_box(box, min_ratio=0.02)
                        was_inside = obj_id in vehicles_in_polygon

                        if inside:
                            newly_in_polygon.add(obj_id)
                            if not was_inside:
                                approach_axis = compute_approach_axis(
                                    vt.centroid_history, np_polygon
                                )
                                vt.approach_axis       = approach_axis
                                vt.polygon_midline     = get_polygon_midline(np_polygon, approach_axis)
                                vt.polygon_entry_frame = frame_index
                                vt.pre_entry_velocity_snapshot = list(vt.velocity_history)
                                triggered_pairs -= {p for p in triggered_pairs if p[0] == obj_id}
                                active_violation_cars.discard(obj_id)
                        else:
                            vt.polygon_entry_frame = None
                            active_violation_cars.discard(obj_id)

                vehicles_in_polygon = newly_in_polygon
                peds_in_polygon     = newly_in_ped_polygon

                # ── frames_missing counter for active violations ───────────────
                for track_id in list(active_violations.keys()):
                    if track_id in current_veh_ids:
                        active_violations[track_id]["frames_missing"] = 0
                        active_violations[track_id]["last_seen_frame"] = frame_index
                    else:
                        active_violations[track_id]["frames_missing"] += 1
                        if active_violations[track_id]["frames_missing"] >= MISSING_FRAMES_FINALIZE:
                            _finalise_active_violation(
                                track_id, active_violations[track_id], enforcement_pipeline
                            )
                            del active_violations[track_id]

                # ── Age out missing pedestrian tracks ─────────────────────────
                for gone_id in list(ped_tracks.keys()):
                    if gone_id not in current_ped_ids:
                        ped_tracks[gone_id].frames_outside_count += 1
                        if ped_tracks[gone_id].frames_outside_count > TRACK_RESET_FRAMES:
                            del ped_tracks[gone_id]

                # ── Clean up disappeared vehicle tracks ───────────────────────
                for gone_id in list(veh_tracks.keys()):
                    if gone_id not in current_veh_ids:
                        triggered_pairs -= {p for p in triggered_pairs if p[0] == gone_id}
                        active_violation_cars.discard(gone_id)
                        plate_bbox_cache.pop(gone_id, None)
                        plate_text_cache.pop(gone_id, None)
                        plate_ocr_pending.discard(gone_id)
                        wrong_dir_flagged.discard(gone_id)
                        clear_plate_votes(gone_id)
                        del veh_tracks[gone_id]
                        # active_violations entry kept until frames_missing threshold

                # ── Second pass: violation checks + drawing ───────────────────
                for box, cls, obj_id in zip(boxes, classes, ids):
                    obj_id = int(obj_id)
                    x1, y1, x2, y2 = box
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    if cls != 0:  # vehicle
                        vt = veh_tracks.get(obj_id)

                        # ── A: Crosswalk violation ────────────────────────────
                        if (
                            vt is not None
                            and vt.polygon_entry_frame is not None
                            and (frame_index - vt.polygon_entry_frame) <= ENTRY_EVAL_WINDOW_FRAMES
                        ):
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
                                        # Plate deferred: not visible while car is on crosswalk
                                        plate_number=None,
                                    )
                                    snap_path = _save_violation_snapshot(
                                        frame=frame,
                                        violation_box=box,
                                        all_boxes=boxes,
                                        all_classes=classes,
                                        polygon=polygon,
                                        event=event,
                                    )
                                    event.snapshot_path = snap_path
                                    future = enforcement_pipeline.submit_violation(
                                        frame.copy(), event
                                    )
                                    if obj_id not in active_violations:
                                        active_violations[obj_id] = {
                                            "violation_id":       event.violation_id,
                                            "violation_type":     event.violation_type,
                                            "triggered_at_frame": frame_index,
                                            "best_plate":         None,
                                            "best_confidence":    0.0,
                                            "plate_saved":        False,
                                            "frames_missing":     0,
                                            "last_seen_frame":    frame_index,
                                            "last_ocr_frame":     0,
                                            "future":             future,
                                        }

                        # ── B: Wrong-direction detection ──────────────────────
                        # Gate: ≥8 frames history, centroid inside road Y band, not flagged.
                        if ENABLE_WRONG_DIR and (
                            vt is not None
                            and len(vt.centroid_history) >= 8
                            and obj_id not in wrong_dir_flagged
                            and ROAD_Y_RANGE[0] <= cy <= ROAD_Y_RANGE[1]
                        ):
                            if _compute_vehicle_direction(vt) == "REVERSE":
                                wrong_dir_flagged.add(obj_id)

                                wd_violation = types.SimpleNamespace(
                                    violation_type="WRONG_DIRECTION",
                                    severity="HIGH",
                                )
                                wd_ped = types.SimpleNamespace(velocity_history=[])
                                wd_event = build_event(
                                    frame_index=frame_index,
                                    box=box,
                                    car_id=obj_id,
                                    violation=wd_violation,
                                    polygon=polygon,
                                    ped_track=wd_ped,
                                    veh_track=vt,
                                    plate_number=None,
                                )
                                wd_snap = _save_violation_snapshot(
                                    frame=frame,
                                    violation_box=box,
                                    all_boxes=boxes,
                                    all_classes=classes,
                                    polygon=polygon,
                                    event=wd_event,
                                )
                                wd_event.snapshot_path = wd_snap
                                future = enforcement_pipeline.submit_violation(
                                    frame.copy(), wd_event
                                )
                                if obj_id not in active_violations:
                                    active_violations[obj_id] = {
                                        "violation_id":       wd_event.violation_id,
                                        "violation_type":     "WRONG_DIRECTION",
                                        "triggered_at_frame": frame_index,
                                        "best_plate":         None,
                                        "best_confidence":    0.0,
                                        "plate_saved":        False,
                                        "frames_missing":     0,
                                        "last_seen_frame":    frame_index,
                                        "last_ocr_frame":     0,
                                        "future":             future,
                                    }

                    # ── Drawing ───────────────────────────────────────────────
                    obj_class = "person" if cls == 0 else "vehicle"

                    is_wrong_dir     = cls != 0 and obj_id in wrong_dir_flagged
                    violation_active = cls != 0 and obj_id in active_violation_cars

                    if is_wrong_dir:
                        box_color = _ORANGE_BGR
                    elif violation_active:
                        box_color = (0, 0, 255)
                    elif cls != 0 and obj_id in vehicles_in_polygon:
                        box_color = (0, 255, 255)
                    else:
                        box_color = (0, 255, 0)

                    draw_box(frame, box, obj_class, box_color)

                    if cls == 0:  # pedestrian labels
                        pt = ped_tracks.get(obj_id)
                        if pt:
                            cv2.putText(frame, pt.state, (cx, cy + 18),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            direction = _ped_direction(pt)
                            if direction != "STATIC":
                                cv2.putText(frame, direction, (cx, cy - 40),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                    else:  # vehicle labels
                        vt = veh_tracks.get(obj_id)
                        state_label = (
                            "INSIDE" if (vt and vt.polygon_entry_frame is not None) else "OUTSIDE"
                        )
                        (tw, _), _ = cv2.getTextSize(state_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.putText(frame, state_label, (int(x2) - tw, int(y1) - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                        if violation_active:
                            cv2.putText(frame, "VIOLATION", (cx, cy - 25),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                        if is_wrong_dir:
                            # Text overlay above the box
                            cv2.putText(
                                frame, "<- WRONG DIR", (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, _ORANGE_BGR, 2,
                            )
                            # Arrow from trailing centroid → current centroid
                            if vt is not None and len(vt.centroid_history) >= 2:
                                hist = list(vt.centroid_history)
                                tail = (int(hist[-2][0]), int(hist[-2][1]))
                                head = (int(hist[-1][0]), int(hist[-1][1]))
                                cv2.arrowedLine(frame, tail, head, _ORANGE_BGR, 2, tipLength=0.4)

                        # ── Deferred plate capture status label ───────────────
                        if obj_id in active_violations:
                            av = active_violations[obj_id]
                            if av["plate_saved"] and av["best_plate"] not in (None, "UNREAD"):
                                cap_label = f"PLATE: {av['best_plate']}"
                                cap_color = (0, 240, 60)    # green — confirmed
                            elif av["plate_saved"]:
                                cap_label = "PLATE: UNREAD"
                                cap_color = (0, 200, 200)
                            else:
                                cap_label = "CAPTURING..."
                                cap_color = (0, 220, 255)   # yellow — in progress
                            cv2.putText(frame, cap_label, (int(x1), int(y2) + 18),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, cap_color, 2)

                        # ── Plate detection + OCR ─────────────────────────────
                        # Skip while car is inside the crosswalk zone — plate not
                        # visible from this camera angle until after exit.
                        currently_inside = vt is not None and vt.polygon_entry_frame is not None
                        if ENABLE_PLATE_DETECTOR and not currently_inside and plate_detector is not None:
                            x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
                            x1c = max(0, x1i)
                            y1c = max(0, y1i)
                            x2c = min(frame.shape[1] - 1, x2i)
                            y2c = min(frame.shape[0] - 1, y2i)
                            if (x2c - x1c) >= 32 and (y2c - y1c) >= 32:
                                try:
                                    crop = frame[y1c:y2c, x1c:x2c]
                                    pbbox, _pconf = plate_detector._best_box(crop)
                                    if pbbox is not None:
                                        px1, py1, px2, py2 = pbbox
                                        plate_bbox_cache[obj_id] = (
                                            x1c + px1, y1c + py1,
                                            x1c + px2, y1c + py2,
                                        )
                                        bw_ok = (x2c - x1c) >= 80
                                        bh_ok = (y2c - y1c) >= 40

                                        # Deferred OCR for violated vehicles
                                        if ENABLE_DEFERRED_OCR and (
                                            obj_id in active_violations
                                            and not active_violations[obj_id]["plate_saved"]
                                            and obj_id not in plate_ocr_pending
                                            and bw_ok and bh_ok
                                            and ocr_engine is not None
                                            and (frame_index - active_violations[obj_id]["last_ocr_frame"])
                                                >= OCR_RETRY_INTERVAL
                                        ):
                                            plate_crop = _get_plate_region(
                                                frame, box, plate_detector
                                            )
                                            if plate_crop is not None:
                                                active_violations[obj_id]["last_ocr_frame"] = frame_index
                                                plate_ocr_pending.add(obj_id)
                                                threading.Thread(
                                                    target=_run_deferred_ocr,
                                                    args=(
                                                        ocr_engine, obj_id, plate_crop,
                                                        active_violations, plate_ocr_pending,
                                                        enforcement_pipeline,
                                                    ),
                                                    daemon=True,
                                                ).start()

                                        # Regular one-shot OCR for non-violated vehicles
                                        elif ENABLE_REGULAR_OCR and (
                                            obj_id not in active_violations
                                            and ocr_engine is not None
                                            and obj_id not in plate_ocr_pending
                                            and obj_id not in plate_text_cache
                                            and bw_ok and bh_ok
                                        ):
                                            ppx1o = max(0, int(x1c + px1))
                                            ppy1o = max(0, int(y1c + py1))
                                            ppx2o = min(frame.shape[1] - 1, int(x1c + px2))
                                            ppy2o = min(frame.shape[0] - 1, int(y1c + py2))
                                            if ppx2o - ppx1o > 8 and ppy2o - ppy1o > 8:
                                                plate_crop_reg = frame[ppy1o:ppy2o, ppx1o:ppx2o].copy()
                                                plate_ocr_pending.add(obj_id)
                                                threading.Thread(
                                                    target=_run_ocr_for_vehicle,
                                                    args=(
                                                        ocr_engine, obj_id, plate_crop_reg,
                                                        plate_text_cache, plate_ocr_pending,
                                                    ),
                                                    daemon=True,
                                                ).start()
                                    else:
                                        plate_bbox_cache.pop(obj_id, None)
                                except Exception:
                                    pass

                        plate_box  = plate_bbox_cache.get(obj_id)
                        plate_text = plate_text_cache.get(obj_id)
                        if plate_box is not None:
                            ppx1, ppy1, ppx2, ppy2 = [int(v) for v in plate_box]
                            cv2.rectangle(frame, (ppx1, ppy1), (ppx2, ppy2), (0, 255, 255), 2)
                            if plate_text:
                                plate_label = plate_text
                                plate_color = (0, 240, 60) if not plate_text.startswith("~") else (0, 200, 255)
                            elif obj_id in plate_ocr_pending:
                                plate_label = "OCR..."
                                plate_color = (100, 200, 255)
                            else:
                                plate_label = "PLATE"
                                plate_color = (0, 255, 255)
                            cv2.putText(frame, plate_label, (ppx1, ppy1 - 4),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, plate_color, 1)

            elif _cached_nms_data is not None:
                # Inference still running — redraw last known boxes + labels
                c_boxes, c_classes, c_ids = _cached_nms_data
                for box, cls, obj_id in zip(c_boxes, c_classes, c_ids):
                    obj_id  = int(obj_id)
                    x1, y1, x2, y2 = box
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    obj_class = "person" if cls == 0 else "vehicle"
                    is_wrong_dir     = cls != 0 and obj_id in wrong_dir_flagged
                    violation_active = cls != 0 and obj_id in active_violation_cars
                    if is_wrong_dir:
                        box_color = _ORANGE_BGR
                    elif violation_active:
                        box_color = (0, 0, 255)
                    elif cls != 0 and obj_id in vehicles_in_polygon:
                        box_color = (0, 255, 255)
                    else:
                        box_color = (0, 255, 0)
                    draw_box(frame, box, obj_class, box_color)

                    if cls == 0:
                        pt = ped_tracks.get(obj_id)
                        if pt:
                            cv2.putText(frame, pt.state, (cx, cy + 18),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    else:
                        vt = veh_tracks.get(obj_id)
                        state_label = "INSIDE" if (vt and vt.polygon_entry_frame is not None) else "OUTSIDE"
                        (tw, _), _ = cv2.getTextSize(state_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.putText(frame, state_label, (int(x2) - tw, int(y1) - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        if violation_active:
                            cv2.putText(frame, "VIOLATION", (cx, cy - 25),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                        if is_wrong_dir:
                            cv2.putText(frame, "<- WRONG DIR", (int(x1), int(y1) - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, _ORANGE_BGR, 2)
                        if obj_id in active_violations:
                            av = active_violations[obj_id]
                            if av["plate_saved"] and av["best_plate"] not in (None, "UNREAD"):
                                cap_label = f"PLATE: {av['best_plate']}"
                                cap_color = (0, 240, 60)
                            elif av["plate_saved"]:
                                cap_label = "PLATE: UNREAD"
                                cap_color = (0, 200, 200)
                            else:
                                cap_label = "CAPTURING..."
                                cap_color = (0, 220, 255)
                            cv2.putText(frame, cap_label, (int(x1), int(y2) + 18),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, cap_color, 2)

            cv2.imshow(WINDOW_NAME, frame)

            # Pace display to source FPS so the video runs at real-time speed.
            now = time.perf_counter()
            budget_left_ms = int((_frame_delay - (now - _last_display_t)) * 1000)
            key = cv2.waitKey(max(1, budget_left_ms)) & 0xFF
            _last_display_t = time.perf_counter()

            if key == 27:
                print("[INFO] Stopped by user (ESC).")
                break

            elif key in (ord('r'), ord('R')):
                # ── Re-draw polygon ──────────────────────────────────────────────
                print("[INFO] Entering polygon re-draw mode — left-click to add points, right-click to finish.")
                editor.points.clear()
                editor.done = False
                while not editor.done:
                    ret2, cal_frame = cap.read()
                    if not ret2:
                        editor.done = True
                        break
                    editor.draw(cal_frame)
                    cv2.putText(cal_frame, "LEFT: add point  RIGHT: finish  ESC: cancel",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.imshow(WINDOW_NAME, cal_frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        editor.done = False
                        break
                if editor.done and editor.get_polygon() is not None:
                    editor.save()
                    polygon      = editor.get_polygon()
                    np_polygon   = np.array(polygon, dtype=np.float32)
                    crosswalk    = CrosswalkZone(polygon)
                    pw = float(np_polygon[:, 0].max() - np_polygon[:, 0].min())
                    ph = float(np_polygon[:, 1].max() - np_polygon[:, 1].min())
                    default_approach_axis   = "from_bottom" if ph >= pw else "from_left"
                    default_polygon_midline = get_polygon_midline(np_polygon, default_approach_axis)
                    print("[INFO] Polygon updated.")
                else:
                    print("[INFO] Polygon re-draw cancelled.")

    finally:
        # Finalise any violations still awaiting plate capture before shutdown
        for track_id, av in list(active_violations.items()):
            _finalise_active_violation(track_id, av, enforcement_pipeline)
        active_violations.clear()

        worker.stop()
        enforcement_pipeline.shutdown()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
