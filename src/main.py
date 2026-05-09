from __future__ import annotations
import argparse
import os
import queue as _queue
import re
import threading
import time
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
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SNAPSHOTS_DIR = _PROJECT_ROOT / "static" / "snapshots"
from detector.tracker import (
    IDMerger,
    PedestrianTrack,
    VehicleTrack,
    apply_cross_class_nms,
)
from detector.segmentation import (
    SegmentedYOLODetector,
    VEHICLE_CLASS_NAMES,
    class_label,
    class_color,
)
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
from vision.draw import _put_label, draw_segmentation_mask, draw_all_masks
from vision.stabilizer import VideoStabilizer
WINDOW_NAME = "Crosswalk Violation System"
_AMBER_BGR  = (11, 158, 245)
_RED_BGR    = (0, 0, 239)
_BLUE_BGR   = (255, 0, 0)
_ORANGE_BGR = (0, 165, 255)
DIRECTION_THRESHOLD: int = 15
ROAD_Y_RANGE: tuple = (350, 1080)
MISSING_FRAMES_FINALIZE: int = 15
OCR_RETRY_INTERVAL: int = 5
OCR_HIGH_CONF_THRESHOLD: float = 0.75
OCR_MIN_ACCEPT_CONF: float = 0.35
ENABLE_PLATE_DETECTOR: bool = False
ENABLE_DEFERRED_OCR: bool   = False
ENABLE_WRONG_DIR: bool      = True

def _apply_runtime_settings() -> None:
    global DIRECTION_THRESHOLD, ROAD_Y_RANGE, MISSING_FRAMES_FINALIZE
    global OCR_RETRY_INTERVAL, OCR_HIGH_CONF_THRESHOLD, OCR_MIN_ACCEPT_CONF
    global ENABLE_PLATE_DETECTOR, ENABLE_DEFERRED_OCR, ENABLE_WRONG_DIR
    rt = settings.runtime
    DIRECTION_THRESHOLD      = rt.direction_threshold
    ROAD_Y_RANGE             = (rt.road_y_min, rt.road_y_max)
    MISSING_FRAMES_FINALIZE  = rt.missing_frames_finalize
    OCR_RETRY_INTERVAL       = rt.ocr_retry_interval
    OCR_HIGH_CONF_THRESHOLD  = rt.ocr_high_conf_threshold
    OCR_MIN_ACCEPT_CONF      = rt.ocr_min_accept_conf
    ENABLE_PLATE_DETECTOR    = rt.enable_plate_detector
    ENABLE_DEFERRED_OCR      = rt.enable_deferred_ocr
    ENABLE_WRONG_DIR         = rt.enable_wrong_dir

_apply_runtime_settings()
_violations_lock = threading.Lock()
_plate_votes: defaultdict = defaultdict(list)
_plate_votes_lock = threading.Lock()
def submit_plate_reading(track_id: int, text: str, conf: float) -> None:
    with _plate_votes_lock:
        _plate_votes[track_id].append((text, conf))
def get_best_plate(track_id: int):
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
    with _plate_votes_lock:
        _plate_votes.pop(track_id, None)


def _remap_masks(
    filtered_boxes: np.ndarray,
    raw_boxes: np.ndarray,
    raw_masks: list,
) -> list:
    n = len(filtered_boxes)
    if n == 0 or not raw_masks:
        return [None] * n
    out: list = []
    for fb in filtered_boxes:
        found = None
        for j, rb in enumerate(raw_boxes):
            if j < len(raw_masks) and np.allclose(fb, rb, atol=1.0):
                found = raw_masks[j]
                break
        out.append(found)
    return out


class _InferenceWorker:
    def __init__(self, detector: SegmentedYOLODetector, stabilizer=None) -> None:
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
            raw_boxes: np.ndarray = np.empty((0, 4), dtype=np.float32)
            raw_seg_masks: list = []
            if result and result[0].boxes.id is not None:
                raw_boxes = result[0].boxes.xyxy.cpu().numpy()
                if len(raw_boxes) > 0:
                    raw_seg_masks = self._detector.segment_frame(stable_frame, raw_boxes)
            try:
                self._out.get_nowait()
            except _queue.Empty:
                pass
            self._out.put((stable_frame, result, raw_boxes, raw_seg_masks))
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
def clean_plate_text(raw: str) -> str:
    text = re.sub(r"[^A-Z0-9]", "", raw.upper().strip())
    return text if len(text) >= 4 else ""
def _compute_vehicle_direction(vt: VehicleTrack) -> str:
    positions = list(vt.centroid_history)
    if len(positions) < 8:
        return "STATIONARY"
    recent = positions[-8:]
    dx = recent[-1][0] - recent[0][0]
    avg_x_motion = sum(
        abs(recent[i + 1][0] - recent[i][0]) for i in range(len(recent) - 1)
    ) / (len(recent) - 1)
    if avg_x_motion < 10.0:
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
    plate_top = y2c - max(int(bh * 0.25), 20)
    region = frame[plate_top:y2c, x1c:x2c]
    return region if region.size > 0 else None
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
def _run_deferred_ocr(
    ocr_engine: OCREngine,
    track_id: int,
    plate_crop: np.ndarray,
    active_violations: Dict[int, dict],
    plate_ocr_pending: Set[int],
    pipeline: EnforcementPipeline,
) -> None:
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
def main():
    parser = argparse.ArgumentParser(description="Crosswalk Violation System")
    parser.add_argument(
        "--no-stabilize", action="store_true",
        help="Disable video stabilisation",
    )
    parser.add_argument(
        "--video", metavar="PATH",
        help="Path to video file (overrides VIDEO_PATH env / config default)",
    )
    args = parser.parse_args()
    enable_stabilization = not args.no_stabilize
    video_source = args.video if args.video else VIDEO_PATH
    print(f"[INFO] Starting pipeline with source: {video_source}")
    print(f"[INFO] Model: {MODEL_PATH} | conf={CONF_THRESHOLD} | imgsz={IMG_SIZE}")
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")
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
    detector = SegmentedYOLODetector(
        model_path=MODEL_PATH,
        seg_model_path=(
            settings.segmentation.seg_model_path
            if settings.segmentation.enabled else ""
        ),
        classes=DETECTION_CLASSES,
        conf=CONF_THRESHOLD,
        imgsz=IMG_SIZE,
        run_every_n_frames=settings.segmentation.run_every_n_frames,
    )
    stabilizer = VideoStabilizer() if enable_stabilization else None
    worker     = _InferenceWorker(detector, stabilizer)
    id_merger  = IDMerger(proximity_px=40.0, min_frames=3)
    enforcement_pipeline = EnforcementPipeline(settings)
    _plate_thread.join()
    _ocr_thread.join()
    plate_detector = _plate_box[0]
    ocr_engine     = _ocr_box[0]
    ped_tracks: Dict[int, PedestrianTrack] = {}
    veh_tracks: Dict[int, VehicleTrack]    = {}
    vehicles_in_polygon: Set[int]   = set()
    peds_in_polygon: Set[int]       = set()
    active_violation_cars: Set[int] = set()
    triggered_pairs: Set[tuple]     = set()
    _cached_nms_data: Optional[tuple] = None
    _last_stable_frame: Optional[np.ndarray] = None
    plate_bbox_cache: Dict[int, tuple] = {}
    plate_text_cache: Dict[int, str]   = {}
    plate_ocr_pending: Set[int]        = set()
    wrong_dir_flagged: Set[int] = set()
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
            _raw_boxes_for_seg: np.ndarray = np.empty((0, 4), dtype=np.float32)
            _raw_seg_masks: list = []
            if worker_result is not None:
                frame, results, _raw_boxes_for_seg, _raw_seg_masks = worker_result
                _last_stable_frame = frame
            else:
                if _last_stable_frame is not None:
                    frame = _last_stable_frame.copy()
                results = None
            cv2.polylines(frame, [np.array(polygon, dtype=np.int32)], True, (0, 255, 255), 2)
            if stabilizer is not None:
                stab_label = "Stabilised" if stabilizer.is_stable else "Unstable"
                stab_color = (0, 255, 0) if stabilizer.is_stable else (0, 0, 255)
                cv2.putText(frame, stab_label, (frame.shape[1] - 160, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, stab_color, 2)
            cv2.putText(
                frame,
                f"P:{len(ped_tracks)} V:{len(vehicles_in_polygon)}",
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
                seg_masks = _remap_masks(boxes, _raw_boxes_for_seg, _raw_seg_masks)
                if seg_masks and settings.segmentation.enabled:
                    draw_all_masks(
                        frame, boxes, classes, seg_masks,
                        alpha=settings.segmentation.mask_alpha,
                    )
                _cached_nms_data = (boxes, classes, ids, seg_masks)
                current_ped_ids: Set[int] = set()
                current_veh_ids: Set[int] = set()
                newly_in_polygon: Set[int] = set()
                newly_in_ped_polygon: Set[int] = set()
                for det_i, (box, cls, obj_id) in enumerate(zip(boxes, classes, ids)):
                    obj_id  = int(obj_id)
                    det_mask = seg_masks[det_i] if det_i < len(seg_masks) else None
                    x1, y1, x2, y2 = box
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    if cls == 0:
                        current_ped_ids.add(obj_id)
                        if obj_id not in ped_tracks:
                            ped_tracks[obj_id] = PedestrianTrack(track_id=obj_id)
                        pt = ped_tracks[obj_id]
                        pt.prev_centroid = pt.centroid
                        pt.centroid = (cx, cy)
                        pt.bbox = (x1, y1, x2, y2)
                        pt.mask = det_mask
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
                        if crosswalk.intersects_mask_foot(det_mask, frame.shape, box_fallback=box):
                            newly_in_ped_polygon.add(obj_id)
                    else:
                        current_veh_ids.add(obj_id)
                        if obj_id not in veh_tracks:
                            veh_tracks[obj_id] = VehicleTrack(track_id=obj_id)
                        vt = veh_tracks[obj_id]
                        vt.prev_centroid = vt.centroid
                        vt.centroid = (cx, cy)
                        vt.vehicle_class = int(cls)
                        vt.mask = det_mask
                        if vt.prev_centroid is not None:
                            vt.velocity_history.append((
                                cx - vt.prev_centroid[0],
                                cy - vt.prev_centroid[1],
                            ))
                        vt.centroid_history.append((cx, cy))
                        inside = crosswalk.intersects_mask(
                            det_mask, frame.shape, box_fallback=box,
                        )
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
                peds_in_polygon = {
                    pid for pid, pt in ped_tracks.items()
                    if pt.state in ("ENTERING", "CROSSING", "CLEARING")
                }
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
                for gone_id in list(ped_tracks.keys()):
                    if gone_id not in current_ped_ids:
                        ped_tracks[gone_id].frames_outside_count += 1
                        if ped_tracks[gone_id].frames_outside_count > TRACK_RESET_FRAMES:
                            del ped_tracks[gone_id]
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
                for det_i, (box, cls, obj_id) in enumerate(zip(boxes, classes, ids)):
                    obj_id = int(obj_id)
                    x1, y1, x2, y2 = box
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    if cls != 0:
                        vt = veh_tracks.get(obj_id)
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
                                    vt.last_violation_frame = frame_index
                                    event = build_event(
                                        frame_index=frame_index,
                                        box=box,
                                        car_id=obj_id,
                                        violation=v,
                                        polygon=polygon,
                                        ped_track=pt,
                                        veh_track=vt,
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
                        if ENABLE_WRONG_DIR and (
                            vt is not None
                            and len(vt.centroid_history) >= 8
                            and obj_id not in wrong_dir_flagged
                            and ROAD_Y_RANGE[0] <= cy <= ROAD_Y_RANGE[1]
                        ):
                            if _compute_vehicle_direction(vt) == "REVERSE":
                                wrong_dir_flagged.add(obj_id)
                    is_wrong_dir     = cls != 0 and obj_id in wrong_dir_flagged
                    violation_active = cls != 0 and obj_id in active_violation_cars
                    det_mask_i = seg_masks[det_i] if det_i < len(seg_masks) else None
                    if violation_active and det_mask_i is not None:
                        draw_segmentation_mask(frame, det_mask_i, cls=-1, alpha=0.5,
                                               color=(0, 0, 220))
                    elif is_wrong_dir and det_mask_i is not None:
                        draw_segmentation_mask(frame, det_mask_i, cls=-1, alpha=0.45,
                                               color=_AMBER_BGR)
                    lx = max(0, int(x1) + 4)
                    ly = max(14, int(y1) - 4)
                    if cls == 0:
                        pt = ped_tracks.get(obj_id)
                        state_str = pt.state if pt else "?"
                        _SC = {"OUTSIDE": (50,50,50), "ENTERING": (120,60,0), "CROSSING": (0,90,0), "EXITED": (80,40,0)}
                        sc = _SC.get(state_str, (40,40,40))
                        _put_label(frame, f"PED #{obj_id}", lx, ly,       (20,20,20),   (255,255,255))
                        _put_label(frame, state_str,        lx, ly + 18,  sc,           (255,255,255))
                        if pt:
                            direction = _ped_direction(pt)
                            if direction != "STATIC":
                                _put_label(frame, direction, lx, ly + 36, (0,80,120), (0,220,255))
                    else:
                        vt = veh_tracks.get(obj_id)
                        cls_name = class_label(int(cls)).upper()
                        _put_label(frame, f"{cls_name} #{obj_id}", lx, ly, (20,20,20), (255,255,255))
                        in_zone = vt is not None and vt.polygon_entry_frame is not None
                        if in_zone:
                            _put_label(frame, "IN ZONE", lx, ly + 18, (0,80,40), (0,255,180))
                        if violation_active:
                            _put_label(frame, "! VIOLATION", lx, cy,     (0, 0, 150), (255, 255, 255))
                        if is_wrong_dir:
                            _put_label(frame, "WRONG DIR",  lx, cy + 20, (80, 60, 0), (255, 180, 0))
                        if obj_id in active_violations:
                            av = active_violations[obj_id]
                            if av["plate_saved"] and av["best_plate"] not in (None, "UNREAD"):
                                _put_label(frame, av["best_plate"], lx, cy + 40,
                                           (0, 160, 40), (255, 255, 255))
                            elif av["plate_saved"]:
                                _put_label(frame, "UNREAD", lx, cy + 40,
                                           (0, 100, 120), (200, 200, 200))
                            else:
                                _put_label(frame, "CAPTURING", lx, cy + 40,
                                           (0, 130, 180), (255, 255, 255))
                        currently_inside = in_zone
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
                c_boxes, c_classes, c_ids, c_masks = (
                    _cached_nms_data if len(_cached_nms_data) == 4
                    else (*_cached_nms_data, [])
                )
                if c_masks and settings.segmentation.enabled:
                    draw_all_masks(
                        frame, c_boxes, c_classes, c_masks,
                        alpha=settings.segmentation.mask_alpha,
                    )
                for c_i, (box, cls, obj_id) in enumerate(zip(c_boxes, c_classes, c_ids)):
                    obj_id  = int(obj_id)
                    x1, y1, x2, y2 = box
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    is_wrong_dir     = cls != 0 and obj_id in wrong_dir_flagged
                    violation_active = cls != 0 and obj_id in active_violation_cars
                    c_mask_i = c_masks[c_i] if c_i < len(c_masks) else None
                    if violation_active and c_mask_i is not None:
                        draw_segmentation_mask(frame, c_mask_i, cls=-1, alpha=0.5,
                                               color=(0, 0, 220))
                    elif is_wrong_dir and c_mask_i is not None:
                        draw_segmentation_mask(frame, c_mask_i, cls=-1, alpha=0.45,
                                               color=_AMBER_BGR)
                    lx_c = max(0, int(x1) + 4)
                    ly_c = max(14, int(y1) - 4)
                    if cls == 0:
                        pt = ped_tracks.get(obj_id)
                        state_str = pt.state if pt else "?"
                        _SC = {"OUTSIDE": (50,50,50), "ENTERING": (120,60,0), "CROSSING": (0,90,0), "EXITED": (80,40,0)}
                        sc = _SC.get(state_str, (40,40,40))
                        _put_label(frame, f"PED #{obj_id}", lx_c, ly_c,      (20,20,20),  (255,255,255))
                        _put_label(frame, state_str,        lx_c, ly_c + 18, sc,          (255,255,255))
                        if pt:
                            direction = _ped_direction(pt)
                            if direction != "STATIC":
                                _put_label(frame, direction, lx_c, ly_c + 36, (0,80,120), (0,220,255))
                    else:
                        vt = veh_tracks.get(obj_id)
                        cls_name = class_label(int(cls)).upper()
                        _put_label(frame, f"{cls_name} #{obj_id}", lx_c, ly_c, (20,20,20), (255,255,255))
                        if vt is not None and vt.polygon_entry_frame is not None:
                            _put_label(frame, "IN ZONE", lx_c, ly_c + 18, (0,80,40), (0,255,180))
                        if violation_active:
                            _put_label(frame, "! VIOLATION", lx_c, cy,     (0, 0, 150), (255, 255, 255))
                        if is_wrong_dir:
                            _put_label(frame, "WRONG DIR",  lx_c, cy + 20, (80, 60, 0), (255, 180, 0))
                        if obj_id in active_violations:
                            av = active_violations[obj_id]
                            if av["plate_saved"] and av["best_plate"] not in (None, "UNREAD"):
                                _put_label(frame, av["best_plate"], lx_c, cy + 40,
                                           (0, 160, 40), (255, 255, 255))
                            elif av["plate_saved"]:
                                _put_label(frame, "UNREAD", lx_c, cy + 40,
                                           (0, 100, 120), (200, 200, 200))
                            else:
                                _put_label(frame, "CAPTURING", lx_c, cy + 40,
                                           (0, 130, 180), (255, 255, 255))
            cv2.imshow(WINDOW_NAME, frame)
            now = time.perf_counter()
            budget_left_ms = int((_frame_delay - (now - _last_display_t)) * 1000)
            key = cv2.waitKey(max(1, budget_left_ms)) & 0xFF
            _last_display_t = time.perf_counter()
            if key == 27:
                print("[INFO] Stopped by user (ESC).")
                break
            elif key in (ord('r'), ord('R')):
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
        for track_id, av in list(active_violations.items()):
            _finalise_active_violation(track_id, av, enforcement_pipeline)
        active_violations.clear()
        worker.stop()
        enforcement_pipeline.shutdown()
        cap.release()
        cv2.destroyAllWindows()
if __name__ == "__main__":
    main()
