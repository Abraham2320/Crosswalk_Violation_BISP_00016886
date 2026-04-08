"""
live_processor.py
─────────────────
Full YOLO + tracking + violation-detection pipeline running on the live camera
stream.  Reads raw frames from camera_manager, annotates them with the polygon
overlay, bounding boxes and state labels, then pushes the result back into
camera_manager so the MJPEG feed shows the processed output.

Usage (via Flask app):
    from live_processor import live_proc
    live_proc.start()
    stats = live_proc.get_stats()
    live_proc.stop()
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Set

import cv2
import numpy as np

# ── Add src/ to import path ───────────────────────────────────────────────────
_SRC = Path(__file__).parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# src-level imports (loaded lazily inside _load_components to keep app startup fast)
_src_loaded = False

# ── Colours (BGR) — matches main.py ──────────────────────────────────────────
_AMBER  = (11, 158, 245)    # #F59E0B  amber — polygon / crosswalk outline
_RED    = (0,   0, 255)     # red — vehicle violating
_YELLOW = (0,  255, 255)    # yellow — vehicle in zone
_GREEN  = (0,  255,   0)    # green — default box colour
_BLUE   = (255,  0,   0)    # blue — person in snapshot
_WHITE  = (255, 255, 255)   # white — text labels
_CYAN   = (0,  255, 255)    # cyan — HUD counter

_POLYGON_PATH = Path(__file__).parent / "crosswalk_polygon.json"
_SNAPSHOTS    = Path(__file__).parent / "static" / "snapshots"


def _polygon_path_for(cam_id: str) -> Path:
    """Return the per-camera polygon file path."""
    if cam_id in ("default", "cam2"):
        return _POLYGON_PATH
    return Path(__file__).parent / f"crosswalk_polygon_{cam_id}.json"


def _draw_zone_overlay(frame: np.ndarray, polygon: np.ndarray, color, alpha: float = 0.15) -> None:
    """Identical to main.py draw_zone_overlay."""
    overlay = frame.copy()
    cv2.fillPoly(overlay, [polygon], color)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.polylines(frame, [polygon], True, color, 2)


# alias used inside _process
_draw_zone = _draw_zone_overlay


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_polygon_from_disk(path: Optional[Path] = None) -> Optional[np.ndarray]:
    p = path if path is not None else _POLYGON_PATH
    if p.exists():
        try:
            pts = json.loads(p.read_text())
            if not isinstance(pts, list) or len(pts) < 4:
                return None
            return np.array(pts, dtype=np.int32)
        except Exception:
            pass
    return None


def _ped_direction(velocity_history) -> str:
    if not velocity_history:
        return "STATIC"
    dy = sum(d for _, d in velocity_history)
    if dy > 5:
        return "DOWN"
    if dy < -5:
        return "UP"
    return "STATIC"


def _speed(vt) -> Optional[float]:
    h = getattr(vt, "pre_entry_velocity_snapshot", None)
    if not h:
        return None
    mags = [(dx ** 2 + dy ** 2) ** 0.5 for dx, dy in h]
    return round(sum(mags) / len(mags), 2) if mags else None


def _intersects_box(box, polygon_4: np.ndarray, min_ratio: float = 0.02) -> bool:
    """Convex polygon ∩ bounding-box overlap check (uses first 4 polygon points)."""
    x1, y1, x2, y2 = map(int, box)
    box_pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    inter, _ = cv2.intersectConvexConvex(polygon_4[:4].astype(np.float32), box_pts)
    area = (x2 - x1) * (y2 - y1)
    return (inter / area) >= min_ratio if area > 0 else False


# ── LiveProcessor ─────────────────────────────────────────────────────────────

class LiveProcessor:
    """
    Background detection pipeline.  One thread reads camera frames, runs YOLO,
    updates tracks, checks violations, draws overlays, and writes the annotated
    JPEG back to camera_manager for MJPEG serving.
    """

    def __init__(self, cam_id: str = "default") -> None:
        self._cam_id    = cam_id
        self._poly_path = _polygon_path_for(cam_id)
        self._running  = False
        self._starting = False
        self._thread: Optional[threading.Thread] = None
        self._start_thread: Optional[threading.Thread] = None
        self._lock     = threading.Lock()
        self.last_error = ""

        # ── Live stats (written by detect thread, read by Flask) ──────────────
        self.ped_total   = 0
        self.veh_total   = 0
        self.ped_in_zone = 0
        self.veh_in_zone = 0
        self.session_violations  = 0
        self.active_violation    = False
        self._violation_until    = 0.0

        # ── Components (lazy) ─────────────────────────────────────────────────
        self._detector  = None
        self._pipeline  = None
        self._id_merger = None
        self._polygon:  Optional[np.ndarray] = None
        self._crosswalk = None
        self._upper_poly = None
        self._lower_poly = None
        self._split_ratio = 0.32
        self._stabilizer = None
        self._stabilizer_enabled = os.getenv("LIVE_ENABLE_STABILIZATION", "1") != "0"
        self._stabilizer_initialized = False

        # ── Plate detection ───────────────────────────────────────────────────
        self._plate_detector = None
        self._plate_bbox_cache: Dict[int, tuple] = {}   # obj_id -> (fx1,fy1,fx2,fy2) in frame coords
        self._ocr_engine    = None
        self._plate_text_cache:  Dict[int, str] = {}   # obj_id -> detected plate text
        self._plate_ocr_pending: Set[int] = set()       # obj_ids currently being OCR'd

        # ── Tracking state ────────────────────────────────────────────────────
        self._ped_tracks:         Dict = {}
        self._veh_tracks:         Dict = {}
        self._vehicles_in_polygon: Set[int] = set()
        self._active_viol_cars:   Set[int] = set()
        self._triggered_pairs:    Set[tuple] = set()
        self._frame_index = 0
        # Frame-skip bookkeeping (fully initialised here so _loop never hits AttributeError)
        self._skip_counter = 0
        self._last_boxes   = None   # (boxes, classes, ids, confs) cache for inter-frame draw
        self._detect_every = 1

    # ── Public API ─────────────────────────────────────────────────────────────

    def _start_blocking(self) -> bool:
        """Blocking startup path (loads model/pipeline and starts worker thread)."""
        try:
            self._load_components()
        except Exception as exc:
            self.last_error = str(exc)
            print(f"[LiveProcessor] Component load failed: {exc}")
            return False

        self._reset_tracks()
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print("[LiveProcessor] Detection started.")
        return True

    def start(self) -> bool:
        with self._lock:
            if self._running:
                return True
            if self._starting:
                return True
            self._starting = True
            self.last_error = ""

        try:
            return self._start_blocking()
        finally:
            with self._lock:
                self._starting = False

    def start_async(self) -> bool:
        """Non-blocking startup for web route so UI does not wait on model loading."""
        with self._lock:
            if self._running or self._starting:
                return True
            self._starting = True
            self.last_error = ""

        # Load the polygon synchronously so get_stats() reflects it immediately
        # (the route returns before the background thread finishes loading the model)
        self.reload_polygon()

        def _worker() -> None:
            try:
                self._start_blocking()
            finally:
                with self._lock:
                    self._starting = False

        self._start_thread = threading.Thread(target=_worker, daemon=True)
        self._start_thread.start()
        return True

    def stop(self) -> None:
        self._running = False
        self._starting = False
        # Import here to avoid circular import at module load time
        if self._cam_id == "default":
            from stream import camera_manager
            camera_manager.clear_annotated()
        else:
            from stream import registry
            registry.get(self._cam_id).clear_annotated()
        print(f"[LiveProcessor:{self._cam_id}] Detection stopped.")

    def reload_polygon(self) -> None:
        """Re-read polygon from disk (called after user saves a new polygon)."""
        from geometry.crosswalk import CrosswalkZone

        poly = _load_polygon_from_disk(self._poly_path)
        with self._lock:
            self._polygon = poly
            self._crosswalk = None
            self._upper_poly = None
            self._lower_poly = None
            if poly is not None and len(poly) >= 4:
                cw = CrosswalkZone(poly.tolist())
                self._crosswalk = cw
                self._upper_poly, self._lower_poly = cw.get_split_polygons(ratio=self._split_ratio)
        if poly is not None:
            print(f"[LiveProcessor] Polygon loaded ({len(poly)} points).")

    def get_stats(self) -> dict:
        with self._lock:
            return {
                "active":             self._running,
                "starting":           self._starting,
                "ped_total":          self.ped_total,
                "veh_total":          self.veh_total,
                "ped_in_zone":        self.ped_in_zone,
                "veh_in_zone":        self.veh_in_zone,
                "session_violations": self.session_violations,
                "active_violation":   self.active_violation,
                "polygon_loaded":     self._polygon is not None,
                "last_error":         self.last_error,
            }

    # ── Internals ──────────────────────────────────────────────────────────────

    def _load_components(self) -> None:
        from config import settings as cfg
        from detector.yolo_detector import YOLODetector
        from detector.tracker import IDMerger
        from services.pipeline import EnforcementPipeline
        from vision.stabilizer import VideoStabilizer

        live_conf  = float(os.getenv("LIVE_DETECTION_CONFIDENCE", str(cfg.models.detection_confidence)))
        live_imgsz = int(os.getenv("LIVE_IMAGE_SIZE", str(cfg.models.image_size)))
        # LIVE_MODEL_PATH lets you use a lighter model for the stream
        # (e.g. yolov8n.pt) while the batch pipeline uses the heavier one.
        live_model = os.getenv("LIVE_MODEL_PATH", cfg.models.detection_model_path)

        print(f"[LiveProcessor] model={live_model}  conf={live_conf}  imgsz={live_imgsz}")

        self._detector  = YOLODetector(
            live_model,
            cfg.models.detection_classes,
            live_conf,
            live_imgsz,
        )
        self._pipeline  = EnforcementPipeline(cfg)
        self._id_merger = IDMerger(proximity_px=40.0, min_frames=3)

        # ── Plate detector (uses custom YOLO model or Haar fallback) ──────────
        try:
            from alpr.detector import LicensePlateDetector
            self._plate_detector = LicensePlateDetector(cfg)
            model_type = "custom YOLO" if self._plate_detector._detector is not None else "Haar cascade"
            print(f"[LiveProcessor:{self._cam_id}] Plate detector ready ({model_type})")
        except Exception as exc:
            print(f"[LiveProcessor:{self._cam_id}] Plate detector unavailable: {exc}")
            self._plate_detector = None

        # ── OCR engine ────────────────────────────────────────────────────────
        try:
            from OCR.engine import OCREngine
            self._ocr_engine = OCREngine(cfg)
            print(f"[LiveProcessor:{self._cam_id}] OCR engine ready ({cfg.models.ocr_backend})")
        except Exception as exc:
            print(f"[LiveProcessor:{self._cam_id}] OCR engine unavailable: {exc}")
            self._ocr_engine = None
        self._split_ratio = cfg.runtime.split_ratio
        self._stabilizer = VideoStabilizer() if self._stabilizer_enabled else None
        self._stabilizer_initialized = False
        self._detect_every = max(1, int(os.getenv("LIVE_DETECT_EVERY_N_FRAMES", "1")))
        self.reload_polygon()

    def _reset_tracks(self) -> None:
        self._ped_tracks.clear()
        self._veh_tracks.clear()
        self._vehicles_in_polygon.clear()
        self._active_viol_cars.clear()
        self._triggered_pairs.clear()
        self._plate_bbox_cache.clear()
        self._plate_text_cache.clear()
        self._plate_ocr_pending.clear()
        self._frame_index = 0
        self.ped_total = self.veh_total = 0
        self.ped_in_zone = self.veh_in_zone = 0
        self.session_violations = 0
        self.active_violation   = False
        self._stabilizer_initialized = False
        self._detect_every = 1
        self._last_boxes   = None   # last (boxes, classes, ids, confs) for inter-frame drawing
        self._skip_counter = 0

    def _loop(self) -> None:
        if self._cam_id == "default":
            from stream import camera_manager as cam_stream
        else:
            from stream import registry
            cam_stream = registry.get(self._cam_id)
        while self._running:
            raw = cam_stream.get_numpy()
            if raw is None:
                time.sleep(0.010)
                continue
            try:
                self._skip_counter += 1
                run_detect = (self._skip_counter % self._detect_every == 0)
                annotated = self._process(raw, run_detect=run_detect)
            except Exception as exc:
                import traceback
                print(f"[LiveProcessor:{self._cam_id}] Frame error: {exc}")
                traceback.print_exc()
                annotated = raw
            _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 82])
            cam_stream.set_annotated_jpeg(buf.tobytes())

    # ── Frame processing ───────────────────────────────────────────────────────

    def _process(self, frame: np.ndarray, run_detect: bool = True) -> np.ndarray:
        from detector.tracker import (
            PedestrianTrack, VehicleTrack, apply_cross_class_nms,
        )
        from logic.violation import (
            TRACK_RESET_FRAMES, check_violation, compute_approach_axis,
            get_polygon_midline, update_pedestrian_state,
        )
        from vision.draw import draw_box

        self._frame_index += 1

        # ── Stabilizer (mirrors main.py frame_index == 1 check) ───────────────
        if self._stabilizer is not None:
            if not self._stabilizer_initialized:
                self._stabilizer.init_reference(frame)
                self._stabilizer_initialized = True
            else:
                frame = self._stabilizer.stabilize(frame)

        h, w = frame.shape[:2]

        # ── Get polygon info (thread-safe) ────────────────────────────────────
        with self._lock:
            polygon   = self._polygon
            crosswalk = self._crosswalk
            upper_poly = self._upper_poly
            lower_poly = self._lower_poly

        # ── Compute polygon geometry (needed by track logic below) ────────────
        if polygon is not None:
            np_poly = polygon.astype(np.float32)
            pw      = float(np_poly[:, 0].max() - np_poly[:, 0].min())
            ph      = float(np_poly[:, 1].max() - np_poly[:, 1].min())
            d_axis  = "from_bottom" if ph >= pw else "from_left"
            d_mid   = get_polygon_midline(np_poly, d_axis)
        else:
            np_poly = None
            d_axis  = "from_bottom"
            d_mid   = h / 2.0

        # ── Run YOLO on the CLEAN frame — mirrors main.py exactly ─────────────
        # Detection MUST happen before any drawing so the model sees unmodified
        # pixels.  The previous order fed polygon lines/zone-colour overlays into
        # YOLO, which confused the detector → fewer valid track IDs returned →
        # the early-return path fired every frame → annotated_jpeg was never
        # updated → the MJPEG feed froze on dark/placeholder content.
        results = self._detector.detect(frame) if (self._detector and run_detect) else None

        # ── Draw polygon zone overlay (AFTER detection, same order as main.py) ─
        if polygon is not None and crosswalk is not None:
            crosswalk.draw(frame)
            crosswalk.draw_half_split(frame, ratio=self._split_ratio)
            if upper_poly is not None and lower_poly is not None:
                _draw_zone_overlay(frame, upper_poly, (255, 0, 0), alpha=0.15)
                _draw_zone_overlay(frame, lower_poly, (0, 255, 0), alpha=0.15)

        # ── Stabilizer label — mirrors main.py ────────────────────────────────
        if self._stabilizer is not None:
            stab_label = "Stabilised" if self._stabilizer.is_stable else "Unstable"
            stab_color = (0, 255, 0) if self._stabilizer.is_stable else (0, 0, 255)
            cv2.putText(frame, stab_label, (w - 160, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, stab_color, 2)

        # ── HUD — mirrors main.py ─────────────────────────────────────────────
        cv2.putText(
            frame,
            f"P:{self.ped_total}  V:{self.veh_total}",
            (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            _CYAN,
            2,
        )

        # ── No detections or ByteTrack not yet tracking ───────────────────────
        if not results or results[0].boxes.id is None:
            # On skipped frames draw last known boxes (keeps the feed smooth)
            if self._last_boxes is not None and not run_detect:
                self._draw_last_boxes(frame)
            return frame

        boxes   = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy().astype(int)
        ids     = results[0].boxes.id.cpu().numpy().astype(int)
        confs   = results[0].boxes.conf.cpu().numpy()

        boxes, classes, ids, confs = apply_cross_class_nms(
            boxes, classes, ids, confs, iou_threshold=0.5
        )
        ids = self._id_merger.update(ids, boxes)
        self._last_boxes = (boxes, classes, ids, confs)  # cache for frame-skip

        cur_ped: Set[int] = set()
        cur_veh: Set[int] = set()
        newly_in: Set[int] = set()

        # ── First pass: update track objects — mirrors main.py ────────────────
        for box, cls, obj_id in zip(boxes, classes, ids):
            obj_id = int(obj_id)
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            if cls == 0:  # pedestrian
                cur_ped.add(obj_id)
                if obj_id not in self._ped_tracks:
                    self._ped_tracks[obj_id] = PedestrianTrack(track_id=obj_id)
                pt = self._ped_tracks[obj_id]
                pt.prev_centroid = pt.centroid
                pt.centroid = (cx, cy)
                pt.bbox = (x1, y1, x2, y2)   # store bbox for box-intersection FSM
                if pt.prev_centroid is not None:
                    pt.velocity_history.append((
                        cx - pt.prev_centroid[0],
                        cy - pt.prev_centroid[1],
                    ))
                pt.frames_outside_count = 0
                if np_poly is not None:
                    update_pedestrian_state(pt, np_poly, d_mid, d_axis, self._frame_index)

            else:  # vehicle
                cur_veh.add(obj_id)
                if obj_id not in self._veh_tracks:
                    self._veh_tracks[obj_id] = VehicleTrack(track_id=obj_id)
                vt = self._veh_tracks[obj_id]
                vt.prev_centroid = vt.centroid
                vt.centroid = (cx, cy)
                if vt.prev_centroid is not None:
                    vt.velocity_history.append((
                        cx - vt.prev_centroid[0],
                        cy - vt.prev_centroid[1],
                    ))
                vt.centroid_history.append((cx, cy))

                inside    = (crosswalk is not None and crosswalk.intersects_box(box, min_ratio=0.02))
                was_inside = obj_id in self._vehicles_in_polygon

                if inside:
                    newly_in.add(obj_id)
                    if not was_inside:
                        ax = (
                            compute_approach_axis(vt.centroid_history, np_poly)
                            if np_poly is not None else d_axis
                        )
                        vt.approach_axis             = ax
                        vt.polygon_midline           = (
                            get_polygon_midline(np_poly, ax) if np_poly is not None else d_mid
                        )
                        vt.polygon_entry_frame       = self._frame_index
                        vt.pre_entry_velocity_snapshot = list(vt.velocity_history)
                        self._triggered_pairs -= {p for p in self._triggered_pairs if p[0] == obj_id}
                        self._active_viol_cars.discard(obj_id)
                else:
                    vt.polygon_entry_frame = None
                    self._active_viol_cars.discard(obj_id)

        self._vehicles_in_polygon = newly_in

        # ── Age out missing tracks — mirrors main.py ──────────────────────────
        for gone_id in list(self._ped_tracks):
            if gone_id not in cur_ped:
                self._ped_tracks[gone_id].frames_outside_count += 1
                if self._ped_tracks[gone_id].frames_outside_count > TRACK_RESET_FRAMES:
                    del self._ped_tracks[gone_id]

        for gone_id in list(self._veh_tracks):
            if gone_id not in cur_veh:
                self._triggered_pairs -= {p for p in self._triggered_pairs if p[0] == gone_id}
                self._active_viol_cars.discard(gone_id)
                self._plate_bbox_cache.pop(gone_id, None)
                self._plate_text_cache.pop(gone_id, None)
                self._plate_ocr_pending.discard(gone_id)
                del self._veh_tracks[gone_id]

        # ── Second pass: violation checks + drawing — mirrors main.py ─────────
        for box, cls, obj_id in zip(boxes, classes, ids):
            obj_id = int(obj_id)
            x1, y1, x2, y2 = box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            if cls != 0:  # vehicle
                vt = self._veh_tracks.get(obj_id)
                if vt is not None and vt.polygon_entry_frame == self._frame_index:
                    for ped_id, pt in self._ped_tracks.items():
                        pair = (obj_id, ped_id)
                        if pair in self._triggered_pairs:
                            continue
                        v = check_violation(
                            car_track=vt,
                            ped_track=pt,
                            polygon=np_poly,
                            frame_number=self._frame_index,
                            approach_axis=vt.approach_axis or d_axis,
                            polygon_midline=vt.polygon_midline or d_mid,
                        )
                        if v is not None:
                            self._triggered_pairs.add(pair)
                            self._active_viol_cars.add(obj_id)
                            with self._lock:
                                self.session_violations += 1
                                self._violation_until = time.monotonic() + 5.0
                            self._submit_violation(
                                frame, box, np_poly, vt, pt, v,
                                all_boxes=boxes, all_classes=classes,
                            )

            # ── Draw box — same color scheme as main.py ───────────────────────
            obj_class       = "person" if cls == 0 else "vehicle"
            violation_active = cls != 0 and obj_id in self._active_viol_cars

            if cls == 0:  # pedestrian: state-based colour
                pt_draw = self._ped_tracks.get(obj_id)
                ped_state = pt_draw.state if pt_draw else "OUTSIDE"
                box_color = (
                    _YELLOW if ped_state == "CROSSING"
                    else _AMBER  if ped_state in ("ENTERING", "CLEARING")
                    else _GREEN
                )
            else:  # vehicle
                box_color = (
                    _RED    if violation_active
                    else _YELLOW if obj_id in self._vehicles_in_polygon
                    else _GREEN
                )
            draw_box(frame, box, obj_class, box_color)

            if cls == 0:
                pt = self._ped_tracks.get(obj_id)
                if pt:
                    cv2.putText(
                        frame, pt.state, (cx, cy + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, _WHITE, 1,
                    )
                    direction = _ped_direction(pt.velocity_history)
                    if direction != "STATIC":
                        cv2.putText(
                            frame, direction, (cx, cy - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2,
                        )
            else:
                vt = self._veh_tracks.get(obj_id)
                state_label = "INSIDE" if (vt and vt.polygon_entry_frame is not None) else "OUTSIDE"
                cv2.putText(
                    frame, state_label, (cx, cy + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, _WHITE, 1,
                )
                if violation_active:
                    cv2.putText(
                        frame, "VIOLATION", (cx, cy - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, _RED, 2,
                    )

                # ── Plate detection + OCR overlay ─────────────────────────────
                if run_detect and self._plate_detector is not None:
                    x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
                    x1c = max(0, x1i); y1c = max(0, y1i)
                    x2c = min(w - 1, x2i); y2c = min(h - 1, y2i)
                    if (x2c - x1c) >= 32 and (y2c - y1c) >= 32:
                        try:
                            crop = frame[y1c:y2c, x1c:x2c]
                            pbbox, _pconf = self._plate_detector._best_box(crop)
                            if pbbox is not None:
                                px1, py1, px2, py2 = pbbox
                                frame_coords = (
                                    x1c + px1, y1c + py1, x1c + px2, y1c + py2,
                                )
                                self._plate_bbox_cache[obj_id] = frame_coords

                                # Launch OCR once per vehicle ID (async, non-blocking)
                                if (
                                    self._ocr_engine is not None
                                    and obj_id not in self._plate_ocr_pending
                                    and obj_id not in self._plate_text_cache
                                ):
                                    ppx1o = max(0, int(x1c + px1))
                                    ppy1o = max(0, int(y1c + py1))
                                    ppx2o = min(w - 1, int(x1c + px2))
                                    ppy2o = min(h - 1, int(y1c + py2))
                                    if ppx2o - ppx1o > 8 and ppy2o - ppy1o > 8:
                                        plate_crop_img = frame[ppy1o:ppy2o, ppx1o:ppx2o].copy()
                                        self._plate_ocr_pending.add(obj_id)
                                        import threading as _t
                                        _t.Thread(
                                            target=self._run_ocr_for_vehicle,
                                            args=(obj_id, plate_crop_img),
                                            daemon=True,
                                        ).start()
                            else:
                                self._plate_bbox_cache.pop(obj_id, None)
                        except Exception:
                            pass

                plate_box  = self._plate_bbox_cache.get(obj_id)
                plate_text = self._plate_text_cache.get(obj_id)
                if plate_box is not None:
                    ppx1, ppy1, ppx2, ppy2 = [int(v) for v in plate_box]
                    cv2.rectangle(frame, (ppx1, ppy1), (ppx2, ppy2), (0, 255, 255), 2)
                    if plate_text:
                        label = plate_text
                        color = (0, 240, 60) if not plate_text.startswith("~") else (0, 200, 255)
                    elif obj_id in self._plate_ocr_pending:
                        label = "OCR..."
                        color = (100, 200, 255)
                    else:
                        label = "PLATE"
                        color = (0, 255, 255)
                    cv2.putText(
                        frame, label, (ppx1, ppy1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
                    )

        # ── Update live stats ─────────────────────────────────────────────────
        ped_in_zone = sum(
            1 for pid, pt in self._ped_tracks.items()
            if pid in cur_ped and pt.state in ("ENTERING", "CROSSING", "CLEARING")
        )
        with self._lock:
            self.ped_total   = len(cur_ped)
            self.veh_total   = len(cur_veh)
            self.ped_in_zone = ped_in_zone
            self.veh_in_zone = len(newly_in)
            self.active_violation = time.monotonic() < self._violation_until

        return frame

    def _draw_last_boxes(self, frame: np.ndarray) -> None:
        """Re-draw last known bounding boxes on inter-detection frames."""
        if self._last_boxes is None:
            return
        from vision.draw import draw_box
        boxes, classes, ids, _ = self._last_boxes
        for box, cls, obj_id in zip(boxes, classes, ids):
            obj_id = int(obj_id)
            obj_class = "person" if cls == 0 else "vehicle"
            if cls == 0:
                pt_draw = self._ped_tracks.get(obj_id)
                ped_state = pt_draw.state if pt_draw else "OUTSIDE"
                color = (
                    _YELLOW if ped_state == "CROSSING"
                    else _AMBER  if ped_state in ("ENTERING", "CLEARING")
                    else _GREEN
                )
            else:
                viol = obj_id in self._active_viol_cars
                zone = obj_id in self._vehicles_in_polygon
                color = _RED if viol else _YELLOW if zone else _GREEN
            draw_box(frame, box, obj_class, color)
            # Plate overlay from cache on skipped frames
            if cls != 0:
                plate_box  = self._plate_bbox_cache.get(obj_id)
                plate_text = self._plate_text_cache.get(obj_id)
                if plate_box is not None:
                    ppx1, ppy1, ppx2, ppy2 = [int(v) for v in plate_box]
                    cv2.rectangle(frame, (ppx1, ppy1), (ppx2, ppy2), (0, 255, 255), 2)
                    label = plate_text if plate_text else ("OCR..." if obj_id in self._plate_ocr_pending else "PLATE")
                    color = (0, 240, 60) if plate_text and not plate_text.startswith("~") else (0, 200, 255)
                    cv2.putText(frame, label, (ppx1, ppy1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    def _draw_hud(self, frame: np.ndarray) -> None:
        """Kept for backward compatibility; logic now inlined in _process."""
        pass

    def _run_ocr_for_vehicle(self, obj_id: int, plate_crop: np.ndarray) -> None:
        """Run OCR on plate crop in a background thread; cache result in _plate_text_cache."""
        import tempfile
        import os as _os
        tmp_path = None
        try:
            tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            cv2.imwrite(tmp.name, plate_crop)
            tmp.close()
            tmp_path = tmp.name
            result = self._ocr_engine.recognize(tmp_path)
            if result.plate_text:
                self._plate_text_cache[obj_id] = result.plate_text
            elif result.raw_text and len(result.raw_text.strip()) >= 3:
                self._plate_text_cache[obj_id] = f"~{result.raw_text.strip()[:10]}"
        except Exception:
            pass
        finally:
            self._plate_ocr_pending.discard(obj_id)
            if tmp_path:
                try:
                    _os.unlink(tmp_path)
                except Exception:
                    pass

    def _submit_violation(self, frame, vbox, np_poly, vt, pt, violation,
                            all_boxes=None, all_classes=None) -> None:
        """Save annotated snapshot (identical to main.py _save_violation_snapshot) and submit."""
        import threading as _t
        from schemas import ViolationEvent
        from config import settings as cfg

        _SNAPSHOTS.mkdir(parents=True, exist_ok=True)
        x1, y1, x2, y2 = [int(v) for v in vbox]

        # Use cached OCR plate if available (only accept confirmed plates, not raw ~ guesses)
        cached_plate = self._plate_text_cache.get(vt.track_id)
        clean_plate = cached_plate if (cached_plate and not cached_plate.startswith("~")) else None

        event = ViolationEvent.create(
            vehicle_id=vt.track_id,
            frame_index=self._frame_index,
            vehicle_bbox=(x1, y1, x2, y2),
            vehicle_zone=None,
            polygon=[tuple(map(int, p)) for p in np_poly.tolist()],
            pedestrian_direction=_ped_direction(pt.velocity_history),
            pedestrian_zone=None,
            confidence=1.0,
            location=cfg.runtime.location_name,
            violation_type=violation.violation_type,
            severity=violation.severity,
            vehicle_speed_estimate=_speed(vt),
            plate_number=clean_plate,
        )

        # ── Annotated snapshot — exact copy of main.py _save_violation_snapshot
        try:
            snap = frame.copy()
            fh, fw = snap.shape[:2]

            # Red semi-transparent banner (top 40 px, opacity 0.6)
            overlay = snap.copy()
            cv2.rectangle(overlay, (0, 0), (fw, 40), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.6, snap, 0.4, 0, snap)

            # All bounding boxes: blue for peds, red for other vehicles
            if all_boxes is not None and all_classes is not None:
                for b, c in zip(all_boxes, all_classes):
                    bx1, by1, bx2, by2 = [int(v) for v in b]
                    color = (255, 0, 0) if int(c) == 0 else (0, 0, 255)
                    cv2.rectangle(snap, (bx1, by1), (bx2, by2), color, 2)

            # Offending vehicle bbox bright red, 3 px
            cv2.rectangle(snap, (x1, y1), (x2, y2), (0, 0, 255), 3)

            # Polygon outline in amber
            cv2.polylines(snap, [np_poly.astype(np.int32)], True, _AMBER, 2)

            # Text overlay inside banner (same as main.py)
            ts = event.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            plate_str = event.plate_number or "CHECKING..."
            lines = [
                "VIOLATION DETECTED",
                f"Type: {violation.violation_type}",
                f"Vehicle ID: {vt.track_id}",
                f"Plate: {plate_str}",
                f"Time: {ts}",
            ]
            y_off = 14
            for line in lines:
                cv2.putText(
                    snap, line, (8, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA,
                )
                y_off += 16

            fname = f"snapshot_{event.violation_id}_{self._frame_index}.jpg"
            cv2.imwrite(str(_SNAPSHOTS / fname), snap)
            event.snapshot_path = f"snapshots/{fname}"

            # Save plate crop if we have a cached detection for this vehicle
            plate_bbox_frame = self._plate_bbox_cache.get(vt.track_id)
            if plate_bbox_frame is not None:
                try:
                    fh_f, fw_f = frame.shape[:2]
                    ppx1, ppy1, ppx2, ppy2 = plate_bbox_frame
                    ppx1 = max(0, ppx1); ppy1 = max(0, ppy1)
                    ppx2 = min(fw_f - 1, ppx2); ppy2 = min(fh_f - 1, ppy2)
                    if ppx2 > ppx1 and ppy2 > ppy1:
                        plate_crop = frame[ppy1:ppy2, ppx1:ppx2]
                        plate_fname = f"plate_{event.violation_id}.jpg"
                        cv2.imwrite(str(_SNAPSHOTS / plate_fname), plate_crop)
                        event.plate_crop_path = f"snapshots/{plate_fname}"
                        print(f"[LiveProcessor] Plate crop saved: {plate_fname}")
                except Exception as exc:
                    print(f"[LiveProcessor] Plate crop save failed: {exc}")
        except Exception as exc:
            print(f"[LiveProcessor] Snapshot failed: {exc}")

        if self._pipeline:
            _t.Thread(
                target=lambda: self._pipeline.submit_violation(frame.copy(), event),
                daemon=True,
            ).start()


# ── LiveProcessorRegistry ────────────────────────────────────────────────────

class LiveProcessorRegistry:
    """Manages one LiveProcessor per camera ID (lazy init)."""

    def __init__(self) -> None:
        self._procs: dict[str, LiveProcessor] = {}
        self._lock = threading.Lock()

    def get(self, cam_id: str) -> LiveProcessor:
        with self._lock:
            if cam_id not in self._procs:
                self._procs[cam_id] = LiveProcessor(cam_id=cam_id)
            return self._procs[cam_id]

    def statuses(self) -> dict:
        with self._lock:
            return {cid: proc.get_stats() for cid, proc in self._procs.items()}


proc_registry = LiveProcessorRegistry()


# ── Module-level singleton (backward compat for existing /admin/live routes) ─
live_proc = LiveProcessor()
