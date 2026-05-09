from __future__ import annotations
import json
import os
import queue as _queue
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Set
import cv2
import numpy as np
_SRC = Path(__file__).parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
_src_loaded = False
_AMBER  = (11, 158, 245)
_RED    = (0,   0, 255)
_YELLOW = (0,  255, 255)
_GREEN  = (0,  255,   0)
_BLUE   = (255,  0,   0)
_WHITE  = (255, 255, 255)
_CYAN   = (0,  255, 255)
_ORANGE = (0,  165, 255)
_DIRECTION_THRESHOLD   = 15
_MIN_MOVING_SPEED_PX   = 2.0
_ROAD_Y_MIN = int(os.getenv("LIVE_ROAD_Y_MIN", "0"))
_ROAD_Y_MAX = int(os.getenv("LIVE_ROAD_Y_MAX", "9999"))
_FORWARD_SIGN = int(os.getenv("LIVE_FORWARD_DIR_SIGN", "1"))
_CONF_VEH  = float(os.getenv("LIVE_CONF_VEH",  "0.30"))
_CONF_PED  = float(os.getenv("LIVE_CONF_PED",  "0.25"))
_AREA_VEH  = float(os.getenv("LIVE_AREA_VEH",  "1400.0"))
_AREA_PED  = float(os.getenv("LIVE_AREA_PED",  "600.0"))
_POLYGON_PATH = Path(__file__).parent / "crosswalk_polygon.json"
_SNAPSHOTS    = Path(__file__).parent / "static" / "snapshots"

class _LiveInferenceWorker:
    def __init__(self, detector, clahe=None, stabilizer=None) -> None:
        self._detector  = detector
        self._clahe     = clahe
        self._stabilizer = stabilizer
        self._stabilizer_initialized = False
        self._in:  _queue.Queue = _queue.Queue(maxsize=1)
        self._out: _queue.Queue = _queue.Queue(maxsize=1)
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="live-yolo-worker")
        self._thread.start()

    def _run(self) -> None:
        while True:
            item = self._in.get()
            if item is None:
                break
            frame, frame_idx = item
            if self._stabilizer is not None:
                if not self._stabilizer_initialized:
                    self._stabilizer.init_reference(frame)
                    self._stabilizer_initialized = True
                    stable = frame
                else:
                    stable = self._stabilizer.stabilize(frame)
            else:
                stable = frame
            detect_input = self._clahe.apply(stable) if self._clahe is not None else stable
            result = self._detector.detect(detect_input)
            raw_boxes: np.ndarray = np.empty((0, 4), dtype=np.float32)
            raw_seg_masks: list = []
            if result and result[0].boxes.id is not None:
                raw_boxes = result[0].boxes.xyxy.cpu().numpy()
                if len(raw_boxes) > 0:
                    raw_seg_masks = self._detector.segment_frame(detect_input, raw_boxes)
            try:
                self._out.get_nowait()
            except _queue.Empty:
                pass
            self._out.put((stable, detect_input, result, frame_idx, raw_boxes, raw_seg_masks))

    def submit(self, frame: np.ndarray, frame_idx: int) -> None:
        try:
            self._in.put_nowait((frame.copy(), frame_idx))
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
def _polygon_path_for(cam_id: str) -> Path:
    if cam_id in ("default", "cam2"):
        return _POLYGON_PATH
    return Path(__file__).parent / f"crosswalk_polygon_{cam_id}.json"
def _draw_zone_overlay(frame: np.ndarray, polygon: np.ndarray, color, alpha: float = 0.15) -> None:
    overlay = frame.copy()
    cv2.fillPoly(overlay, [polygon], color)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.polylines(frame, [polygon], True, color, 2)
_draw_zone = _draw_zone_overlay
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
def _compute_vehicle_direction(vt) -> str:
    if _FORWARD_SIGN == 0:
        return "STATIONARY"
    positions = list(vt.centroid_history)
    if len(positions) < 8:
        return "STATIONARY"
    recent = positions[-8:]
    dx = recent[-1][0] - recent[0][0]
    avg_x_motion = sum(
        abs(recent[i + 1][0] - recent[i][0]) for i in range(len(recent) - 1)
    ) / (len(recent) - 1)
    if avg_x_motion < 5.0:
        return "STATIONARY"
    signed_dx = dx * _FORWARD_SIGN
    if signed_dx < -_DIRECTION_THRESHOLD:
        return "REVERSE"
    if signed_dx > _DIRECTION_THRESHOLD:
        return "FORWARD"
    return "STATIONARY"
def _vehicle_speed(vt) -> float:
    h = list(vt.velocity_history)
    if not h:
        return 0.0
    recent = h[-8:]
    return sum((dx ** 2 + dy ** 2) ** 0.5 for dx, dy in recent) / len(recent)
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


def _intersects_box(box, polygon_4: np.ndarray, min_ratio: float = 0.02) -> bool:
    x1, y1, x2, y2 = map(int, box)
    box_pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    inter, _ = cv2.intersectConvexConvex(polygon_4[:4].astype(np.float32), box_pts)
    area = (x2 - x1) * (y2 - y1)
    return (inter / area) >= min_ratio if area > 0 else False
class LiveProcessor:
    def __init__(self, cam_id: str = "default") -> None:
        self._cam_id    = cam_id
        self._poly_path = _polygon_path_for(cam_id)
        self._running  = False
        self._starting = False
        self._thread: Optional[threading.Thread] = None
        self._start_thread: Optional[threading.Thread] = None
        self._lock     = threading.Lock()
        self.last_error = ""
        self.ped_total   = 0
        self.veh_total   = 0
        self.ped_in_zone = 0
        self.veh_in_zone = 0
        self.session_violations  = 0
        self.active_violation    = False
        self._violation_until    = 0.0
        self._detector  = None
        self._worker: Optional[_LiveInferenceWorker] = None
        self._pipeline  = None
        self._id_merger = None
        self._polygon:  Optional[np.ndarray] = None
        self._crosswalk = None
        self._upper_poly = None
        self._lower_poly = None
        self._split_ratio = 0.32
        self._show_split_overlay = True
        self._stabilizer = None
        self._stabilizer_enabled = os.getenv("LIVE_ENABLE_STABILIZATION", "0") != "0"
        self._stabilizer_initialized = False
        self._enable_plate_detector = os.getenv("LIVE_ENABLE_PLATE_DETECTOR", "1") != "0"
        self._enable_ocr            = os.getenv("LIVE_ENABLE_OCR", "1") != "0"
        self._enable_wrong_dir      = os.getenv("LIVE_ENABLE_WRONG_DIR", "1") != "0"
        self._enable_segmentation   = os.getenv("SEGMENTATION_ENABLED", "1") == "1"
        self._seg_masks: list = []
        self._clahe        = None
        self._clean_frame: Optional[np.ndarray] = None
        self._plate_detector = None
        self._plate_bbox_cache: Dict[int, tuple] = {}
        self._ocr_engine    = None
        self._plate_text_cache:  Dict[int, str] = {}
        self._plate_ocr_pending: Set[int] = set()
        self._wrong_dir_flagged: Set[int] = set()
        self._post_viol_collectors: Dict[int, dict] = {}
        self._ped_tracks:         Dict = {}
        self._veh_tracks:         Dict = {}
        self._vehicles_in_polygon: Set[int] = set()
        self._active_viol_cars:   Set[int] = set()
        self._triggered_pairs:    Set[tuple] = set()
        self._frame_index = 0
        self._skip_counter = 0
        self._last_boxes   = None
        self._detect_every = 1
        self._live_model_path = ""
        self._plate_model_mode = "unavailable"
        self._ocr_backend = "none"
        self._ocr_gpu_enabled = False
        self.frames_total = 0
        self.frames_detected = 0
        self.frames_skipped = 0
        self.last_person_count = 0
        self.last_vehicle_count = 0
        self.last_frame_ms = 0.0
        self.avg_frame_ms = 0.0
        self.last_detect_ms = 0.0
        self.avg_detect_ms = 0.0
        self.last_ocr_ms = 0.0
        self.avg_ocr_ms = 0.0
        self.ocr_attempts = 0
        self.ocr_accepted = 0
        self.ocr_rejected = 0
        self.violations_submitted = 0
    def _start_blocking(self) -> bool:
        try:
            self._load_components()
        except Exception as exc:
            self.last_error = str(exc)
            print(f"[LiveProcessor] Component load failed: {exc}")
            return False
        self._worker = _LiveInferenceWorker(
            detector=self._detector,
            clahe=self._clahe,
            stabilizer=self._stabilizer,
        )
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
        with self._lock:
            if self._running or self._starting:
                return True
            self._starting = True
            self.last_error = ""
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
        if self._worker is not None:
            self._worker.stop()
            self._worker = None
        if self._cam_id == "default":
            from stream import camera_manager
            camera_manager.clear_annotated()
        else:
            from stream import registry
            registry.get(self._cam_id).clear_annotated()
        print(f"[LiveProcessor:{self._cam_id}] Detection stopped.")
    def reload_polygon(self) -> None:
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
                "cam_id":             self._cam_id,
                "active":             self._running,
                "starting":           self._starting,
                "ped_total":          self.ped_total,
                "veh_total":          self.veh_total,
                "ped_in_zone":        self.ped_in_zone,
                "veh_in_zone":        self.veh_in_zone,
                "session_violations": self.session_violations,
                "active_violation":   self.active_violation,
                "polygon_loaded":     self._polygon is not None,
                "polygon_points":     int(len(self._polygon)) if self._polygon is not None else 0,
                "detect_every_n":     self._detect_every,
                "model_detection":    self._live_model_path,
                "model_plate":        self._plate_model_mode,
                "ocr_backend":        self._ocr_backend,
                "ocr_gpu":            self._ocr_gpu_enabled,
                "frames_total":       self.frames_total,
                "frames_detected":    self.frames_detected,
                "frames_skipped":     self.frames_skipped,
                "last_person_count":  self.last_person_count,
                "last_vehicle_count": self.last_vehicle_count,
                "last_frame_ms":      round(self.last_frame_ms, 2),
                "avg_frame_ms":       round(self.avg_frame_ms, 2),
                "last_detect_ms":     round(self.last_detect_ms, 2),
                "avg_detect_ms":      round(self.avg_detect_ms, 2),
                "last_ocr_ms":        round(self.last_ocr_ms, 2),
                "avg_ocr_ms":         round(self.avg_ocr_ms, 2),
                "ocr_attempts":       self.ocr_attempts,
                "ocr_accepted":       self.ocr_accepted,
                "ocr_rejected":       self.ocr_rejected,
                "ocr_pending":        len(self._plate_ocr_pending),
                "plate_cached":       len(self._plate_text_cache),
                "violations_submitted":   self.violations_submitted,
                "last_error":            self.last_error,
                "enable_plate_detector": self._enable_plate_detector,
                "enable_ocr":            self._enable_ocr,
                "enable_wrong_dir":      self._enable_wrong_dir,
                "enable_segmentation":   self._enable_segmentation,
            }
    def reset_session_outputs(self) -> None:
        with self._lock:
            self.session_violations = 0
            self.active_violation = False
            self._violation_until = 0.0
            self._active_viol_cars.clear()
            self._triggered_pairs.clear()
            self._plate_bbox_cache.clear()
            self._plate_text_cache.clear()
            self._plate_ocr_pending.clear()
            self.frames_total = 0
            self.frames_detected = 0
            self.frames_skipped = 0
            self.last_person_count = 0
            self.last_vehicle_count = 0
            self.last_frame_ms = 0.0
            self.avg_frame_ms = 0.0
            self.last_detect_ms = 0.0
            self.avg_detect_ms = 0.0
            self.last_ocr_ms = 0.0
            self.avg_ocr_ms = 0.0
            self.ocr_attempts = 0
            self.ocr_accepted = 0
            self.ocr_rejected = 0
            self.violations_submitted = 0
    def _ema(self, prev: float, current: float, alpha: float = 0.15) -> float:
        if prev <= 0.0:
            return current
        return (1.0 - alpha) * prev + alpha * current
    def _load_components(self) -> None:
        if self._detector is not None and self._pipeline is not None and self._id_merger is not None:
            self.reload_polygon()
            return
        from config import settings as cfg
        from detector.segmentation import SegmentedYOLODetector
        from detector.tracker import IDMerger
        from services.pipeline import EnforcementPipeline
        from vision.stabilizer import VideoStabilizer
        live_conf  = float(os.getenv("LIVE_DETECTION_CONFIDENCE", str(cfg.models.detection_confidence)))
        live_imgsz = int(os.getenv("LIVE_IMAGE_SIZE", str(cfg.models.image_size)))
        live_model = os.getenv("LIVE_MODEL_PATH", cfg.models.detection_model_path)
        self._live_model_path = live_model
        seg_model  = cfg.segmentation.seg_model_path if self._enable_segmentation else ""
        seg_every  = cfg.segmentation.run_every_n_frames
        print(f"[LiveProcessor] model={live_model}  conf={live_conf}  imgsz={live_imgsz}"
              f"  seg={'on' if self._enable_segmentation else 'off'}")
        self._detector = SegmentedYOLODetector(
            model_path=live_model,
            seg_model_path=seg_model,
            classes=cfg.models.detection_classes,
            conf=live_conf,
            imgsz=live_imgsz,
            run_every_n_frames=seg_every,
        )
        self._pipeline  = EnforcementPipeline(cfg)
        self._id_merger = IDMerger(proximity_px=40.0, min_frames=3)
        if self._enable_plate_detector:
            try:
                from alpr.detector import LicensePlateDetector
                self._plate_detector = LicensePlateDetector(cfg)
                model_type = "custom YOLO" if self._plate_detector._detector is not None else "Haar cascade"
                self._plate_model_mode = model_type
                print(f"[LiveProcessor:{self._cam_id}] Plate detector ready ({model_type})")
            except Exception as exc:
                print(f"[LiveProcessor:{self._cam_id}] Plate detector unavailable: {exc}")
                self._plate_detector = None
                self._plate_model_mode = "unavailable"
        else:
            self._plate_detector  = None
            self._plate_model_mode = "disabled"
            print(f"[LiveProcessor:{self._cam_id}] Plate detector disabled (LIVE_ENABLE_PLATE_DETECTOR=0)")
        if self._enable_ocr:
            try:
                from OCR.engine import OCREngine
                self._ocr_engine = OCREngine(cfg)
                self._ocr_backend = cfg.models.ocr_backend
                self._ocr_gpu_enabled = os.getenv("OCR_USE_GPU", "1") != "0"
                print(f"[LiveProcessor:{self._cam_id}] OCR engine ready ({cfg.models.ocr_backend})")
            except Exception as exc:
                print(f"[LiveProcessor:{self._cam_id}] OCR engine unavailable: {exc}")
                self._ocr_engine = None
                self._ocr_backend = "none"
                self._ocr_gpu_enabled = False
        else:
            self._ocr_engine = None
            self._ocr_backend = "disabled"
            self._ocr_gpu_enabled = False
            print(f"[LiveProcessor:{self._cam_id}] OCR disabled (LIVE_ENABLE_OCR=0)")
        self._split_ratio = cfg.runtime.split_ratio
        self._show_split_overlay = cfg.runtime.show_split_overlay
        self._stabilizer = VideoStabilizer() if self._stabilizer_enabled else None
        self._stabilizer_initialized = False
        try:
            from vision.preprocessing import CLAHEPreprocessor
            from config import Config as _Cfg
            if _Cfg.CLAHE_ENABLED:
                self._clahe = CLAHEPreprocessor(
                    clip_limit=_Cfg.CLAHE_CLIP_LIMIT,
                    tile_size=_Cfg.CLAHE_TILE_SIZE,
                )
                print(f"[LiveProcessor:{self._cam_id}] CLAHE enabled "
                      f"(clip={_Cfg.CLAHE_CLIP_LIMIT}, tile={_Cfg.CLAHE_TILE_SIZE})")
            else:
                self._clahe = None
        except Exception as _clahe_exc:
            print(f"[LiveProcessor:{self._cam_id}] CLAHE unavailable: {_clahe_exc}")
            self._clahe = None
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
        self._wrong_dir_flagged.clear()
        self._post_viol_collectors.clear()
        self._frame_index = 0
        self.ped_total = self.veh_total = 0
        self.ped_in_zone = self.veh_in_zone = 0
        self.session_violations = 0
        self.active_violation   = False
        self._stabilizer_initialized = False
        self._detect_every = 1
        self._last_boxes   = None
        self._seg_masks    = []
        self._skip_counter = 0
        self.frames_total = 0
        self.frames_detected = 0
        self.frames_skipped = 0
        self.last_person_count = 0
        self.last_vehicle_count = 0
        self.last_frame_ms = 0.0
        self.avg_frame_ms = 0.0
        self.last_detect_ms = 0.0
        self.avg_detect_ms = 0.0
        self.last_ocr_ms = 0.0
        self.avg_ocr_ms = 0.0
        self.ocr_attempts = 0
        self.ocr_accepted = 0
        self.ocr_rejected = 0
        self.violations_submitted = 0
    def _loop(self) -> None:
        if self._cam_id == "default":
            from stream import camera_manager as cam_stream
        else:
            from stream import registry
            cam_stream = registry.get(self._cam_id)
        _last_seq = 0
        _last_stable: Optional[np.ndarray] = None
        _last_detect_input: Optional[np.ndarray] = None
        _last_results = None
        _last_raw_boxes: np.ndarray = np.empty((0, 4), dtype=np.float32)
        _last_raw_seg_masks: list = []
        while self._running:
            raw, new_seq = cam_stream.get_numpy_if_new(_last_seq)
            if raw is not None:
                _last_seq = new_seq
                self._frame_index += 1
                self.frames_total += 1
                if self._worker is not None:
                    self._skip_counter += 1
                    if self._skip_counter % self._detect_every == 0:
                        self._worker.submit(raw, self._frame_index)
            worker_result = self._worker.get_result() if self._worker is not None else None
            got_new_detection = False
            if worker_result is not None:
                stable, detect_input, results, _fi, raw_boxes, raw_seg_masks = worker_result
                _last_stable        = stable
                _last_detect_input  = detect_input
                _last_results       = results
                _last_raw_boxes     = raw_boxes
                _last_raw_seg_masks = raw_seg_masks
                got_new_detection   = True
                self.frames_detected += 1
            else:
                self.frames_skipped += 1
            render_frame = (_last_stable if _last_stable is not None
                            else raw if raw is not None else None)
            if render_frame is None:
                time.sleep(0.005)
                continue
            try:
                annotated = self._process(
                    render_frame,
                    detect_input=_last_detect_input,
                    results=_last_results,
                    got_new_detection=got_new_detection,
                    raw_boxes_for_seg=_last_raw_boxes,
                    raw_seg_masks=_last_raw_seg_masks,
                )
            except Exception as exc:
                import traceback
                print(f"[LiveProcessor:{self._cam_id}] Frame error: {exc}")
                traceback.print_exc()
                annotated = render_frame
            _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 82])
            cam_stream.set_annotated_jpeg(buf.tobytes())
            if raw is None and worker_result is None:
                time.sleep(0.005)
    def _process(
        self,
        frame: np.ndarray,
        detect_input: Optional[np.ndarray] = None,
        results=None,
        got_new_detection: bool = False,
        raw_boxes_for_seg: Optional[np.ndarray] = None,
        raw_seg_masks: Optional[list] = None,
    ) -> np.ndarray:
        from detector.tracker import (
            PedestrianTrack, VehicleTrack, apply_cross_class_nms,
        )
        from logic.violation import (
            ENTRY_EVAL_WINDOW_FRAMES, TRACK_RESET_FRAMES, check_violation, compute_approach_axis,
            get_polygon_midline, update_pedestrian_state,
        )
        from vision.draw import draw_all_masks
        from detector.segmentation import class_label, class_color
        t_frame_start = time.perf_counter()
        h, w = frame.shape[:2]
        with self._lock:
            polygon   = self._polygon
            crosswalk = self._crosswalk
            upper_poly = self._upper_poly
            lower_poly = self._lower_poly
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
        _detect_frame = detect_input if detect_input is not None else frame
        self._clean_frame = frame.copy()
        if polygon is not None and crosswalk is not None:
            crosswalk.draw(frame)
            if self._show_split_overlay:
                crosswalk.draw_half_split(frame, ratio=self._split_ratio)
                if upper_poly is not None and lower_poly is not None:
                    _draw_zone_overlay(frame, upper_poly, (255, 0, 0), alpha=0.15)
                    _draw_zone_overlay(frame, lower_poly, (0, 255, 0), alpha=0.15)
        if self._stabilizer is not None:
            stab_label = "Stabilised" if self._stabilizer.is_stable else "Unstable"
            stab_color = (0, 255, 0) if self._stabilizer.is_stable else (0, 0, 255)
            cv2.putText(frame, stab_label, (w - 160, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, stab_color, 2)
        cv2.putText(
            frame,
            f"P:{self.ped_total}  V:{self.veh_total}",
            (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            _CYAN,
            2,
        )
        if not results or results[0].boxes.id is None:
            if self._last_boxes is not None:
                self._draw_last_boxes(frame)
            self._draw_info_hud(frame)
            return frame
        boxes   = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy().astype(int)
        ids     = results[0].boxes.id.cpu().numpy().astype(int)
        confs   = results[0].boxes.conf.cpu().numpy()
        self.last_person_count = int((classes == 0).sum())
        self.last_vehicle_count = int((classes != 0).sum())
        boxes, classes, ids, confs = apply_cross_class_nms(
            boxes, classes, ids, confs, iou_threshold=0.35
        )
        ids = self._id_merger.update(ids, boxes)
        if len(boxes) > 0:
            areas   = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            is_veh  = classes != 0
            conf_ok = np.where(is_veh, confs >= _CONF_VEH, confs >= _CONF_PED)
            area_ok = np.where(is_veh, areas >= _AREA_VEH, areas >= _AREA_PED)
            keep    = conf_ok & area_ok
            boxes   = boxes[keep]
            classes = classes[keep]
            ids     = ids[keep]
            confs   = confs[keep]
        _raw_boxes = raw_boxes_for_seg if raw_boxes_for_seg is not None else np.empty((0, 4), dtype=np.float32)
        _raw_masks = raw_seg_masks if raw_seg_masks is not None else []
        seg_masks = (
            _remap_masks(boxes, _raw_boxes, _raw_masks)
            if self._enable_segmentation and len(boxes) > 0
            else []
        )
        self._seg_masks = seg_masks
        self._last_boxes = (boxes, classes, ids, confs, seg_masks)
        if seg_masks and self._enable_segmentation:
            from config import settings as _cfg
            draw_all_masks(frame, boxes, classes, seg_masks,
                           alpha=_cfg.segmentation.mask_alpha)
        cur_ped: Set[int] = set()
        cur_veh: Set[int] = set()
        newly_in: Set[int] = set()
        for det_i, (box, cls, obj_id) in enumerate(zip(boxes, classes, ids)):
            obj_id   = int(obj_id)
            det_mask = seg_masks[det_i] if det_i < len(seg_masks) else None
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            if cls == 0:
                cur_ped.add(obj_id)
                if obj_id not in self._ped_tracks:
                    self._ped_tracks[obj_id] = PedestrianTrack(track_id=obj_id)
                pt = self._ped_tracks[obj_id]
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
                if np_poly is not None:
                    update_pedestrian_state(pt, np_poly, d_mid, d_axis, self._frame_index)
            else:
                cur_veh.add(obj_id)
                if obj_id not in self._veh_tracks:
                    self._veh_tracks[obj_id] = VehicleTrack(track_id=obj_id)
                vt = self._veh_tracks[obj_id]
                vt.prev_centroid = vt.centroid
                vt.centroid = (cx, cy)
                vt.bbox = (x1, y1, x2, y2)
                vt.frames_outside_count = 0
                vt.vehicle_class = int(cls)
                vt.mask = det_mask
                if vt.prev_centroid is not None:
                    vt.velocity_history.append((
                        cx - vt.prev_centroid[0],
                        cy - vt.prev_centroid[1],
                    ))
                vt.centroid_history.append((cx, cy))
                inside = (
                    crosswalk is not None
                    and crosswalk.intersects_mask(det_mask, frame.shape, box_fallback=box)
                )
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
        for gone_id in list(self._ped_tracks):
            if gone_id not in cur_ped:
                self._ped_tracks[gone_id].frames_outside_count += 1
                if self._ped_tracks[gone_id].frames_outside_count > TRACK_RESET_FRAMES:
                    del self._ped_tracks[gone_id]
        _VEH_GRACE = 15
        for gone_id in list(self._veh_tracks):
            if gone_id not in cur_veh:
                self._veh_tracks[gone_id].frames_outside_count += 1
                if self._veh_tracks[gone_id].frames_outside_count <= _VEH_GRACE:
                    continue
                _pv_gone = self._post_viol_collectors.pop(gone_id, None)
                if _pv_gone and _pv_gone['crops'] and self._ocr_engine is not None:
                    import threading as _t
                    _t.Thread(
                        target=self._run_multi_ocr_for_violation,
                        args=(_pv_gone['violation_id'], list(_pv_gone['crops'])),
                        daemon=True,
                    ).start()
                self._triggered_pairs -= {p for p in self._triggered_pairs if p[0] == gone_id}
                self._active_viol_cars.discard(gone_id)
                self._plate_bbox_cache.pop(gone_id, None)
                self._plate_text_cache.pop(gone_id, None)
                self._plate_ocr_pending.discard(gone_id)
                self._wrong_dir_flagged.discard(gone_id)
                del self._veh_tracks[gone_id]
        for det_i, (box, cls, obj_id) in enumerate(zip(boxes, classes, ids)):
            obj_id = int(obj_id)
            x1, y1, x2, y2 = box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            if cls != 0:
                vt = self._veh_tracks.get(obj_id)
                if (
                    vt is not None
                    and vt.polygon_entry_frame is not None
                    and (self._frame_index - vt.polygon_entry_frame) <= ENTRY_EVAL_WINDOW_FRAMES
                    and _vehicle_speed(vt) >= _MIN_MOVING_SPEED_PX
                ):
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
            if (
                cls != 0
                and self._enable_wrong_dir
                and vt is not None
                and len(vt.centroid_history) >= 8
                and obj_id not in self._wrong_dir_flagged
                and _ROAD_Y_MIN <= cy <= _ROAD_Y_MAX
                and _vehicle_speed(vt) >= _MIN_MOVING_SPEED_PX
            ):
                if _compute_vehicle_direction(vt) == "REVERSE":
                    self._wrong_dir_flagged.add(obj_id)
            violation_active = cls != 0 and obj_id in self._active_viol_cars
            is_wrong_dir     = cls != 0 and obj_id in self._wrong_dir_flagged
            det_mask_i = seg_masks[det_i] if det_i < len(seg_masks) else None
            if violation_active and det_mask_i is not None:
                from vision.draw import draw_segmentation_mask
                draw_segmentation_mask(frame, det_mask_i, cls=-1, alpha=0.55, color=_RED)
            elif is_wrong_dir and det_mask_i is not None:
                from vision.draw import draw_segmentation_mask
                draw_segmentation_mask(frame, det_mask_i, cls=-1, alpha=0.45, color=_ORANGE)
            lx = max(0, int(x1) + 4)
            ly_state = max(14, int(y1) - 22)
            ly_class = max(14, int(y1) - 4)
            if cls == 0:
                pt = self._ped_tracks.get(obj_id)
                state_str = pt.state if pt else "?"
                _SC = {"OUTSIDE": (160,160,160), "ENTERING": (0,200,255), "CROSSING": (30,220,30), "EXITED": (200,100,0)}
                sc  = _SC.get(state_str, _WHITE)
                cv2.putText(frame, state_str, (lx, ly_state), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 3, cv2.LINE_AA)
                cv2.putText(frame, state_str, (lx, ly_state), cv2.FONT_HERSHEY_SIMPLEX, 0.55, sc,      2, cv2.LINE_AA)
                cv2.putText(frame, "PED",     (lx, ly_class), cv2.FONT_HERSHEY_SIMPLEX, 0.5,  (0,0,0), 3, cv2.LINE_AA)
                cv2.putText(frame, "PED",     (lx, ly_class), cv2.FONT_HERSHEY_SIMPLEX, 0.5,  _WHITE,  1, cv2.LINE_AA)
                if pt:
                    direction = _ped_direction(pt.velocity_history)
                    if direction != "STATIC":
                        cv2.putText(frame, direction, (lx, ly_class + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0),     3, cv2.LINE_AA)
                        cv2.putText(frame, direction, (lx, ly_class + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,220,255), 2, cv2.LINE_AA)
            else:
                vt = self._veh_tracks.get(obj_id)
                cls_name = class_label(int(cls)).upper()
                if violation_active:
                    _vstate_lbl = "VIOLATION"
                    _vstate_col = _RED
                elif is_wrong_dir:
                    _vstate_lbl = "WRONG DIR"
                    _vstate_col = _ORANGE
                elif vt is not None and vt.polygon_entry_frame is not None:
                    _vstate_lbl = "IN ZONE"
                    _vstate_col = (0, 255, 180)
                else:
                    _vstate_lbl = None
                    _vstate_col = _WHITE
                if _vstate_lbl is not None:
                    cv2.putText(frame, _vstate_lbl, (lx, ly_state), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0),      3, cv2.LINE_AA)
                    cv2.putText(frame, _vstate_lbl, (lx, ly_state), cv2.FONT_HERSHEY_SIMPLEX, 0.55, _vstate_col,  2, cv2.LINE_AA)
                cv2.putText(frame, cls_name, (lx, ly_class), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3, cv2.LINE_AA)
                cv2.putText(frame, cls_name, (lx, ly_class), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _WHITE,  1, cv2.LINE_AA)
                if violation_active:
                    cv2.putText(frame, "! VIOLATION", (lx, int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 4, cv2.LINE_AA)
                    cv2.putText(frame, "! VIOLATION", (lx, int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, _RED,    2, cv2.LINE_AA)
                if got_new_detection and self._plate_detector is not None:
                    x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
                    x1c = max(0, x1i); y1c = max(0, y1i)
                    x2c = min(w - 1, x2i); y2c = min(h - 1, y2i)
                    if (x2c - x1c) >= 32 and (y2c - y1c) >= 32:
                        try:
                            _clean = self._clean_frame if self._clean_frame is not None else frame
                            crop = _clean[y1c:y2c, x1c:x2c]
                            pbbox, _pconf = self._plate_detector._best_box(crop)
                            if pbbox is not None:
                                px1, py1, px2, py2 = pbbox
                                frame_coords = (
                                    x1c + px1, y1c + py1, x1c + px2, y1c + py2,
                                )
                                self._plate_bbox_cache[obj_id] = frame_coords
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
                                        plate_crop_img = _clean[ppy1o:ppy2o, ppx1o:ppx2o].copy()
                                        self._plate_ocr_pending.add(obj_id)
                                        import threading as _t
                                        _t.Thread(
                                            target=self._run_ocr_for_vehicle,
                                            args=(obj_id, plate_crop_img),
                                            daemon=True,
                                        ).start()
                                _pv = self._post_viol_collectors.get(obj_id)
                                if _pv is not None and len(_pv['crops']) < 8:
                                    _pvx1 = max(0, int(x1c + px1)); _pvy1 = max(0, int(y1c + py1))
                                    _pvx2 = min(w - 1, int(x1c + px2)); _pvy2 = min(h - 1, int(y1c + py2))
                                    if _pvx2 - _pvx1 > 8 and _pvy2 - _pvy1 > 8:
                                        _pv['crops'].append(_clean[_pvy1:_pvy2, _pvx1:_pvx2].copy())
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
        for _pv_vid in list(self._post_viol_collectors):
            _pv_col = self._post_viol_collectors[_pv_vid]
            _pv_col['countdown'] -= 1
            if _pv_col['countdown'] <= 0:
                _pv_col = self._post_viol_collectors.pop(_pv_vid)
                if _pv_col['crops'] and self._ocr_engine is not None:
                    import threading as _t
                    _t.Thread(
                        target=self._run_multi_ocr_for_violation,
                        args=(_pv_col['violation_id'], list(_pv_col['crops'])),
                        daemon=True,
                    ).start()
        ped_in_zone = sum(
            1 for pid, pt in self._ped_tracks.items()
            if pt.state in ("ENTERING", "CROSSING", "CLEARING")
        )
        with self._lock:
            self.ped_total   = len(self._ped_tracks)
            self.veh_total   = len(cur_veh)
            self.ped_in_zone = ped_in_zone
            self.veh_in_zone = len(newly_in)
            self.active_violation = time.monotonic() < self._violation_until
            self.last_frame_ms = (time.perf_counter() - t_frame_start) * 1000.0
            self.avg_frame_ms = self._ema(self.avg_frame_ms, self.last_frame_ms)
        self._draw_info_hud(frame)
        return frame
    def _draw_last_boxes(self, frame: np.ndarray) -> None:
        if self._last_boxes is None:
            return
        from vision.draw import draw_all_masks
        from detector.segmentation import class_label, class_color
        if len(self._last_boxes) == 5:
            boxes, classes, ids, _, cached_masks = self._last_boxes
        else:
            boxes, classes, ids, _ = self._last_boxes
            cached_masks = []
        if cached_masks and self._enable_segmentation:
            draw_all_masks(frame, boxes, classes, cached_masks, alpha=0.35)
        for c_i, (box, cls, obj_id) in enumerate(zip(boxes, classes, ids)):
            obj_id = int(obj_id)
            x1, y1, x2, y2 = box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            c_mask_i = cached_masks[c_i] if c_i < len(cached_masks) else None
            lx2 = max(0, int(x1) + 4)
            ly2_state = max(14, int(y1) - 22)
            ly2_class = max(14, int(y1) - 4)
            if cls == 0:
                pt_draw  = self._ped_tracks.get(obj_id)
                st       = pt_draw.state if pt_draw else "OUTSIDE"
                _SC2 = {"OUTSIDE": (160,160,160), "ENTERING": (0,200,255), "CROSSING": (30,220,30), "EXITED": (200,100,0)}
                sc2  = _SC2.get(st, (255,255,255))
                cv2.putText(frame, st,    (lx2, ly2_state), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 3, cv2.LINE_AA)
                cv2.putText(frame, st,    (lx2, ly2_state), cv2.FONT_HERSHEY_SIMPLEX, 0.55, sc2,     2, cv2.LINE_AA)
                cv2.putText(frame, "PED", (lx2, ly2_class), cv2.FONT_HERSHEY_SIMPLEX, 0.5,  (0,0,0), 3, cv2.LINE_AA)
                cv2.putText(frame, "PED", (lx2, ly2_class), cv2.FONT_HERSHEY_SIMPLEX, 0.5,  (255,255,255), 1, cv2.LINE_AA)
            else:
                viol = obj_id in self._active_viol_cars
                wd   = obj_id in self._wrong_dir_flagged
                if viol and c_mask_i is not None:
                    from vision.draw import draw_segmentation_mask
                    draw_segmentation_mask(frame, c_mask_i, cls=-1, alpha=0.55, color=_RED)
                elif wd and c_mask_i is not None:
                    from vision.draw import draw_segmentation_mask
                    draw_segmentation_mask(frame, c_mask_i, cls=-1, alpha=0.45, color=_ORANGE)
                cls_name2 = class_label(int(cls)).upper()
                vt2 = self._veh_tracks.get(obj_id)
                if viol:
                    _v2state_lbl = "VIOLATION"
                    _v2state_col = _RED
                elif wd:
                    _v2state_lbl = "WRONG DIR"
                    _v2state_col = _ORANGE
                elif vt2 is not None and vt2.polygon_entry_frame is not None:
                    _v2state_lbl = "IN ZONE"
                    _v2state_col = (0, 255, 180)
                else:
                    _v2state_lbl = None
                    _v2state_col = _WHITE
                if _v2state_lbl is not None:
                    cv2.putText(frame, _v2state_lbl, (lx2, ly2_state), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0),       3, cv2.LINE_AA)
                    cv2.putText(frame, _v2state_lbl, (lx2, ly2_state), cv2.FONT_HERSHEY_SIMPLEX, 0.55, _v2state_col,  2, cv2.LINE_AA)
                cv2.putText(frame, cls_name2, (lx2, ly2_class), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0),       3, cv2.LINE_AA)
                cv2.putText(frame, cls_name2, (lx2, ly2_class), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                if viol:
                    cv2.putText(frame, "! VIOLATION", (lx2, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 4, cv2.LINE_AA)
                    cv2.putText(frame, "! VIOLATION", (lx2, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.9, _RED,    2, cv2.LINE_AA)
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
        pass
    def _draw_info_hud(self, frame: np.ndarray) -> None:
        fh, fw = frame.shape[:2]
        fps = (1000.0 / self.avg_frame_ms) if self.avg_frame_ms > 0 else 0.0
        font  = cv2.FONT_HERSHEY_SIMPLEX
        sc    = 0.48
        thick = 1
        pad   = 7
        lh    = 19
        fps_col = (0, 255, 0) if fps >= 20 else (0, 180, 255) if fps >= 10 else (0, 60, 255)
        flags = [
            ("PLT",  self._enable_plate_detector),
            ("OCR",  self._enable_ocr),
            ("WRD",  self._enable_wrong_dir),
            ("SEG",  self._enable_segmentation),
            ("STAB", self._stabilizer_enabled),
        ]
        row1a = f"FPS {fps:.1f}"
        row1b = f"det {self.avg_detect_ms:.0f}ms"
        (r1a_w, r1_h), _ = cv2.getTextSize(row1a, font, sc, thick)
        (r1b_w, _),    _ = cv2.getTextSize(row1b, font, sc, thick)
        row1_w = r1a_w + 10 + r1b_w
        flag_gap = 8
        flag_parts = [(lbl, en, cv2.getTextSize(lbl, font, sc, thick)[0]) for lbl, en in flags]
        row2_w = sum(fw_ for _, _, (fw_, _) in flag_parts) + flag_gap * (len(flag_parts) - 1)
        box_w = max(row1_w, row2_w) + 2 * pad
        box_h = lh * 2 + 2 * pad
        margin = 10
        x0 = fw - box_w - margin
        y0 = fh - box_h - margin
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (15, 15, 15), cv2.FILLED)
        cv2.addWeighted(overlay, 0.62, frame, 0.38, 0, frame)
        y1 = y0 + pad + r1_h
        cv2.putText(frame, row1a, (x0 + pad, y1), font, sc, fps_col, thick, cv2.LINE_AA)
        cv2.putText(frame, row1b, (x0 + pad + r1a_w + 10, y1), font, sc, (140, 140, 140), thick, cv2.LINE_AA)
        y2 = y0 + pad + lh + r1_h
        xc = x0 + pad
        for lbl, enabled, (fw_, fh_) in flag_parts:
            col = (0, 210, 60) if enabled else (65, 65, 65)
            cv2.putText(frame, lbl, (xc, y2), font, sc, col, thick, cv2.LINE_AA)
            xc += fw_ + flag_gap
    def _run_ocr_for_vehicle(self, obj_id: int, plate_crop: np.ndarray) -> None:
        t0 = time.perf_counter()
        try:
            result = self._ocr_engine.recognize_array(plate_crop)
            self.ocr_attempts += 1
            if result.plate_text:
                self._plate_text_cache[obj_id] = result.plate_text
                self.ocr_accepted += 1
                if self._pipeline is not None:
                    try:
                        self._pipeline.repository.upsert_vehicle(result.plate_text)
                    except Exception:
                        pass
            elif result.raw_text and len(result.raw_text.strip()) >= 3:
                self._plate_text_cache[obj_id] = f"~{result.raw_text.strip()[:10]}"
                self.ocr_rejected += 1
            else:
                self.ocr_rejected += 1
        except Exception:
            self.ocr_rejected += 1
        finally:
            ocr_ms = (time.perf_counter() - t0) * 1000.0
            self.last_ocr_ms = ocr_ms
            self.avg_ocr_ms = self._ema(self.avg_ocr_ms, ocr_ms)
            self._plate_ocr_pending.discard(obj_id)
    def _run_multi_ocr_for_violation(self, violation_id: str, crops: list) -> None:
        from config import settings as _cfg
        plates_dir = _cfg.storage.plate_crops_dir
        best_text: Optional[str] = None
        best_conf = 0.0
        for i, crop in enumerate(crops):
            try:
                fname = plates_dir / f"{violation_id}_f{i}.jpg"
                cv2.imwrite(str(fname), crop)
                result = self._ocr_engine.recognize_array(crop)
                if result and result.plate_text and result.confidence > best_conf:
                    best_text = result.plate_text
                    best_conf = result.confidence
            except Exception:
                pass
        if best_text and self._pipeline:
            try:
                self._pipeline.update_violation_plate(violation_id, best_text, best_conf)
                print(f"[LiveProcessor] Multi-frame OCR {violation_id}: {best_text} ({best_conf:.2f})")
            except Exception as exc:
                print(f"[LiveProcessor] Multi-frame OCR update failed: {exc}")

    def _submit_violation(self, frame, vbox, np_poly, vt, pt, violation,
                            all_boxes=None, all_classes=None) -> None:
        import threading as _t
        from schemas import ViolationEvent
        from config import settings as cfg
        _SNAPSHOTS.mkdir(parents=True, exist_ok=True)
        x1, y1, x2, y2 = [int(v) for v in vbox]
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
        try:
            snap = frame.copy()
            fh, fw = snap.shape[:2]
            overlay = snap.copy()
            cv2.rectangle(overlay, (0, 0), (fw, 40), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.6, snap, 0.4, 0, snap)
            if all_boxes is not None and all_classes is not None:
                for b, c in zip(all_boxes, all_classes):
                    bx1, by1, bx2, by2 = [int(v) for v in b]
                    color = (255, 0, 0) if int(c) == 0 else (0, 0, 255)
                    cv2.rectangle(snap, (bx1, by1), (bx2, by2), color, 2)
            cv2.rectangle(snap, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.polylines(snap, [np_poly.astype(np.int32)], True, _AMBER, 2)
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
            plate_bbox_frame = self._plate_bbox_cache.get(vt.track_id)
            if plate_bbox_frame is not None:
                try:
                    src = self._clean_frame if self._clean_frame is not None else frame
                    fh_f, fw_f = src.shape[:2]
                    ppx1, ppy1, ppx2, ppy2 = plate_bbox_frame
                    ppx1 = max(0, ppx1); ppy1 = max(0, ppy1)
                    ppx2 = min(fw_f - 1, ppx2); ppy2 = min(fh_f - 1, ppy2)
                    if ppx2 > ppx1 and ppy2 > ppy1:
                        plate_crop = src[ppy1:ppy2, ppx1:ppx2]
                        plate_fname = f"plate_{event.violation_id}.jpg"
                        cv2.imwrite(str(_SNAPSHOTS / plate_fname), plate_crop)
                        event.plate_crop_path = f"snapshots/{plate_fname}"
                        print(f"[LiveProcessor] Plate crop saved: {plate_fname}")
                except Exception as exc:
                    print(f"[LiveProcessor] Plate crop save failed: {exc}")
        except Exception as exc:
            print(f"[LiveProcessor] Snapshot failed: {exc}")
        if self._pipeline:
            self.violations_submitted += 1
            self._post_viol_collectors[vt.track_id] = {
                'violation_id': event.violation_id,
                'crops': [],
                'countdown': 30,
            }
            _t.Thread(
                target=lambda: self._pipeline.submit_violation(frame.copy(), event),
                daemon=True,
            ).start()
class LiveProcessorRegistry:
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
live_proc = LiveProcessor()
