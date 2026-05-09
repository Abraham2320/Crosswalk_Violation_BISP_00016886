from __future__ import annotations
import io
import os
import threading
import time
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
def _make_placeholder() -> bytes:
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    img[:] = (15, 17, 23)
    cv2.putText(img, "NO CAMERA CONNECTED", (140, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 80, 80), 2)
    cv2.putText(img, "Enter camera source and click CONNECT", (90, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1)
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return buf.tobytes()
_PLACEHOLDER = _make_placeholder()
class CameraStream:
    def __init__(self) -> None:
        self._cap:    Optional[cv2.VideoCapture] = None
        self._frame:  Optional[np.ndarray] = None
        self._jpeg:   bytes = _PLACEHOLDER
        self._annotated_jpeg: Optional[bytes] = None
        self._lock    = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.source:  str = ""
        self.connected: bool = False
        self.fps:     float = 0.0
        self.width:   int   = 0
        self.height:  int   = 0
        self.error:   str   = ""
        self._frame_seq: int = 0
    def start(self, source: str | int) -> bool:
        self.stop()
        self.source = str(source)
        self.error  = ""
        src = int(source) if str(source).isdigit() else source
        cap, used_source = self._open_with_fallback(src)
        if cap is None:
            self.error = f"Cannot open usable source: {source}"
            return False
        self._cap     = cap
        self.source   = str(used_source)
        self.connected = True
        self.fps      = cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.width    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._running = True
        self._thread  = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return True
    def _open_with_fallback(self, src: str | int) -> tuple[Optional[cv2.VideoCapture], str | int]:
        candidates: list[tuple[str | int, Optional[int]]] = []
        candidates.append((src, None))
        if isinstance(src, int):
            candidates.append((src, cv2.CAP_DSHOW))
            candidates.append((src, cv2.CAP_MSMF))
        import os as _os
        _demo = _os.getenv("DEMO_VIDEO_SOURCE", "Videos/v2.mp4")
        if isinstance(src, int) and _demo:
            candidates.append((_demo, None))
        for candidate_src, backend in candidates:
            cap = cv2.VideoCapture(candidate_src) if backend is None else cv2.VideoCapture(candidate_src, backend)
            if not cap.isOpened():
                cap.release()
                continue
            usable = False
            for _ in range(5):
                ok, frame = cap.read()
                if ok and frame is not None and not self._is_near_black(frame):
                    usable = True
                    break
            if not usable:
                cap.release()
                continue
            return cap, candidate_src
        return None, src
    @staticmethod
    def _is_near_black(frame: np.ndarray) -> bool:
        if frame is None or frame.size == 0:
            return True
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean = float(gray.mean())
        std = float(gray.std())
        return mean < 6.0 and std < 22.0
    def stop(self) -> None:
        self._running  = False
        self.connected = False
        if self._cap:
            self._cap.release()
            self._cap = None
        with self._lock:
            self._jpeg  = _PLACEHOLDER
            self._frame = None
    def set_annotated_jpeg(self, jpeg: bytes) -> None:
        with self._lock:
            self._annotated_jpeg = jpeg
    def clear_annotated(self) -> None:
        with self._lock:
            self._annotated_jpeg = None
    def get_jpeg(self) -> bytes:
        with self._lock:
            return self._annotated_jpeg if self._annotated_jpeg is not None else self._jpeg
    def get_numpy(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._frame.copy() if self._frame is not None else None
    def get_numpy_if_new(self, last_seq: int) -> tuple:
        with self._lock:
            if self._frame is None or self._frame_seq == last_seq:
                return None, last_seq
            return self._frame.copy(), self._frame_seq
    def status(self) -> dict:
        return {
            "connected": self.connected,
            "source":    self.source,
            "fps":       round(self.fps, 1),
            "width":     self.width,
            "height":    self.height,
            "error":     self.error,
        }
    @staticmethod
    def _is_local_video_file(source: str) -> bool:
        s = str(source).strip()
        if s.isdigit():
            return False
        if s.startswith(("rtsp://", "http://", "https://", "mjpeg://")):
            return False
        return Path(s).suffix.lower() in {".mp4", ".avi", ".mkv", ".mov", ".ts", ".m4v", ".wmv"}
    def _capture_loop(self) -> None:
        is_video_file = self._is_local_video_file(self.source)
        target_interval = 1.0 / max(self.fps, 1.0)
        while self._running and self._cap is not None:
            t0  = time.monotonic()
            ret, frame = self._cap.read()
            if not ret:
                if is_video_file:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                self.connected = False
                self.error = "Stream ended or connection lost."
                with self._lock:
                    self._jpeg = _PLACEHOLDER
                break
            if not is_video_file and self._is_near_black(frame):
                elapsed = time.monotonic() - t0
                sleep   = target_interval - elapsed
                if sleep > 0:
                    time.sleep(sleep)
                continue
            _, buf = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75]
            )
            with self._lock:
                self._jpeg     = buf.tobytes()
                self._frame    = frame
                self._frame_seq += 1
            elapsed = time.monotonic() - t0
            sleep   = target_interval - elapsed
            if sleep > 0:
                time.sleep(sleep)
camera_manager = CameraStream()
CAMERA_CONFIGS: dict[str, dict] = {
    "cam1": {"label": "Camera 1 - Phone Camera",  "source": "http://localhost:4747/video", "demo": "Videos/v1.mp4"},
    "cam2": {"label": "Camera 2", "demo": "Videos/v2.mp4"},
    "cam3": {"label": "Camera 3",      "demo": "Videos/v3.mp4"},
}
class CameraRegistry:
    def __init__(self) -> None:
        self._streams: dict[str, CameraStream] = {}
        self._lock = threading.Lock()
    def get(self, cam_id: str) -> CameraStream:
        with self._lock:
            if cam_id not in self._streams:
                self._streams[cam_id] = CameraStream()
            return self._streams[cam_id]
    def statuses(self) -> dict:
        return {
            cam_id: {"config": cfg, "stream": self.get(cam_id).status()}
            for cam_id, cfg in CAMERA_CONFIGS.items()
        }
registry = CameraRegistry()
def mjpeg_generator(cam: CameraStream):
    while True:
        frame = cam.get_jpeg()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )
        time.sleep(0.033)
