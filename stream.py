"""
stream.py — Thread-safe MJPEG camera stream manager.

Supports:
  - Webcam:         source = 0  (or 1, 2 … for additional cameras)
  - RTSP IP camera: source = "rtsp://admin:pass@192.168.1.100:554/stream1"
  - HTTP MJPEG:     source = "http://192.168.1.100:8080/video"
  - Video file:     source = "Videos/v2.mp4"  (useful for demo)

Usage in Flask:
  from stream import camera_manager
  camera_manager.start("rtsp://...")
  frame_bytes = camera_manager.get_jpeg()

──────────────────────────────────────────────────────────────
CONNECTING A CAMERA
──────────────────────────────────────────────────────────────

OFFLINE (local USB / built-in webcam):
  Set CAMERA_SOURCE=0 in your .env file (0 = first camera).
  Multiple cameras: 0, 1, 2 …

ONLINE — Hikvision / Dahua RTSP:
  rtsp://admin:PASSWORD@CAMERA_IP:554/Streaming/Channels/101
  Example: rtsp://admin:Admin123@192.168.1.64:554/Streaming/Channels/101

ONLINE — Generic IP camera (ONVIF):
  rtsp://USERNAME:PASSWORD@CAMERA_IP:554/stream1

ONLINE — Budget IP cameras (often on port 80):
  http://CAMERA_IP:80/video.cgi?resolution=VGA

ONLINE — Reolink / TP-Link Tapo:
  rtsp://admin:PASSWORD@CAMERA_IP:554/h264Preview_01_main

TIP: Test your RTSP URL with VLC → Media → Open Network Stream.
"""
from __future__ import annotations

import io
import os
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ── Placeholder frame (dark image shown when no camera connected) ────────────
def _make_placeholder() -> bytes:
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    img[:] = (15, 17, 23)   # --bg-primary colour
    cv2.putText(img, "NO CAMERA CONNECTED", (140, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 80, 80), 2)
    cv2.putText(img, "Enter camera source and click CONNECT", (90, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1)
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return buf.tobytes()


_PLACEHOLDER = _make_placeholder()


# ── CameraStream ─────────────────────────────────────────────────────────────

class CameraStream:
    """
    Captures frames from any OpenCV-compatible source in a background daemon
    thread, so Flask routes never block waiting for a frame.
    """

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

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self, source: str | int) -> bool:
        """Open the camera source.  Returns True on success."""
        self.stop()                    # release any existing capture
        self.source = str(source)
        self.error  = ""

        # Numeric string → int index
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
        """
        Open source and verify the first frame is usable.
        On Windows, some camera indexes open successfully but return near-black frames.
        Falls back to the demo video file when all camera indexes fail.
        """
        candidates: list[tuple[str | int, Optional[int]]] = []

        # Primary candidate
        candidates.append((src, None))

        # Windows-specific backend fallbacks for camera indexes.
        if isinstance(src, int):
            candidates.append((src, cv2.CAP_DSHOW))
            candidates.append((src, cv2.CAP_MSMF))
            # If selected index is black/unusable, try common webcam indexes.
            for alt in (0, 1):
                if alt != src:
                    candidates.append((alt, cv2.CAP_DSHOW))
                    candidates.append((alt, cv2.CAP_MSMF))
                    candidates.append((alt, None))

        # Final fallback: demo video file (works even without any camera attached)
        import os as _os
        _demo = _os.getenv("DEMO_VIDEO_SOURCE", "Videos/v2.mp4")
        if isinstance(src, int) and _demo:
            candidates.append((_demo, None))

        for candidate_src, backend in candidates:
            cap = cv2.VideoCapture(candidate_src) if backend is None else cv2.VideoCapture(candidate_src, backend)
            if not cap.isOpened():
                cap.release()
                continue

            ok, frame = cap.read()
            if not ok or frame is None:
                cap.release()
                continue

            if self._is_near_black(frame):
                cap.release()
                continue

            return cap, candidate_src

        return None, src

    @staticmethod
    def _is_near_black(frame: np.ndarray) -> bool:
        """Heuristic: reject likely-empty/black frames returned by bad camera backends."""
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
        """Called by live_processor to replace the raw stream with an annotated frame."""
        with self._lock:
            self._annotated_jpeg = jpeg

    def clear_annotated(self) -> None:
        """Stop serving annotated frames (revert to raw stream)."""
        with self._lock:
            self._annotated_jpeg = None

    def get_jpeg(self) -> bytes:
        """Return the latest annotated frame if available, otherwise the raw frame."""
        with self._lock:
            return self._annotated_jpeg if self._annotated_jpeg is not None else self._jpeg

    def get_numpy(self) -> Optional[np.ndarray]:
        """Return the latest raw numpy frame (thread-safe copy)."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def status(self) -> dict:
        return {
            "connected": self.connected,
            "source":    self.source,
            "fps":       round(self.fps, 1),
            "width":     self.width,
            "height":    self.height,
            "error":     self.error,
        }

    # ── Background thread ─────────────────────────────────────────────────────

    @staticmethod
    def _is_local_video_file(source: str) -> bool:
        """Return True if source is a local file path (not a camera index or network URL)."""
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
                    # Loop the video: seek back to the first frame and continue
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                # Real camera/stream disconnected
                self.connected = False
                self.error = "Stream ended or connection lost."
                with self._lock:
                    self._jpeg = _PLACEHOLDER
                break

            _, buf = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75]
            )
            with self._lock:
                self._jpeg  = buf.tobytes()
                self._frame = frame

            elapsed = time.monotonic() - t0
            sleep   = target_interval - elapsed
            if sleep > 0:
                time.sleep(sleep)


# ── Singleton ────────────────────────────────────────────────────────────────

camera_manager = CameraStream()


# ── MJPEG generator (pass to Flask Response) ─────────────────────────────────

def mjpeg_generator(cam: CameraStream):
    """Yields MJPEG multipart frames suitable for Flask streaming Response."""
    while True:
        frame = cam.get_jpeg()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )
        time.sleep(0.033)   # cap at ~30 fps to the browser
