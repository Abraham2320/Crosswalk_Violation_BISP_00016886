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

        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            cap.release()
            self.error = f"Cannot open source: {source}"
            return False

        self._cap     = cap
        self.connected = True
        self.fps      = cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.width    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._running = True
        self._thread  = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._running  = False
        self.connected = False
        if self._cap:
            self._cap.release()
            self._cap = None
        with self._lock:
            self._jpeg  = _PLACEHOLDER
            self._frame = None

    def get_jpeg(self) -> bytes:
        """Return the latest frame as a JPEG bytes object (thread-safe)."""
        with self._lock:
            return self._jpeg

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

    def _capture_loop(self) -> None:
        target_interval = 1.0 / max(self.fps, 1.0)
        while self._running and self._cap is not None:
            t0  = time.monotonic()
            ret, frame = self._cap.read()
            if not ret:
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
