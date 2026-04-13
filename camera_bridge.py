"""
camera_bridge.py — Run this on your LOCAL Windows PC to forward your webcam
to the Colab GPU instance so it can process live frames.

WHY THIS EXISTS:
  Colab cannot access your PC's USB webcam directly.  This script captures
  your webcam and serves it as an MJPEG HTTP stream.  You then expose that
  stream via ngrok, giving Colab a public URL it can open as CAMERA_SOURCE.

ARCHITECTURE:
  [PC webcam] → [this script on port 8080] → [ngrok tunnel] → [Colab CAMERA_SOURCE]
  Colab reads frames as if they're a regular IP camera stream.

SETUP (one-time):
  1. Make sure Flask is installed in your venv:
       pip install flask opencv-python
  2. Make sure ngrok is installed:
       winget install ngrok.ngrok
  3. Add your ngrok auth token (one-time):
       ngrok config add-authtoken YOUR_TOKEN
  4. Get your free static ngrok domain at: https://dashboard.ngrok.com/cloud-edge/domains

USAGE:
  Option A — use the start_camera_bridge.bat helper (double-click it)
  Option B — run manually:
       # Terminal 1:
       python camera_bridge.py
       # Terminal 2:
       ngrok http 8080 --domain=YOUR-STATIC-DOMAIN.ngrok-free.app

  Then in Colab Cell B3, set:
       CAMERA_SOURCE = "https://YOUR-STATIC-DOMAIN.ngrok-free.app/video"

TESTING:
  While the bridge is running, open http://localhost:8080 in your browser —
  you should see your webcam feed.  If that works, the ngrok URL will too.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Generator

import cv2
from flask import Flask, Response, render_template_string

# ── Configuration ─────────────────────────────────────────────────────────────
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))   # 0 = first webcam, 1 = second, …
PORT         = int(os.getenv("BRIDGE_PORT",  "8080"))
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "75"))  # 75 = good balance of quality vs bandwidth
TARGET_FPS   = int(os.getenv("TARGET_FPS",   "15"))  # keep low to avoid saturating the tunnel

# ── Camera setup ──────────────────────────────────────────────────────────────
_cap  = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)   # CAP_DSHOW = faster on Windows
_lock = threading.Lock()
_latest_jpeg: bytes = b""
_running = True


def _capture_loop() -> None:
    """Background thread: continuously captures and JPEG-encodes the latest frame."""
    global _latest_jpeg, _running
    interval = 1.0 / TARGET_FPS
    while _running:
        t0 = time.monotonic()
        ret, frame = _cap.read()
        if ret and frame is not None:
            ok, buf = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
            )
            if ok:
                with _lock:
                    _latest_jpeg = buf.tobytes()
        elapsed = time.monotonic() - t0
        sleep = max(0.0, interval - elapsed)
        time.sleep(sleep)


# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)

_INDEX_HTML = """<!doctype html>
<html>
<head>
  <title>Camera Bridge</title>
  <style>
    body { background:#111; color:#eee; font-family:sans-serif; text-align:center; padding:2rem; }
    img  { max-width:100%; border:2px solid #444; border-radius:4px; }
    p    { color:#888; font-size:.9rem; margin-top:1rem; }
  </style>
</head>
<body>
  <h2>Camera Bridge — local feed preview</h2>
  <img src="/video" alt="webcam feed" />
  <p>If you can see your webcam here, the ngrok URL is working correctly.</p>
</body>
</html>"""


def _mjpeg_generator() -> Generator[bytes, None, None]:
    while True:
        with _lock:
            frame = _latest_jpeg
        if frame:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame
                + b"\r\n"
            )
        time.sleep(1.0 / TARGET_FPS)


@app.route("/video")
def video_feed() -> Response:
    return Response(
        _mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/")
def index() -> str:
    return _INDEX_HTML


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not _cap.isOpened():
        print(f"ERROR: Cannot open camera {CAMERA_INDEX}.")
        print("Try a different index: set CAMERA_INDEX=1 (or 2) before running.")
        raise SystemExit(1)

    w = int(_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera {CAMERA_INDEX} opened: {w}x{h} @ {TARGET_FPS} fps target")
    print(f"JPEG quality: {JPEG_QUALITY}")
    print()
    print(f"Bridge running at:  http://localhost:{PORT}")
    print(f"Video stream at:    http://localhost:{PORT}/video")
    print()
    print("Next step: open a second terminal and run:")
    print("  ngrok http 8080 --domain=YOUR-STATIC-DOMAIN.ngrok-free.app")
    print()
    print("Then set CAMERA_SOURCE in Colab Cell B3 to your ngrok URL + /video")

    t = threading.Thread(target=_capture_loop, daemon=True)
    t.start()

    try:
        app.run(host="0.0.0.0", port=PORT, threaded=True)
    finally:
        _running = False
        _cap.release()
