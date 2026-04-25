"""
camera_bridge.py — Run this on your LOCAL Windows PC to forward a camera
(USB webcam OR DroidCam phone camera) to the Colab GPU instance.

WHY THIS EXISTS:
  Colab cannot access your PC's webcam or phone camera directly.  This script
  captures from any OpenCV-compatible source and serves it as an MJPEG HTTP
  stream on port 8080.  You then expose that with a Cloudflare Quick Tunnel
  (no account) so Colab can connect.

ARCHITECTURE:
  [DroidCam / webcam] → [this script :8080] → [cloudflared tunnel] → [Colab CAMERA_SOURCE]
  Colab reads frames as if from a regular IP camera stream.

─────────────────────────────────────────────────────────────────────────────
MODE A — DroidCam (recommended: your phone becomes an IP camera)
─────────────────────────────────────────────────────────────────────────────
  1. Install "DroidCam" on your Android phone (free, by Dev47Apps).
  2. Connect phone and PC to the SAME Wi-Fi.
  3. Open DroidCam on the phone — it shows the phone's local IP and port
     (default port 4747).
  4. Set the env var before running:
       set DROIDCAM_URL=http://192.168.X.X:4747/video

     Or on macOS/Linux:
       export DROIDCAM_URL=http://192.168.X.X:4747/video

  5. Run this bridge:
       python camera_bridge.py

  6. In a second terminal, expose it via Cloudflare (no account needed):
       cloudflared tunnel --url http://localhost:8080

  7. Copy the *.trycloudflare.com URL and paste it into Colab Cell B3:
       CAMERA_SOURCE = "https://YOUR-ID.trycloudflare.com/video"

─────────────────────────────────────────────────────────────────────────────
MODE B — USB / built-in webcam (original behaviour)
─────────────────────────────────────────────────────────────────────────────
  Leave DROIDCAM_URL unset.  Adjust CAMERA_INDEX if you have multiple cameras:
       set CAMERA_INDEX=1
       python camera_bridge.py

TESTING:
  While the bridge is running, open http://localhost:8080 in your browser —
  you should see the camera feed.  If that works, the Cloudflare URL will too.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Generator

import cv2
from flask import Flask, Response

# ── Configuration ─────────────────────────────────────────────────────────────
DROIDCAM_URL = os.getenv("DROIDCAM_URL", "")          # e.g. http://192.168.1.5:4747/video
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))    # fallback webcam index
PORT         = int(os.getenv("BRIDGE_PORT",  "8080"))
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "75"))
TARGET_FPS   = int(os.getenv("TARGET_FPS",   "15"))   # keep low — tunnel bandwidth is limited

# ── Camera setup ──────────────────────────────────────────────────────────────
if DROIDCAM_URL:
    # DroidCam (or any MJPEG/RTSP URL) — do NOT use CAP_DSHOW (Windows-only, int-index only)
    _source = DROIDCAM_URL
    _cap    = cv2.VideoCapture(DROIDCAM_URL)
    _mode   = f"DroidCam  {DROIDCAM_URL}"
else:
    # Local USB / built-in webcam
    _source = CAMERA_INDEX
    _cap    = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)  # CAP_DSHOW = faster on Windows
    _mode   = f"Webcam index {CAMERA_INDEX}"

_lock = threading.Lock()
_latest_jpeg: bytes = b""
_running = True


def _capture_loop() -> None:
    """Background thread: continuously captures and JPEG-encodes the latest frame.
    For URL sources (DroidCam) it automatically reconnects on disconnect."""
    global _latest_jpeg, _running, _cap
    interval = 1.0 / TARGET_FPS
    consecutive_failures = 0

    while _running:
        t0 = time.monotonic()
        ret, frame = _cap.read()

        if ret and frame is not None:
            consecutive_failures = 0
            ok, buf = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
            )
            if ok:
                with _lock:
                    _latest_jpeg = buf.tobytes()
        else:
            consecutive_failures += 1
            # After 30 consecutive blank frames (~2 s at 15 fps), try to reconnect
            if consecutive_failures >= 30 and DROIDCAM_URL:
                print(f"[bridge] Lost connection — reconnecting to {DROIDCAM_URL} ...")
                _cap.release()
                time.sleep(2.0)
                _cap = cv2.VideoCapture(DROIDCAM_URL)
                consecutive_failures = 0

        elapsed = time.monotonic() - t0
        time.sleep(max(0.0, interval - elapsed))


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
  <h2>Camera Bridge — live preview</h2>
  <img src="/video" alt="camera feed" />
  <p>If you can see the camera here, the Cloudflare tunnel URL is working correctly.</p>
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
        if DROIDCAM_URL:
            print(f"ERROR: Cannot connect to DroidCam at {DROIDCAM_URL}")
            print("Check that:")
            print("  1. DroidCam app is running on your phone")
            print("  2. Phone and PC are on the same Wi-Fi")
            print("  3. The IP address in DROIDCAM_URL matches what the app shows")
        else:
            print(f"ERROR: Cannot open camera {CAMERA_INDEX}.")
            print("Try a different index: set CAMERA_INDEX=1 (or 2) before running.")
        raise SystemExit(1)

    w = int(_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Source  : {_mode}")
    print(f"Size    : {w}x{h}  |  Target FPS: {TARGET_FPS}  |  JPEG quality: {JPEG_QUALITY}")
    print()
    print(f"Bridge  : http://localhost:{PORT}")
    print(f"Stream  : http://localhost:{PORT}/video")
    print()
    print("Next — expose this port via Cloudflare (no account needed):")
    print("  cloudflared tunnel --url http://localhost:8080")
    print()
    print("Then set CAMERA_SOURCE in Colab Cell B3 to:")
    print("  https://YOUR-ID.trycloudflare.com/video")

    t = threading.Thread(target=_capture_loop, daemon=True)
    t.start()

    try:
        app.run(host="0.0.0.0", port=PORT, threaded=True)
    finally:
        _running = False
        _cap.release()
