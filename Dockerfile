# ── Stage 1: dependencies ──────────────────────────────────────────────────────
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# System libs required by OpenCV and EasyOCR
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        libgomp1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Stage 2: application ───────────────────────────────────────────────────────
FROM base AS app

COPY . .

# Ensure runtime directories exist (DB, snapshots, artifacts)
RUN mkdir -p static/snapshots artifacts/frames artifacts/vehicles \
             artifacts/plates artifacts/invoices artifacts/reports \
             Videos

# Pre-download YOLOv8n weights so the container doesn't reach out at runtime.
# This makes the image larger but the app starts instantly offline.
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" || true

# Default environment — override at runtime via Railway/Render/Fly env vars or .env
ENV FLASK_ENV=production \
    CAMERA_SOURCE=0 \
    LOCATION_NAME="Crosswalk A" \
    LOCATION_CODE="CW-A-01" \
    LOCATION_LATITUDE=41.2963 \
    LOCATION_LONGITUDE=69.2798 \
    DEFAULT_FINE_AMOUNT=150000 \
    AUTHORITY_NAME="WIUT Traffic Enforcement Unit" \
    # 1 worker + 4 threads fits in 512 MB RAM; scale up on larger instances
    WEB_CONCURRENCY=1

EXPOSE 5000

# Gunicorn: 1 worker (enough for demo), 4 threads for concurrent requests,
# 120 s timeout for slow CV operations.
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-5000} --workers ${WEB_CONCURRENCY:-1} --threads 4 --timeout 120 app:app"]
