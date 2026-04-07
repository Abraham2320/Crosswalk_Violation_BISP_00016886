# ── Stage 1: build ────────────────────────────────────────────────────────────
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

# ── Stage 2: app ──────────────────────────────────────────────────────────────
FROM base AS app

COPY . .

# Ensure runtime directories exist
RUN mkdir -p static/snapshots artifacts/frames artifacts/vehicles \
             artifacts/plates artifacts/invoices artifacts/reports

# Default environment (override at runtime via -e or .env)
ENV FLASK_ENV=production \
    CAMERA_SOURCE=0 \
    LOCATION_NAME="Crosswalk A" \
    LOCATION_CODE="CW-A-01" \
    LOCATION_LATITUDE=41.2963 \
    LOCATION_LONGITUDE=69.2798 \
    DEFAULT_FINE_AMOUNT=150000 \
    AUTHORITY_NAME="WIUT Traffic Enforcement Unit"

EXPOSE 5000

# Gunicorn for production; falls back to Flask dev server if gunicorn missing
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:5000 --workers 2 --threads 4 --timeout 120 app:app 2>/dev/null || python app.py"]
