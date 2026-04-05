# Crosswalk Violation Detection System

A real-time computer vision system that detects vehicles violating pedestrian right-of-way at crosswalks using YOLOv8 object detection and ByteTrack multi-object tracking. Violations are logged to SQLite, cropped evidence frames are saved automatically, and an AI-powered analytics dashboard provides traffic safety insights.

---

## Project Structure

```
Crosswalk_Violation/
├── run_system.py              # Main entry point (video detection + chatbot)
├── run_api.py                 # FastAPI REST API entry point
├── dashboard.py               # Streamlit analytics dashboard
├── generate_demo_data.py      # Seed database with 200 demo violations
├── bytetrack.yaml             # Custom ByteTrack tracker configuration
├── crosswalk_polygon.json     # Saved crosswalk polygon (auto-created on first run)
├── crosswalk_violations.db    # SQLite database (auto-created)
├── requirements.txt
├── artifacts/
│   ├── frames/                # Full violation frame captures
│   ├── vehicles/              # Vehicle crop images
│   ├── plates/                # Plate crop images
│   ├── invoices/              # Generated invoice files (.txt / .pdf)
│   └── reports/               # JSON violation reports
└── src/
    ├── main.py                # Core detection loop
    ├── config.py              # All settings (env-var overrideable)
    ├── schemas.py             # Shared dataclasses
    ├── chatbot.py             # Terminal AI chatbot
    ├── alpr/                  # License plate detector
    ├── api/                   # FastAPI routes, models, services
    ├── capture/               # Evidence frame/crop capture
    ├── detector/              # YOLOv8 wrapper + ByteTrack + NMS + ID merger
    ├── geometry/              # Crosswalk polygon zone logic
    ├── logic/                 # Violation FSM (enter/inside/exit states)
    ├── OCR/                   # EasyOCR plate recognition engine
    ├── reporting/             # Invoice generator + LLM report service
    ├── services/              # Async enforcement pipeline
    ├── storage/               # SQLite / SQLAlchemy repository
    └── vision/                # Frame drawing, FPS counter, stabiliser
```

---

## Installation

**Requirements:** Python 3.9+

```bash
# 1. Clone / unzip the project
cd Crosswalk_Violation

# 2. Create and activate a virtual environment
python -m venv CV_venv
# Windows:
CV_venv\Scripts\activate
# macOS / Linux:
source CV_venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

> **GPU note:** YOLOv8 and EasyOCR run on CPU automatically if no CUDA GPU is detected, but processing speed will be significantly slower (see Known Limitations).

---

## Quick Start: Seed Demo Data

If you don't have a video file ready, seed the database with 200 realistic sample violations so the dashboard and chatbot work immediately:

```bash
python generate_demo_data.py
```

---

## Running the Violation Detector

```bash
python run_system.py --video path/to/video.mp4
```

Or set the path via environment variable and run without the flag:

```bash
set VIDEO_PATH=Videos/v2.mp4   # Windows
python run_system.py
```

### Drawing the Crosswalk Polygon

On the **first run** (or when `crosswalk_polygon.json` does not exist) the system enters **calibration mode**:

1. A window opens showing the first frame of the video.
2. **Left-click** to place polygon corner points around the crosswalk area (4+ points recommended, going clockwise from top-left).
3. **Right-click** to finalise the polygon and begin detection.

The polygon is saved to `crosswalk_polygon.json` and reloaded automatically on every subsequent run. To redefine the polygon, delete `crosswalk_polygon.json` before starting.

### Keyboard Controls

| Key | Action |
|-----|--------|
| `Esc` | Stop detection and exit |

### Optional flags

```bash
python run_system.py --video path/to/video.mp4   # override video path
python run_system.py --no-stabilize              # disable ORB video stabilisation
python run_system.py --chatbot                   # launch AI chatbot instead
```

---

## Running the Analytics Dashboard

```bash
streamlit run dashboard.py
```

Open `http://localhost:8501` in your browser.

**Dashboard sections:**
- KPI cards — total violations, unique vehicles, peak hour, plate recognition rate
- Violations by hour — line chart
- Top offending vehicles — bar chart
- Day × hour heatmap
- Paginated violation log table
- Sidebar filters — date range, confidence threshold, plate filter
- **Generate AI Summary** button — calls Claude API for traffic pattern analysis (requires `ANTHROPIC_API_KEY`)

```bash
# Set the API key before launching (Windows):
set ANTHROPIC_API_KEY=sk-ant-...
streamlit run dashboard.py
```

---

## Running the AI Chatbot

```bash
python run_system.py --chatbot
```

The chatbot loads summary statistics from the database and enters an interactive loop. Ask questions like:

- *"Which vehicle violated the most?"*
- *"What time of day has the highest violation rate?"*
- *"Suggest measures to reduce violations at peak hours."*

Type `quit` or `exit` to leave.

```bash
# Requires API key:
set ANTHROPIC_API_KEY=sk-ant-...
python run_system.py --chatbot
```

---

## Running the REST API (optional)

```bash
python -m uvicorn api.main:app --app-dir src --reload
```

Endpoints:

| Method | Path | Description |
|--------|------|-------------|
| GET | `/violations` | List violations (paginated) |
| GET | `/violations/{id}` | Single violation |
| POST | `/violations` | Create violation record |
| GET | `/vehicles/{plate}` | Vehicle history by plate |
| GET | `/analytics` | Aggregated stats |
| GET | `/health` | Health check |

---

## Environment Variables

All settings have sensible defaults. Override via environment variables or a `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `VIDEO_PATH` | `Videos/v2.mp4` | Input video file |
| `DETECTION_MODEL_PATH` | `yolov8n.pt` | YOLOv8 model weights |
| `DETECTION_CONFIDENCE` | `0.35` | Minimum detection confidence |
| `OCR_BACKEND` | `easyocr` | OCR engine (`easyocr` or `paddleocr`) |
| `OCR_CONFIDENCE_THRESHOLD` | `0.35` | Minimum OCR confidence to accept plate |
| `PLATE_REGEX` | `^[A-Z0-9]{5,10}$` | Regex to validate plate strings |
| `SQLITE_FALLBACK_URL` | `sqlite:///crosswalk_violations.db` | SQLite database path |
| `LOCATION_NAME` | `Crosswalk A` | Location label in reports |
| `AUTHORITY_NAME` | `City Traffic Enforcement Unit` | Issuing authority name |
| `DEFAULT_FINE_AMOUNT` | `150.00` | Fine amount in USD |
| `ANTHROPIC_API_KEY` | *(none)* | Required for dashboard AI summary and chatbot |

---

## How Detection Works

```
Video frame
  └─ YOLOv8 detects: person, car, bus, truck, motorbike
      └─ ByteTrack assigns persistent IDs
          └─ Cross-class NMS removes duplicate vehicle boxes (IoU > 0.5)
              └─ ID Merger resolves split-ID bug (centroids < 40 px for 3+ frames)
                  └─ FSM tracks each object: OUTSIDE → ENTER → INSIDE → EXIT
                      └─ Violation triggered when vehicle ENTERS while pedestrian is inside
                          └─ Evidence capture: full frame + vehicle crop saved
                              └─ ALPR: Haar cascade → EasyOCR → regex validation
                                  └─ Report + invoice generated → SQLite record saved
```

Video stabilisation (ORB + RANSAC homography) keeps the crosswalk polygon locked to the road even when the camera shifts.

---

## Known Limitations

### Camera Stability
The ORB-based video stabiliser requires enough texture in the scene to extract keypoints. In low-contrast or night environments fewer than 8 inliers may be found, causing the stabiliser to fall back to the previous frame's transform. A "Unstable" overlay appears on affected frames.

### Lighting Conditions
EasyOCR plate recognition degrades significantly in poor lighting, strong glare, or partial occlusion. Night-time plates are often unreadable without infrared illumination. Confidence thresholds can be lowered via `OCR_CONFIDENCE_THRESHOLD` but this increases false positives.

### GPU Requirement for Real-Time Speed
Without a CUDA GPU, YOLOv8 inference runs at approximately 5–10 FPS on modern CPUs (vs 30+ FPS on GPU). EasyOCR model loading adds ~10 seconds on first startup. Use `yolov8n.pt` (nano) for the fastest CPU performance at the cost of detection accuracy.

### ByteTrack ID Switching
ByteTrack occasionally reassigns IDs when objects are occluded for more than 2 seconds (60 frames at `track_buffer=60`). The centroid-proximity ID merger (`IDMerger`) recovers most cases but cannot handle extreme occlusion or very fast vehicles.

### Single Camera, Fixed Angle
The system assumes a single static or near-static camera. Pan/tilt/zoom cameras are not supported. Wide-angle distortion is not corrected.
