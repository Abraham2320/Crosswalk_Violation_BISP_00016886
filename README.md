# Crosswalk Violation Enforcement System

Production-ready extension of the existing YOLOv8 crosswalk prototype into a modular enforcement platform for smart-city and SaaS deployments.

## Architecture

```text
Camera Stream
  -> Detection Service (YOLOv8 + tracking + FSM trigger)
  -> Evidence Capture
  -> ALPR (vehicle crop -> plate detector -> plate crop -> OCR)
  -> Reporting / Invoice
  -> API / Dashboard
  -> PostgreSQL
```

## Project Structure

```text
alembic/                     Alembic migrations
src/
  alpr/                      License plate detection on vehicle crops
  api/
    main.py                  FastAPI entrypoint
    dependencies.py          DI wiring
    models/                  Pydantic response/request models
    routes/                  Violations, vehicles, analytics endpoints
    services/                API orchestration layer
  capture/                   Frame and vehicle evidence capture
  detector/                  Existing YOLOv8 detector and tracker
  geometry/                  Existing crosswalk polygon logic
  logic/                     Existing FSM violation logic with trigger events
  OCR/                       OCR preprocessing and recognition
  reporting/                 LLM report generation and invoice output
  services/                  Async enforcement pipeline
  storage/                   SQLAlchemy/sqlite repository layer
  vision/                    Existing drawing/FPS helpers
run_api.py                   API launcher
run_system.py                Detection launcher
docker-compose.yml           API + PostgreSQL stack
Dockerfile                   Container image
```

## Core Behavior

1. Real-time detection runs through the existing `src/main.py` loop.
2. A violation event is created only on `FALSE -> TRUE`.
3. Evidence is saved once per violation: full frame, vehicle crop, timestamp, UUID.
4. ALPR runs only on the vehicle crop.
5. OCR applies grayscale, contrast equalization, thresholding, text cleanup, and regex validation.
6. Violations, vehicles, and invoices are stored in the database.
7. FastAPI exposes violations, vehicles, and analytics.
8. The reporting module generates structured JSON plus formatted legal text and an invoice artifact.

## Database

Primary deployment target:

- PostgreSQL via SQLAlchemy
- Alembic migrations included

Runtime compatibility:

- If SQLAlchemy is unavailable in the local detection environment, the detector still runs with an internal SQLite fallback so the CV loop is not blocked.

Tables:

- `violations`
- `vehicles`
- `invoices`

## Environment

Copy `.env.example` and adjust values as needed.

Important variables:

```bash
DATABASE_URL=postgresql+psycopg://crosswalk:crosswalk@db:5432/crosswalk
SQLITE_FALLBACK_URL=sqlite:///crosswalk_violations.db
VIDEO_PATH=Videos/v2.mp4
DETECTION_MODEL_PATH=yolov8n.pt
PLATE_MODEL_PATH=models/license_plate.pt
OCR_BACKEND=easyocr
LLM_PROVIDER=mock
OPENAI_MODEL=gpt-5.4-mini
AUTHORITY_NAME=City Traffic Enforcement Unit
LOCATION_NAME=Crosswalk A
LOCATION_CODE=CW-A-01
DEFAULT_FINE_AMOUNT=150.00
```

## Local Run

Detection runtime:

```powershell
CV_venv\Scripts\python.exe src/main.py
```

API runtime:

```powershell
CV_venv\Scripts\python.exe -m uvicorn api.main:app --app-dir src --reload
```

## Docker Run

Start PostgreSQL and API:

```bash
docker compose up --build
```

API will be available at `http://localhost:8000`.

## Alembic

Run migrations:

```bash
alembic upgrade head
```

Create a new migration:

```bash
alembic revision --autogenerate -m "describe change"
```

## API Endpoints

- `GET /violations`
- `GET /violations/{id}`
- `POST /violations`
- `GET /vehicles/{plate}`
- `GET /analytics`
- `GET /health`

## Testing

Executed and passing in `CV_venv`:

```powershell
CV_venv\Scripts\python.exe -m unittest discover -s tests -v
```

Coverage includes:

- violation trigger transition behavior
- OCR cleaning and validation
- full pipeline persistence path

## Notes

- The detector entrypoint was extended, not rewritten.
- ALPR no longer falls back to the full frame.
- `reportlab` remains optional at runtime for the detector; when absent, invoice generation falls back to text output so live enforcement does not crash.
