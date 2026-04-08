# Crosswalk Violation Detection System

> **Real-time AI-powered pedestrian right-of-way enforcement**  
> Tashkent, Uzbekistan Â· April 2026  
> Authority: **WIUT Traffic Enforcement Unit**

A production-grade computer vision system that automatically detects and records vehicles that fail to yield to pedestrians at marked crosswalks. The system uses **YOLOv8** object detection combined with **ByteTrack** multi-object tracking, a spatial Finite State Machine (FSM) for violation logic, automatic license plate recognition (ALPR), and a full enforcement pipeline that issues invoices, generates reports, and stores everything in a database. Two web interfaces â€” a Flask admin panel and a Streamlit analytics dashboard â€” provide real-time monitoring and historical analysis.

---

## Table of Contents

1. [Live Violation Screenshots](#live-violation-screenshots)
2. [System Overview](#system-overview)
3. [Full Project Structure](#full-project-structure)
4. [Technology Stack](#technology-stack)
5. [Architecture & Pipeline](#architecture--pipeline)
6. [Module Reference](#module-reference)
7. [Database Schema](#database-schema)
8. [Installation](#installation)
9. [Configuration (All Environment Variables)](#configuration-all-environment-variables)
10. [Running the System](#running-the-system)
11. [Flask Admin Panel](#flask-admin-panel)
12. [Streamlit Analytics Dashboard](#streamlit-analytics-dashboard)
13. [REST API](#rest-api)
14. [AI Chatbot](#ai-chatbot)
15. [Deployment](#deployment)
16. [Performance](#performance)
17. [Known Errors & Limitations](#known-errors--limitations)
18. [Tests](#tests)
19. [Evidence Artifacts](#evidence-artifacts)
20. [Demo Data](#demo-data)

---

## Live Violation Screenshots

The following screenshots are **real captures** from the live detection system running on Amir Temur Avenue, Tashkent (April 7, 2026). Each image is annotated automatically at the moment of violation trigger.

### Screenshot 1 â€” Vehicle 32 Â· Frame 103 Â· 2026-04-07 21:01:29

![Violation snapshot 1](static/snapshots/snapshot_184a6ece-78e4-4604-8f82-c63d42aa7c1a_103.jpg)

**What the overlay shows:**
| Element | Colour | Meaning |
|---------|--------|---------|
| Red semi-transparent banner (top) | Red | VIOLATION DETECTED header; shows type, vehicle ID, plate, timestamp |
| **Bold red rectangle** (vehicle) | Red, 3 px | Offending vehicle â€” a dark SUV (Vehicle ID 32) that entered the crosswalk |
| **Blue rectangle** (person) | Blue | Pedestrian currently crossing ("CLEARING" state â€” moving UP toward exit zone) |
| **Amber polygon** | Amber / orange | Crosswalk zone boundary drawn during calibration |
| **Green overlay** (lower half) | Green, 15 % fill | Lower sub-zone of the crosswalk (below the 32 % split line) |
| **Blue overlay** (upper half) | Blue, 15 % fill | Upper sub-zone of the crosswalk |
| "Stabilised" label (top-right) | Green | ORB video stabiliser running correctly; homography accepted |
| `P:1 V:4` counter (top-left) | Cyan | 1 pedestrian tracked, 4 vehicles tracked in this frame |
| `OUTSIDE` labels (right vehicles) | White | Other vehicles â€” outside the crosswalk zone, no violation |

---

### Screenshot 2 â€” Vehicle 110 Â· Frame 512 Â· 2026-04-07 21:02:58

![Violation snapshot 2](static/snapshots/snapshot_5e7fc750-7d1e-43f5-afc7-1025311d5596_512.jpg)

**What the overlay shows:** Same scene ~90 seconds later. Vehicle ID has changed to 110 (ByteTrack re-issued ID after a brief occlusion). The pedestrian is in "CROSSING" state, already past the midline. The dark SUV has entered the crosswalk from the left side while the pedestrian still has right-of-way. Three vehicles to the right are correctly labelled OUTSIDE.

---

### Screenshot 3 â€” Vehicle 186 Â· Frame 988 Â· 2026-04-07 20:48:11

![Violation snapshot 3](static/snapshots/snapshot_86b77f71-8fa8-4e15-9dac-adf2ad9667c6_988.jpg)

**What the overlay shows:** The offending vehicle is further into the crosswalk zone. The pedestrian is labelled "CROSSING / UP". The crosswalk polygon correctly tracks the zebra-crossing road markings. Other vehicles remain in the OUTSIDE state onscreen right.

---

### Annotation Legend (All Screenshots)

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  RED BANNER  VIOLATION DETECTED                                  â”‚
â”‚              Type: FAILED_TO_YIELD                               â”‚
â”‚              Vehicle ID: <id>                                    â”‚
â”‚              Plate: UNDETECTED (or plate number)                 â”‚
â”‚              Time: YYYY-MM-DD HH:MM:SS                           â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚  AMBER outline  â† Crosswalk polygon zone (user-defined)          â”‚
â”‚  BLUE fill      â† Upper sub-zone (32% split)                     â”‚
â”‚  GREEN fill     â† Lower sub-zone                                 â”‚
â”‚  CYAN split lineâ† Midline dividing upper / lower zones           â”‚
â”‚  RED box (3px)  â† Offending vehicle bounding box                 â”‚
â”‚  BLUE box       â† Person bounding box                            â”‚
â”‚  GREEN box      â† Other vehicles (outside zone)                  â”‚
â”‚  "UP"/"DOWN"    â† Pedestrian direction of travel                 â”‚
â”‚  "CROSSING"     â† Pedestrian FSM state                           â”‚
â”‚  "CLEARING"     â† Pedestrian about to exit the zone              â”‚
â”‚  "OUTSIDE"      â† Vehicle/pedestrian not in the crosswalk        â”‚
â”‚  "Stabilised"   â† ORB homography accepted this frame             â”‚
â”‚  "Unstable"     â† Stabiliser fell back to previous transform     â”‚
â”‚  P:N V:N        â† Live count of tracked pedestrians / vehicles   â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

---

## System Overview

```
Camera / Video File
       â”‚
       â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  Video Stabiliser (ORB + RANSAC homography)                  â”‚
â”‚  Keeps polygon locked on road despite camera vibration       â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
           â”‚ stabilised frame
           â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  YOLOv8 Object Detector (yolov8n.pt or yolov8x.pt)           â”‚
â”‚  Classes: person(0) car(2) motorcycle(3) bus(5) truck(7)     â”‚
â”‚  Confidence â‰¥ 0.35 Â· Image size: 960 px                      â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
           â”‚ raw [bbox, class, conf]
           â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  ByteTrack Multi-Object Tracker (bytetrack.yaml)             â”‚
â”‚  Assigns persistent integer IDs across frames                â”‚
â”‚  track_buffer=60 frames Â· match_thresh=0.85                  â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
           â”‚ [bbox, class, id, conf]
           â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  Cross-Class NMS                                             â”‚
â”‚  Removes duplicate vehicle boxes (IoU > 0.50)                â”‚
â”‚     â†“                                                        â”‚
â”‚  IDMerger                                                    â”‚
â”‚  Merges split IDs when centroids < 40 px for â‰¥ 3 frames      â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
           â”‚ clean [bbox, class, id, conf]
           â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  CrosswalkZone (geometry/crosswalk.py)                       â”‚
â”‚  Polygon intersection test for each bounding box             â”‚
â”‚  Splits polygon into upper/lower sub-zones (ratio=0.32)      â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
           â”‚ zone membership (inside/outside + sub-zone)
           â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  Violation FSM (logic/violation.py)                          â”‚
â”‚  PedestrianTrack FSM: OUTSIDEâ†’ENTERINGâ†’CROSSINGâ†’CLEARING     â”‚
â”‚  VehicleTrack FSM:    OUTSIDEâ†’ENTERâ†’INSIDEâ†’EXIT              â”‚
â”‚  Trigger: vehicle ENTERS while pedestrian is CROSSING        â”‚
â”‚  Conditions: same zone, direction, approach axis             â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
           â”‚ ViolationEvent (UUID, timestamp, bbox, plate, â€¦)
           â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  EnforcementPipeline (services/pipeline.py)  [ThreadPoolÃ—2] â”‚
â”‚  â‘  EvidenceBuilder   â†’ full frame + vehicle crop saved       â”‚
â”‚  â‘¡ LicensePlateDetector  (YOLO model or Haar cascade)        â”‚
â”‚  â‘¢ OCREngine (EasyOCR)  â†’ plate text + regex validation      â”‚
â”‚  â‘£ LLMReportService  â†’ structured JSON report                â”‚
â”‚  â‘¤ InvoiceGenerator  â†’ .pdf (ReportLab) or .txt fallback     â”‚
â”‚  â‘¥ ViolationRepository  â†’ SQLite / PostgreSQL record         â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
           â”‚
    â”Œâ”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
    â–¼                         â–¼
Flask Admin Panel         Streamlit Dashboard
app.py  (port 5000)       dashboard.py (port 8501)
```

---

## Full Project Structure

```
Crosswalk_Violation/
â”‚
â”œâ”€â”€ app.py                        # Flask admin panel + violator portal
â”œâ”€â”€ auth.py                       # Login, session lockout, login_required decorator
â”œâ”€â”€ database.py                   # Flask-side SQLite helpers, admin_users, audit_log
â”œâ”€â”€ live_processor.py             # Background YOLO pipeline for live camera feed
â”œâ”€â”€ stream.py                     # Thread-safe MJPEG camera stream manager
â”œâ”€â”€ dashboard.py                  # Streamlit analytics dashboard
â”œâ”€â”€ run_system.py                 # CLI entry point (video file detector + chatbot)
â”œâ”€â”€ run_api.py                    # FastAPI REST API entry point
â”œâ”€â”€ generate_demo_data.py         # Seed 100â€“200 demo violations + snapshot images
â”‚
â”œâ”€â”€ bytetrack.yaml                # Custom ByteTrack tuning config
â”œâ”€â”€ crosswalk_polygon.json        # Default camera polygon (cam2 / demo)
â”œâ”€â”€ crosswalk_polygon_cam1.json   # Per-camera polygon for cam1
â”œâ”€â”€ crosswalk_polygon_cam3.json   # Per-camera polygon for cam3
â”‚
â”œâ”€â”€ requirements.txt              # Python dependencies
â”œâ”€â”€ Dockerfile                    # Multi-stage Docker image
â”œâ”€â”€ docker-compose.yml            # Web + dashboard services
â”œâ”€â”€ Procfile                      # Render/Heroku deployment
â”œâ”€â”€ render.yaml                   # Render.com IaC config
â”œâ”€â”€ alembic.ini                   # Alembic migration config
â”œâ”€â”€ alembic/
â”‚   â”œâ”€â”€ env.py
â”‚   â”œâ”€â”€ script.py.mako
â”‚   â””â”€â”€ versions/
â”‚       â””â”€â”€ 20260319_000001_initial_schema.py
â”‚
â”œâ”€â”€ models/
â”‚   â”œâ”€â”€ plate_detector.pt         # Custom YOLOv8 plate detection model
â”‚   â””â”€â”€ README.md
â”œâ”€â”€ yolov8n.pt                    # YOLOv8 nano (fast, CPU-friendly)
â”œâ”€â”€ yolov8x.pt                    # YOLOv8 extra-large (highest accuracy)
â”‚
â”œâ”€â”€ artifacts/
â”‚   â”œâ”€â”€ frames/                   # Full frame at moment of violation
â”‚   â”œâ”€â”€ vehicles/                 # Cropped vehicle image
â”‚   â”œâ”€â”€ plates/                   # Cropped plate image
â”‚   â”œâ”€â”€ invoices/                 # Issued invoices (.txt or .pdf)
â”‚   â””â”€â”€ reports/                  # JSON violation + LLM report files
â”‚
â”œâ”€â”€ static/
â”‚   â”œâ”€â”€ css/                      # Admin panel stylesheets
â”‚   â”œâ”€â”€ js/                       # Admin panel JavaScript
â”‚   â””â”€â”€ snapshots/                # Annotated violation snapshot images (served by Flask)
â”‚
â”œâ”€â”€ templates/
â”‚   â”œâ”€â”€ base.html                 # Shared admin layout
â”‚   â”œâ”€â”€ admin/
â”‚   â”‚   â”œâ”€â”€ login.html            # Admin login page
â”‚   â”‚   â”œâ”€â”€ dashboard.html        # Admin home (KPIs, charts)
â”‚   â”‚   â”œâ”€â”€ violations.html       # Paginated violation list
â”‚   â”‚   â”œâ”€â”€ violation_detail.html # Single violation + map + evidence
â”‚   â”‚   â”œâ”€â”€ vehicles.html         # Vehicle registry
â”‚   â”‚   â”œâ”€â”€ cameras.html          # Camera management
â”‚   â”‚   â”œâ”€â”€ camera_detail.html    # Live stream + per-camera stats
â”‚   â”‚   â”œâ”€â”€ live.html             # Live MJPEG stream view
â”‚   â”‚   â”œâ”€â”€ analytics.html        # Charts and AI summary
â”‚   â”‚   â”œâ”€â”€ audit.html            # Admin audit log
â”‚   â”‚   â””â”€â”€ invoice_view.html     # Invoice viewer
â”‚   â””â”€â”€ portal/
â”‚       â”œâ”€â”€ index.html            # Public violator lookup
â”‚       â””â”€â”€ results.html          # Violation results for a plate
â”‚
â”œâ”€â”€ src/
â”‚   â”œâ”€â”€ __init__.py
â”‚   â”œâ”€â”€ main.py                   # Core video detection loop
â”‚   â”œâ”€â”€ config.py                 # AppSettings (ModelSettings, RuntimeSettings, StorageSettings)
â”‚   â”œâ”€â”€ schemas.py                # Shared dataclasses (ViolationEvent, EvidenceBundle, OCRResultâ€¦)
â”‚   â”œâ”€â”€ chatbot.py                # Claude-powered terminal chatbot
â”‚   â”‚
â”‚   â”œâ”€â”€ alpr/
â”‚   â”‚   â””â”€â”€ detector.py           # LicensePlateDetector (YOLO â†’ Haar cascade fallback)
â”‚   â”‚
â”‚   â”œâ”€â”€ api/
â”‚   â”‚   â”œâ”€â”€ main.py               # FastAPI app factory
â”‚   â”‚   â”œâ”€â”€ app.py                # FastAPI instance
â”‚   â”‚   â”œâ”€â”€ dependencies.py       # DB session dependency injection
â”‚   â”‚   â”œâ”€â”€ models/               # Pydantic request/response schemas
â”‚   â”‚   â”œâ”€â”€ routes/               # violations, vehicles, analytics, health
â”‚   â”‚   â””â”€â”€ services/             # Business logic for API layer
â”‚   â”‚
â”‚   â”œâ”€â”€ capture/
â”‚   â”‚   â””â”€â”€ service.py            # EvidenceBuilder (save frame + vehicle crop + metadata)
â”‚   â”‚
â”‚   â”œâ”€â”€ detector/
â”‚   â”‚   â”œâ”€â”€ yolo_detector.py      # YOLODetector (YOLO.track wrapper)
â”‚   â”‚   â””â”€â”€ tracker.py            # IDMerger, apply_cross_class_nms, PedestrianTrack, VehicleTrack, ObjectFSM
â”‚   â”‚
â”‚   â”œâ”€â”€ geometry/
â”‚   â”‚   â”œâ”€â”€ crosswalk.py          # CrosswalkZone (polygon draw, intersects_box, split polygons)
â”‚   â”‚   â””â”€â”€ polygon_editor.py     # Mouse-click polygon calibration tool
â”‚   â”‚
â”‚   â”œâ”€â”€ logic/
â”‚   â”‚   â””â”€â”€ violation.py          # Violation FSM, check_violation, compute_approach_axis
â”‚   â”‚
â”‚   â”œâ”€â”€ OCR/
â”‚   â”‚   â””â”€â”€ engine.py             # OCREngine (EasyOCR or PaddleOCR, preprocessing, regex validation)
â”‚   â”‚
â”‚   â”œâ”€â”€ reporting/
â”‚   â”‚   â”œâ”€â”€ invoice.py            # InvoiceGenerator (ReportLab PDF or .txt fallback)
â”‚   â”‚   â””â”€â”€ llm_service.py        # LLMReportService (OpenAI or mock structured report)
â”‚   â”‚
â”‚   â”œâ”€â”€ services/
â”‚   â”‚   â””â”€â”€ pipeline.py           # EnforcementPipeline (async ThreadPoolExecutor orchestrator)
â”‚   â”‚
â”‚   â”œâ”€â”€ storage/
â”‚   â”‚   â”œâ”€â”€ database.py           # SQLAlchemy ORM models + ViolationRepository
â”‚   â”‚   â””â”€â”€ csv_writer.py         # CSV export helper
â”‚   â”‚
â”‚   â”œâ”€â”€ ui/                       # (reserved for future UI helpers)
â”‚   â”‚
â”‚   â””â”€â”€ vision/
â”‚       â”œâ”€â”€ draw.py               # draw_box helper
â”‚       â”œâ”€â”€ fps.py                # FPS counter
â”‚       â””â”€â”€ stabilizer.py         # VideoStabilizer (ORB + RANSAC homography)
â”‚
â”œâ”€â”€ tests/
â”‚   â”œâ”€â”€ conftest.py
â”‚   â”œâ”€â”€ test_violation_logic.py         # Unit tests: FSM violation trigger
â”‚   â”œâ”€â”€ test_ocr_engine.py              # Unit tests: OCR pipeline
â”‚   â””â”€â”€ test_pipeline_integration.py   # Integration test: full enforcement pipeline
â”‚
â””â”€â”€ Videos/                       # Video files for testing (not committed to git)
```

---

## Technology Stack

### Computer Vision & AI

| Library | Version | Purpose |
|---------|---------|---------|
| **Ultralytics YOLOv8** | `>=8.2.0` | Object detection (people, cars, trucks, buses, motorcycles) + ByteTrack tracking |
| **OpenCV** (`opencv-python`) | `>=4.9.0` | Frame capture, ORB keypoints, homography warp, drawing, JPEG encode/decode |
| **NumPy** | `>=1.26.0` | Array operations throughout pipeline |
| **EasyOCR** | `>=1.7.1` | License plate text recognition (CPU-supported; GPU optional) |
| **supervision** | `>=0.21.0` | ByteTrack/tracker utilities |
| **lap** | `>=0.5.12` | Linear Assignment Problem solver (required by ByteTrack) |

**Models used:**
- `yolov8n.pt` â€” YOLOv8 Nano (3.2M params, fastest on CPU, ~5â€“10 FPS CPU)
- `yolov8x.pt` â€” YOLOv8 XLarge (68.2M params, highest accuracy, requires GPU for real-time)
- `models/plate_detector.pt` â€” Custom-trained YOLOv8 model for license plate localisation

### Web Framework & API

| Library | Version | Purpose |
|---------|---------|---------|
| **Flask** | `>=2.3.0` | Admin panel + violator portal web application |
| **Werkzeug** | `>=2.3.0` | Password hashing (`werkzeug.security`), WSGI utilities |
| **FastAPI** | `>=0.115.0` | Optional REST API server (`run_api.py`) |
| **Uvicorn** | `>=0.30.0` | ASGI server for FastAPI |
| **Pydantic** | `>=2.7.0` | Request/response validation for FastAPI |
| **Gunicorn** | `>=22.0.0` | Production WSGI server (2 workers, 4 threads) |

### Database & Storage

| Library | Version | Purpose |
|---------|---------|---------|
| **SQLAlchemy** | `>=2.0.0` | ORM for violations, vehicles, invoices tables |
| **Alembic** | `>=1.13.0` | Database migration management |
| **psycopg** | `>=3.1.0` | PostgreSQL driver (primary; SQLite used as fallback) |
| **SQLite** | built-in | Default local database (`crosswalk_violations.db`) |

### Analytics & AI

| Library | Version | Purpose |
|---------|---------|---------|
| **Streamlit** | `>=1.35.0` | Analytics dashboard web app |
| **Pandas** | `>=2.1.0` | Data manipulation for dashboard charts |
| **Anthropic Claude** | `>=0.28.0` | AI-generated traffic summary (dashboard) and chatbot |
| **OpenAI** | optional | Alternative LLM provider for report generation |

### Reporting

| Library | Version | Purpose |
|---------|---------|---------|
| **ReportLab** | `>=4.0.0` | PDF invoice generation; falls back to `.txt` if not installed |
| **Pillow** | `>=10.0.0` | Image utilities for snapshot post-processing |

### Infrastructure

| Tool | Purpose |
|------|---------|
| **Docker** | Container image (`python:3.12-slim` base) |
| **docker-compose** | Runs Flask (port 5000) + Streamlit (port 8501) together |
| **Render.com** | Cloud deployment (`render.yaml`) |
| **python-dotenv** | `.env` file support for all environment variables |

---

## Architecture & Pipeline

### Detection Loop (per frame)

```
1. Read frame from cv2.VideoCapture
2. VideoStabilizer.stabilize(frame)
   â””â”€ ORB features â†’ BFMatcher â†’ Lowe ratio test â†’ RANSAC homography
   â””â”€ warpPerspective (inverse H) â†’ stabilised frame
   â””â”€ Fallback: return original frame if < 12 RANSAC inliers

3. YOLODetector.detect(frame)
   â””â”€ YOLO.track(frame, persist=True, tracker=bytetrack.yaml)
   â””â”€ classes=[0,2,3,5,7]  conf=0.35  imgsz=960

4. apply_cross_class_nms(boxes, classes, ids, confs, iou=0.50)
   â””â”€ For every vehicle pair: if IoU > 0.50 keep higher-confidence, drop other

5. IDMerger.update(ids, boxes)
   â””â”€ For each pair of IDs: if centroid distance < 40 px for â‰¥ 3 frames
   â””â”€ Remap higher ID â†’ lower ID (permanent in-session)

6. For each detection:
   â”œâ”€ Pedestrian (class 0):
   â”‚   â””â”€ PedestrianTrack: update centroid, velocity_history (deque 10)
   â”‚   â””â”€ FSM: OUTSIDE â†’ ENTERING â†’ CROSSING â†’ CLEARING â†’ EXITED
   â”‚   â””â”€ CrosswalkZone.intersects_box(bbox, min_ratio=0.02)
   â”‚
   â””â”€ Vehicle (class 2,3,5,7):
       â””â”€ VehicleTrack: update centroid, velocity_history (deque 20)
       â””â”€ compute_approach_axis (from_top/bottom/left/right)
       â””â”€ CrosswalkZone.intersects_box â†’ inside/outside
       â””â”€ ObjectFSM: OUTSIDE â†’ ENTER â†’ INSIDE â†’ EXIT

7. check_violation(car_id, ped_id, car_state, ped_state, â€¦)
   â””â”€ Trigger if: car_state == ENTER AND ped_state in (CROSSING, CLEARING)
   â””â”€ AND (car_id, ped_id) not already in triggered_pairs
   â””â”€ Returns Violation(type="FAILED_TO_YIELD", severity="HIGH")

8. EnforcementPipeline.submit_violation(frame, event)       [background thread]
   â””â”€ EvidenceBuilder.capture_event â†’ artifacts/frames/ + artifacts/vehicles/
   â””â”€ LicensePlateDetector.detect â†’ YOLO plate model OR Haar cascade
   â””â”€ OCREngine.recognize â†’ preprocess â†’ EasyOCR â†’ clean â†’ regex validate
   â””â”€ LLMReportService.generate â†’ mock JSON report (or OpenAI call)
   â””â”€ InvoiceGenerator.generate â†’ .pdf (ReportLab) or .txt fallback
   â””â”€ ViolationRepository.save_violation â†’ SQLite/PostgreSQL INSERT
```

### Violation FSM States

```
Pedestrian:              Vehicle:
  OUTSIDE                  OUTSIDE
     â”‚ enters bbox              â”‚ enters bbox
     â–¼                          â–¼
  ENTERING               â•â•â–º ENTER  â—„â”€â”€ VIOLATION TRIGGER HERE
     â”‚ crosses midline           â”‚
     â–¼                          â–¼
  CROSSING                   INSIDE
     â”‚ moves to exit zone        â”‚
     â–¼                          â–¼
  CLEARING                    EXIT
     â”‚ bbox leaves                â”‚
     â–¼                          â–¼
   EXITED                    OUTSIDE
```

### Crosswalk Zone Split (ratio = 0.32)

The polygon is divided into upper (top 32%) and lower (bottom 68%) sub-zones by interpolating along the left and right edges. This split determines which zone the pedestrian is in relative to the vehicle's approach direction.

```
p0 â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ p1
â”‚  upper sub-zone (32%) â”‚
â”œâ”€â”€â”€â”€ split line â”€â”€â”€â”€â”€â”€â”€â”€â”¤  â† cyan line drawn on frame
â”‚  lower sub-zone (68%) â”‚
p3 â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ p2
```

---

## Module Reference

### `src/config.py` â€” `AppSettings`

Three nested dataclasses, all fields overrideable via environment variable:

- **`ModelSettings`** â€” YOLOv8 model path, confidence, image size, OCR backend, plate regex, LLM provider
- **`RuntimeSettings`** â€” video path, polygon path, FPS, location name/code/GPS, fine amount, camera source, Yandex Maps key
- **`StorageSettings`** â€” database URL, SQLite fallback URL, output directory tree

### `src/schemas.py` â€” Data Models

| Dataclass | Fields | Purpose |
|-----------|--------|---------|
| `ViolationEvent` | violation_id, timestamp, vehicle_id, frame_index, vehicle_bbox, polygon, pedestrian_direction, confidence, plate_number, snapshot_path, â€¦ | Core event object created at trigger time |
| `EvidenceBundle` | event, frame_path, vehicle_crop_path, vehicle_crop_bbox, frame_shape | Passed through enforcement pipeline |
| `PlateDetectionResult` | plate_bbox, plate_crop_path, source, confidence | Output of ALPR stage |
| `OCRResult` | plate_text, confidence, raw_text, accepted | Output of OCR stage |
| `ReportPayload` | violation_id, timestamp, plate_number, violation_type, location, fine_amount, authority_name | Input to LLM / invoice |
| `ReportResult` | report_json, report_text, invoice_path | Output of LLM stage |
| `InvoiceRecordData` | violation_id, amount, status, pdf_path | Stored in invoices table |

### `src/detector/tracker.py`

- **`apply_cross_class_nms`** â€” removes duplicate vehicle bounding boxes (IoU > threshold), keeps higher-confidence detection
- **`IDMerger`** â€” resolves ByteTrack ID-switch bug: if two IDs have centroids within 40 px for 3+ consecutive frames, remaps the higher ID to the lower permanently for the session
- **`PedestrianTrack`** â€” stores FSM state, entry/exit frames, centroid, velocity deque(10), bbox
- **`VehicleTrack`** â€” stores approach axis, polygon midline, centroid, velocity deque(20), pre-entry velocity snapshot for speed estimation
- **`ObjectFSM`** â€” simple OUTSIDE/ENTER/INSIDE/EXIT state machine based on inside/outside boolean transitions

### `src/vision/stabilizer.py` â€” `VideoStabilizer`

ORB-based video stabilisation:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `MAX_FEATURES` | 1000 | ORB keypoints per frame |
| `MATCH_RATIO` | 0.75 | Lowe ratio-test threshold |
| `MIN_INLIERS` | 12 | Minimum RANSAC inliers to accept homography |
| `MAX_TRANSLATION_FRAC` | 0.35 | Reject shifts > 35% of frame width/height |
| `MIN_SCALE` | 0.75 | Reject extreme zoom-out |
| `MAX_SCALE` | 1.35 | Reject extreme zoom-in |
| `MAX_PERSPECTIVE_TERM` | 0.0025 | Reject strong projective skew |

On failure (< 12 inliers), returns the **original frame unwarped** (fail-open) rather than applying a stale transform.

### `src/OCR/engine.py` â€” `OCREngine`

Pre-processing chain before OCR:
1. BGR â†’ Grayscale
2. `cv2.equalizeHist` â€” normalise contrast
3. `cv2.GaussianBlur(3,3)` â€” reduce noise
4. `cv2.threshold` (Otsu binary) â€” binarise

Post-processing:
1. Strip non-alphanumeric characters (`[^A-Z0-9]`)
2. Validate against `PLATE_REGEX = ^[A-Z0-9]{5,10}$`
3. Check `confidence >= OCR_CONFIDENCE_THRESHOLD (0.35)`

Supports **EasyOCR** (default) and **PaddleOCR** (uncomment in requirements.txt).

### `src/alpr/detector.py` â€” `LicensePlateDetector`

Two-tier detection:
1. **Primary:** custom `models/plate_detector.pt` YOLOv8 model (if file exists)
2. **Fallback:** OpenCV Haar cascade (`haarcascade_russian_plate_number.xml`)

The detected plate crop is saved to `artifacts/plates/<violation_id>.jpg` and passed to OCREngine.

### `stream.py` â€” `CameraStream`

Thread-safe MJPEG streaming supporting:
- USB/built-in webcam: `CAMERA_SOURCE=0`
- RTSP IP camera (Hikvision/Dahua): `rtsp://admin:pass@IP:554/Streaming/Channels/101`
- HTTP MJPEG: `http://IP:80/video.cgi`
- Video file (demo): `Videos/v2.mp4`

Frames are read in a background daemon thread; Flask routes call `camera_manager.get_jpeg()` without blocking.

### `live_processor.py` â€” `LiveProcessor`

Runs the full YOLO + tracking + violation FSM pipeline on the live camera stream. Pushes annotated frames back into `camera_manager` for the MJPEG feed. Supports per-camera polygon files:

| Camera ID | Polygon File |
|-----------|-------------|
| `default` / `cam2` | `crosswalk_polygon.json` |
| `cam1` | `crosswalk_polygon_cam1.json` |
| `cam3` | `crosswalk_polygon_cam3.json` |

---

## Database Schema

### SQLite / PostgreSQL via SQLAlchemy ORM

#### `violations` table

| Column | Type | Description |
|--------|------|-------------|
| `id` | TEXT (UUID) PK | Violation UUID |
| `timestamp` | DATETIME (UTC) | When violation occurred |
| `plate_number` | TEXT nullable | Recognised plate string (NULL if unreadable) |
| `vehicle_id` | INTEGER | ByteTrack assigned ID |
| `vehicle_image_path` | TEXT | Path to vehicle crop image |
| `frame_image_path` | TEXT | Path to full frame image |
| `plate_image_path` | TEXT nullable | Path to plate crop |
| `report_path` | TEXT nullable | Path to JSON report |
| `invoice_path` | TEXT nullable | Path to invoice file |
| `violation_type` | TEXT | `crosswalk_violation` or `FAILED_TO_YIELD` |
| `severity` | TEXT | `HIGH` or `LOW` |
| `pedestrian_direction` | TEXT | `UP`, `DOWN`, or `STATIC` |
| `confidence` | FLOAT | Detection confidence |
| `status` | TEXT | `processed` (plate found) or `pending` |
| `location` | TEXT | Location name |
| `location_name` | TEXT | Full location label |
| `vehicle_speed_estimate` | FLOAT nullable | Estimated speed in px/frame |
| `snapshot_path` | TEXT nullable | Path to annotated snapshot |
| `plate_crop_path` | TEXT nullable | Alternate plate crop path |
| `llm_report_json` | TEXT nullable | Full JSON from LLM service |
| `llm_report_text` | TEXT nullable | Plain text LLM summary |
| `created_at` | DATETIME (UTC) | Record creation time |

#### `vehicles` table

| Column | Type | Description |
|--------|------|-------------|
| `id` | TEXT (UUID) PK | Vehicle UUID |
| `plate_number` | TEXT UNIQUE | Plate string |
| `owner_name` | TEXT nullable | Owner (if linked) |
| `violations_count` | INTEGER | Total violation count |

#### `invoices` table

| Column | Type | Description |
|--------|------|-------------|
| `id` | TEXT (UUID) PK | Invoice UUID |
| `violation_id` | TEXT FK â†’ violations | Linked violation |
| `amount` | NUMERIC(10,2) | Fine amount |
| `issued_at` | DATETIME | Issue time |
| `status` | TEXT | `issued` / `paid` / `cancelled` |
| `pdf_path` | TEXT | Path to invoice file |

#### `admin_users` table (Flask-side)

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER PK | Auto-increment |
| `username` | TEXT UNIQUE | Admin username |
| `password_hash` | TEXT | Werkzeug PBKDF2 hash |
| `created_at` | TIMESTAMP | Creation time |

#### `audit_log` table

Tracks all admin actions (login, status changes, exports) with IP address and timestamp.

---

## Installation

**Requirements:** Python 3.9+, Windows / Linux / macOS

```bash
# 1. Clone / unzip the project
cd Crosswalk_Violation

# 2. Create and activate a virtual environment
python -m venv CV_venv
# Windows:
CV_venv\Scripts\activate
# macOS / Linux:
source CV_venv/bin/activate

# 3. Install all dependencies
pip install -r requirements.txt
```

**For PDF invoice generation** (optional):
```bash
pip install reportlab
```

**For PaddleOCR** instead of EasyOCR (optional):
```bash
pip install paddleocr
# Then set OCR_BACKEND=paddleocr in .env
```

> **GPU note:** YOLOv8 and EasyOCR automatically detect CUDA. On CPU, `yolov8n.pt` runs at ~5â€“10 FPS. On a modern GPU (RTX 3060+), `yolov8x.pt` reaches 30+ FPS.

---

## Configuration (All Environment Variables)

Create a `.env` file in the project root. All variables have defaults.

```env
# â”€â”€ Models â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
DETECTION_MODEL_PATH=yolov8n.pt           # or yolov8x.pt for higher accuracy
PLATE_MODEL_PATH=models/plate_detector.pt
DETECTION_CONFIDENCE=0.35
PLATE_CONFIDENCE=0.25
IMAGE_SIZE=960
OCR_BACKEND=easyocr                       # or paddleocr
OCR_CONFIDENCE_THRESHOLD=0.35
PLATE_REGEX=^[A-Z0-9]{5,10}$

# â”€â”€ LLM / AI â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
LLM_PROVIDER=mock                         # or openai
OPENAI_MODEL=gpt-4o-mini
ANTHROPIC_API_KEY=sk-ant-...              # for dashboard AI summary + chatbot

# â”€â”€ Runtime â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
VIDEO_PATH=Videos/v2.mp4
POLYGON_PATH=crosswalk_polygon.json
TARGET_FPS=15
HISTORY_LENGTH=8
PEDESTRIAN_DIRECTION_THRESHOLD=3
CROSSWALK_SPLIT_RATIO=0.32
PIPELINE_WORKERS=2
OCR_WORKERS=1

# â”€â”€ Location â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
LOCATION_NAME=Crosswalk A
LOCATION_CODE=CW-A-01
AUTHORITY_NAME=WIUT Traffic Enforcement Unit
DEFAULT_FINE_AMOUNT=150000                # in local currency (UZS)
LOCATION_LATITUDE=41.2963
LOCATION_LONGITUDE=69.2798

# â”€â”€ Camera (for live mode) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
CAMERA_SOURCE=0                           # 0 = webcam, or RTSP URL
# CAMERA_SOURCE=rtsp://admin:pass@192.168.1.64:554/Streaming/Channels/101

# â”€â”€ Database â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
DATABASE_URL=postgresql+psycopg://crosswalk:crosswalk@localhost:5432/crosswalk
SQLITE_FALLBACK_URL=sqlite:///crosswalk_violations.db

# â”€â”€ Maps â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
YANDEX_MAPS_API_KEY=your_key_here

# â”€â”€ Flask â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
SECRET_KEY=change-in-production
```

---

## Running the System

### 1. Seed demo data (no camera required)

```bash
python generate_demo_data.py
python generate_demo_data.py --count 200   # custom count
```

### 2. Video file detector (CLI)

```bash
python run_system.py --video Videos/v2.mp4
```

**First run â€” Calibration Mode:**
1. A window shows the first video frame
2. **Left-click** â€” add polygon corner points (â‰¥ 4, clockwise from top-left)
3. **Right-click** â€” finalise polygon and start detection
4. Polygon saved to `crosswalk_polygon.json` and reused on future runs
5. To reset: delete `crosswalk_polygon.json`

**Optional flags:**
```bash
python run_system.py --video path/to/file.mp4  # override video path
python run_system.py --no-stabilize            # disable ORB stabilisation
python run_system.py --chatbot                 # launch AI chatbot instead
```

**Keyboard Controls:**

| Key | Action |
|-----|--------|
| `Esc` | Stop detection and exit |

### 3. Flask Admin Panel (live camera mode)

```bash
# Windows (set env vars first)
set CAMERA_SOURCE=0
set ANTHROPIC_API_KEY=sk-ant-...
python app.py

# Or with gunicorn (production):
gunicorn app:app --bind 0.0.0.0:5000 --workers 2 --threads 4
```

Open `http://localhost:5000`

Default admin credentials (seeded on first run):
- Username: `admin`
- Password: `admin123` â† **change immediately in production**

---

## Flask Admin Panel

**URL:** `http://localhost:5000`

The Flask application (`app.py`) provides a full web-based enforcement management system.

### Admin Routes

| Route | Description |
|-------|-------------|
| `/admin/login` | Admin login (max 5 attempts, 15-minute lockout) |
| `/admin/dashboard` | KPI cards, recent violations, hourly chart |
| `/admin/violations` | Paginated violation list with search/filter |
| `/admin/violations/<id>` | Full violation detail: snapshot, evidence images, Yandex map pin, invoice download, LLM report |
| `/admin/vehicles` | Vehicle registry with violation counts |
| `/admin/cameras` | Camera management (add/remove camera sources) |
| `/admin/cameras/<id>` | Per-camera live stream + statistics |
| `/admin/live` | Live MJPEG stream with real-time detection overlay |
| `/admin/analytics` | Charts + AI-generated traffic summary (Claude API) |
| `/admin/audit` | Admin action audit log |
| `/admin/export` | Export violations as CSV |
| `/admin/logout` | End session |

### Public Violator Portal

| Route | Description |
|-------|-------------|
| `/portal` | Public plate lookup form |
| `/portal/results?plate=<plate>` | Show all violations for a plate number |

### Security Features

- Session-based authentication with `login_required` decorator
- PBKDF2 password hashing via Werkzeug
- Brute-force protection: 5 failed attempts â†’ 15-minute session lockout
- All admin actions written to `audit_log` table
- `WAL` journal mode on SQLite for concurrent reads
- `KMP_DUPLICATE_LIB_OK=TRUE` set at startup to prevent Windows OMP conflicts

---

## Streamlit Analytics Dashboard

```bash
streamlit run dashboard.py
# Open: http://localhost:8501
```

### Sections

| Section | Description |
|---------|-------------|
| **KPI Cards** | Total violations Â· Unique vehicles Â· Peak hour Â· Plate recognition rate |
| **Violations by Hour** | Line chart (with Pandas resampling) |
| **Top Offending Vehicles** | Bar chart of vehicle IDs by frequency |
| **Day Ã— Hour Heatmap** | 7-day Ã— 24-hour violation density matrix |
| **Violation Log Table** | Paginated sortable table |
| **Generate AI Summary** | Calls Claude API for traffic pattern analysis narrative |

### Sidebar Filters

- Date range picker
- Minimum confidence slider
- Plate captured / not captured / all radio

```bash
# Set API key before launching:
set ANTHROPIC_API_KEY=sk-ant-...
streamlit run dashboard.py
```

---

## REST API

```bash
python -m uvicorn api.main:app --app-dir src --reload
# Docs: http://localhost:8000/docs
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/violations` | List violations (paginated, filterable) |
| `GET` | `/violations/{id}` | Single violation detail |
| `POST` | `/violations` | Create a violation record |
| `GET` | `/vehicles/{plate}` | Vehicle history by plate number |
| `GET` | `/analytics` | Aggregated stats (total, peak hour, top plates) |
| `GET` | `/health` | Health check â€” returns `{"status": "ok"}` |

---

## AI Chatbot

```bash
set ANTHROPIC_API_KEY=sk-ant-...
python run_system.py --chatbot
```

The chatbot loads statistics from SQLite and uses Claude (`claude-3-*`) in a terminal conversation loop. Example questions:

- *"Which vehicle violated the most?"*
- *"What time of day has the highest violation rate?"*
- *"What is today's violation count?"*
- *"Suggest measures to reduce violations at peak hours."*

Statistics pre-loaded into context:
- Total violations
- Top 5 offending vehicles (ID + count)
- Peak hour (most violations)
- Plate detection rate (%)

Type `quit` or `exit` to leave.

---

## Deployment

### Docker Compose (recommended)

```bash
# Build and start both services:
docker-compose up --build -d

# Flask admin panel:  http://localhost:5000
# Streamlit dashboard: http://localhost:8501
```

**Environment:**
- Copy `.env.example` to `.env` and fill in `SECRET_KEY`, `ANTHROPIC_API_KEY`, `YANDEX_MAPS_API_KEY`
- For webcam access inside container on Linux, uncomment the `devices` section in `docker-compose.yml`

### Render.com (cloud)

Push to GitHub â†’ connect repo on Render â†’ it uses `render.yaml` automatically:

```yaml
startCommand: gunicorn app:app --bind 0.0:$PORT --workers 2 --threads 4 --timeout 120
```

Set secret env vars in the Render dashboard: `YANDEX_MAPS_API_KEY`, `ANTHROPIC_API_KEY`, `SECRET_KEY`

### Heroku / Railway

```
Procfile:
web: gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --threads 4 --timeout 120
```

---

## Performance

### Detection Speed (measured on CPU â€” Intel Core i7-12th Gen)

| Model | Image Size | Inference FPS (CPU) | Inference FPS (GPU RTX 3060) |
|-------|-----------|---------------------|------------------------------|
| `yolov8n.pt` (nano) | 960 px | ~5â€“10 FPS | 60+ FPS |
| `yolov8x.pt` (xlarge) | 960 px | ~1â€“3 FPS | 30â€“40 FPS |

### EasyOCR

- **First startup:** ~10 seconds to load model weights into memory
- **Per plate recognition:** ~0.5â€“1.5 seconds on CPU (depends on image resolution)
- **GPU:** ~50â€“100 ms per plate

### Video Stabiliser

- ORB feature extraction: ~8 ms/frame on CPU
- BFMatcher + RANSAC: ~5 ms/frame
- warpPerspective: ~2 ms/frame

### Enforcement Pipeline (background thread)

| Stage | Typical Duration |
|-------|-----------------|
| Evidence capture (save frame + crop) | ~20 ms |
| Plate detection (Haar cascade) | ~25 ms |
| OCR preprocessing + EasyOCR | ~500 ms â€“ 1500 ms |
| Mock LLM report generation | ~1 ms |
| Invoice generation (.txt) | ~1 ms |
| SQLite INSERT | ~5 ms |

The pipeline runs in a `ThreadPoolExecutor(max_workers=2)` so it does not block the main detection loop.

### Database

- SQLite with WAL journal mode â€” supports concurrent reads from Flask + background writes from detector
- Demo dataset: 200 violations generated in ~3 seconds

---

## Known Errors & Limitations

### Error: `libiomp5md.dll already initialized` (Windows)

**Cause:** Conflict when importing PyTorch (via YOLOv8) alongside OpenCV in the same process on Windows.

**Fix applied:** `app.py` sets `os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"` before any imports. This prevents the crash.

---

### Error: `Cannot open video` / `RuntimeError: Cannot open video`

**Cause:** `VIDEO_PATH` env variable or `--video` argument points to a non-existent file, or the codec is not supported by the installed OpenCV build.

**Fix:** Verify the path exists. Convert video to H.264/MP4 if using an unusual codec:
```bash
ffmpeg -i input.avi -c:v libx264 output.mp4
```

---

### Error: `Polygon missing or invalid` / `RuntimeError: Polygon missing or invalid`

**Cause:** `crosswalk_polygon.json` exists but contains fewer than 4 points, is malformed JSON, or right-click was not used to finalise during calibration.

**Fix:** Delete `crosswalk_polygon.json` and re-run to recalibrate.

---

### Error: `ModuleNotFoundError: No module named 'easyocr'`

**Cause:** EasyOCR not installed or virtual environment not activated.

**Fix:**
```bash
CV_venv\Scripts\activate
pip install easyocr
```

---

### Error: `ModuleNotFoundError: No module named 'reportlab'`

**Cause:** ReportLab not installed.

**Fix:** The system falls back to `.txt` invoices automatically. To enable PDF:
```bash
pip install reportlab
```

---

### Error: `anthropic.AuthenticationError` / Chatbot/Dashboard shows no AI summary

**Cause:** `ANTHROPIC_API_KEY` not set.

**Fix:**
```bash
set ANTHROPIC_API_KEY=sk-ant-...
```
The dashboard still loads without it; only the AI summary button is affected. The chatbot will fail to start without a key.

---

### Warning: `Unstable` shown on frame (orange label)

**Cause:** ORB homography estimation found fewer than 12 RANSAC inliers. Common in:
- Low-contrast or overexposed scenes
- Night time / rain
- Camera rapidly panning

**Behaviour:** The original (unstabilised) frame is used. Detection continues. Polygon may drift slightly from the road markings.

---

### Limitation: ByteTrack ID switching on occlusion

ByteTrack re-issues IDs after an object is lost for more than `track_buffer=60` frames (~2 seconds at 30 FPS). The `IDMerger` recovers most cases using centroid proximity but cannot handle:
- Extreme occlusion (object fully hidden > 2 seconds)
- Very fast vehicles passing through the zone in one frame

**Observed in live footage:** Vehicle IDs changed from 32 â†’ 110 â†’ 186 for the same dark SUV across three captured violations in the sample screenshots (same physical vehicle, different ByteTrack IDs after re-detection cycles).

---

### Limitation: Plate recognition rate (~20% on live footage)

The Haar cascade fallback struggles with:
- Oblique camera angles (plates appear skewed)
- Russian/Uzbek plate formats at distance
- Plates with mud, shadows, or partial occlusion

All three screenshots show `Plate: UNDETECTED`. The custom `models/plate_detector.pt` (if trained on local plates) significantly improves this rate.

---

### Limitation: Single-zone polygon, fixed camera only

- No support for PTZ (pan-tilt-zoom) cameras
- Wide-angle lens distortion is not corrected
- Only the crosswalk polygon region is monitored; sidewalks and approach lanes are not analysed

---

### Limitation: GPU required for real-time on high-resolution video

`yolov8n.pt` at 960 px on CPU produces ~5â€“10 FPS â€” sufficient for 15 FPS target video. At higher resolutions or with `yolov8x.pt`, a CUDA-capable GPU is required.

---

## Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test files
python -m pytest tests/test_violation_logic.py -v
python -m pytest tests/test_ocr_engine.py -v
python -m pytest tests/test_pipeline_integration.py -v
```

### `test_violation_logic.py`

Tests the `ViolationDetector` FSM:
1. **Trigger once on ENTER:** Violation fires exactly once when vehicle transitions OUTSIDE â†’ ENTER with pedestrian CROSSING in same zone
2. **No re-trigger on INSIDE:** Second `evaluate_vehicle` call with state INSIDE returns `None` trigger (no duplicate)
3. **Clear after EXIT:** Active violation removed when vehicle leaves zone

### `test_ocr_engine.py`

Tests the OCR preprocessing and validation pipeline.

### `test_pipeline_integration.py`

Integration test: creates a minimal `EvidenceBundle` and runs through the full `EnforcementPipeline` (with mock LLM), verifying DB record creation and file outputs.

---

## Evidence Artifacts

Each violation generates up to 6 evidence files:

```
artifacts/
â”œâ”€â”€ frames/<violation_id>.jpg          # Full frame at moment of trigger
â”œâ”€â”€ vehicles/<violation_id>.jpg        # Cropped vehicle bounding box
â”œâ”€â”€ plates/<violation_id>.jpg          # Cropped plate region (if detected)
â”œâ”€â”€ invoices/<violation_id>.txt (.pdf) # Fine notice
â””â”€â”€ reports/
    â”œâ”€â”€ <violation_id>.json            # Raw ViolationEvent metadata
    â””â”€â”€ <violation_id>_report.json     # LLM-generated enforcement report
```

**Example report (`artifacts/reports/<id>_report.json`):**
```json
{
  "report": "Violation 0f79827e ... recorded on 2026-03-19T13:53:27Z for vehicle UNREADABLE 
            at Crosswalk A. The vehicle failed to yield to a pedestrian moving STATIC 
            within the marked crosswalk.",
  "legal_explanation": "The observed conduct constitutes a crosswalk right-of-way violation 
                        because the vehicle entered the crosswalk while a pedestrian had 
                        lawful priority of movement.",
  "fine_amount": 150.0,
  "payment_instructions": "Pay $150.00 to City Traffic Enforcement Unit within 30 days 
                           using the reference number listed on the invoice.",
  "violation_summary": "Failure to yield to a pedestrian in the marked crosswalk."
}
```

**Example invoice (`artifacts/invoices/<id>.txt`):**
```
City Traffic Enforcement Unit
Violation ID: 0f79827e-3470-471e-bdbd-ebac4300aa1f
Plate Number: UNREADABLE
Violation Type: crosswalk_violation
Date: 2026-03-19T13:53:27.170232+00:00
Fine Amount: $150.00
Payment due within 30 days.
```

Static snapshots (annotated frames served by Flask):
```
static/snapshots/snapshot_<violation_id>_<frame_index>.jpg
```

---

## Demo Data

`generate_demo_data.py` seeds the database with realistic Tashkent crosswalk violations:

- **Uzbek plate formats:** `01A123BC` (Tashkent city), `30A111HJ` (Tashkent oblast), `40â€“70` region plates
- **Locations:** 4 real Tashkent crosswalks with GPS coordinates
- **Hour distribution:** weighted toward rush hours (07:00â€“09:00, 17:00â€“19:00)
- **Plate capture rate:** ~80% detected, ~20% `NULL`
- **Pedestrian directions:** UP / DOWN / STATIC
- **Violation types:** crosswalk_violation, FAILED_TO_YIELD

```bash
python generate_demo_data.py              # default 100 records
python generate_demo_data.py --count 200  # 200 records
```

---

## License

See [LICENSE](LICENSE) for terms.
