from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts"


def _env_path(name: str, default: Path) -> Path:
    return Path(os.getenv(name, str(default)))


@dataclass(slots=True)
class ModelSettings:
    detection_model_path: str = os.getenv("DETECTION_MODEL_PATH", "yolov8n.pt")
    license_plate_model_path: str = os.getenv("PLATE_MODEL_PATH", "models/license_plate.pt")
    detection_classes: List[int] = field(default_factory=lambda: [0, 2, 3, 5, 7])
    plate_classes: List[int] = field(default_factory=list)
    detection_confidence: float = float(os.getenv("DETECTION_CONFIDENCE", "0.35"))
    plate_confidence: float = float(os.getenv("PLATE_CONFIDENCE", "0.25"))
    image_size: int = int(os.getenv("IMAGE_SIZE", "960"))
    ocr_backend: str = os.getenv("OCR_BACKEND", "easyocr")
    ocr_confidence_threshold: float = float(os.getenv("OCR_CONFIDENCE_THRESHOLD", "0.35"))
    plate_regex: str = os.getenv("PLATE_REGEX", r"^[A-Z0-9]{5,10}$")
    llm_provider: str = os.getenv("LLM_PROVIDER", "mock")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")


@dataclass(slots=True)
class RuntimeSettings:
    video_path: str = os.getenv("VIDEO_PATH", "Videos/v2.mp4")
    polygon_path: str = os.getenv("POLYGON_PATH", "crosswalk_polygon.json")
    history_length: int = int(os.getenv("HISTORY_LENGTH", "8"))
    pedestrian_direction_threshold: int = int(
        os.getenv("PEDESTRIAN_DIRECTION_THRESHOLD", "3")
    )
    split_ratio: float = float(os.getenv("CROSSWALK_SPLIT_RATIO", "0.32"))
    target_fps: int = int(os.getenv("TARGET_FPS", "15"))
    location_name: str = os.getenv("LOCATION_NAME", "Crosswalk A")
    authority_name: str = os.getenv("AUTHORITY_NAME", "City Traffic Enforcement Unit")
    default_fine_amount: float = float(os.getenv("DEFAULT_FINE_AMOUNT", "150.00"))
    max_workers: int = int(os.getenv("PIPELINE_WORKERS", "2"))
    ocr_workers: int = int(os.getenv("OCR_WORKERS", "1"))
    location_code: str = os.getenv("LOCATION_CODE", "CW-A-01")

    # ── Camera source for live streaming (used by stream.py + app.py) ────────
    # Set to webcam index (0, 1, 2…) or a full RTSP/HTTP URL:
    #   CAMERA_SOURCE=0                              # first USB/built-in webcam
    #   CAMERA_SOURCE=rtsp://admin:pass@192.168.1.64:554/stream1
    #   CAMERA_SOURCE=http://192.168.1.100:8080/video
    camera_source: str = os.getenv("CAMERA_SOURCE", "0")

    # ── Yandex Maps API key (used in admin invoices + violation detail) ───────
    # Get your key at: https://developer.tech.yandex.com/services/
    # Then add to your .env file:  YANDEX_MAPS_API_KEY=<your_key>
    yandex_maps_api_key: str = os.getenv("YANDEX_MAPS_API_KEY", "")

    # ── Location GPS coordinates (set per crosswalk camera) ──────────────────
    # Used to pin violations on the map.  Tashkent examples:
    #   41.2963,69.2798  → Amir Temur Ave crosswalk
    #   41.2959,69.2697  → Navoi Street crosswalk
    location_latitude:  float = float(os.getenv("LOCATION_LATITUDE",  "41.2963"))
    location_longitude: float = float(os.getenv("LOCATION_LONGITUDE", "69.2798"))


@dataclass(slots=True)
class StorageSettings:
    database_url: str = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg://crosswalk:crosswalk@localhost:5432/crosswalk",
    )
    sqlite_fallback_url: str = os.getenv(
        "SQLITE_FALLBACK_URL",
        f"sqlite:///{(PROJECT_ROOT / 'crosswalk_violations.db').as_posix()}",
    )
    output_dir: Path = field(default_factory=lambda: _env_path("OUTPUT_DIR", DEFAULT_OUTPUT_DIR))
    frames_dir: Path = field(init=False)
    vehicle_crops_dir: Path = field(init=False)
    plate_crops_dir: Path = field(init=False)
    invoices_dir: Path = field(init=False)
    reports_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.frames_dir = self.output_dir / "frames"
        self.vehicle_crops_dir = self.output_dir / "vehicles"
        self.plate_crops_dir = self.output_dir / "plates"
        self.invoices_dir = self.output_dir / "invoices"
        self.reports_dir = self.output_dir / "reports"


@dataclass(slots=True)
class AppSettings:
    models: ModelSettings = field(default_factory=ModelSettings)
    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)
    storage: StorageSettings = field(default_factory=StorageSettings)

    def ensure_directories(self) -> None:
        for path in (
            self.storage.output_dir,
            self.storage.frames_dir,
            self.storage.vehicle_crops_dir,
            self.storage.plate_crops_dir,
            self.storage.invoices_dir,
            self.storage.reports_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


settings = AppSettings()
settings.ensure_directories()


VIDEO_PATH = settings.runtime.video_path
MODEL_PATH = settings.models.detection_model_path
DETECTION_CLASSES = settings.models.detection_classes
CONF_THRESHOLD = settings.models.detection_confidence
IMG_SIZE = settings.models.image_size
HISTORY = settings.runtime.history_length
MIN_SPEED = 2.0
