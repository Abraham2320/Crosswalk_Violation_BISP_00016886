from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
PROJECT_ROOT = Path(__file__).resolve().parent.parent
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None
if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts"
def _env_path(name: str, default: Path) -> Path:
    return Path(os.getenv(name, str(default)))
@dataclass(slots=True)
class ModelSettings:
    detection_model_path: str = os.getenv("DETECTION_MODEL_PATH", "yolov8l.pt")
    license_plate_model_path: str = os.getenv("PLATE_MODEL_PATH", "models/license_plate.pt")
    detection_classes: List[int] = field(default_factory=lambda: [0, 2, 3, 5, 7])
    plate_classes: List[int] = field(default_factory=list)
    detection_confidence: float = float(os.getenv("DETECTION_CONFIDENCE", "0.35"))
    plate_confidence: float = float(os.getenv("PLATE_CONFIDENCE", "0.25"))
    image_size: int = int(os.getenv("IMAGE_SIZE", "960"))
    ocr_backend: str = os.getenv("OCR_BACKEND", "easyocr")
    ocr_confidence_threshold: float = float(os.getenv("OCR_CONFIDENCE_THRESHOLD", "0.40"))
    plate_regex: str = os.getenv("PLATE_REGEX", r"^[A-Z0-9]{5,10}$")
    llm_provider: str = os.getenv("LLM_PROVIDER", "anthropic")
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
@dataclass(slots=True)
class RuntimeSettings:
    video_path: str = os.getenv("VIDEO_PATH", "Videos/v2.mp4")
    polygon_path: str = os.getenv("POLYGON_PATH", "crosswalk_polygon.json")
    history_length: int = int(os.getenv("HISTORY_LENGTH", "8"))
    pedestrian_direction_threshold: int = int(
        os.getenv("PEDESTRIAN_DIRECTION_THRESHOLD", "3")
    )
    split_ratio: float = float(os.getenv("CROSSWALK_SPLIT_RATIO", "0.12"))
    show_split_overlay: bool = os.getenv("SHOW_SPLIT_OVERLAY", "1") != "0"
    target_fps: int = int(os.getenv("TARGET_FPS", "15"))
    location_name: str = os.getenv("LOCATION_NAME", "Crosswalk A")
    authority_name: str = os.getenv("AUTHORITY_NAME", "City Traffic Enforcement Unit")
    default_fine_amount: float = float(os.getenv("DEFAULT_FINE_AMOUNT", "150.00"))
    max_workers: int = int(os.getenv("PIPELINE_WORKERS", "2"))
    ocr_workers: int = int(os.getenv("OCR_WORKERS", "1"))
    location_code: str = os.getenv("LOCATION_CODE", "CW-A-01")
    camera_source: str = os.getenv("CAMERA_SOURCE", "0")
    yandex_maps_api_key: str = os.getenv("YANDEX_MAPS_API_KEY", "")
    location_latitude:  float = float(os.getenv("LOCATION_LATITUDE",  "41.2963"))
    location_longitude: float = float(os.getenv("LOCATION_LONGITUDE", "69.2798"))
    direction_threshold: int = int(os.getenv("DIRECTION_THRESHOLD", "15"))
    road_y_min: int = int(os.getenv("ROAD_Y_MIN", "350"))
    road_y_max: int = int(os.getenv("ROAD_Y_MAX", "1080"))
    missing_frames_finalize: int = int(os.getenv("MISSING_FRAMES_FINALIZE", "15"))
    ocr_retry_interval: int = int(os.getenv("OCR_RETRY_INTERVAL", "5"))
    ocr_high_conf_threshold: float = float(os.getenv("OCR_HIGH_CONF_THRESHOLD", "0.75"))
    ocr_min_accept_conf: float = float(os.getenv("OCR_MIN_ACCEPT_CONF", "0.35"))
    enable_plate_detector: bool = os.getenv("ENABLE_PLATE_DETECTOR", "0") == "1"
    enable_deferred_ocr: bool = os.getenv("ENABLE_DEFERRED_OCR", "0") == "1"
    enable_regular_ocr: bool = os.getenv("ENABLE_REGULAR_OCR", "0") == "1"
    enable_wrong_dir: bool = os.getenv("ENABLE_WRONG_DIR", "1") == "1"
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
class SegmentationSettings:
    enabled: bool = os.getenv("SEGMENTATION_ENABLED", "1") == "1"
    seg_model_path: str = os.getenv("SEG_MODEL_PATH", "yolov8l-seg.pt")
    mask_alpha: float = float(os.getenv("MASK_ALPHA", "0.35"))
    mask_min_ratio: float = float(os.getenv("MASK_MIN_RATIO", "0.02"))
    run_every_n_frames: int = int(os.getenv("SEG_EVERY_N_FRAMES", "3"))

@dataclass(slots=True)
class AppSettings:
    models: ModelSettings = field(default_factory=ModelSettings)
    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)
    storage: StorageSettings = field(default_factory=StorageSettings)
    segmentation: SegmentationSettings = field(default_factory=SegmentationSettings)
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
class Config:
    AUDIT_ENABLED: bool = os.getenv("AUDIT_ENABLED", "true").lower() == "true"
    AUDIT_LOG_DIR: Path = Path(os.getenv("AUDIT_LOG_DIR", "audit/logs"))
    AUDIT_SNAPSHOT_DIR: Path = Path(os.getenv("AUDIT_SNAPSHOT_DIR", "audit/snapshots"))
    AUDIT_CLIP_DIR: Path = Path(os.getenv("AUDIT_CLIP_DIR", "audit/clips"))
    AUDIT_VIDEO_DIR: Path = Path(os.getenv("AUDIT_VIDEO_DIR", "audit/videos"))
    AUDIT_FINDINGS_CSV: Path = Path(os.getenv("AUDIT_FINDINGS_CSV", "audit/findings.csv"))
    AUDIT_CATEGORIES_PATH: Path = Path(os.getenv("AUDIT_CATEGORIES_PATH", "audit/categories.json"))
    AUDIT_CLIP_PRE_FRAMES: int = int(os.getenv("AUDIT_CLIP_PRE_FRAMES", "30"))
    AUDIT_CLIP_POST_FRAMES: int = int(os.getenv("AUDIT_CLIP_POST_FRAMES", "30"))
    CLAHE_ENABLED: bool = os.getenv("CLAHE_ENABLED", "true").lower() == "true"
    CLAHE_CLIP_LIMIT: float = float(os.getenv("CLAHE_CLIP_LIMIT", "2.0"))
    CLAHE_TILE_SIZE: int = int(os.getenv("CLAHE_TILE_SIZE", "8"))
    GLARE_DETECTION_ENABLED: bool = os.getenv("GLARE_DETECTION_ENABLED", "true").lower() == "true"
    GLARE_LUMINANCE_THRESHOLD: int = int(os.getenv("GLARE_LUMINANCE_THRESHOLD", "240"))
    GLARE_REJECT_THRESHOLD: float = float(os.getenv("GLARE_REJECT_THRESHOLD", "0.4"))
    ORB_MIN_MATCHES: int = int(os.getenv("ORB_MIN_MATCHES", "30"))
    STAB_MATCH_WARNING_THRESHOLD: int = int(os.getenv("STAB_MATCH_WARNING_THRESHOLD", "50"))
    POSE_ENABLED: bool = os.getenv("POSE_ENABLED", "false").lower() == "true"
    POSE_MODEL_PATH: str = os.getenv("POSE_MODEL_PATH", "yolov8n-pose.pt")
    VIOLATION_CONFIRM_FRAMES: int = int(os.getenv("VIOLATION_CONFIRM_FRAMES", "4"))
    PED_MIN_FRAMES_TO_QUALIFY: int = int(os.getenv("PED_MIN_FRAMES_TO_QUALIFY", "3"))
    VIOLATION_GRACE_FRAMES: int = int(os.getenv("VIOLATION_GRACE_FRAMES", "2"))
    YOLO_CONF_THRESHOLD: float = float(os.getenv("YOLO_CONF_THRESHOLD", "0.35"))
    YOLO_MODEL_PATH: str = os.getenv("YOLO_MODEL_PATH", "yolov8s.pt")
