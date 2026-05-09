from __future__ import annotations
import json
import tempfile
import unittest
import sys
from pathlib import Path
import cv2
import numpy as np
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from config import AppSettings
from schemas import OCRResult, PlateDetectionResult, ReportResult, ViolationEvent
from services.pipeline import EnforcementPipeline
class StubPlateDetector:
    def __init__(self, plate_path: Path):
        self.plate_path = plate_path
    def detect(self, evidence):
        return PlateDetectionResult(
            plate_bbox=(0, 0, 10, 10),
            plate_crop_path=self.plate_path,
            source="vehicle_crop",
            confidence=0.9,
        )
class StubOCREngine:
    def recognize(self, image_path: str):
        return OCRResult(plate_text="ABC123", confidence=0.95, raw_text="ABC123", accepted=True)
class StubReportService:
    def generate(self, payload):
        return ReportResult(
            report_json={
                "report": "stub",
                "legal_explanation": "stub",
                "fine_amount": payload.fine_amount,
                "payment_instructions": "stub",
                "violation_summary": "stub",
            },
            report_text="stub",
        )
class StubInvoiceGenerator:
    def __init__(self, invoice_path: Path):
        self.invoice_path = invoice_path
    def generate(self, payload):
        self.invoice_path.write_bytes(b"pdf")
        return self.invoice_path
class PipelineIntegrationTests(unittest.TestCase):
    def test_pipeline_persists_violation(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            settings = AppSettings()
            settings.storage.database_url = "sqlite:///ignored.db"
            settings.storage.sqlite_fallback_url = f"sqlite:///{(tmp_path / 'test.db').as_posix()}"
            settings.storage.output_dir = tmp_path / "artifacts"
            settings.storage.__post_init__()
            settings.ensure_directories()
            pipeline = EnforcementPipeline(settings)
            plate_path = settings.storage.plate_crops_dir / "plate.jpg"
            cv2.imwrite(str(plate_path), np.full((20, 60, 3), 255, dtype=np.uint8))
            pipeline.plate_detector = StubPlateDetector(plate_path)
            pipeline.ocr_engine = StubOCREngine()
            pipeline.report_service = StubReportService()
            pipeline.invoice_generator = StubInvoiceGenerator(
                settings.storage.invoices_dir / "invoice.pdf"
            )
            frame = np.full((120, 160, 3), 255, dtype=np.uint8)
            event = ViolationEvent.create(
                vehicle_id=99,
                frame_index=1,
                vehicle_bbox=(10, 20, 80, 90),
                vehicle_zone="upper",
                polygon=[(0, 0), (100, 0), (100, 100), (0, 100)],
                pedestrian_direction="UP",
                pedestrian_zone="upper",
                confidence=0.88,
                location="Test Crosswalk",
            )
            violation_id = pipeline.submit_violation(frame, event).result(timeout=10)
            record = pipeline.repository.get_violation(violation_id)
            self.assertIsNotNone(record)
            self.assertEqual(record.plate_number, "ABC123")
            self.assertTrue(Path(record.frame_image_path).exists())
            self.assertTrue(Path(record.vehicle_image_path).exists())
            self.assertTrue(Path(record.invoice_path).exists())
            self.assertEqual(json.loads(record.llm_report_json)["report"], "stub")
            pipeline.shutdown()
if __name__ == "__main__":
    unittest.main()
