from __future__ import annotations

import json
from concurrent.futures import Future, ThreadPoolExecutor

from alpr.detector import LicensePlateDetector
from capture.service import EvidenceBuilder
from config import AppSettings
from OCR.engine import OCREngine
from reporting.invoice import InvoiceGenerator
from reporting.llm_service import LLMReportService
from schemas import EvidenceBundle, InvoiceRecordData, ReportPayload, ViolationEvent
from storage.database import Database, ViolationRepository


class EnforcementPipeline:
    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.db = Database(settings)
        self.db.create_all()
        self.repository = ViolationRepository(self.db)
        self.evidence_builder = EvidenceBuilder(settings)
        self.plate_detector = LicensePlateDetector(settings)
        self._ocr_engine = None  # Lazy-load OCR engine to avoid EasyOCR initialization hang
        self.report_service = LLMReportService(settings)
        self.invoice_generator = InvoiceGenerator(settings)
        self.executor = ThreadPoolExecutor(max_workers=settings.runtime.max_workers)
    
    @property
    def ocr_engine(self) -> OCREngine:
        """Lazy-load OCR engine on first access to avoid EasyOCR initialization hang."""
        if self._ocr_engine is None:
            self._ocr_engine = OCREngine(self.settings)
        return self._ocr_engine

    def submit_violation(self, frame, event: ViolationEvent) -> Future:
        # Pass raw frame + event to the background thread; disk I/O happens there
        return self.executor.submit(self._process_violation, frame.copy(), event)

    def _process_violation(self, frame, event: ViolationEvent) -> str:
        evidence = self.evidence_builder.capture_event(frame, event)
        return self._process_evidence(evidence)

    def _process_evidence(self, evidence: EvidenceBundle) -> str:
        plate_detection = self.plate_detector.detect(evidence)
        ocr_result = (
            self.ocr_engine.recognize(str(plate_detection.plate_crop_path))
            if plate_detection.plate_crop_path
            else None
        )
        pipeline_plate = ocr_result.plate_text if ocr_result and ocr_result.accepted else None
        # Fall back to the plate already detected by the live OCR cache (passed via event)
        plate_number = pipeline_plate or evidence.event.plate_number or None
        if plate_number:
            self.repository.upsert_vehicle(plate_number)

        fine_amount = self.settings.runtime.default_fine_amount
        report_payload = ReportPayload(
            violation_id=evidence.event.violation_id,
            timestamp=evidence.event.timestamp.isoformat(),
            plate_number=plate_number or "UNREADABLE",
            violation_type=evidence.event.violation_type,
            pedestrian_direction=evidence.event.pedestrian_direction,
            location=evidence.event.location,
            location_code=self.settings.runtime.location_code,
            authority_name=self.settings.runtime.authority_name,
            fine_amount=fine_amount,
            plate_crop_path=evidence.event.plate_crop_path,
            snapshot_path=evidence.event.snapshot_path,
        )
        report_result = self.report_service.generate(report_payload)
        invoice_path = self.invoice_generator.generate(report_payload)
        report_result.invoice_path = invoice_path
        self.repository.create_invoice(
            InvoiceRecordData(
                violation_id=evidence.event.violation_id,
                amount=fine_amount,
                status="issued",
                pdf_path=str(invoice_path),
            )
        )

        report_path = self.settings.storage.reports_dir / f"{evidence.event.violation_id}_report.json"
        report_path.write_text(json.dumps(report_result.report_json, indent=2), encoding="utf-8")

        payload = {
            "id": evidence.event.violation_id,
            "timestamp": evidence.event.timestamp,
            "plate_number": plate_number,
            "vehicle_id": evidence.event.vehicle_id,
            "vehicle_image_path": str(evidence.vehicle_crop_path),
            "frame_image_path": str(evidence.frame_path),
            "plate_image_path": str(plate_detection.plate_crop_path) if plate_detection.plate_crop_path else None,
            "report_path": str(report_path),
            "invoice_path": str(invoice_path),
            "violation_type": evidence.event.violation_type,
            "severity": evidence.event.severity,
            "pedestrian_direction": evidence.event.pedestrian_direction,
            "status": "processed" if plate_number else "pending",
            "location": evidence.event.location,
            "location_name": evidence.event.location,
            "confidence": evidence.event.confidence,
            "created_at": evidence.event.timestamp,
            "llm_report_json": json.dumps(report_result.report_json),
            "llm_report_text": report_result.report_text,
            "snapshot_path": evidence.event.snapshot_path,
            "plate_crop_path": evidence.event.plate_crop_path,
            "vehicle_speed_estimate": evidence.event.vehicle_speed_estimate,
        }
        self.repository.save_violation(payload)
        return evidence.event.violation_id

    def update_violation_plate(
        self,
        violation_id: str,
        plate_number: str,
        confidence: float,
    ) -> None:
        """Queue a plate-number update for a violation already saved to the DB."""
        self.executor.submit(self._do_plate_update, violation_id, plate_number, confidence)

    def _do_plate_update(
        self,
        violation_id: str,
        plate_number: str,
        confidence: float,
    ) -> None:
        is_real = plate_number and plate_number not in ("UNREAD", "UNREADABLE")
        if is_real:
            try:
                self.repository.upsert_vehicle(plate_number)
            except Exception:
                pass
        try:
            self.repository.update_plate_number(violation_id, plate_number, confidence)
        except Exception as exc:
            print(f"[WARN] plate update failed for {violation_id}: {exc}")

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=False)
