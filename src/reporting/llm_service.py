from __future__ import annotations
import json
from dataclasses import asdict
from typing import Any, Dict
from config import AppSettings
from schemas import ReportPayload, ReportResult
class LLMReportService:
    def __init__(self, settings: AppSettings):
        self.settings = settings
        self._client = None
        if settings.models.llm_provider.lower() == "anthropic":
            try:
                import anthropic
                api_key = settings.models.anthropic_api_key or None
                self._client = anthropic.Anthropic(api_key=api_key)
            except Exception:
                self._client = None
    def build_prompt(self, payload: ReportPayload) -> str:
        return (
            "Generate a formal traffic violation package in JSON with keys "
            "`report`, `legal_explanation`, `fine_amount`, `payment_instructions`, `violation_summary`. "
            "Write in a legal-administrative tone suitable for a municipal enforcement record. "
            "Return ONLY valid JSON — no markdown, no code fences. "
            f"Input: {json.dumps(asdict(payload), ensure_ascii=True)}"
        )
    def generate(self, payload: ReportPayload) -> ReportResult:
        if self._client is not None:
            try:
                message = self._client.messages.create(
                    model=self.settings.models.anthropic_model,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": self.build_prompt(payload)}],
                )
                text = message.content[0].text.strip()
                try:
                    structured = json.loads(text)
                except json.JSONDecodeError:
                    structured = {
                        "report": text,
                        "legal_explanation": text,
                        "fine_amount": payload.fine_amount,
                        "payment_instructions": "Refer to the issuing authority.",
                        "violation_summary": text,
                    }
                return ReportResult(report_json=structured, report_text=text)
            except Exception:
                pass
        structured = self._mock_response(payload)
        return ReportResult(report_json=structured, report_text=structured["report"])
    def _mock_response(self, payload: ReportPayload) -> Dict[str, Any]:
        report = (
            f"Violation {payload.violation_id} recorded on {payload.timestamp} for vehicle "
            f"{payload.plate_number} at {payload.location} ({payload.location_code}). The vehicle failed to yield to a "
            f"pedestrian moving {payload.pedestrian_direction} within the marked crosswalk."
        )
        explanation = (
            "The observed conduct constitutes a crosswalk right-of-way violation because the "
            "vehicle entered the crosswalk while a pedestrian had lawful priority of movement."
        )
        payment = (
            f"Pay ${payload.fine_amount:.2f} to {payload.authority_name} within 30 days "
            "using the reference number listed on the invoice."
        )
        return {
            "report": report,
            "legal_explanation": explanation,
            "fine_amount": payload.fine_amount,
            "payment_instructions": payment,
            "violation_summary": "Failure to yield to a pedestrian in the marked crosswalk.",
        }
