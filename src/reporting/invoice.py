from __future__ import annotations

from pathlib import Path

from config import AppSettings
from schemas import ReportPayload


class InvoiceGenerator:
    def __init__(self, settings: AppSettings):
        self.settings = settings

    def generate(self, payload: ReportPayload) -> Path:
        try:
            return self._generate_pdf(payload)
        except ModuleNotFoundError:
            return self._generate_text(payload)

    def _generate_pdf(self, payload: ReportPayload) -> Path:
        from reportlab.lib.pagesizes import LETTER  # type: ignore
        from reportlab.pdfgen import canvas  # type: ignore

        invoice_path = self.settings.storage.invoices_dir / f"{payload.violation_id}.pdf"
        pdf = canvas.Canvas(str(invoice_path), pagesize=LETTER)
        pdf.setFont("Helvetica-Bold", 16)
        pdf.drawString(72, 740, self.settings.runtime.authority_name)
        pdf.setFont("Helvetica", 11)
        pdf.drawString(72, 710, f"Violation ID: {payload.violation_id}")
        pdf.drawString(72, 690, f"Plate Number: {payload.plate_number}")
        pdf.drawString(72, 670, f"Violation Type: {payload.violation_type}")
        pdf.drawString(72, 650, f"Date: {payload.timestamp}")
        pdf.drawString(72, 630, f"Fine Amount: ${payload.fine_amount:.2f}")
        pdf.drawString(72, 610, f"Authority: {payload.authority_name}")
        pdf.drawString(72, 590, "Payment due within 30 days.")
        pdf.save()
        return invoice_path

    def _generate_text(self, payload: ReportPayload) -> Path:
        invoice_path = self.settings.storage.invoices_dir / f"{payload.violation_id}.txt"
        invoice_path.write_text(
            "\n".join(
                [
                    self.settings.runtime.authority_name,
                    f"Violation ID: {payload.violation_id}",
                    f"Plate Number: {payload.plate_number}",
                    f"Violation Type: {payload.violation_type}",
                    f"Date: {payload.timestamp}",
                    f"Fine Amount: ${payload.fine_amount:.2f}",
                    "Payment due within 30 days.",
                ]
            ),
            encoding="utf-8",
        )
        return invoice_path
