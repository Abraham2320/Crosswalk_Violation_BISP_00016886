from __future__ import annotations

from pathlib import Path

from config import AppSettings
from schemas import ReportPayload

_STATIC = Path(__file__).parent.parent.parent / "static"


def _resolve_image(rel_path: str | None) -> Path | None:
    """Return absolute path if the image file exists, else None."""
    if not rel_path:
        return None
    p = _STATIC / rel_path
    return p if p.exists() else None


class InvoiceGenerator:
    def __init__(self, settings: AppSettings):
        self.settings = settings

    def generate(self, payload: ReportPayload) -> Path:
        try:
            return self._generate_pdf(payload)
        except ModuleNotFoundError:
            return self._generate_text(payload)

    def _generate_pdf(self, payload: ReportPayload) -> Path:
        from reportlab.lib.pagesizes import LETTER          # type: ignore
        from reportlab.lib.utils import ImageReader         # type: ignore
        from reportlab.pdfgen import canvas                 # type: ignore

        invoice_path = self.settings.storage.invoices_dir / f"{payload.violation_id}.pdf"
        W, H = LETTER   # 612 × 792 pts
        pdf = canvas.Canvas(str(invoice_path), pagesize=LETTER)

        # ── Header ──────────────────────────────────────────────────────────────
        pdf.setFillColorRGB(0.10, 0.10, 0.10)
        pdf.rect(0, H - 60, W, 60, fill=1, stroke=0)
        pdf.setFillColorRGB(1, 1, 1)
        pdf.setFont("Helvetica-Bold", 15)
        pdf.drawString(36, H - 38, self.settings.runtime.authority_name)
        pdf.setFont("Helvetica", 9)
        pdf.drawRightString(W - 36, H - 38, "TRAFFIC VIOLATION NOTICE")

        # ── Violation details ────────────────────────────────────────────────────
        pdf.setFillColorRGB(0.1, 0.1, 0.1)
        y = H - 90
        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(36, y, f"Violation ID:  {payload.violation_id}")
        y -= 22
        pdf.setFont("Helvetica", 11)
        pdf.drawString(36, y, f"Type:          {payload.violation_type.replace('_', ' ').title()}")
        y -= 18
        pdf.drawString(36, y, f"Date / Time:   {payload.timestamp}")
        y -= 18
        pdf.drawString(36, y, f"Location:      {payload.location}  [{payload.location_code}]")
        y -= 18
        pdf.drawString(36, y, f"Authority:     {payload.authority_name}")

        # ── Plate section ────────────────────────────────────────────────────────
        y -= 30
        pdf.setFillColorRGB(0.94, 0.94, 0.94)
        pdf.rect(30, y - 8, W - 60, 30, fill=1, stroke=0)
        pdf.setFillColorRGB(0.1, 0.1, 0.1)
        pdf.setFont("Helvetica-Bold", 14)
        plate_display = payload.plate_number if payload.plate_number != "UNREADABLE" else "— UNREADABLE —"
        pdf.drawString(36, y + 6, f"License Plate:  {plate_display}")
        y -= 14

        # Plate crop image
        plate_img_path = _resolve_image(payload.plate_crop_path)
        if plate_img_path:
            try:
                img = ImageReader(str(plate_img_path))
                iw, ih = img.getSize()
                max_w, max_h = 240, 72
                scale = min(max_w / max(iw, 1), max_h / max(ih, 1))
                dw, dh = iw * scale, ih * scale
                y -= dh + 8
                pdf.drawImage(img, 36, y, width=dw, height=dh,
                              preserveAspectRatio=True, mask='auto')
                pdf.setFont("Helvetica", 8)
                pdf.setFillColorRGB(0.5, 0.5, 0.5)
                pdf.drawString(36, y - 10, "Plate crop captured at time of violation")
                y -= 18
            except Exception:
                pass

        # ── Fine ────────────────────────────────────────────────────────────────
        y -= 22
        pdf.setFillColorRGB(0.85, 0.12, 0.12)
        pdf.setFont("Helvetica-Bold", 13)
        pdf.drawString(36, y, f"Fine Amount:   ${payload.fine_amount:.2f}")
        pdf.setFillColorRGB(0.1, 0.1, 0.1)
        pdf.setFont("Helvetica", 10)
        y -= 16
        pdf.drawString(36, y, "Payment due within 30 days of this notice.")

        # ── Incident snapshot ────────────────────────────────────────────────────
        snap_path = _resolve_image(payload.snapshot_path)
        if snap_path:
            try:
                y -= 28
                pdf.setFont("Helvetica-Bold", 10)
                pdf.drawString(36, y, "Incident Photo:")
                y -= 4
                img = ImageReader(str(snap_path))
                iw, ih = img.getSize()
                max_w = W - 72
                max_h = min(y - 50, 220)   # leave 50pt margin at bottom
                scale = min(max_w / max(iw, 1), max_h / max(ih, 1))
                dw, dh = iw * scale, ih * scale
                y -= dh + 4
                if y > 40:
                    pdf.drawImage(img, 36, y, width=dw, height=dh,
                                  preserveAspectRatio=True, mask='auto')
            except Exception:
                pass

        # ── Footer ──────────────────────────────────────────────────────────────
        pdf.setFont("Helvetica", 8)
        pdf.setFillColorRGB(0.5, 0.5, 0.5)
        pdf.drawCentredString(W / 2, 28, f"ID: {payload.violation_id}")

        pdf.save()
        return invoice_path

    def _generate_text(self, payload: ReportPayload) -> Path:
        invoice_path = self.settings.storage.invoices_dir / f"{payload.violation_id}.txt"
        plate_line = f"Plate Number:   {payload.plate_number}"
        if payload.plate_crop_path:
            plate_line += f"\nPlate Image:    {payload.plate_crop_path}"
        if payload.snapshot_path:
            plate_line += f"\nIncident Photo: {payload.snapshot_path}"
        invoice_path.write_text(
            "\n".join([
                self.settings.runtime.authority_name,
                "TRAFFIC VIOLATION NOTICE",
                "",
                f"Violation ID:   {payload.violation_id}",
                plate_line,
                f"Type:           {payload.violation_type}",
                f"Date / Time:    {payload.timestamp}",
                f"Location:       {payload.location} [{payload.location_code}]",
                f"Fine Amount:    ${payload.fine_amount:.2f}",
                "",
                "Payment due within 30 days.",
            ]),
            encoding="utf-8",
        )
        return invoice_path
