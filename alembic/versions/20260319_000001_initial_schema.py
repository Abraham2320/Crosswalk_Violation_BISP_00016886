"""initial schema

Revision ID: 20260319_000001
Revises:
Create Date: 2026-03-19 19:05:00
"""

from alembic import op
import sqlalchemy as sa


revision = "20260319_000001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "vehicles",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("plate_number", sa.String(length=32), nullable=False, unique=True),
        sa.Column("owner_name", sa.String(length=128), nullable=True),
        sa.Column("violations_count", sa.Integer(), nullable=False, server_default="0"),
    )
    op.create_index("ix_vehicles_plate_number", "vehicles", ["plate_number"], unique=True)

    op.create_table(
        "violations",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("plate_number", sa.String(length=32), nullable=True),
        sa.Column("vehicle_id", sa.Integer(), nullable=False),
        sa.Column("vehicle_image_path", sa.Text(), nullable=False),
        sa.Column("frame_image_path", sa.Text(), nullable=False),
        sa.Column("plate_image_path", sa.Text(), nullable=True),
        sa.Column("report_path", sa.Text(), nullable=True),
        sa.Column("invoice_path", sa.Text(), nullable=True),
        sa.Column("violation_type", sa.String(length=64), nullable=False),
        sa.Column("pedestrian_direction", sa.String(length=32), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="0"),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="pending"),
        sa.Column("location", sa.String(length=128), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("llm_report_json", sa.Text(), nullable=True),
        sa.Column("llm_report_text", sa.Text(), nullable=True),
        sa.Column("vehicle_ref_id", sa.String(length=64), sa.ForeignKey("vehicles.id", ondelete="SET NULL"), nullable=True),
        sa.Column("severity", sa.String(length=32), nullable=False, server_default="HIGH"),
        sa.Column("snapshot_path", sa.Text(), nullable=True),
        sa.Column("location_name", sa.String(length=255), nullable=True),
        sa.Column("vehicle_speed_estimate", sa.Float(), nullable=True),
        sa.Column("plate_crop_path", sa.Text(), nullable=True),
    )
    op.create_index("ix_violations_plate_number", "violations", ["plate_number"], unique=False)
    op.create_index("ix_violations_plate_timestamp", "violations", ["plate_number", "timestamp"], unique=False)

    op.create_table(
        "invoices",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("violation_id", sa.String(length=64), sa.ForeignKey("violations.id", ondelete="CASCADE"), nullable=False, unique=True),
        sa.Column("amount", sa.Numeric(10, 2), nullable=False),
        sa.Column("issued_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="issued"),
        sa.Column("pdf_path", sa.Text(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("invoices")
    op.drop_index("ix_violations_plate_timestamp", table_name="violations")
    op.drop_index("ix_violations_plate_number", table_name="violations")
    op.drop_table("violations")
    op.drop_index("ix_vehicles_plate_number", table_name="vehicles")
    op.drop_table("vehicles")
