"""
app.py — Flask web application entry point.

Admin panel  →  /admin/*        (login required)
Violator portal → /portal       (public)
"""
from __future__ import annotations

import csv
import io
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from flask import (
    Flask,
    jsonify,
    redirect,
    render_template,
    request,
    Response,
    session,
    url_for,
)

from auth import (
    LOCKOUT_MINUTES,
    check_admin_credentials,
    clear_failed_attempts,
    get_lockout_remaining,
    is_locked_out,
    login_required,
    record_failed_attempt,
)
from database import db_connection, init_db, log_audit
from stream import camera_manager, mjpeg_generator

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

app = Flask(__name__)
app.secret_key = "cw-enforcement-secret-change-in-production-please"

with app.app_context():
    init_db()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rows(rows) -> list:
    return [dict(r) for r in rows]


def _row(row) -> dict | None:
    return dict(row) if row else None


def _admin() -> str:
    return session.get("admin", "system")


def _disp_location(v: dict) -> str:
    """Prefer location_name, fall back to location."""
    return v.get("location_name") or v.get("location") or ""


# ---------------------------------------------------------------------------
# ── PUBLIC ROUTES ────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return redirect(url_for("portal"))


@app.route("/portal")
def portal():
    return render_template("portal/index.html")


@app.route("/portal/lookup", methods=["POST"])
def portal_lookup():
    plate = request.form.get("plate_number", "").upper().replace(" ", "").strip()
    if not plate:
        return render_template("portal/index.html", error="Please enter a plate number.")

    with db_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, timestamp, violation_type, severity,
                   location, location_name, snapshot_path,
                   plate_number, confidence
            FROM   violations
            WHERE  plate_number = ?
            ORDER  BY timestamp DESC
            """,
            (plate,),
        ).fetchall()

    log_audit("PORTAL_LOOKUP", target=plate, username="portal")

    violations = _rows(rows)
    for v in violations:
        v["display_location"] = _disp_location(v)

    return render_template(
        "portal/results.html",
        plate_number=plate,
        violations=violations,
    )


# ---------------------------------------------------------------------------
# ── ADMIN AUTH ───────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if "admin" in session:
        return redirect(url_for("admin_dashboard"))

    error = None
    lockout_remaining = 0

    if request.method == "POST":
        if is_locked_out():
            lockout_remaining = get_lockout_remaining()
            m, s = divmod(lockout_remaining, 60)
            error = f"Too many failed attempts. Try again in {m}m {s:02d}s."
        else:
            username = request.form.get("username", "").strip()
            password = request.form.get("password", "")
            if check_admin_credentials(username, password):
                clear_failed_attempts()
                session["admin"] = username
                log_audit("ADMIN_LOGIN", target=username, username=username)
                return redirect(url_for("admin_dashboard"))
            else:
                record_failed_attempt()
                if is_locked_out():
                    lockout_remaining = get_lockout_remaining()
                    error = (
                        f"Too many failed attempts. "
                        f"Account locked for {LOCKOUT_MINUTES} minutes."
                    )
                else:
                    remaining = 5 - session.get("failed_attempts", 0)
                    error = f"Invalid credentials. {remaining} attempt(s) remaining."

    return render_template(
        "admin/login.html",
        error=error,
        lockout_remaining=lockout_remaining,
    )


@app.route("/admin/logout")
def admin_logout():
    session.clear()
    return redirect(url_for("admin_login"))


# ---------------------------------------------------------------------------
# ── ADMIN DASHBOARD ──────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

@app.route("/admin/dashboard")
@login_required
def admin_dashboard():
    with db_connection() as conn:
        total_violations = conn.execute(
            "SELECT COUNT(*) FROM violations"
        ).fetchone()[0]

        high_severity_count = conn.execute(
            "SELECT COUNT(*) FROM violations WHERE severity = 'HIGH'"
        ).fetchone()[0]

        violations_today = conn.execute(
            "SELECT COUNT(*) FROM violations WHERE DATE(timestamp) = DATE('now')"
        ).fetchone()[0]

        unique_vehicles = conn.execute(
            "SELECT COUNT(DISTINCT vehicle_id) FROM violations"
        ).fetchone()[0]

        with_plate = conn.execute(
            """SELECT COUNT(*) FROM violations
               WHERE plate_number IS NOT NULL AND plate_number != ''"""
        ).fetchone()[0]

        plate_detected_rate = (
            round(with_plate / total_violations * 100, 1) if total_violations else 0.0
        )

        recent_rows = conn.execute(
            """SELECT id, timestamp, plate_number, vehicle_id,
                      violation_type, severity, location, location_name,
                      confidence, snapshot_path
               FROM   violations
               ORDER  BY timestamp DESC LIMIT 10"""
        ).fetchall()

        hourly_rows = conn.execute(
            """SELECT CAST(strftime('%H', timestamp) AS INTEGER) AS hour,
                      COUNT(*) AS cnt
               FROM   violations GROUP BY hour"""
        ).fetchall()
        hourly_map = {r["hour"]: r["cnt"] for r in hourly_rows}
        hourly_data = [hourly_map.get(h, 0) for h in range(24)]

        top_rows = conn.execute(
            """SELECT COALESCE(NULLIF(plate_number,''), 'VEH-' || vehicle_id) AS label,
                      COUNT(*) AS cnt
               FROM   violations
               GROUP  BY COALESCE(NULLIF(plate_number,''), vehicle_id)
               ORDER  BY cnt DESC LIMIT 5"""
        ).fetchall()

    recent = _rows(recent_rows)
    for v in recent:
        v["display_location"] = _disp_location(v)

    return render_template(
        "admin/dashboard.html",
        total_violations=total_violations,
        high_severity_count=high_severity_count,
        violations_today=violations_today,
        unique_vehicles=unique_vehicles,
        plate_detected_rate=plate_detected_rate,
        recent_violations=recent,
        hourly_data=hourly_data,
        top_vehicles=_rows(top_rows),
    )


@app.route("/admin/dashboard/data")
@login_required
def admin_dashboard_data():
    """JSON endpoint for live stat refresh (called every 60 s by dashboard.js)."""
    with db_connection() as conn:
        total_violations = conn.execute(
            "SELECT COUNT(*) FROM violations"
        ).fetchone()[0]
        violations_today = conn.execute(
            "SELECT COUNT(*) FROM violations WHERE DATE(timestamp) = DATE('now')"
        ).fetchone()[0]
    return jsonify(
        total_violations=total_violations,
        violations_today=violations_today,
    )


# ---------------------------------------------------------------------------
# ── ADMIN VIOLATIONS LIST ────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

@app.route("/admin/violations")
@login_required
def admin_violations():
    severity      = request.args.get("severity", "").strip()
    vtype         = request.args.get("violation_type", "").strip()
    date_from     = request.args.get("date_from", "").strip()
    date_to       = request.args.get("date_to", "").strip()
    plate         = request.args.get("plate", "").upper().strip()
    page          = max(1, int(request.args.get("page", 1)))
    per_page      = int(request.args.get("per_page", 25))

    conditions, params = [], []
    if severity:
        conditions.append("severity = ?"); params.append(severity)
    if vtype:
        conditions.append("violation_type = ?"); params.append(vtype)
    if date_from:
        conditions.append("DATE(timestamp) >= ?"); params.append(date_from)
    if date_to:
        conditions.append("DATE(timestamp) <= ?"); params.append(date_to)
    if plate:
        conditions.append("plate_number LIKE ?"); params.append(f"%{plate}%")

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    with db_connection() as conn:
        total_count = conn.execute(
            f"SELECT COUNT(*) FROM violations {where}", params
        ).fetchone()[0]

        offset = (page - 1) * per_page
        rows = conn.execute(
            f"""SELECT id, timestamp, plate_number, vehicle_id,
                       violation_type, severity, location, location_name,
                       confidence, snapshot_path
                FROM   violations {where}
                ORDER  BY timestamp DESC
                LIMIT  ? OFFSET ?""",
            params + [per_page, offset],
        ).fetchall()

    total_pages = max(1, math.ceil(total_count / per_page))
    violations = _rows(rows)
    for v in violations:
        v["display_location"] = _disp_location(v)

    return render_template(
        "admin/violations.html",
        violations=violations,
        current_page=page,
        total_pages=total_pages,
        total_count=total_count,
        per_page=per_page,
        filters=dict(
            severity=severity,
            violation_type=vtype,
            date_from=date_from,
            date_to=date_to,
            plate=plate,
        ),
    )


# ---------------------------------------------------------------------------
# ── ADMIN VIOLATION DETAIL ───────────────────────────────────────────────────
# ---------------------------------------------------------------------------

@app.route("/admin/violations/<string:violation_id>")
@login_required
def admin_violation_detail(violation_id: str):
    with db_connection() as conn:
        row = conn.execute(
            "SELECT * FROM violations WHERE id = ?", (violation_id,)
        ).fetchone()

    if not row:
        return render_template("admin/404.html"), 404

    log_audit("VIEW_VIOLATION", target=violation_id, username=_admin())
    v = _row(row)
    v["display_location"] = _disp_location(v)
    return render_template("admin/violation_detail.html", violation=v)


# ---------------------------------------------------------------------------
# ── ADMIN VEHICLES ───────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

@app.route("/admin/vehicles")
@login_required
def admin_vehicles():
    with db_connection() as conn:
        rows = conn.execute(
            """SELECT
                   COALESCE(NULLIF(plate_number,''), '') AS plate_number,
                   vehicle_id,
                   COUNT(*)                              AS total_violations,
                   MIN(timestamp)                        AS first_seen,
                   MAX(timestamp)                        AS last_seen,
                   SUM(CASE WHEN severity='HIGH' THEN 1 ELSE 0 END) AS high_count,
                   SUM(CASE WHEN severity='LOW'  THEN 1 ELSE 0 END) AS low_count
               FROM   violations
               GROUP  BY COALESCE(NULLIF(plate_number,''), CAST(vehicle_id AS TEXT))
               ORDER  BY total_violations DESC"""
        ).fetchall()
    return render_template(
        "admin/vehicles.html",
        vehicles=_rows(rows),
        vehicle_detail=None,
        identifier=None,
    )


@app.route("/admin/vehicles/<string:identifier>")
@login_required
def admin_vehicle_detail(identifier: str):
    with db_connection() as conn:
        rows = conn.execute(
            """SELECT id, timestamp, plate_number, vehicle_id,
                      violation_type, severity, location, location_name,
                      confidence, snapshot_path
               FROM   violations
               WHERE  plate_number = ? OR CAST(vehicle_id AS TEXT) = ?
               ORDER  BY timestamp DESC""",
            (identifier, identifier),
        ).fetchall()

    if not rows:
        return render_template("admin/404.html"), 404

    violations = _rows(rows)
    for v in violations:
        v["display_location"] = _disp_location(v)

    return render_template(
        "admin/vehicles.html",
        vehicles=None,
        vehicle_detail=violations,
        identifier=identifier,
    )


# ---------------------------------------------------------------------------
# ── ADMIN ANALYTICS ──────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

@app.route("/admin/analytics")
@login_required
def admin_analytics():
    with db_connection() as conn:
        hourly_rows = conn.execute(
            """SELECT CAST(strftime('%H', timestamp) AS INTEGER) AS hour,
                      COUNT(*) AS cnt
               FROM   violations GROUP BY hour"""
        ).fetchall()
        hourly_heatmap = {r["hour"]: r["cnt"] for r in hourly_rows}

        daily_rows = conn.execute(
            """SELECT DATE(timestamp) AS day, COUNT(*) AS cnt
               FROM   violations
               WHERE  timestamp >= DATE('now', '-30 days')
               GROUP  BY day ORDER BY day"""
        ).fetchall()
        daily_counts = [{"day": r["day"], "cnt": r["cnt"]} for r in daily_rows]

        type_rows = conn.execute(
            "SELECT violation_type, COUNT(*) AS cnt FROM violations GROUP BY violation_type"
        ).fetchall()
        violation_type_split = {r["violation_type"]: r["cnt"] for r in type_rows}

        sev_rows = conn.execute(
            "SELECT severity, COUNT(*) AS cnt FROM violations GROUP BY severity"
        ).fetchall()
        severity_split = {r["severity"]: r["cnt"] for r in sev_rows}

        loc_rows = conn.execute(
            """SELECT COALESCE(NULLIF(location_name,''), location) AS loc,
                      COUNT(*) AS cnt
               FROM   violations
               GROUP  BY loc ORDER BY cnt DESC LIMIT 10"""
        ).fetchall()
        top_locations = [{"location": r["loc"], "cnt": r["cnt"]} for r in loc_rows]

    return render_template(
        "admin/analytics.html",
        hourly_heatmap=hourly_heatmap,
        daily_counts=daily_counts,
        violation_type_split=violation_type_split,
        severity_split=severity_split,
        top_locations=top_locations,
    )


# ---------------------------------------------------------------------------
# ── ADMIN CSV EXPORT ─────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

@app.route("/admin/export/csv")
@login_required
def admin_export_csv():
    severity  = request.args.get("severity", "").strip()
    vtype     = request.args.get("violation_type", "").strip()
    date_from = request.args.get("date_from", "").strip()
    date_to   = request.args.get("date_to", "").strip()

    conditions, params = [], []
    if severity:
        conditions.append("severity = ?"); params.append(severity)
    if vtype:
        conditions.append("violation_type = ?"); params.append(vtype)
    if date_from:
        conditions.append("DATE(timestamp) >= ?"); params.append(date_from)
    if date_to:
        conditions.append("DATE(timestamp) <= ?"); params.append(date_to)

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    with db_connection() as conn:
        rows = conn.execute(
            f"""SELECT id, timestamp, plate_number, vehicle_id,
                       violation_type, severity,
                       COALESCE(NULLIF(location_name,''), location) AS location_name,
                       confidence, snapshot_path
                FROM   violations {where}
                ORDER  BY timestamp DESC""",
            params,
        ).fetchall()

    log_audit("EXPORT_CSV", target="all", username=_admin())

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "id", "timestamp", "plate_number", "car_id",
        "violation_type", "severity", "location_name", "confidence", "snapshot_path",
    ])
    for r in rows:
        writer.writerow([
            r["id"], r["timestamp"], r["plate_number"], r["vehicle_id"],
            r["violation_type"], r["severity"], r["location_name"],
            r["confidence"], r["snapshot_path"],
        ])

    return Response(
        buf.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=violations_export.csv"},
    )


# ---------------------------------------------------------------------------
# ── ADMIN AUDIT LOG ──────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

@app.route("/admin/audit")
@login_required
def admin_audit():
    with db_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM audit_log ORDER BY timestamp DESC LIMIT 100"
        ).fetchall()
    return render_template("admin/audit.html", entries=_rows(rows))


# ---------------------------------------------------------------------------
# ── TEMPLATE CONTEXT ─────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

@app.context_processor
def inject_globals():
    return {
        "current_year":     datetime.now(timezone.utc).year,
        "yandex_maps_key":  os.getenv("YANDEX_MAPS_API_KEY", ""),
        "camera_source_default": os.getenv("CAMERA_SOURCE", "0"),
    }


# ---------------------------------------------------------------------------
# ── ADMIN LIVE STREAM ────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

@app.route("/admin/live")
@login_required
def admin_live():
    return render_template(
        "admin/live.html",
        camera_status=camera_manager.status(),
        default_source=os.getenv("CAMERA_SOURCE", "0"),
    )


@app.route("/admin/live/feed")
@login_required
def admin_live_feed():
    """MJPEG stream endpoint — embed as <img src='/admin/live/feed'>."""
    return Response(
        mjpeg_generator(camera_manager),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/admin/live/start", methods=["POST"])
@login_required
def admin_live_start():
    source = request.form.get("source", "0").strip()
    ok = camera_manager.start(source)
    log_audit("CAMERA_START", target=source, username=_admin())
    return jsonify(ok=ok, **camera_manager.status())


@app.route("/admin/live/stop", methods=["POST"])
@login_required
def admin_live_stop():
    camera_manager.stop()
    log_audit("CAMERA_STOP", username=_admin())
    return jsonify(ok=True, **camera_manager.status())


@app.route("/admin/live/status")
@login_required
def admin_live_status():
    return jsonify(**camera_manager.status())


@app.route("/admin/live/recent")
@login_required
def admin_live_recent():
    """Latest 15 violations as JSON — polled every 3 s by the live page."""
    with db_connection() as conn:
        rows = conn.execute(
            """SELECT id, timestamp, plate_number, vehicle_id,
                      violation_type, severity, snapshot_path
               FROM   violations
               ORDER  BY timestamp DESC LIMIT 15"""
        ).fetchall()
    return jsonify(violations=[dict(r) for r in rows])


# ---------------------------------------------------------------------------
# ── ADMIN INVOICE ────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

@app.route("/admin/violations/<string:violation_id>/invoice")
@login_required
def admin_invoice(violation_id: str):
    with db_connection() as conn:
        row = conn.execute(
            "SELECT * FROM violations WHERE id = ?", (violation_id,)
        ).fetchone()
    if not row:
        return render_template("admin/404.html"), 404

    v = _row(row)
    v["display_location"] = _disp_location(v)

    # Payment deadline = violation date + 30 days
    deadline = "—"
    if v.get("timestamp"):
        try:
            ts_str = v["timestamp"][:19].replace("T", " ")
            ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
            deadline = (ts + timedelta(days=30)).strftime("%d %B %Y")
        except Exception:
            pass

    log_audit("VIEW_INVOICE", target=violation_id, username=_admin())
    return render_template(
        "admin/invoice_view.html",
        violation=v,
        deadline=deadline,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
