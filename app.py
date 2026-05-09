from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import csv
import io
import json
import math
import re
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
from werkzeug.security import generate_password_hash
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
from src.i18n import get_locale, t_for, SUPPORTED_LANGS
try:
    from audit_ui import audit_bp as _audit_bp
    _HAS_AUDIT_BP = True
except Exception as _e:
    _HAS_AUDIT_BP = False
    print(f"[WARN] audit_ui not available: {_e}")
try:
    from live_processor import live_proc, proc_registry
    from stream import camera_manager, mjpeg_generator, registry as cam_registry, CAMERA_CONFIGS
except Exception:
    _FALLBACK_JPEG = b""
    class _NullCamera:
        def __init__(self):
            self.source = "unavailable"
        def start(self, source):
            self.source = str(source)
            return False
        def stop(self):
            return None
        def get_jpeg(self):
            return _FALLBACK_JPEG
        def status(self):
            return {
                "connected": False,
                "source": self.source,
                "fps": 0.0,
                "width": 0,
                "height": 0,
                "error": "Live camera runtime is unavailable in this deployment.",
            }
    class _NullProc:
        def start_async(self):
            return False
        def stop(self):
            return None
        def reload_polygon(self):
            return None
        def get_stats(self):
            return {
                "active": False,
                "session_violations": 0,
                "last_error": "Live detection runtime is unavailable in this deployment.",
            }
    class _NullRegistry:
        def __init__(self, factory):
            self._factory = factory
            self._items = {}
        def get(self, key):
            if key not in self._items:
                self._items[key] = self._factory()
            return self._items[key]
    def mjpeg_generator(_manager):
        payload = b""
        while True:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + payload + b"\r\n"
    camera_manager = _NullCamera()
    cam_registry = _NullRegistry(_NullCamera)
    live_proc = _NullProc()
    proc_registry = _NullRegistry(_NullProc)
    CAMERA_CONFIGS = {}
try:
    import anthropic as _anthropic
except ImportError:
    _anthropic = None
PROJECT_ROOT = Path(__file__).resolve().parent
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None
if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env")
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "cw-enforcement-dev-secret-change-in-production")
with app.app_context():
    init_db()
if _HAS_AUDIT_BP:
    app.register_blueprint(_audit_bp)
def _template_context() -> dict:
    lang = get_locale()
    return {
        "current_year": datetime.now(timezone.utc).year,
        "yandex_maps_key": os.getenv("YANDEX_MAPS_API_KEY", ""),
        "camera_source_default": os.getenv("CAMERA_SOURCE", "0"),
        "t": t_for(lang),
        "lang": lang,
        "SUPPORTED_LANGS": SUPPORTED_LANGS,
        "yandex_maps_api_key": os.getenv("YANDEX_MAPS_API_KEY", ""),
        "location_latitude": os.getenv("LOCATION_LATITUDE", "41.2963"),
        "location_longitude": os.getenv("LOCATION_LONGITUDE", "69.2798"),
        "location_name": os.getenv("LOCATION_NAME", "Crosswalk A"),
    }
app.jinja_env.globals.update(
    t=lambda key, **kwargs: t_for(get_locale())(key, **kwargs),
    SUPPORTED_LANGS=SUPPORTED_LANGS,
)
def _rows(rows) -> list:
    return [dict(r) for r in rows]
def _row(row) -> dict | None:
    return dict(row) if row else None
def _admin() -> str:
    return session.get("admin", "system")
def _disp_location(v: dict) -> str:
    return v.get("location_name") or v.get("location") or ""
_CAM_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{2,32}$")
def _camera_rows() -> list[dict]:
    with db_connection() as conn:
        rows = conn.execute(
            """
            SELECT cam_id, label, default_source, demo_source,
                   location_name, latitude, longitude, tags
            FROM cameras
            WHERE is_active = 1
            ORDER BY cam_id
            """
        ).fetchall()
    return [dict(r) for r in rows]
def _refresh_camera_configs() -> None:
    rows = _camera_rows()
    CAMERA_CONFIGS.clear()
    for r in rows:
        CAMERA_CONFIGS[r["cam_id"]] = {
            "label": r["label"],
            "source": r["default_source"],
            "demo": r["demo_source"] or "",
            "location_name": r.get("location_name") or "Crosswalk A",
            "latitude": float(r.get("latitude") or 41.2963),
            "longitude": float(r.get("longitude") or 69.2798),
            "tags": [t.strip() for t in (r.get("tags") or "").split(",") if t.strip()],
        }
def _all_admin_users() -> list[dict]:
    with db_connection() as conn:
        rows = conn.execute(
            "SELECT id, username, created_at FROM admin_users ORDER BY username"
        ).fetchall()
    return [dict(r) for r in rows]
_refresh_camera_configs()
def _chat_db_context() -> str:
    try:
        with db_connection() as conn:
            total = conn.execute("SELECT COUNT(*) FROM violations").fetchone()[0]
            today = conn.execute(
                "SELECT COUNT(*) FROM violations WHERE DATE(timestamp) = DATE('now')"
            ).fetchone()[0]
            high = conn.execute(
                "SELECT COUNT(*) FROM violations WHERE severity='HIGH'"
            ).fetchone()[0]
            unique_v = conn.execute(
                "SELECT COUNT(DISTINCT vehicle_id) FROM violations"
            ).fetchone()[0]
            peak_row = conn.execute(
                "SELECT CAST(strftime('%H',timestamp) AS INTEGER) AS hr, COUNT(*) AS n "
                "FROM violations GROUP BY hr ORDER BY n DESC LIMIT 1"
            ).fetchone()
            peak_hour = f"{peak_row[0]:02d}:00" if peak_row else "N/A"
            top5 = conn.execute(
                "SELECT COALESCE(NULLIF(plate_number,''),'VEH-'||vehicle_id) AS lbl, "
                "COUNT(*) AS n FROM violations GROUP BY lbl ORDER BY n DESC LIMIT 5"
            ).fetchall()
            top5_str = ", ".join(f"{r[0]} ({r[1]}x)" for r in top5) or "N/A"
            plate_rate_row = conn.execute(
                "SELECT SUM(CASE WHEN plate_number IS NOT NULL AND plate_number!='' THEN 1 ELSE 0 END)*100.0/MAX(COUNT(*),1) "
                "FROM violations"
            ).fetchone()
            plate_rate = f"{plate_rate_row[0]:.1f}%" if plate_rate_row and plate_rate_row[0] else "N/A"
    except Exception:
        return "Database unavailable."
    cam = camera_manager.status()
    det = live_proc.get_stats()
    return (
        f"Crosswalk Violation System — Live Status\n"
        f"- Total violations recorded: {total}\n"
        f"- Violations today: {today}\n"
        f"- High-severity violations: {high}\n"
        f"- Unique vehicles detected: {unique_v}\n"
        f"- Peak violation hour: {peak_hour}\n"
        f"- Top offenders: {top5_str}\n"
        f"- Plate recognition rate: {plate_rate}\n"
        f"- Camera: {'LIVE at ' + (cam.get('source') or 'unknown') if cam.get('connected') else 'OFFLINE'}\n"
        f"- Detection: {'ACTIVE' if det.get('active') else 'INACTIVE'}, "
        f"session violations: {det.get('session_violations', 0)}\n"
        f"- Location: {os.getenv('LOCATION_NAME','Crosswalk A')}, "
        f"{os.getenv('LOCATION_LATITUDE','41.2963')}, {os.getenv('LOCATION_LONGITUDE','69.2798')}"
    )
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
@app.route("/set-language")
def set_language():
    lang = request.args.get("lang", "en")
    if lang not in SUPPORTED_LANGS:
        lang = "en"
    session["lang"] = lang
    next_url = request.args.get("next") or request.referrer or url_for("index")
    return redirect(next_url)
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
@app.route("/admin/admins")
@login_required
def admin_admins():
    users = _all_admin_users()
    return render_template(
        "admin/admins.html",
        admins=users,
        error=request.args.get("error", ""),
        ok=request.args.get("ok", ""),
    )
@app.route("/admin/admins/add", methods=["POST"])
@login_required
def admin_admins_add():
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")
    if not re.fullmatch(r"[A-Za-z0-9_.-]{3,32}", username):
        return redirect(url_for("admin_admins", error="invalid-username"))
    if len(password) < 8:
        return redirect(url_for("admin_admins", error="weak-password"))
    with db_connection() as conn:
        exists = conn.execute(
            "SELECT 1 FROM admin_users WHERE username = ?",
            (username,),
        ).fetchone()
        if exists:
            return redirect(url_for("admin_admins", error="username-exists"))
        conn.execute(
            "INSERT INTO admin_users (username, password_hash) VALUES (?, ?)",
            (username, generate_password_hash(password)),
        )
    log_audit("ADMIN_USER_ADD", target=username, username=_admin())
    return redirect(url_for("admin_admins", ok="admin-added"))
@app.route("/admin/admins/<int:admin_id>/delete", methods=["POST"])
@login_required
def admin_admins_delete(admin_id: int):
    with db_connection() as conn:
        target = conn.execute(
            "SELECT id, username FROM admin_users WHERE id = ?",
            (admin_id,),
        ).fetchone()
        if target is None:
            return redirect(url_for("admin_admins", error="admin-not-found"))
        if target["username"] == _admin():
            return redirect(url_for("admin_admins", error="cannot-delete-self"))
        total_admins = conn.execute("SELECT COUNT(*) FROM admin_users").fetchone()[0]
        if total_admins <= 1:
            return redirect(url_for("admin_admins", error="last-admin"))
        conn.execute("DELETE FROM admin_users WHERE id = ?", (admin_id,))
    log_audit("ADMIN_USER_DELETE", target=target["username"], username=_admin())
    return redirect(url_for("admin_admins", ok="admin-deleted"))
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
                       confidence, snapshot_path, plate_crop_path
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
    _lat = os.getenv("LOCATION_LATITUDE",  "41.2963")
    _lng = os.getenv("LOCATION_LONGITUDE", "69.2798")
    try:
        v["latitude"]  = float(_lat)
        v["longitude"] = float(_lng)
    except (TypeError, ValueError):
        v["latitude"]  = 41.2963
        v["longitude"] = 69.2798
    return render_template("admin/violation_detail.html", violation=v)
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
@app.route("/admin/audit")
@login_required
def admin_audit():
    with db_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM audit_log ORDER BY timestamp DESC LIMIT 100"
        ).fetchall()
    return render_template("admin/audit.html", entries=_rows(rows))
@app.context_processor
def inject_globals():
    return _template_context()
@app.route("/admin/live")
@login_required
def admin_live():
    demo_source = os.getenv("DEMO_VIDEO_SOURCE", "Videos/v2.mp4")
    return render_template(
        "admin/live.html",
        camera_status=camera_manager.status(),
        default_source=os.getenv("CAMERA_SOURCE", demo_source),
        demo_source=demo_source,
    )
@app.route("/admin/live/feed")
@login_required
def admin_live_feed():
    return Response(
        mjpeg_generator(camera_manager),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )
@app.route("/admin/live/snapshot")
@login_required
def admin_live_snapshot():
    jpeg = camera_manager.get_jpeg()
    return Response(jpeg, mimetype="image/jpeg",
                    headers={"Cache-Control": "no-store"})
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
def _build_pipeline_checks(cam_connected: bool, cam_source: str, proc_obj) -> dict:
    checks = {}
    try:
        import cv2
        checks["opencv"] = {"ok": True, "detail": f"OpenCV {cv2.__version__}"}
        try:
            cv2.ORB_create(nfeatures=250)
            checks["orb"] = {"ok": True, "detail": "ORB feature extractor available"}
        except Exception as exc:
            checks["orb"] = {"ok": False, "detail": f"ORB unavailable: {exc}"}
    except Exception as exc:
        checks["opencv"] = {"ok": False, "detail": f"OpenCV import failed: {exc}"}
        checks["orb"] = {"ok": False, "detail": "ORB unavailable because OpenCV failed"}
    try:
        from config import settings as cfg
        model_path = Path(cfg.models.detection_model_path)
        checks["yolo"] = {
            "ok": model_path.exists(),
            "detail": f"Model path: {model_path}",
        }
    except Exception as exc:
        checks["yolo"] = {"ok": False, "detail": f"YOLO config failed: {exc}"}
    try:
        from logic.violation import check_violation, update_pedestrian_state
        checks["fsm_rules"] = {
            "ok": callable(check_violation) and callable(update_pedestrian_state),
            "detail": "FSM + violation rule functions imported",
        }
    except Exception as exc:
        checks["fsm_rules"] = {"ok": False, "detail": f"Rules import failed: {exc}"}
    pipeline_initialized = getattr(proc_obj, "_pipeline", None) is not None
    pipeline_error = getattr(proc_obj, "last_error", "")
    checks["pipeline"] = {
        "ok": pipeline_initialized or not bool(pipeline_error),
        "detail": (
            "Pipeline initialized"
            if pipeline_initialized
            else f"Pipeline start failed: {pipeline_error}"
            if pipeline_error
            else "Pipeline ready (starts when detection starts)"
        ),
    }
    checks["camera"] = {
        "ok": bool(cam_connected),
        "detail": cam_source or "No source connected",
    }
    return checks
@app.route("/admin/live/checks")
@login_required
def admin_live_checks():
    cam_status = camera_manager.status()
    checks = _build_pipeline_checks(
        cam_connected=cam_status.get("connected", False),
        cam_source=cam_status.get("source") or "",
        proc_obj=live_proc,
    )
    return jsonify(ok=all(v.get("ok") for v in checks.values()), checks=checks)
@app.route("/admin/live/recent")
@login_required
def admin_live_recent():
    with db_connection() as conn:
        rows = conn.execute(
            """SELECT id, timestamp, plate_number, vehicle_id,
                      violation_type, severity, snapshot_path
               FROM   violations
               ORDER  BY timestamp DESC LIMIT 15"""
        ).fetchall()
    return jsonify(violations=[dict(r) for r in rows])
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
    deadline = "—"
    if v.get("timestamp"):
        try:
            ts_str = v["timestamp"][:19].replace("T", " ")
            ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
            deadline = (ts + timedelta(days=30)).strftime("%d %B %Y")
        except Exception:
            pass
    _lat = os.getenv("LOCATION_LATITUDE",  "41.2963")
    _lng = os.getenv("LOCATION_LONGITUDE", "69.2798")
    try:
        v["latitude"]  = float(_lat)
        v["longitude"] = float(_lng)
    except (TypeError, ValueError):
        v["latitude"]  = 41.2963
        v["longitude"] = 69.2798
    v["location_address"] = os.getenv("LOCATION_ADDRESS", "")
    log_audit("VIEW_INVOICE", target=violation_id, username=_admin())
    return render_template(
        "admin/invoice_view.html",
        violation=v,
        deadline=deadline,
    )
@app.route("/admin/live/detection/start", methods=["POST"])
@login_required
def admin_live_detection_start():
    cam = camera_manager.status()
    if not cam.get("connected"):
        return jsonify(
            ok=False,
            error="Camera is not connected. Connect a source first.",
            **live_proc.get_stats(),
        ), 400
    polygon_path = PROJECT_ROOT / "crosswalk_polygon.json"
    if not polygon_path.exists():
        return jsonify(
            ok=False,
            error="No crosswalk zone defined. Draw and save a polygon first.",
            **live_proc.get_stats(),
        ), 400
    try:
        pts = json.loads(polygon_path.read_text())
        if not isinstance(pts, list) or len(pts) < 4:
            return jsonify(
                ok=False,
                error="Crosswalk zone has fewer than 4 points. Re-draw the polygon.",
                **live_proc.get_stats(),
            ), 400
    except Exception:
        return jsonify(
            ok=False,
            error="Polygon file is invalid. Re-draw the zone.",
            **live_proc.get_stats(),
        ), 400
    ok = live_proc.start_async()
    log_audit("DETECTION_START", username=_admin())
    return jsonify(ok=ok, **live_proc.get_stats())
@app.route("/admin/live/detection/stop", methods=["POST"])
@login_required
def admin_live_detection_stop():
    live_proc.stop()
    log_audit("DETECTION_STOP", username=_admin())
    return jsonify(ok=True, **live_proc.get_stats())
@app.route("/admin/live/detection/status")
@login_required
def admin_live_detection_status():
    return jsonify(**live_proc.get_stats())
@app.route("/admin/live/polygon", methods=["GET"])
@login_required
def admin_live_polygon_get():
    polygon_path = PROJECT_ROOT / "crosswalk_polygon.json"
    if polygon_path.exists():
        try:
            pts = json.loads(polygon_path.read_text())
            if not isinstance(pts, list) or len(pts) < 4:
                return jsonify(points=[])
            return jsonify(points=pts)
        except Exception:
            pass
    return jsonify(points=[])
@app.route("/admin/live/polygon", methods=["POST"])
@login_required
def admin_live_polygon_save():
    data   = request.get_json(silent=True) or {}
    points = data.get("points", [])
    polygon_path = PROJECT_ROOT / "crosswalk_polygon.json"
    if not points:
        if polygon_path.exists():
            polygon_path.unlink()
        live_proc.reload_polygon()
        log_audit("POLYGON_CLEAR", username=_admin())
        return jsonify(ok=True, count=0)
    if not isinstance(points, list) or len(points) < 4:
        return jsonify(ok=False, error="Need at least 4 points"), 400
    try:
        normalized = []
        for p in points:
            if not isinstance(p, (list, tuple)) or len(p) != 2:
                return jsonify(ok=False, error="Invalid point format"), 400
            normalized.append([int(p[0]), int(p[1])])
    except Exception:
        return jsonify(ok=False, error="Polygon points must be numeric"), 400
    polygon_path.write_text(json.dumps(normalized))
    live_proc.reload_polygon()
    log_audit("POLYGON_SAVE", target=f"{len(normalized)} pts", username=_admin())
    return jsonify(ok=True, count=len(normalized))
_CHAT_SYSTEM = (
    "You are a traffic safety analyst assistant for a crosswalk violation detection "
    "system in Tashkent, Uzbekistan. You have direct access to live violation data shown "
    "at the start of each message. You can perform data analysis, identify trends, "
    "compare time periods, explain detection logic, and give actionable recommendations. "
    "Be concise, factual, and use markdown formatting (bold, lists) where helpful. "
    "Answer only in English unless the user asks otherwise."
)
_CHAT_MODEL = "claude-sonnet-4-6"
@app.route("/api/chat", methods=["POST"])
@login_required
def api_chat():
    if _anthropic is None:
        return jsonify(ok=False, error="anthropic package not installed"), 500
    data    = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    history = data.get("history") or []
    if not message:
        return jsonify(ok=False, error="Empty message"), 400
    db_ctx   = _chat_db_context()
    first_content = f"Current system data:\n{db_ctx}\n\nUser question: {message}"
    messages = []
    for h in history[-12:]:
        role    = h.get("role", "user")
        content = h.get("content", "")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": first_content})
    try:
        client = _anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))
        resp   = client.messages.create(
            model=_CHAT_MODEL,
            max_tokens=1024,
            system=_CHAT_SYSTEM,
            messages=messages,
        )
        answer = resp.content[0].text
        return jsonify(ok=True, answer=answer)
    except Exception as exc:
        return jsonify(ok=False, error=str(exc)), 500
@app.route("/admin/api/summary", methods=["POST"])
@login_required
def admin_api_summary():
    if _anthropic is None:
        return jsonify(ok=False, error="anthropic package not installed"), 500
    db_ctx = _chat_db_context()
    prompt = (
        f"Based on the following data, generate a concise executive summary "
        f"(3-5 short paragraphs) covering key metrics, notable patterns, "
        f"current system status, and one or two actionable recommendations.\n\n"
        f"{db_ctx}"
    )
    try:
        client = _anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))
        resp   = client.messages.create(
            model=_CHAT_MODEL,
            max_tokens=900,
            system=_CHAT_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        summary = resp.content[0].text
        return jsonify(ok=True, summary=summary)
    except Exception as exc:
        return jsonify(ok=False, error=str(exc)), 500
@app.route("/admin/chatbot")
@login_required
def admin_chatbot():
    with db_connection() as conn:
        total = conn.execute("SELECT COUNT(*) FROM violations").fetchone()[0]
        today = conn.execute(
            "SELECT COUNT(*) FROM violations WHERE DATE(timestamp) = DATE('now')"
        ).fetchone()[0]
        this_week = conn.execute(
            "SELECT COUNT(*) FROM violations WHERE timestamp >= DATE('now','-7 days')"
        ).fetchone()[0]
        high_sev = conn.execute(
            "SELECT COUNT(*) FROM violations WHERE severity='HIGH'"
        ).fetchone()[0]
        unique_veh = conn.execute(
            "SELECT COUNT(DISTINCT vehicle_id) FROM violations"
        ).fetchone()[0]
        plate_rate_row = conn.execute(
            "SELECT ROUND(SUM(CASE WHEN plate_number IS NOT NULL AND plate_number!='' THEN 1 ELSE 0 END)*100.0/MAX(COUNT(*),1),1) FROM violations"
        ).fetchone()
        plate_rate = plate_rate_row[0] if plate_rate_row and plate_rate_row[0] else 0.0
        type_rows = conn.execute(
            "SELECT violation_type, COUNT(*) AS cnt FROM violations GROUP BY violation_type ORDER BY cnt DESC"
        ).fetchall()
        recent_rows = conn.execute(
            "SELECT timestamp, violation_type, plate_number, severity FROM violations ORDER BY timestamp DESC LIMIT 5"
        ).fetchall()
    det = live_proc.get_stats()
    stats = {
        "total": total, "today": today, "this_week": this_week,
        "high_severity": high_sev, "unique_vehicles": unique_veh,
        "plate_rate": plate_rate,
        "detection_active": det.get("active", False),
        "session_violations": det.get("session_violations", 0),
        "types": [{"type": r["violation_type"], "count": r["cnt"]} for r in type_rows],
        "recent": [dict(r) for r in recent_rows],
    }
    return render_template("admin/chatbot.html", stats=stats, model=_CHAT_MODEL)
@app.context_processor
def inject_global_vars():
    return _template_context()
@app.route("/admin/cameras")
@login_required
def admin_cameras():
    _refresh_camera_configs()
    statuses = {}
    for cam_id, cfg in CAMERA_CONFIGS.items():
        cam  = cam_registry.get(cam_id)
        proc = proc_registry.get(cam_id)
        statuses[cam_id] = {
            "config":     cfg,
            "stream":     cam.status(),
            "detection":  proc.get_stats(),
        }
    return render_template(
        "admin/cameras.html",
        camera_statuses=statuses,
        error=request.args.get("error", ""),
        ok=request.args.get("ok", ""),
    )
@app.route("/admin/cameras/add", methods=["POST"])
@login_required
def admin_cameras_add():
    cam_id = request.form.get("cam_id", "").strip()
    label = request.form.get("label", "").strip()
    source = request.form.get("source", "").strip() or "0"
    demo_source = request.form.get("demo_source", "").strip()
    location_name = request.form.get("location_name", "").strip() or label
    tags_raw = request.form.get("tags", "").strip()
    lat_raw = request.form.get("latitude", "").strip() or "41.2963"
    lng_raw = request.form.get("longitude", "").strip() or "69.2798"
    if not _CAM_ID_PATTERN.fullmatch(cam_id):
        return redirect(url_for("admin_cameras", error="invalid-cam-id"))
    if not label:
        return redirect(url_for("admin_cameras", error="label-required"))
    try:
        latitude = float(lat_raw)
        longitude = float(lng_raw)
    except ValueError:
        return redirect(url_for("admin_cameras", error="invalid-location"))
    if not (-90.0 <= latitude <= 90.0 and -180.0 <= longitude <= 180.0):
        return redirect(url_for("admin_cameras", error="invalid-location"))
    tags = ", ".join([t.strip() for t in tags_raw.split(",") if t.strip()])
    with db_connection() as conn:
        exists = conn.execute(
            "SELECT 1 FROM cameras WHERE cam_id = ?",
            (cam_id,),
        ).fetchone()
        if exists:
            return redirect(url_for("admin_cameras", error="cam-id-exists"))
        conn.execute(
            """
            INSERT INTO cameras (
                cam_id, label, default_source, demo_source,
                location_name, latitude, longitude, tags, is_active
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
            """,
            (cam_id, label, source, demo_source, location_name, latitude, longitude, tags),
        )
    _refresh_camera_configs()
    log_audit("CAMERA_ADD", target=f"{cam_id}:{source}", username=_admin())
    return redirect(url_for("admin_cameras", ok="camera-added"))
@app.route("/admin/cameras/<string:cam_id>/location", methods=["POST"])
@login_required
def admin_camera_location_update(cam_id: str):
    if cam_id not in CAMERA_CONFIGS:
        return redirect(url_for("admin_cameras", error="camera-not-found"))
    location_name = request.form.get("location_name", "").strip() or CAMERA_CONFIGS[cam_id].get("label", cam_id)
    tags_raw = request.form.get("tags", "").strip()
    lat_raw = request.form.get("latitude", "").strip()
    lng_raw = request.form.get("longitude", "").strip()
    try:
        latitude = float(lat_raw)
        longitude = float(lng_raw)
    except ValueError:
        return redirect(url_for("admin_camera_detail", cam_id=cam_id, error="invalid-location"))
    if not (-90.0 <= latitude <= 90.0 and -180.0 <= longitude <= 180.0):
        return redirect(url_for("admin_camera_detail", cam_id=cam_id, error="invalid-location"))
    tags = ", ".join([t.strip() for t in tags_raw.split(",") if t.strip()])
    with db_connection() as conn:
        conn.execute(
            """
            UPDATE cameras
            SET location_name = ?, latitude = ?, longitude = ?, tags = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE cam_id = ?
            """,
            (location_name, latitude, longitude, tags, cam_id),
        )
    _refresh_camera_configs()
    log_audit("CAMERA_LOCATION_UPDATE", target=f"{cam_id}:{location_name}", username=_admin())
    return redirect(url_for("admin_camera_detail", cam_id=cam_id, ok="location-updated"))
@app.route("/admin/cameras/<string:cam_id>/delete", methods=["POST"])
@login_required
def admin_cameras_delete(cam_id: str):
    with db_connection() as conn:
        row = conn.execute(
            "SELECT cam_id FROM cameras WHERE cam_id = ? AND is_active = 1",
            (cam_id,),
        ).fetchone()
        if row is None:
            return redirect(url_for("admin_cameras", error="camera-not-found"))
        active_count = conn.execute(
            "SELECT COUNT(*) FROM cameras WHERE is_active = 1"
        ).fetchone()[0]
        if active_count <= 1:
            return redirect(url_for("admin_cameras", error="last-camera"))
        conn.execute("DELETE FROM cameras WHERE cam_id = ?", (cam_id,))
    try:
        cam_registry.get(cam_id).stop()
    except Exception:
        pass
    try:
        proc_registry.get(cam_id).stop()
    except Exception:
        pass
    poly_path = PROJECT_ROOT / (
        "crosswalk_polygon.json"
        if cam_id in ("cam2", "default")
        else f"crosswalk_polygon_{cam_id}.json"
    )
    if poly_path.exists():
        poly_path.unlink()
    _refresh_camera_configs()
    log_audit("CAMERA_DELETE", target=cam_id, username=_admin())
    return redirect(url_for("admin_cameras", ok="camera-deleted"))
@app.route("/admin/cameras/<string:cam_id>")
@login_required
def admin_camera_detail(cam_id: str):
    _refresh_camera_configs()
    if cam_id not in CAMERA_CONFIGS:
        return render_template("admin/404.html"), 404
    cam  = cam_registry.get(cam_id)
    proc = proc_registry.get(cam_id)
    cfg  = CAMERA_CONFIGS[cam_id]
    demo = cfg.get("demo", "")
    return render_template(
        "admin/camera_detail.html",
        cam_id=cam_id,
        cam_label=cfg.get("label", cam_id),
        demo_source=demo,
        camera_location_name=cfg.get("location_name", cfg.get("label", cam_id)),
        camera_latitude=cfg.get("latitude", 41.2963),
        camera_longitude=cfg.get("longitude", 69.2798),
        camera_tags=cfg.get("tags", []),
        camera_status=cam.status(),
        detection_stats=proc.get_stats(),
        error=request.args.get("error", ""),
        ok=request.args.get("ok", ""),
    )
@app.route("/admin/cameras/<string:cam_id>/feed")
@login_required
def admin_camera_feed(cam_id: str):
    if cam_id not in CAMERA_CONFIGS:
        return jsonify(ok=False, error="Unknown camera"), 404
    cam = cam_registry.get(cam_id)
    return Response(
        mjpeg_generator(cam),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )
@app.route("/admin/cameras/<string:cam_id>/snapshot")
@login_required
def admin_camera_snapshot(cam_id: str):
    if cam_id not in CAMERA_CONFIGS:
        return jsonify(ok=False, error="Unknown camera"), 404
    cam = cam_registry.get(cam_id)
    return Response(cam.get_jpeg(), mimetype="image/jpeg",
                    headers={"Cache-Control": "no-store"})
@app.route("/admin/cameras/<string:cam_id>/start", methods=["POST"])
@login_required
def admin_camera_start(cam_id: str):
    cfg    = CAMERA_CONFIGS.get(cam_id)
    if cfg is None:
        return jsonify(ok=False, error="Unknown camera"), 404
    source = (
        request.form.get("source", "").strip()
        or cfg.get("source", "").strip()
        or cfg.get("demo", "0")
    )
    cam    = cam_registry.get(cam_id)
    ok     = cam.start(source)
    log_audit("CAMERA_START", target=f"{cam_id}:{source}", username=_admin())
    return jsonify(ok=ok, **cam.status())
@app.route("/admin/cameras/<string:cam_id>/stop", methods=["POST"])
@login_required
def admin_camera_stop(cam_id: str):
    if cam_id not in CAMERA_CONFIGS:
        return jsonify(ok=False, error="Unknown camera"), 404
    cam = cam_registry.get(cam_id)
    cam.stop()
    log_audit("CAMERA_STOP", target=cam_id, username=_admin())
    return jsonify(ok=True, **cam.status())
@app.route("/admin/cameras/<string:cam_id>/status")
@login_required
def admin_camera_status(cam_id: str):
    if cam_id not in CAMERA_CONFIGS:
        return jsonify(ok=False, error="Unknown camera"), 404
    return jsonify(**cam_registry.get(cam_id).status())
@app.route("/admin/cameras/<string:cam_id>/detection/start", methods=["POST"])
@login_required
def admin_camera_detection_start(cam_id: str):
    if cam_id not in CAMERA_CONFIGS:
        return jsonify(ok=False, error="Unknown camera"), 404
    cam = cam_registry.get(cam_id)
    if not cam.status().get("connected"):
        return jsonify(ok=False, error="Camera not connected"), 400
    poly_path = PROJECT_ROOT / (
        "crosswalk_polygon.json"
        if cam_id in ("cam2", "default")
        else f"crosswalk_polygon_{cam_id}.json"
    )
    if not poly_path.exists():
        return jsonify(ok=False, error="No crosswalk zone defined. Draw and save a polygon first."), 400
    try:
        pts = json.loads(poly_path.read_text())
        if not isinstance(pts, list) or len(pts) < 4:
            return jsonify(ok=False, error="Crosswalk zone has fewer than 4 points. Re-draw the polygon."), 400
    except Exception:
        return jsonify(ok=False, error="Polygon file is invalid. Re-draw the zone."), 400
    proc = proc_registry.get(cam_id)
    ok   = proc.start_async()
    log_audit("DETECTION_START", target=cam_id, username=_admin())
    return jsonify(ok=ok, **proc.get_stats())
@app.route("/admin/cameras/<string:cam_id>/detection/stop", methods=["POST"])
@login_required
def admin_camera_detection_stop(cam_id: str):
    if cam_id not in CAMERA_CONFIGS:
        return jsonify(ok=False, error="Unknown camera"), 404
    proc = proc_registry.get(cam_id)
    proc.stop()
    log_audit("DETECTION_STOP", target=cam_id, username=_admin())
    return jsonify(ok=True, **proc.get_stats())
@app.route("/admin/cameras/<string:cam_id>/detection/status")
@login_required
def admin_camera_detection_status(cam_id: str):
    if cam_id not in CAMERA_CONFIGS:
        return jsonify(ok=False, error="Unknown camera"), 404
    return jsonify(**proc_registry.get(cam_id).get_stats())
@app.route("/admin/cameras/<string:cam_id>/detection/reset", methods=["POST"])
@login_required
def admin_camera_detection_reset(cam_id: str):
    if cam_id not in CAMERA_CONFIGS:
        return jsonify(ok=False, error="Unknown camera"), 404
    proc = proc_registry.get(cam_id)
    proc.reset_session_outputs()
    log_audit("DETECTION_RESET", target=cam_id, username=_admin())
    return jsonify(ok=True, **proc.get_stats())
@app.route("/admin/cameras/<string:cam_id>/checks")
@login_required
def admin_camera_checks(cam_id: str):
    if cam_id not in CAMERA_CONFIGS:
        return jsonify(ok=False, error="Unknown camera"), 404
    cam = cam_registry.get(cam_id)
    proc = proc_registry.get(cam_id)
    cam_status = cam.status()
    checks = _build_pipeline_checks(
        cam_connected=cam_status.get("connected", False),
        cam_source=cam_status.get("source") or "",
        proc_obj=proc,
    )
    return jsonify(ok=all(v.get("ok") for v in checks.values()), checks=checks)
@app.route("/admin/cameras/<string:cam_id>/polygon", methods=["GET"])
@login_required
def admin_camera_polygon_get(cam_id: str):
    if cam_id not in CAMERA_CONFIGS:
        return jsonify(ok=False, error="Unknown camera"), 404
    poly_path = PROJECT_ROOT / (
        "crosswalk_polygon.json"
        if cam_id in ("cam2", "default")
        else f"crosswalk_polygon_{cam_id}.json"
    )
    if poly_path.exists():
        try:
            pts = json.loads(poly_path.read_text())
            if isinstance(pts, list) and len(pts) >= 4:
                return jsonify(points=pts)
        except Exception:
            pass
    return jsonify(points=[])
@app.route("/admin/cameras/<string:cam_id>/polygon", methods=["POST"])
@login_required
def admin_camera_polygon_save(cam_id: str):
    if cam_id not in CAMERA_CONFIGS:
        return jsonify(ok=False, error="Unknown camera"), 404
    data   = request.get_json(silent=True) or {}
    points = data.get("points", [])
    poly_path = PROJECT_ROOT / (
        "crosswalk_polygon.json"
        if cam_id in ("cam2", "default")
        else f"crosswalk_polygon_{cam_id}.json"
    )
    if not points:
        if poly_path.exists():
            poly_path.unlink()
        proc_registry.get(cam_id).reload_polygon()
        log_audit("POLYGON_CLEAR", target=cam_id, username=_admin())
        return jsonify(ok=True, count=0)
    if not isinstance(points, list) or len(points) < 4:
        return jsonify(ok=False, error="Need at least 4 points"), 400
    try:
        normalized = [[int(p[0]), int(p[1])] for p in points]
    except Exception:
        return jsonify(ok=False, error="Polygon points must be numeric"), 400
    poly_path.write_text(json.dumps(normalized))
    proc_registry.get(cam_id).reload_polygon()
    log_audit("POLYGON_SAVE", target=f"{cam_id}/{len(normalized)} pts", username=_admin())
    return jsonify(ok=True, count=len(normalized))
@app.route("/admin/cameras/<string:cam_id>/recent")
@login_required
def admin_camera_recent(cam_id: str):
    if cam_id not in CAMERA_CONFIGS:
        return jsonify(ok=False, error="Unknown camera"), 404
    with db_connection() as conn:
        rows = conn.execute(
            """SELECT id, timestamp, plate_number, vehicle_id,
                      violation_type, severity, snapshot_path
               FROM   violations
               ORDER  BY timestamp DESC LIMIT 15"""
        ).fetchall()
    return jsonify(violations=[dict(r) for r in rows])
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True, use_reloader=False)
