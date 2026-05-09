from __future__ import annotations
from datetime import datetime, timedelta, timezone
from functools import wraps
from flask import redirect, session, url_for
from werkzeug.security import check_password_hash
from database import db_connection
MAX_ATTEMPTS: int = 5
LOCKOUT_MINUTES: int = 15
def check_admin_credentials(username: str, password: str) -> bool:
    with db_connection() as conn:
        row = conn.execute(
            "SELECT password_hash FROM admin_users WHERE username = ?",
            (username,),
        ).fetchone()
    if row is None:
        return False
    return check_password_hash(row["password_hash"], password)
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "admin" not in session:
            return redirect(url_for("admin_login"))
        return f(*args, **kwargs)
    return decorated
def is_locked_out() -> bool:
    lockout_until = session.get("lockout_until")
    if lockout_until is None:
        return False
    return datetime.now(timezone.utc) < datetime.fromisoformat(lockout_until)
def get_lockout_remaining() -> int:
    lockout_until = session.get("lockout_until")
    if lockout_until is None:
        return 0
    delta = datetime.fromisoformat(lockout_until) - datetime.now(timezone.utc)
    return max(0, int(delta.total_seconds()))
def record_failed_attempt() -> None:
    attempts = session.get("failed_attempts", 0) + 1
    session["failed_attempts"] = attempts
    if attempts >= MAX_ATTEMPTS:
        until = datetime.now(timezone.utc) + timedelta(minutes=LOCKOUT_MINUTES)
        session["lockout_until"] = until.isoformat()
        session["failed_attempts"] = 0
def clear_failed_attempts() -> None:
    session.pop("failed_attempts", None)
    session.pop("lockout_until", None)
