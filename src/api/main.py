from __future__ import annotations
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from api.dependencies import get_repository
from api.routes.analytics import router as analytics_router
from api.routes.violations import router as violations_router
from api.routes.vehicles import router as vehicles_router
app = FastAPI(title="Crosswalk Violation Enforcement API", version="1.0.0")
app.include_router(violations_router)
app.include_router(vehicles_router)
app.include_router(analytics_router)
@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    repository = get_repository()
    rows = repository.list_violations(limit=25)
    body = "".join(
        f"<tr><td>{row.id}</td><td>{row.plate_number or 'UNREADABLE'}</td>"
        f"<td>{row.timestamp.isoformat()}</td><td>{row.status}</td><td>{row.location}</td></tr>"
        for row in rows
    )
    return f"""
    <html>
      <head>
        <title>Crosswalk Violations</title>
        <style>
          body {{ font-family: 'Segoe UI', sans-serif; margin: 2rem; background:
          table {{ width: 100%; border-collapse: collapse; background: white; }}
          th, td {{ padding: 0.75rem; border-bottom: 1px solid
          h1 {{ letter-spacing: 0.04em; }}
        </style>
      </head>
      <body>
        <h1>Crosswalk Violation Dashboard</h1>
        <table>
          <thead>
            <tr><th>ID</th><th>Plate</th><th>Timestamp</th><th>Status</th><th>Location</th></tr>
          </thead>
          <tbody>{body}</tbody>
        </table>
      </body>
    </html>
    """
