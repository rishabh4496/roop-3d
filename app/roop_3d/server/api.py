"""Production same-origin workstation serving with API-safe SPA fallback."""
from pathlib import Path
from fastapi.responses import FileResponse, HTMLResponse
from starlette.exceptions import HTTPException
from starlette.staticfiles import StaticFiles
from roop_3d.audit.isolation import safe_path


class WorkstationFiles(StaticFiles):
    async def get_response(self, path, scope):
        portable = path.replace("\\", "/").lstrip("/")
        if portable == "api" or portable.startswith("api/") or portable == "openapi.json":
            raise HTTPException(404)
        try:
            return await super().get_response(path, scope)
        except HTTPException as error:
            ui_route = portable.split("/", 1)[0] in {"projects", "workstation"}
            if error.status_code != 404 or Path(path).suffix or not ui_route:
                raise
            return await super().get_response("index.html", scope)


def mount_workstation(app, root: Path) -> None:
    ui = safe_path(root, "roop_3d_ui/dist")
    if ui.is_dir() and (ui / "index.html").is_file():
        safe_path(root, "roop_3d_ui/dist/index.html")
        app.mount("/", WorkstationFiles(directory=str(ui), html=True), name="workstation")
    else:
        @app.get("/", response_class=HTMLResponse)
        def build_required():
            return '<!doctype html><html lang="en"><title>Roop 3D</title><h1>Roop 3D</h1><p>Build the workstation with npm ci and npm run build in app/roop_3d_ui.</p><a href="/api/v1/health">Service health</a></html>'
