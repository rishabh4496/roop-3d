"""Stage 12 routes for the optional external-Blender workflow (``roop_3d.io.blender_bridge``).

* ``GET  /api/v1/blender/exporter-script``  the standalone script to run inside Blender 4.x (text; roop-3d never executes it)
* ``GET  /api/v1/blender/status``           whether a Blender MCP server answers on ``localhost:<port>``
* ``POST /api/v1/facesets/{id}/identity-asset/send-to-blender``  build the ``.glb`` and have Blender import it

The host is never a parameter: the bridge is loopback-only by construction, so these routes cannot be used to reach another machine.
"""
from __future__ import annotations

from fastapi import APIRouter, Query, Request
from fastapi.responses import Response
from starlette.concurrency import run_in_threadpool

from roop_3d.core.storage import owned
from roop_3d.io.blender_bridge import DEFAULT_PORT, EXPORTER_SCRIPT, BlenderMCPBridge
from roop_3d.server.identity_assets import TemplateName, _glb, load_document

router = APIRouter(prefix="/api/v1")
STAGING = "roop_3d/.runtime/exports/blender"


@router.get("/blender/exporter-script")
async def blender_exporter_script():
    return Response(content=EXPORTER_SCRIPT, media_type="text/x-python",
                    headers={"Content-Disposition": 'attachment; filename="roop3d_blender_import.py"', "Cache-Control": "no-cache"})


@router.get("/blender/status")
async def blender_status(port: int = Query(default=DEFAULT_PORT, ge=1, le=65535)):
    bridge = BlenderMCPBridge(port=port, timeout=1.0)
    available = await run_in_threadpool(bridge.is_available)
    return {"host": bridge.host, "port": bridge.port, "available": available, "dialect": bridge.dialect,
            "hint": None if available else "Start the Blender MCP server in Blender, or download /api/v1/blender/exporter-script"}


@router.post("/facesets/{faceset_id}/identity-asset/send-to-blender")
async def send_to_blender(
    faceset_id: str, request: Request, template: TemplateName = Query(default="canonical"),
    topology: str | None = Query(default=None, max_length=64),
    jaw_width: float = Query(default=0.0), cheekbone_prominence: float = Query(default=0.0),
    chin_projection: float = Query(default=0.0), nose_bridge: float = Query(default=0.0),
    brow_arch: float = Query(default=0.0), face_fullness: float = Query(default=0.0),
    albedo_prior: bool = Query(default=False), shape_keys: bool = Query(default=False), normal_map: bool = Query(default=True),
    port: int = Query(default=DEFAULT_PORT, ge=1, le=65535),
):
    sliders = {"jaw_width": jaw_width, "cheekbone_prominence": cheekbone_prominence, "chin_projection": chin_projection,
               "nose_bridge": nose_bridge, "brow_arch": brow_arch, "face_fullness": face_fullness}
    document = await run_in_threadpool(load_document, request, faceset_id)
    data = await run_in_threadpool(_glb, document, template, topology, sliders, albedo_prior, shape_keys, normal_map)
    staging = owned(request.app.state.root, STAGING)
    report = await run_in_threadpool(BlenderMCPBridge(port=port).send_glb, data, staging, f"{faceset_id}_head")
    return {"sent": True, "faceset_id": faceset_id, "port": port, "bytes": report["bytes"], "path": report["path"],
            "imported_objects": report["imported_objects"]}
