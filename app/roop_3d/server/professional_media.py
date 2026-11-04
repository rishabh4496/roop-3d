"""Stage 15 media API: what the media layer can do here, the export presets, and input validation.

* ``GET  /api/v1/media/capabilities``    verified / unsupported feature matrix with reasons.
* ``GET  /api/v1/media/export-profiles`` the presets; ``?verify=true`` encodes and validates each on this machine.
* ``POST /api/v1/media/validate``        structured findings for a local video (``deep`` decodes everything).

Read-only apart from the throw-away self-test clips written under the runtime's ``selftest`` folder.
"""
from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import Field
from starlette.concurrency import run_in_threadpool

from roop_3d.core.settings import Contract
from roop_3d.core.storage import owned
from roop_3d.media.capabilities import media_capabilities, selftest_profile
from roop_3d.media.export_profiles import PROFILES
from roop_3d.media.validation import validate_media

router = APIRouter(prefix="/api/v1/media")


class ValidateBody(Contract):
    path: str = Field(min_length=1, max_length=1024)
    deep: bool = False


@router.get("/capabilities")
def capabilities() -> dict:
    return media_capabilities()


@router.get("/export-profiles")
async def export_profiles(request: Request, verify: bool = False) -> dict:
    scratch = owned(request.app.state.root, "roop_3d/.runtime/selftest")
    scratch.mkdir(parents=True, exist_ok=True)
    result = []
    for profile in PROFILES.values():
        item = profile.as_dict()
        item["verification"] = (await run_in_threadpool(selftest_profile, profile, scratch) if verify
                                else {"state": "UNKNOWN", "reason": "not verified yet: request ?verify=true to encode and validate it here"})
        result.append(item)
    return {"profiles": result, "default": "standard_mp4", "processing_bit_depth": 8}


@router.post("/validate")
async def validate(body: ValidateBody, request: Request) -> dict:
    path = request.app.state.media.path(body.path)
    return await run_in_threadpool(validate_media, path, deep=body.deep)
