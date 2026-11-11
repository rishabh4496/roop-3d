"""Read-only route for the five built-in CC0 studio lighting environments (``rendering/assets/stage10_cc0_probes.npz``).

    GET /api/v1/rendering/studio-environments

Each environment is a compact, linear-light reduction of a CC0 Poly Haven HDRI. The route returns what a viewer needs to
preview the lighting: a name, the colour temperature, the licence and source page, nine RGB spherical-harmonic coefficients
(already in the RGB order a browser shader expects: the bundle stores BGR) and a small tone-mapped preview image.

This is a preview aid. The probes are not part of the render pipeline yet (see ``rendering/RENDERING_MODELS.md``), so choosing
one in the UI never changes an output.
"""
from __future__ import annotations

import base64
from functools import lru_cache
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter
from starlette.concurrency import run_in_threadpool

from roop_3d.core.errors import DomainError

router = APIRouter(prefix="/api/v1")

THUMBNAIL_SIZE = (128, 64)          # width, height of the equirectangular preview


def _thumbnail(environment_bgr: np.ndarray) -> str:
    """Tone-mapped (Reinhard, display gamma) PNG data URL of a linear BGR equirectangular map."""
    linear = np.clip(np.asarray(environment_bgr, np.float32), 0.0, None)
    mapped = (linear / (1.0 + linear)) ** (1.0 / 2.2)
    resized = cv2.resize(mapped, THUMBNAIL_SIZE, interpolation=cv2.INTER_CUBIC)
    ok, png = cv2.imencode(".png", np.clip(resized * 255.0 + 0.5, 0, 255).astype(np.uint8))
    if not ok:                                            # pragma: no cover - cv2 only fails on absurd shapes
        raise DomainError("STUDIO_ENVIRONMENT_ENCODE_FAILED", "Could not encode a lighting preview image", 500)
    return "data:image/png;base64," + base64.b64encode(png.tobytes()).decode("ascii")


@lru_cache(maxsize=1)
def _catalog() -> dict[str, Any]:
    from roop_3d.rendering.physical_camera import ENVIRONMENT_PROBES, load_cc0_preview_assets
    try:
        assets = load_cc0_preview_assets()
    except (FileNotFoundError, ValueError) as error:
        raise DomainError("STUDIO_ENVIRONMENTS_UNAVAILABLE", f"The CC0 lighting bundle could not be loaded: {error}", 503) from error
    pages = {probe["asset_id"]: probe["page_url"] for probe in ENVIRONMENT_PROBES}
    environments = []
    for probe in assets.probes:
        coefficients = np.asarray(probe.sh_coefficients, np.float64)[:, ::-1]        # BGR -> RGB
        environments.append({
            "id": probe.asset_id, "name": probe.name, "kelvin": float(probe.white_balance_kelvin), "license": probe.license_id,
            "page_url": pages.get(probe.asset_id, ""), "sh_basis": "roop_3d.rendering.physical_buffers.evaluate_sh_basis",
            "sh_coefficients": np.round(coefficients, 6).tolist(), "thumbnail": _thumbnail(probe.environment_bgr),
        })
    meta = assets.metadata
    return {"schema": "roop3d.studio_environments.v1", "license": meta.get("license", "CC0-1.0"),
            "source": meta.get("environment_source", ""), "count": len(environments), "environments": environments}


@router.get("/rendering/studio-environments")
async def studio_environments() -> dict[str, Any]:
    return await run_in_threadpool(_catalog)
