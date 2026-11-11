"""Zero-network builders for the core 3D assets, used when a machine is offline or restricted.

Each handler has ``build(target, ...) -> Path`` and ``verify(target, ...) -> dict``. None of them opens a socket:

* ``makehuman_cc0_head``   compiled from the hash-locked MakeHuman CC0 sources when all 130 files are cached locally,
                           otherwise restored from the 386 KB CC0 build that ships with the app (digest-checked). The result
                           is bit-identical to a networked build (asset digest), so nothing is approximated.
* ``anatomical_pack_cc0``  the existing local synthesis (MakeHuman head when installed, analytic CC0 head otherwise).
* ``cc0_probe_bundle``     reduced from the Poly Haven HDRIs when they are cached (md5-checked), otherwise five analytic
                           studio probes. The result is labelled ``procedural`` in its metadata: it is a stand-in, not the
                           Poly Haven bundle.
* ``correspondence_maps``  barycentric MakeHuman <-> FLAME surface maps, pure geometry from two local meshes.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Callable

if __package__:                                      # same rule as local_models: one AuditError class per loading mode
    from roop_3d.audit.isolation import AuditError
else:
    from isolation import AuditError

BUNDLED_MAKEHUMAN = Path(__file__).resolve().parents[1] / "reconstruction" / "assets" / "makehuman_head_v1.npz"


def _ensure_app_on_path() -> None:
    app = str(Path(__file__).resolve().parents[2])
    if app not in sys.path:
        sys.path.insert(0, app)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


# ------------------------------------------------------------------------------------------------ MakeHuman head
def _makehuman_cache_complete(cache_dir: Path) -> bool:
    """True only when every pinned source file is present and matches its locked SHA-256 (so the build cannot download)."""
    _ensure_app_on_path()
    from roop_3d.reconstruction import makehuman_build as builder
    lock = builder.load_lock()["files"]
    for relative in builder._all_paths():
        path = cache_dir / relative
        if relative not in lock or not path.is_file() or _sha256(path) != lock[relative]["sha256"]:
            return False
    return True


def build_makehuman_cc0_head(target: Path, *, bundled: Path | None = None) -> Path:
    _ensure_app_on_path()
    from roop_3d.reconstruction import makehuman_build as builder
    target = Path(target)
    out_dir = target.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    if _makehuman_cache_complete(out_dir / "sources"):
        built = builder.build_asset(out_dir, out_dir / "sources")      # complete verified cache: no download can happen
        return built
    resource = Path(bundled) if bundled else BUNDLED_MAKEHUMAN
    if resource.is_file() and resource.with_name(resource.name + ".json").is_file():
        sidecar = json.loads(resource.with_name(resource.name + ".json").read_text("utf-8"))
        shutil.copyfile(resource, target)
        shutil.copyfile(resource.with_name(resource.name + ".json"), target.with_name(target.name + ".json"))
        problem = verify_makehuman_cc0_head(target, sidecar["asset_digest"])
        if not problem["ok"]:
            target.unlink(missing_ok=True)
            raise AuditError("PROCEDURAL_BUNDLE_CORRUPT", problem["detail"])
        return target
    raise AuditError("PROCEDURAL_SOURCE_MISSING",
                     "MakeHuman CC0 sources are not fully cached and no bundled build ships with this install; "
                     "download makehuman_cc0_sources once (zero-auth) or restore roop_3d/reconstruction/assets/makehuman_head_v1.npz")


def verify_makehuman_cc0_head(target: Path, content_digest: str | None = None) -> dict[str, Any]:
    _ensure_app_on_path()
    import numpy as np
    from roop_3d.reconstruction.makehuman_build import asset_digest
    try:
        with np.load(Path(target), allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        actual = asset_digest(arrays)
    except Exception as error:          # BadZipFile, EOFError, ... : any unreadable file is a failed verification
        return {"ok": False, "detail": f"{type(error).__name__}: {error}"}
    if content_digest and actual != content_digest:
        return {"ok": False, "detail": f"asset digest {actual[:16]}... differs from the pinned {content_digest[:16]}...", "digest": actual}
    return {"ok": True, "detail": "asset digest matches", "digest": actual, "kind": "pinned"}


# ------------------------------------------------------------------------------------------------ anatomical pack
def build_anatomical_pack_cc0(target: Path) -> Path:
    _ensure_app_on_path()
    from roop_3d.reconstruction.anatomical_pack import build_anatomical_pack
    return build_anatomical_pack(Path(target))


def verify_anatomical_pack_cc0(target: Path, expected: str | None = None) -> dict[str, Any]:
    _ensure_app_on_path()
    from roop_3d.reconstruction.anatomical_pack import AnatomicalFullHeadAsset
    try:
        asset = AnatomicalFullHeadAsset.load(Path(target), auto_build=False, verify=True)
    except Exception as error:      # any failure to load or verify is a failed verification, never a crash
        return {"ok": False, "detail": f"{type(error).__name__}: {error}"}
    kind = "pinned" if asset.template_source == "makehuman_cc0" else "procedural"
    return {"ok": True, "detail": f"template_source={asset.template_source}", "kind": kind, "digest": _sha256(Path(target))}


# ------------------------------------------------------------------------------------------------ CC0 probes
_PROBE_RECIPES = (
    # name, asset id, kelvin, ambient BGR, [(azimuth deg, elevation deg, angular radius deg, intensity BGR)]
    ("Neutral Studio", "procedural_neutral_studio", 6000.0, (0.30, 0.30, 0.30),
     [(0, 55, 28, (2.4, 2.4, 2.4)), (-70, 20, 22, (1.1, 1.1, 1.1)), (70, 20, 22, (1.1, 1.1, 1.1))]),
    ("Warm Indoor", "procedural_warm_indoor", 2750.0, (0.10, 0.18, 0.30),
     [(40, 35, 16, (0.8, 1.7, 2.8)), (-120, 10, 30, (0.2, 0.4, 0.7))]),
    ("Cool Overcast", "procedural_cool_overcast", 5500.0, (0.55, 0.52, 0.48),
     [(0, 80, 60, (1.3, 1.25, 1.15))]),
    ("Golden Hour", "procedural_golden_hour", 5400.0, (0.12, 0.17, 0.22),
     [(-95, 8, 9, (0.7, 1.9, 3.4)), (80, 40, 50, (0.45, 0.4, 0.35))]),
    ("Cinematic Rim", "procedural_cinematic_rim", 5500.0, (0.05, 0.05, 0.05),
     [(150, 25, 10, (2.8, 2.6, 2.4)), (-150, 25, 10, (2.2, 2.4, 2.8)), (0, 10, 40, (0.25, 0.25, 0.25))]),
)
PROBE_SOURCE_PROCEDURAL = "Procedural analytic studio probes (offline fallback); not Poly Haven data"


def synthesize_probe_maps(shape: tuple[int, int] = (32, 64)):
    """Five deterministic equirectangular maps (linear BGR): a sky/ground gradient plus soft circular lights. No randomness."""
    import numpy as np
    height, width = shape
    row, column = np.mgrid[0:height, 0:width]
    latitude = np.pi * (0.5 - (row + 0.5) / height)
    longitude = 2.0 * np.pi * ((column + 0.5) / width - 0.5)
    cos_lat = np.cos(latitude)
    directions = np.stack((cos_lat * np.sin(longitude), np.sin(latitude), cos_lat * np.cos(longitude)), axis=-1)
    maps = []
    for _, _, _, ambient, lights in _PROBE_RECIPES:
        sky = 0.6 + 0.4 * np.clip(directions[..., 1:2], -1.0, 1.0)          # brighter overhead, darker below the horizon
        image = np.asarray(ambient, np.float64)[None, None, :] * sky
        for azimuth, elevation, radius, intensity in lights:
            a, e = np.radians(azimuth), np.radians(elevation)
            axis = np.array([np.cos(e) * np.sin(a), np.sin(e), np.cos(e) * np.cos(a)])
            angle = np.arccos(np.clip(directions @ axis, -1.0, 1.0))
            falloff = np.exp(-0.5 * (angle / np.radians(radius)) ** 2)
            image = image + falloff[..., None] * np.asarray(intensity, np.float64)[None, None, :]
        maps.append(image.astype(np.float32))
    return maps


def build_cc0_probe_bundle_offline(target: Path, *, hdri_dir: Path | None = None) -> Path:
    """Poly Haven reduction when the five HDRIs are cached and md5-correct, otherwise the analytic stand-in."""
    _ensure_app_on_path()
    import numpy as np
    from roop_3d.rendering import physical_camera as pc
    target = Path(target)
    if hdri_dir is not None and all((Path(hdri_dir) / f"{p['asset_id']}_1k.hdr").is_file() for p in pc.ENVIRONMENT_PROBES):
        try:
            return pc.build_cc0_probe_bundle(hdri_dir, target)
        except (ValueError, OSError):
            pass        # a wrong or unreadable HDRI is never used; fall through to the analytic probes
    maps = synthesize_probe_maps()
    coefficients = [pc._project_environment_sh(m) for m in maps]
    pore = pc.synthesize_micro_pore_normal(256)
    metadata = {
        "schema": pc.PROBE_BUNDLE_SCHEMA, "license": "CC0-1.0", "environment_license": "CC0-1.0",
        "environment_source": PROBE_SOURCE_PROCEDURAL, "pore_license": "CC0-1.0",
        "pore_source": "Deterministic procedural periodic dimple synthesis by Roop 3D contributors",
        "environment_shape": [32, 64], "pore_resolution": 256,
        "probes": [{"asset_id": recipe[1], "name": recipe[0], "kelvin": recipe[2], "license": "CC0-1.0"} for recipe in _PROBE_RECIPES],
    }
    payload = {
        "schema": np.asarray(pc.PROBE_BUNDLE_SCHEMA),
        "probe_names": np.asarray([r[0] for r in _PROBE_RECIPES]),
        "probe_asset_ids": np.asarray([r[1] for r in _PROBE_RECIPES]),
        "probe_kelvin": np.asarray([r[2] for r in _PROBE_RECIPES], np.float32),
        "environment_maps_bgr": np.asarray(maps, np.float16),
        "sh_coefficients_bgr": np.asarray(coefficients, np.float32),
        "micro_pore_normal_xyz": pore.astype(np.float16),
        "metadata": np.asarray(json.dumps(metadata, sort_keys=True)),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **payload)
    sidecar = dict(metadata, bundle_sha256=_sha256(target))
    target.with_name(target.name + ".json").write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return target


def array_digest(path: Path) -> str:
    """Content digest of an ``.npz`` (zip timestamps make the file bytes non-deterministic; the arrays are)."""
    import numpy as np
    digest = hashlib.sha256()
    with np.load(Path(path), allow_pickle=False) as data:
        for key in sorted(data.files):
            array = np.ascontiguousarray(data[key])
            digest.update(key.encode() + str(array.dtype).encode() + str(array.shape).encode())
            digest.update(array.tobytes() if array.dtype.kind != "U" else "|".join(array.tolist()).encode())
    return digest.hexdigest()


def verify_cc0_probe_bundle(target: Path, expected_digest: str | None = None) -> dict[str, Any]:
    """Valid when internally consistent; ``pinned`` only if its array content digest equals ``expected_digest``."""
    _ensure_app_on_path()
    from roop_3d.rendering.physical_camera import load_cc0_preview_assets
    try:
        assets = load_cc0_preview_assets(Path(target))
        digest = array_digest(Path(target))
    except Exception as error:
        return {"ok": False, "detail": f"{type(error).__name__}: {error}"}
    procedural = str(assets.metadata.get("environment_source", "")).startswith("Procedural")
    if not procedural and expected_digest and digest != expected_digest:
        return {"ok": False, "detail": "bundle is internally consistent but is not the pinned CC0 bundle", "digest": digest}
    return {"ok": True, "detail": "procedural stand-in" if procedural else "pinned CC0 bundle",
            "kind": "procedural" if procedural else "pinned", "digest": digest}


# ------------------------------------------------------------------------------------------------ correspondence maps
def _bridge_arrays(bridge) -> dict[str, Any]:
    import numpy as np
    arrays: dict[str, Any] = {}
    for direction in ("target", "source"):
        block = bridge._maps[direction]
        matrix = block["matrix"].tocsr()
        arrays.update({f"{direction}_indptr": matrix.indptr.astype(np.int64), f"{direction}_indices": matrix.indices.astype(np.int64),
                       f"{direction}_data": matrix.data.astype(np.float64), f"{direction}_shape": np.asarray(matrix.shape, np.int64),
                       f"{direction}_offset": np.asarray(block["offset"], np.float64), f"{direction}_distance": np.asarray(block["distance"], np.float64)})
    return arrays


def compile_correspondence_maps(source, target) -> dict[str, Any]:
    """Build the bridge between two TopologyMesh objects and return its arrays (tests inject synthetic meshes)."""
    _ensure_app_on_path()
    from roop_3d.reconstruction.mesh_bridge import BarycentricMeshBridge
    return _bridge_arrays(BarycentricMeshBridge.build(source, target))


def build_correspondence_maps(target: Path, *, registry=None) -> Path:
    _ensure_app_on_path()
    import numpy as np
    from roop_3d.reconstruction.topology import FLAME_OPEN_ID, MAKEHUMAN_ID, default_registry
    registry = registry or default_registry()
    flame = registry.availability(FLAME_OPEN_ID)
    if not flame["available"]:
        raise AuditError("PROCEDURAL_DEPENDENCY_MISSING", f"FLAME 2023 Open is not installed ({flame['state']}); the maps need both meshes")
    arrays = compile_correspondence_maps(registry.load(MAKEHUMAN_ID), registry.load(FLAME_OPEN_ID))
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **arrays, source_topology=np.asarray(MAKEHUMAN_ID), target_topology=np.asarray(FLAME_OPEN_ID))
    return target


def verify_correspondence_maps(target: Path, expected: str | None = None) -> dict[str, Any]:
    import numpy as np
    try:
        with np.load(Path(target), allow_pickle=False) as data:
            for direction in ("target", "source"):
                indptr, indices, values = data[f"{direction}_indptr"], data[f"{direction}_indices"], data[f"{direction}_data"]
                rows, columns = (int(v) for v in data[f"{direction}_shape"])
                if len(indptr) != rows + 1 or indptr[-1] != len(indices) or (len(indices) and indices.max() >= columns):
                    return {"ok": False, "detail": f"{direction} map is structurally invalid"}
                # Every output vertex is a convex combination of the three corners of one triangle.
                sums = np.add.reduceat(values, indptr[:-1]) if len(values) else np.zeros(0)
                if not np.allclose(sums, 1.0, atol=1e-6):
                    return {"ok": False, "detail": f"{direction} barycentric weights do not sum to 1"}
    except Exception as error:
        return {"ok": False, "detail": f"{type(error).__name__}: {error}"}
    return {"ok": True, "detail": "barycentric rows sum to 1", "kind": "pinned", "digest": array_digest(Path(target))}


class Handler:
    def __init__(self, build: Callable[..., Path], verify: Callable[..., dict[str, Any]]):
        self.build, self.verify = build, verify


HANDLERS: dict[str, Handler] = {
    "makehuman_cc0_head": Handler(build_makehuman_cc0_head, verify_makehuman_cc0_head),
    "anatomical_pack_cc0": Handler(build_anatomical_pack_cc0, verify_anatomical_pack_cc0),
    "cc0_probe_bundle": Handler(build_cc0_probe_bundle_offline, verify_cc0_probe_bundle),
    "correspondence_maps": Handler(build_correspondence_maps, verify_correspondence_maps),
}
