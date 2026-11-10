"""Morphable head topologies: a registry of interchangeable surface parameterisations of a head.

A *topology* is a triangle mesh (vertices, faces, UVs) plus, when the model has one, a linear identity basis. Two are
registered:

* ``makehuman_cc0``  (built in, the default): the MakeHuman head+neck, CC0-1.0, no restriction of any kind. It is the
  topology identity assets are built on, so it is always the one a caller gets when it asks for nothing in particular.
* ``flame2023_open`` (pluggable): FLAME 2023 Open, CC-BY-4.0 (commercial use allowed with attribution). The file is
  user-supplied (never bundled, git-ignored), is pinned by SHA-256 in :mod:`roop_3d.audit.local_models`, and is read with an
  allow-listed unpickler (numpy + scipy.sparse only). FLAME 2020 and every other academic-only MPG model (DECA, EMOCA, MICA,
  VOCA) are not registrable here: the registry refuses any spec whose licence does not allow commercial use.

Every topology is expressed in the identity-asset mesh frame (metres, +x image-right, +y down, +z toward the viewer, outward
CCW triangles), so meshes of different topologies can be laid on top of each other without further bookkeeping. FLAME's native
frame (y up) is mapped with a y mirror and its winding is flipped to keep the faces outward.

``flame2023_open`` ships with no UV layout and no landmark embedding. :meth:`MorphableTopologyRegistry.prepared` derives a UV
layout for it from the CC0 MakeHuman atlas through the surface bridge (see :mod:`roop_3d.reconstruction.mesh_bridge`); the
derived layout is labelled as such, it is not an official FLAME UV set.
"""
from __future__ import annotations

import io
import os
import pickle
import threading
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np

from roop_3d.audit.local_models import OPEN_ARTIFACTS, locate_open_artifact, verify_open_artifact
from roop_3d.audit.isolation import AuditError
from roop_3d.audit.project_paths import project_root
from roop_3d.core.errors import DomainError

MAKEHUMAN_ID = "makehuman_cc0"
FLAME_OPEN_ID = "flame2023_open"
DEFAULT_TOPOLOGY_ID = MAKEHUMAN_ID
FLAME_PATH_ENV = "ROOP3D_FLAME2023_OPEN"

#: Licences that may be registered. Anything else (``LicenseRef-FLAME-Academic``, MPG non-commercial, ...) is rejected.
COMMERCIAL_LICENSES = frozenset({"CC0-1.0", "CC-BY-4.0"})
_FLAME_SHAPE_AXES = 300   # FLAME 2023 Open stores 300 identity + 100 expression columns; only identity is exposed here
_Y_MIRROR = np.array([1.0, -1.0, 1.0])


@dataclass(frozen=True)
class TopologySpec:
    topology_id: str
    title: str
    license_id: str
    license_url: str
    commercial_use: str
    attribution: str
    source: str
    builtin: bool = False
    artifact_id: str | None = None   # key in audit.local_models.OPEN_ARTIFACTS for pluggable (file-backed) topologies

    def public(self) -> dict[str, Any]:
        return {"id": self.topology_id, "title": self.title, "license_id": self.license_id, "license_url": self.license_url,
                "commercial_use": self.commercial_use, "attribution": self.attribution, "source": self.source,
                "builtin": self.builtin}


@dataclass
class TopologyMesh:
    """A topology's rest surface (float64, identity-asset frame) and its optional linear identity basis."""

    topology_id: str
    vertices: np.ndarray                      # (V, 3)
    faces: np.ndarray                         # (F, 3) int64
    uvs: np.ndarray | None = None             # (V, 2) per-vertex, v up, seam-split; None when the model has no UV layout
    shape_basis: np.ndarray | None = None     # (V, 3, K) float32 metres per unit coefficient
    shape_names: tuple[str, ...] = ()
    landmarks: dict[str, np.ndarray] = field(default_factory=dict)   # layout name -> vertex indices
    source_index: np.ndarray | None = None    # (V,) index of the pre-split vertex each vertex came from
    uv_provenance: str = "none"
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def vertex_count(self) -> int:
        return int(len(self.vertices))

    @property
    def shape_axes(self) -> int:
        return 0 if self.shape_basis is None else int(self.shape_basis.shape[2])

    def normals(self) -> np.ndarray:
        from roop_3d.reconstruction.rbf_deform import welded_vertex_normals
        return welded_vertex_normals(self.vertices, self.faces).astype(np.float64)

    def with_vertices(self, vertices: np.ndarray) -> "TopologyMesh":
        return replace(self, vertices=np.asarray(vertices, np.float64))

    def mesh3d(self):
        from roop_3d.reconstruction.geometry_types import Mesh3D
        return Mesh3D(self.vertices, self.faces, uvs=self.uvs)


# ----------------------------------------------------------------------------------------------------------- FLAME file
_ALLOWED_GLOBALS = frozenset({
    ("numpy", "dtype"), ("numpy", "ndarray"),
    ("numpy.core.numeric", "_frombuffer"), ("numpy._core.numeric", "_frombuffer"),
    ("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar"),
    ("collections", "OrderedDict"),
    ("scipy.sparse._csc", "csc_matrix"), ("scipy.sparse._csr", "csr_matrix"),
})


class _AllowListUnpickler(pickle.Unpickler):
    """Reads numpy arrays and scipy sparse matrices; every other global is refused rather than imported."""

    def find_class(self, module: str, name: str):
        if (module, name) not in _ALLOWED_GLOBALS:
            raise pickle.UnpicklingError(f"global {module}.{name} is not allowed in a morphable-model file")
        return super().find_class(module, name)


def read_flame_open(path: Path) -> dict[str, Any]:
    """Parse ``flame2023_Open.pkl`` (already hash-verified by the caller) and validate the arrays it must contain."""
    try:
        with Path(path).open("rb") as stream:
            data = _AllowListUnpickler(io.BytesIO(stream.read()), encoding="latin1").load()
        template = np.asarray(data["v_template"], np.float64)
        faces = np.asarray(data["f"]).astype(np.int64)
        shapedirs = np.asarray(data["shapedirs"])
    except (pickle.UnpicklingError, KeyError, EOFError, ValueError, AttributeError, ImportError) as error:
        raise DomainError("TOPOLOGY_ASSET_INVALID", f"FLAME 2023 Open file cannot be read: {type(error).__name__}: {error}", 422) from error
    ok = (template.ndim == 2 and template.shape[1] == 3 and faces.ndim == 2 and faces.shape[1] == 3
          and shapedirs.ndim == 3 and shapedirs.shape[:2] == template.shape and shapedirs.shape[2] >= _FLAME_SHAPE_AXES
          and 0 <= faces.min() and faces.max() < len(template) and np.isfinite(template).all() and np.isfinite(shapedirs).all())
    if not ok:
        raise DomainError("TOPOLOGY_ASSET_INVALID", "FLAME 2023 Open file does not contain the expected template, faces and shape basis", 422)
    return {"v_template": template, "f": faces, "shapedirs": shapedirs[:, :, :_FLAME_SHAPE_AXES]}


def flame_to_identity_frame(template: np.ndarray, faces: np.ndarray, shapedirs: np.ndarray):
    """FLAME native (y up) -> identity-asset frame (y down). A mirror reverses handedness, so the winding is swapped."""
    return template * _Y_MIRROR, faces[:, [0, 2, 1]], shapedirs * _Y_MIRROR[None, :, None]


# -------------------------------------------------------------------------------------------------------------- registry
Loader = Callable[["MorphableTopologyRegistry", TopologySpec], TopologyMesh]


class MorphableTopologyRegistry:
    """Named, licence-gated head topologies. ``makehuman_cc0`` is always present and is the default."""

    def __init__(self, *, flame_path: Path | str | None = None):
        self._specs: dict[str, TopologySpec] = {}
        self._loaders: dict[str, Loader] = {}
        self._cache: dict[tuple, TopologyMesh] = {}
        self._bridges: dict[tuple, Any] = {}
        self._lock = threading.RLock()
        self._flame_path = Path(flame_path) if flame_path else None
        self.register(_makehuman_spec(), _load_makehuman)
        self.register(_flame_spec(), _load_flame_open)

    # ---------------------------------------------------------------- registration
    def register(self, spec: TopologySpec, loader: Loader) -> None:
        if spec.license_id not in COMMERCIAL_LICENSES or spec.commercial_use != "allow":
            raise DomainError("TOPOLOGY_LICENSE_REFUSED",
                              f"{spec.topology_id}: licence {spec.license_id!r} does not permit commercial use; refusing to register", 422)
        if spec.artifact_id is not None and spec.artifact_id not in OPEN_ARTIFACTS:
            raise DomainError("TOPOLOGY_NOT_PINNED", f"{spec.topology_id}: artifact {spec.artifact_id!r} has no pinned SHA-256 in local_models", 422)
        if spec.topology_id in self._specs:
            raise DomainError("TOPOLOGY_DUPLICATE", f"{spec.topology_id} is already registered", 409)
        self._specs[spec.topology_id] = spec
        self._loaders[spec.topology_id] = loader

    @property
    def default_id(self) -> str:
        return DEFAULT_TOPOLOGY_ID

    def ids(self) -> list[str]:
        return list(self._specs)

    def spec(self, topology_id: str) -> TopologySpec:
        try:
            return self._specs[topology_id]
        except KeyError:
            raise DomainError("TOPOLOGY_UNKNOWN", f"Unknown topology {topology_id!r}; registered: {', '.join(self._specs)}", 422) from None

    # ---------------------------------------------------------------- files
    def artifact_path(self, topology_id: str) -> Path | None:
        spec = self.spec(topology_id)
        if spec.artifact_id is None:
            return None
        candidate = self._flame_path or (Path(os.environ[FLAME_PATH_ENV]) if os.environ.get(FLAME_PATH_ENV) else None)
        if candidate is not None:
            from roop_3d.models.owned_files import owned_path
            try:
                candidate = owned_path(candidate)     # must stay inside this checkout, no links
            except ValueError as error:
                raise DomainError("TOPOLOGY_PATH_REJECTED", f"Morphable-model path rejected: {error}", 422) from error
            return candidate if candidate.is_file() else None
        return locate_open_artifact(spec.artifact_id, project_root())

    def availability(self, topology_id: str) -> dict[str, Any]:
        spec = self.spec(topology_id)
        if spec.builtin and spec.artifact_id is None:
            from roop_3d.reconstruction.makehuman_build import ASSET_NAME, DEFAULT_ASSET_DIR
            built = (DEFAULT_ASSET_DIR / ASSET_NAME).is_file()
            return {"available": built, "state": "ready" if built else "asset_missing",
                    "detail": None if built else "Build once: python -m roop_3d.reconstruction.makehuman_build"}
        path = self.artifact_path(topology_id)
        if path is None:
            return {"available": False, "state": "asset_missing",
                    "detail": f"{OPEN_ARTIFACTS[spec.artifact_id]['filename']} was not found "
                              f"(searched {', '.join(OPEN_ARTIFACTS[spec.artifact_id]['search_paths'])}; override with {FLAME_PATH_ENV})"}
        try:
            verify_open_artifact(spec.artifact_id, path)
        except AuditError as error:
            return {"available": False, "state": "hash_mismatch", "detail": error.detail}
        return {"available": True, "state": "ready", "detail": None}

    def describe(self) -> list[dict[str, Any]]:
        return [{**spec.public(), "default": spec.topology_id == self.default_id, **self.availability(spec.topology_id)}
                for spec in self._specs.values()]

    # ---------------------------------------------------------------- meshes
    def load(self, topology_id: str) -> TopologyMesh:
        """The topology's own rest mesh (for ``flame2023_open`` this has no UVs; see :meth:`prepared`)."""
        spec = self.spec(topology_id)
        with self._lock:
            key = (topology_id, self._stamp(spec))
            if key not in self._cache:
                self._cache = {k: v for k, v in self._cache.items() if k[0] != topology_id}
                self._cache[key] = self._loaders[topology_id](self, spec)
            return self._cache[key]

    def prepared(self, topology_id: str) -> TopologyMesh:
        """:meth:`load`, with a UV layout and a basis in matching vertex order. Derived through the bridge when the model has none."""
        mesh = self.load(topology_id)
        if mesh.uvs is not None or topology_id == self.default_id:
            return mesh
        return self.bridge(self.default_id, topology_id).target

    def bridge(self, source_id: str, target_id: str):
        """Cached :class:`~roop_3d.reconstruction.mesh_bridge.BarycentricMeshBridge` between two registered topologies."""
        from roop_3d.reconstruction.mesh_bridge import BarycentricMeshBridge
        source, target = self.load(source_id), self.load(target_id)
        with self._lock:
            key = (source_id, target_id, self._stamp(self.spec(source_id)), self._stamp(self.spec(target_id)))
            if key not in self._bridges:
                self._bridges[key] = BarycentricMeshBridge.build(source, target)
            return self._bridges[key]

    def fingerprint(self, topology_id: str) -> tuple:
        """Changes whenever the files behind a topology change (used to key caches built from it)."""
        return self._stamp(self.spec(topology_id))

    def _stamp(self, spec: TopologySpec) -> tuple:
        if spec.artifact_id is not None:
            path = self.artifact_path(spec.topology_id)
            return (str(path), path.stat().st_mtime_ns, path.stat().st_size) if path else ("absent",)
        from roop_3d.reconstruction.makehuman_build import ASSET_NAME, DEFAULT_ASSET_DIR
        target = DEFAULT_ASSET_DIR / ASSET_NAME
        return (str(target), target.stat().st_mtime_ns) if target.exists() else ("absent",)


def _makehuman_spec() -> TopologySpec:
    from roop_3d.reconstruction.makehuman_head import LICENSE_ID, LICENSE_NOTICE
    return TopologySpec(
        topology_id=MAKEHUMAN_ID, title="MakeHuman head + neck (CC0)", license_id=LICENSE_ID,
        license_url="https://creativecommons.org/publicdomain/zero/1.0/", commercial_use="allow",
        attribution="None required (CC0-1.0). " + LICENSE_NOTICE, source="https://github.com/makehumancommunity/makehuman",
        builtin=True)


def _flame_spec() -> TopologySpec:
    record = OPEN_ARTIFACTS[FLAME_OPEN_ID]
    return TopologySpec(
        topology_id=FLAME_OPEN_ID, title=record["name"], license_id=record["license_id"], license_url=record["license_url"],
        commercial_use=record["commercial_use"], attribution=record["attribution"], source=record["source"], builtin=False,
        artifact_id=FLAME_OPEN_ID)


def _load_makehuman(registry: MorphableTopologyRegistry, spec: TopologySpec) -> TopologyMesh:
    from roop_3d.reconstruction.makehuman_head import MakeHumanHeadModel
    from roop_3d.reconstruction.makehuman_template import makehuman_template, shape_basis
    mesh = makehuman_template()
    model = MakeHumanHeadModel.load()
    return TopologyMesh(
        topology_id=spec.topology_id, vertices=mesh.vertices.astype(np.float64), faces=mesh.faces.astype(np.int64),
        uvs=mesh.uvs.astype(np.float64), shape_basis=shape_basis(), shape_names=tuple(model.shape_names),
        landmarks={k: np.asarray(v, np.int64) for k, v in mesh.landmark_vertices.items()}, uv_provenance="makehuman_atlas",
        meta={"license_id": spec.license_id})


def _load_flame_open(registry: MorphableTopologyRegistry, spec: TopologySpec) -> TopologyMesh:
    path = registry.artifact_path(spec.topology_id)
    if path is None:
        availability = registry.availability(spec.topology_id)
        raise DomainError("TOPOLOGY_ASSET_MISSING", f"{spec.title} is not installed. {availability['detail']}", 404)
    try:
        verified = verify_open_artifact(spec.artifact_id, path)         # SHA-256 before a single byte is parsed
    except AuditError as error:
        raise DomainError("TOPOLOGY_ASSET_HASH_MISMATCH", error.detail, 422) from error
    raw = read_flame_open(path)
    vertices, faces, basis = flame_to_identity_frame(raw["v_template"], raw["f"], raw["shapedirs"])
    return TopologyMesh(
        topology_id=spec.topology_id, vertices=vertices, faces=faces, uvs=None, shape_basis=basis.astype(np.float32),
        shape_names=tuple(f"shape_{i:03d}" for i in range(basis.shape[2])), uv_provenance="none",
        meta={"license_id": spec.license_id, "sha256": verified["sha256"], "attribution": spec.attribution})


_DEFAULT: MorphableTopologyRegistry | None = None
_DEFAULT_LOCK = threading.Lock()


def default_registry() -> MorphableTopologyRegistry:
    """Process-wide registry (meshes and bridges are cached on it, so the first call is the only expensive one)."""
    global _DEFAULT
    with _DEFAULT_LOCK:
        if _DEFAULT is None:
            _DEFAULT = MorphableTopologyRegistry()
        return _DEFAULT
