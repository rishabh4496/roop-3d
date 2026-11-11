"""Stage 12: a dependency-free binary glTF 2.0 (``.glb``) serializer for head meshes, textures, shape keys and head tracks.

Only the standard library and numpy are used (the PNG encoder is ``zlib`` + ``struct``), so the module imports in milliseconds
and runs anywhere the service runs. It writes exactly one ``.glb`` byte stream:

* ``POSITION`` / ``NORMAL`` / ``TEXCOORD_0`` (and ``TANGENT`` when a normal map is attached), uint16 or uint32 indices
* a PBR material with an optional diffuse albedo PNG and an optional tangent-space normal-map PNG
* shape keys as glTF morph targets (``mesh.extras.targetNames`` carries their names, the convention three.js and Blender read)
* an optional head track as a glTF animation (translation, rotation and, if given, morph-weight channels on the head node)
* optional ``_UPPER_CASE`` per-vertex float attributes and free-form ``extras``

Frames. glTF is right-handed, y up. The project's identity-asset *mesh frame* is y down and z toward the viewer, which is
left-handed, so a mesh in that frame must be given with ``frame="mesh"``: the serializer then mirrors y and swaps the triangle
winding (same maths as ``reconstruction.identity_asset.to_gltf_frame``). Posed meshes are serialized the same way as canonical ones.

Channel order. Every image array in this code base is OpenCV-style BGR(A) uint8, so that is the default (``channel_order="bgr"``).

This module never imports Blender (``bpy``); see ``blender_bridge`` for the isolation rules.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import hashlib
import json
import math
import struct
import threading
from typing import Any, Literal, Mapping, Sequence
import zlib

import numpy as np

from roop_3d.core.errors import DomainError

GLB_MAGIC = b"glTF"
GLB_VERSION = 2
CHUNK_JSON = 0x4E4F534A      # "JSON"
CHUNK_BIN = 0x004E4942       # "BIN\0"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

_FLOAT, _USHORT, _UINT = 5126, 5123, 5125
_ARRAY_BUFFER, _ELEMENT_ARRAY_BUFFER = 34962, 34963
_CLAMP_TO_EDGE, _LINEAR = 33071, 9729
_FLIP_Y = np.array([1.0, -1.0, 1.0], np.float32)
_TYPE_BY_WIDTH = {1: "SCALAR", 2: "VEC2", 3: "VEC3", 4: "VEC4"}

Frame = Literal["gltf", "mesh"]
ChannelOrder = Literal["bgr", "rgb"]


def _fail(code: str, message: str, status: int = 422) -> DomainError:
    return DomainError(code, message, status)


# --------------------------------------------------------------------------------------------------------------- PNG
def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)


def encode_png(image: np.ndarray, *, channel_order: ChannelOrder = "bgr", level: int = 1) -> bytes:
    """Encode a uint8 grey / BGR(A) / RGB(A) array as PNG with the standard library only (filter 0, zlib ``level``)."""
    a = np.asarray(image)
    if a.dtype != np.uint8:
        raise _fail("GLB_TEXTURE_DTYPE", f"Textures must be uint8 arrays, got {a.dtype}")
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[..., 0]
    if a.ndim == 2:
        channels, colour_type = 1, 0
    elif a.ndim == 3 and a.shape[2] in (3, 4):
        channels, colour_type = a.shape[2], 2 if a.shape[2] == 3 else 6
        if channel_order == "bgr":
            a = a[..., [2, 1, 0] if channels == 3 else [2, 1, 0, 3]]
    else:
        raise _fail("GLB_TEXTURE_SHAPE", f"Textures must be (H, W), (H, W, 3) or (H, W, 4), got {a.shape}")
    height, width = a.shape[:2]
    if height == 0 or width == 0:
        raise _fail("GLB_TEXTURE_SHAPE", "Texture has no pixels")
    raw = np.empty((height, 1 + width * channels), np.uint8)
    raw[:, 0] = 0
    raw[:, 1:] = a.reshape(height, width * channels)
    ihdr = struct.pack(">IIBBBBB", width, height, 8, colour_type, 0, 0, 0)
    return PNG_SIGNATURE + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", zlib.compress(raw, level)) + _png_chunk(b"IEND", b"")


def png_size(data: bytes) -> tuple[int, int]:
    """(width, height) of a PNG byte string; raises if it is not one."""
    if len(data) < 33 or data[:8] != PNG_SIGNATURE or data[12:16] != b"IHDR":
        raise _fail("GLB_TEXTURE_NOT_PNG", "Texture bytes are not a PNG image")
    width, height = struct.unpack(">II", data[16:24])
    return int(width), int(height)


class _PngCache:
    """Content-addressed LRU of encoded PNGs: re-serialising an unchanged atlas costs a hash (~ms), not a zlib pass."""

    def __init__(self, max_entries: int = 8, max_bytes: int = 96 * 1024 * 1024):
        self._items: OrderedDict[tuple, bytes] = OrderedDict()
        self._bytes, self._max_entries, self._max_bytes = 0, max_entries, max_bytes
        self._lock = threading.Lock()

    def get(self, image: np.ndarray, channel_order: str, level: int) -> bytes:
        a = np.ascontiguousarray(image)
        key = (hashlib.blake2b(a, digest_size=16).digest(), a.shape, str(a.dtype), channel_order, level)
        with self._lock:
            hit = self._items.get(key)
            if hit is not None:
                self._items.move_to_end(key)
                return hit
        encoded = encode_png(a, channel_order=channel_order, level=level)
        with self._lock:
            if key not in self._items:
                self._items[key] = encoded
                self._bytes += len(encoded)
            while self._items and (len(self._items) > self._max_entries or self._bytes > self._max_bytes):
                _, dropped = self._items.popitem(last=False)
                self._bytes -= len(dropped)
        return encoded


# ---------------------------------------------------------------------------------------------------------- inputs
@dataclass(frozen=True)
class MorphTarget:
    """One shape key: per-vertex position deltas in the same frame as the ``frame`` argument of ``serialize``."""
    name: str
    position_deltas: np.ndarray                    # (V, 3)
    normal_deltas: np.ndarray | None = None        # (V, 3), optional; viewers shade with the base normals when absent


@dataclass(frozen=True)
class HeadTrack:
    """A rigid head animation in the glTF frame (metres, unit quaternions ``x, y, z, w``)."""
    times: np.ndarray                              # (N,) seconds, strictly increasing
    translations: np.ndarray                       # (N, 3)
    rotations: np.ndarray                          # (N, 4) xyzw
    morph_weights: np.ndarray | None = None        # (N, K) keyed to the serializer's morph targets
    name: str = "head_track"
    extras: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_tracked_frames(cls, frames: Sequence[Any], *, name: str = "head_track") -> "HeadTrack":
        """Build a track from ``tracking.video_tracker.TrackedHeadFrame`` items (duck-typed: ``timestamp``, ``frame_index``,
        ``confidence``, ``camera.rotation`` / ``tx`` / ``ty`` / ``scale``). Frames without a camera pose are skipped.

        The tracker's camera is weak-perspective: ``pixel = scale * (R X)[:2] + t``. Rotation (3 DoF) and the in-plane position
        (2 DoF) are exact; depth is *not* metric, so the Z translation is left at 0 and the per-key ``scale`` (pixels per
        metre, which encodes distance) is kept in ``extras`` instead of being faked into a translation. Translation is relative to
        the first tracked frame."""
        kept = [f for f in frames if getattr(f, "camera", None) is not None]
        if not kept:
            raise _fail("GLB_TRACK_EMPTY", "No frame carries a camera pose, so there is no head track to export")
        t0 = float(kept[0].timestamp)
        times = np.array([float(f.timestamp) - t0 for f in kept], np.float64)
        keep = np.concatenate([[True], np.diff(times) > 1e-9])                       # glTF needs strictly increasing times
        kept, times = [f for f, ok in zip(kept, keep) if ok], times[keep]
        xy = np.array([[f.camera.tx / f.camera.scale, -f.camera.ty / f.camera.scale] for f in kept], np.float64)
        xy -= xy[0]
        trans = np.zeros((len(kept), 3), np.float32)
        trans[:, :2] = xy
        flip = np.diag([1.0, -1.0, 1.0])
        quats = _continuous([_matrix_to_xyzw(flip @ np.asarray(f.camera.rotation, np.float64) @ flip) for f in kept])
        extras = {
            "source": "VideoHeadTracker3D weak-perspective camera",
            "dof": "rotation (3) + in-plane translation (2) measured; depth not metric (translation z = 0)",
            "translation_origin": "first tracked frame",
            "scale_px_per_m": [round(float(f.camera.scale), 4) for f in kept],
            "frame_index": [int(f.frame_index) for f in kept],
            "confidence": [round(float(f.confidence), 4) for f in kept],
        }
        return cls(times, trans, quats, None, name, extras)


def _matrix_to_xyzw(r: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion ``x, y, z, w`` (Shepperd)."""
    trace = r[0, 0] + r[1, 1] + r[2, 2]
    if trace > 0.0:
        s = 0.5 / math.sqrt(trace + 1.0)
        q = (r[2, 1] - r[1, 2]) * s, (r[0, 2] - r[2, 0]) * s, (r[1, 0] - r[0, 1]) * s, 0.25 / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = 2.0 * math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
        q = 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s, (r[2, 1] - r[1, 2]) / s
    elif r[1, 1] > r[2, 2]:
        s = 2.0 * math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])
        q = (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s, (r[0, 2] - r[2, 0]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])
        q = (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s, (r[1, 0] - r[0, 1]) / s
    out = np.array(q, np.float64)
    return out / np.linalg.norm(out)


def _continuous(quats: Sequence[np.ndarray]) -> np.ndarray:
    """Flip signs so consecutive quaternions are in the same hemisphere (otherwise LINEAR interpolation takes the long way)."""
    out = np.array(quats, np.float64)
    for i in range(1, len(out)):
        if float(out[i] @ out[i - 1]) < 0.0:
            out[i] = -out[i]
    return out.astype(np.float32)


# ------------------------------------------------------------------------------------------- normal map / tangents
def metres_per_uv_unit(positions: np.ndarray, faces: np.ndarray, uvs: np.ndarray) -> float:
    """Surface metres per UV unit (sum of 3-D edge lengths over sum of UV edge lengths): the scale a height map needs."""
    f = np.asarray(faces, np.int64)
    p, t = np.asarray(positions, np.float64), np.asarray(uvs, np.float64)
    pairs = ((0, 1), (1, 2), (2, 0))
    d3 = sum(np.linalg.norm(p[f[:, a]] - p[f[:, b]], axis=1).sum() for a, b in pairs)
    d2 = sum(np.linalg.norm(t[f[:, a]] - t[f[:, b]], axis=1).sum() for a, b in pairs)
    if d2 <= 1e-12:
        raise _fail("GLB_UV_DEGENERATE", "The UV layout has no area, so no normal map can be derived")
    return float(d3 / d2)


def normal_map_from_height(height_m: np.ndarray, metres_per_texel: float, *, strength: float = 1.0) -> np.ndarray:
    """Tangent-space normal map (BGR uint8, OpenGL / glTF convention: R = +u, G = +v with v up, B = out of the surface)
    from a UV-space height field in metres. Row 0 is the top of the image, i.e. ``v = 1``."""
    h = np.asarray(height_m, np.float64)
    if h.ndim != 2 or h.size == 0 or not np.all(np.isfinite(h)):
        raise _fail("GLB_HEIGHT_INVALID", "Height map must be a finite (H, W) array")
    if not metres_per_texel > 0.0:
        raise _fail("GLB_HEIGHT_INVALID", "metres_per_texel must be positive")
    d_u = np.gradient(h, axis=1) / metres_per_texel
    d_v = -np.gradient(h, axis=0) / metres_per_texel                     # image rows run toward -v
    n = np.stack([-d_u * strength, -d_v * strength, np.ones_like(h)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    rgb = np.rint((n * 0.5 + 0.5) * 255.0).astype(np.uint8)
    return np.ascontiguousarray(rgb[..., ::-1])


def compute_tangents(positions: np.ndarray, normals: np.ndarray, uvs: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Per-vertex ``TANGENT`` (x, y, z, w) for a glTF normal map. ``uvs`` use the OBJ convention (v up). The tangent follows
    +u; the glTF bitangent ``cross(N, T) * w`` follows +v, which is the green channel of the normal map."""
    f = np.asarray(faces, np.int64)
    p, n, t = (np.asarray(x, np.float64) for x in (positions, normals, uvs))
    e1, e2 = p[f[:, 1]] - p[f[:, 0]], p[f[:, 2]] - p[f[:, 0]]
    d1, d2 = t[f[:, 1]] - t[f[:, 0]], t[f[:, 2]] - t[f[:, 0]]
    det = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]
    ok = np.abs(det) > 1e-18
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)[:, None]
    tan = (d2[:, 1:2] * e1 - d1[:, 1:2] * e2) * inv
    bit = (d1[:, 0:1] * e2 - d2[:, 0:1] * e1) * inv
    acc_t, acc_b = np.zeros_like(p), np.zeros_like(p)
    for corner in range(3):
        for axis in range(3):
            acc_t[:, axis] += np.bincount(f[:, corner], weights=tan[:, axis], minlength=len(p))
            acc_b[:, axis] += np.bincount(f[:, corner], weights=bit[:, axis], minlength=len(p))
    tangent = acc_t - n * np.sum(n * acc_t, axis=1, keepdims=True)
    length = np.linalg.norm(tangent, axis=1, keepdims=True)
    fallback = np.cross(n, [0.0, 1.0, 0.0]) + 1e-12                               # vertices no valid triangle reaches
    tangent = np.where(length > 1e-12, tangent / np.maximum(length, 1e-12), fallback / np.linalg.norm(fallback, axis=1, keepdims=True))
    w = np.where(np.sum(np.cross(n, tangent) * acc_b, axis=1) < 0.0, -1.0, 1.0)
    return np.concatenate([tangent, w[:, None]], axis=1).astype(np.float32)


# --------------------------------------------------------------------------------------------------- serializer
class _Packer:
    """Accumulates the BIN chunk with 4-byte alignment and records buffer views / accessors."""

    def __init__(self) -> None:
        self.chunks: list[Any] = []
        self.offset = 0
        self.views: list[dict[str, Any]] = []
        self.accessors: list[dict[str, Any]] = []

    def view(self, data: Any, nbytes: int, target: int | None = None) -> int:
        pad = -self.offset % 4
        if pad:
            self.chunks.append(b"\x00" * pad)
            self.offset += pad
        entry: dict[str, Any] = {"buffer": 0, "byteOffset": self.offset, "byteLength": nbytes}
        if target is not None:
            entry["target"] = target
        self.views.append(entry)
        self.chunks.append(data)
        self.offset += nbytes
        return len(self.views) - 1

    def array(self, values: np.ndarray, component: int, target: int | None, *, bounds: bool = False) -> int:
        dtype = {_FLOAT: "<f4", _USHORT: "<u2", _UINT: "<u4"}[component]
        a = np.ascontiguousarray(values, dtype=dtype)
        width = 1 if a.ndim == 1 else a.shape[1]
        acc: dict[str, Any] = {"bufferView": self.view(a.tobytes(), a.nbytes, target), "componentType": component,
                               "count": int(a.shape[0]), "type": _TYPE_BY_WIDTH[width]}
        if bounds:
            lo, hi = a.min(axis=0), a.max(axis=0)
            acc["min"], acc["max"] = np.atleast_1d(lo).astype(float).tolist(), np.atleast_1d(hi).astype(float).tolist()
        self.accessors.append(acc)
        return len(self.accessors) - 1

    def blob(self, data: bytes) -> int:
        return self.view(data, len(data))


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serialisable")


def _pad(data: bytes, filler: bytes) -> bytes:
    return data + filler * (-len(data) % 4)


class GLBBinarySerializer:
    """Serialize a head mesh (+ textures, shape keys, head track) to ``.glb`` bytes. Thread-safe; holds only a PNG cache."""

    def __init__(self, *, generator: str = "roop-3d GLBBinarySerializer", png_level: int = 1, cache: _PngCache | None = None):
        self.generator, self.png_level = generator, int(png_level)
        self._png = cache or _PngCache()

    # ----- texture handling
    def _image_bytes(self, image: Any, channel_order: ChannelOrder, what: str) -> tuple[bytes, tuple[int, int]]:
        if isinstance(image, (bytes, bytearray, memoryview)):
            data = bytes(image)
            return data, png_size(data)
        if not isinstance(image, np.ndarray):
            raise _fail("GLB_TEXTURE_TYPE", f"{what} must be a uint8 array or PNG bytes")
        data = self._png.get(image, channel_order, self.png_level)
        return data, (int(image.shape[1]), int(image.shape[0]))

    # ----- geometry
    @staticmethod
    def _geometry(mesh: Any, frame: Frame):
        if frame not in ("gltf", "mesh"):
            raise _fail("GLB_FRAME_INVALID", "frame must be 'gltf' or 'mesh'")
        vertices = np.asarray(mesh.vertices, np.float32)
        faces = np.asarray(mesh.faces, np.int64)
        normals = np.asarray(mesh.normals, np.float32)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3 or normals.shape != vertices.shape:
            raise _fail("GLB_MESH_INVALID", "Mesh needs (V, 3) vertices and normals and (F, 3) faces")
        if len(vertices) == 0 or len(faces) == 0:
            raise _fail("GLB_MESH_INVALID", "Mesh is empty")
        if faces.min() < 0 or faces.max() >= len(vertices):
            raise _fail("GLB_MESH_INVALID", "Face index out of bounds")
        if not (np.isfinite(vertices).all() and np.isfinite(normals).all()):
            raise _fail("GLB_MESH_INVALID", "Mesh contains NaN or Inf")
        uvs = None if getattr(mesh, "uvs", None) is None else np.asarray(mesh.uvs, np.float32)
        if uvs is not None and (uvs.shape != (len(vertices), 2) or not np.isfinite(uvs).all()):
            raise _fail("GLB_UV_COUNT_MISMATCH", "GLB needs exactly one finite UV per vertex (weld per-corner UVs first)")
        if frame == "mesh":
            vertices, normals, faces = vertices * _FLIP_Y, normals * _FLIP_Y, faces[:, [0, 2, 1]]
        return vertices, faces, normals, uvs

    # ----- main entry
    def serialize(
        self, mesh: Any, *, frame: Frame = "gltf", albedo: Any = None, normal_map: Any = None,
        morph_targets: Sequence[MorphTarget] = (), morph_weights: Sequence[float] | None = None,
        track: HeadTrack | None = None, vertex_attributes: Mapping[str, np.ndarray] | None = None,
        extras: Mapping[str, Any] | None = None, name: str = "head", channel_order: ChannelOrder = "bgr",
    ) -> bytes:
        """``mesh`` is any object with ``vertices`` (V, 3), ``faces`` (F, 3), ``normals`` (V, 3) and optional ``uvs`` (V, 2),
        such as ``reconstruction.geometry_types.Mesh3D``. ``albedo`` / ``normal_map`` are uint8 arrays (``channel_order``) or PNG
        bytes. Morph-target deltas and ``vertex_attributes`` follow the same ``frame`` as the vertices."""
        vertices, faces, normals, uvs = self._geometry(mesh, frame)
        count = len(vertices)
        if (albedo is not None or normal_map is not None) and uvs is None:
            raise _fail("GLB_TEXTURE_NEEDS_UVS", "Textures need a mesh with UV coordinates")
        pack = _Packer()

        attributes = {"POSITION": pack.array(vertices, _FLOAT, _ARRAY_BUFFER, bounds=True),
                      "NORMAL": pack.array(normals, _FLOAT, _ARRAY_BUFFER)}
        if uvs is not None:
            gltf_uv = uvs.copy()
            gltf_uv[:, 1] = 1.0 - gltf_uv[:, 1]                                        # glTF's v origin is the top
            attributes["TEXCOORD_0"] = pack.array(gltf_uv, _FLOAT, _ARRAY_BUFFER)
        if normal_map is not None:
            attributes["TANGENT"] = pack.array(compute_tangents(vertices, normals, uvs, faces), _FLOAT, _ARRAY_BUFFER)
        for key, values in (vertex_attributes or {}).items():
            if not key.startswith("_") or len(key) < 2 or key != key.upper() or not key[1:].replace("_", "").isalnum():
                raise _fail("GLB_ATTRIBUTE_INVALID", f"Custom glTF attribute {key!r} must look like _UPPER_CASE")
            arr = np.asarray(values, np.float32).reshape(count, -1)
            if arr.shape[1] not in _TYPE_BY_WIDTH:
                raise _fail("GLB_ATTRIBUTE_INVALID", f"{key} must have 1-4 components per vertex")
            if not np.isfinite(arr).all():
                raise _fail("GLB_ATTRIBUTE_INVALID", f"{key} contains NaN or Inf")
            attributes[key] = pack.array(arr, _FLOAT, _ARRAY_BUFFER)

        indices = pack.array(faces.reshape(-1), _USHORT if count <= 65535 else _UINT, _ELEMENT_ARRAY_BUFFER)

        targets: list[dict[str, int]] = []
        for target in morph_targets:
            delta = np.asarray(target.position_deltas, np.float32)
            if delta.shape != vertices.shape or not np.isfinite(delta).all():
                raise _fail("GLB_MORPH_INVALID", f"Morph target {target.name!r} needs finite deltas of shape {vertices.shape}")
            if frame == "mesh":
                delta = delta * _FLIP_Y
            entry = {"POSITION": pack.array(delta, _FLOAT, _ARRAY_BUFFER, bounds=True)}
            if target.normal_deltas is not None:
                nd = np.asarray(target.normal_deltas, np.float32)
                if nd.shape != vertices.shape or not np.isfinite(nd).all():
                    raise _fail("GLB_MORPH_INVALID", f"Morph target {target.name!r} normal deltas have the wrong shape")
                entry["NORMAL"] = pack.array(nd * _FLIP_Y if frame == "mesh" else nd, _FLOAT, _ARRAY_BUFFER)
            targets.append(entry)
        weights = [0.0] * len(targets) if morph_weights is None else [float(w) for w in morph_weights]
        if len(weights) != len(targets):
            raise _fail("GLB_MORPH_INVALID", f"{len(weights)} morph weights for {len(targets)} morph targets")

        material: dict[str, Any] = {"name": "skin", "doubleSided": False,
                                    "pbrMetallicRoughness": {"metallicFactor": 0.0, "roughnessFactor": 0.85}}
        images: list[dict[str, Any]] = []
        textures: list[dict[str, Any]] = []
        sizes: dict[str, tuple[int, int]] = {}
        for slot, image in (("baseColorTexture", albedo), ("normalTexture", normal_map)):
            if image is None:
                continue
            data, sizes[slot] = self._image_bytes(image, channel_order, slot)
            images.append({"bufferView": pack.blob(data), "mimeType": "image/png", "name": slot})
            textures.append({"sampler": 0, "source": len(images) - 1})
            ref = {"index": len(textures) - 1}
            if slot == "baseColorTexture":
                material["pbrMetallicRoughness"]["baseColorTexture"] = ref
            else:
                material["normalTexture"] = {**ref, "scale": 1.0}
        if albedo is None:
            material["pbrMetallicRoughness"]["baseColorFactor"] = [0.8, 0.8, 0.8, 1.0]

        primitive: dict[str, Any] = {"attributes": attributes, "indices": indices, "material": 0, "mode": 4}
        if targets:
            primitive["targets"] = targets
        mesh_json: dict[str, Any] = {"name": name, "primitives": [primitive]}
        if targets:
            mesh_json["weights"] = weights
            mesh_json["extras"] = {"targetNames": [t.name for t in morph_targets]}

        root: dict[str, Any] = {
            "asset": {"version": "2.0", "generator": self.generator},
            "scene": 0, "scenes": [{"name": name, "nodes": [0]}], "nodes": [{"name": name, "mesh": 0}],
            "meshes": [mesh_json], "materials": [material],
        }
        if textures:
            root["samplers"] = [{"magFilter": _LINEAR, "minFilter": _LINEAR, "wrapS": _CLAMP_TO_EDGE, "wrapT": _CLAMP_TO_EDGE}]
            root["textures"], root["images"] = textures, images
        if track is not None:
            root["animations"] = [self._animation(track, pack, len(targets))]
        if extras:
            root["extras"] = dict(extras)

        root["accessors"], root["bufferViews"] = pack.accessors, pack.views
        root["buffers"] = [{"byteLength": -(-pack.offset // 4) * 4}]
        return self._assemble(root, pack)

    # ----- animation
    @staticmethod
    def _animation(track: HeadTrack, pack: _Packer, target_count: int) -> dict[str, Any]:
        times = np.asarray(track.times, np.float32)
        n = len(times)
        trans, rot = np.asarray(track.translations, np.float32), np.asarray(track.rotations, np.float32)
        if n == 0 or trans.shape != (n, 3) or rot.shape != (n, 4):
            raise _fail("GLB_TRACK_INVALID", "Track needs N>0 times, (N, 3) translations and (N, 4) rotations")
        if not (np.isfinite(times).all() and np.isfinite(trans).all() and np.isfinite(rot).all()):
            raise _fail("GLB_TRACK_INVALID", "Track contains NaN or Inf")
        if n > 1 and not np.all(np.diff(times) > 0.0):
            raise _fail("GLB_TRACK_INVALID", "Track times must be strictly increasing")
        norms = np.linalg.norm(rot, axis=1, keepdims=True)
        if np.any(norms < 1e-6):
            raise _fail("GLB_TRACK_INVALID", "Track contains a zero quaternion")
        rot = rot / norms
        time_acc = pack.array(times, _FLOAT, None, bounds=True)                       # glTF requires min/max on animation input
        samplers = [{"input": time_acc, "output": pack.array(trans, _FLOAT, None), "interpolation": "LINEAR"},
                    {"input": time_acc, "output": pack.array(rot, _FLOAT, None), "interpolation": "LINEAR"}]
        channels = [{"sampler": 0, "target": {"node": 0, "path": "translation"}},
                    {"sampler": 1, "target": {"node": 0, "path": "rotation"}}]
        if track.morph_weights is not None:
            w = np.asarray(track.morph_weights, np.float32)
            if target_count == 0 or w.shape != (n, target_count) or not np.isfinite(w).all():
                raise _fail("GLB_TRACK_INVALID", f"Morph weights must be (N, {target_count}) and the mesh needs morph targets")
            samplers.append({"input": time_acc, "output": pack.array(w.reshape(-1), _FLOAT, None), "interpolation": "LINEAR"})
            channels.append({"sampler": 2, "target": {"node": 0, "path": "weights"}})
        out: dict[str, Any] = {"name": track.name, "samplers": samplers, "channels": channels}
        if track.extras:
            out["extras"] = dict(track.extras)
        return out

    # ----- container
    @staticmethod
    def _assemble(root: dict[str, Any], pack: _Packer) -> bytes:
        json_chunk = _pad(json.dumps(root, separators=(",", ":"), default=_json_default).encode("utf-8"), b" ")
        bin_len = -(-pack.offset // 4) * 4
        tail = b"\x00" * (bin_len - pack.offset)
        total = 12 + 8 + len(json_chunk) + 8 + bin_len
        return b"".join([struct.pack("<4sII", GLB_MAGIC, GLB_VERSION, total), struct.pack("<II", len(json_chunk), CHUNK_JSON),
                         json_chunk, struct.pack("<II", bin_len, CHUNK_BIN), *pack.chunks, tail])


# ------------------------------------------------------------------------------------------------------ reader
def parse_glb(data: bytes) -> tuple[dict[str, Any], bytes]:
    """Strictly parse a ``.glb`` byte string into ``(json, bin)``; raises ``DomainError`` on any container violation."""
    if len(data) < 20 or data[:4] != GLB_MAGIC:
        raise _fail("GLB_BAD_MAGIC", "Not a GLB file (magic is not 'glTF')")
    version, length = struct.unpack_from("<II", data, 4)
    if version != GLB_VERSION:
        raise _fail("GLB_BAD_VERSION", f"Unsupported GLB container version {version}")
    if length != len(data):
        raise _fail("GLB_BAD_LENGTH", f"Header says {length} bytes but the stream has {len(data)}")
    chunks: list[tuple[int, bytes]] = []
    pos = 12
    while pos < length:
        if pos + 8 > length:
            raise _fail("GLB_BAD_CHUNK", "Truncated chunk header")
        size, kind = struct.unpack_from("<II", data, pos)
        pos += 8
        if size % 4 or pos + size > length:
            raise _fail("GLB_BAD_CHUNK", "Chunk length is unaligned or runs past the end of the file")
        chunks.append((kind, data[pos:pos + size]))
        pos += size
    if not chunks or chunks[0][0] != CHUNK_JSON:
        raise _fail("GLB_BAD_CHUNK", "The first chunk must be JSON")
    document = json.loads(chunks[0][1].decode("utf-8"))
    bin_chunks = [c for k, c in chunks[1:] if k == CHUNK_BIN]
    return document, (bin_chunks[0] if bin_chunks else b"")


def read_accessor(document: Mapping[str, Any], binary: bytes, index: int) -> np.ndarray:
    """Decode one accessor (float / uint16 / uint32, tightly packed) back to a numpy array, for verification and tooling."""
    acc = document["accessors"][index]
    view = document["bufferViews"][acc["bufferView"]]
    dtype = {_FLOAT: "<f4", _USHORT: "<u2", _UINT: "<u4"}[acc["componentType"]]
    width = {v: k for k, v in _TYPE_BY_WIDTH.items()}[acc["type"]]
    start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    flat = np.frombuffer(binary, dtype=dtype, count=acc["count"] * width, offset=start)
    return flat.reshape(acc["count"], width) if width > 1 else flat
