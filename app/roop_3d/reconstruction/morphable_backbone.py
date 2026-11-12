"""Versioned, local three-tier reconstruction exports.

This is an export boundary, not a claim that arbitrary upstream checkpoints are
ONNX compatible. A bundle must include topology/mask provenance and three graphs
with the contracts documented in docs/dense-face-backbone.md. Missing outputs
are errors; zero-filled substitutes, synthetic wrinkles and MakeHuman fallbacks
are never supplied by this adapter.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import cv2
import numpy as np

from roop_3d.audit.project_paths import application_root
from roop_3d.core.errors import DomainError
from roop_3d.engine.volumetric.preprocess import aligned_crop
from roop_3d.models.owned_files import owned_path
from roop_3d.reconstruction.dense_correspondence import project
from roop_3d.reconstruction.geometry_types import Mesh3D
from roop_3d.reconstruction.displaced_surface import DisplacementSurface, detail_levels, sample_uv_displacement

TOPOLOGIES = {"bfm35709": 35709, "flame2023open": 5023}
SCHEMA = "roop3d.dense_face_bundle.v1"


def bundle_signature():
    """Invalidate cached preview geometry when a local bundle is installed/replaced."""
    path = owned_path(application_root() / "roop_3d/.runtime/models/dense_face/bundle.json")
    if not path.is_file():
        return ("dense_face_missing",)
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    try:
        assets = json.loads(payload)["assets"]
        identities = []
        for role, record in sorted(assets.items()):
            asset = owned_path(path.parent / record["file"])
            if not asset.is_relative_to(path.parent):
                raise ValueError("Bundle asset leaves its directory")
            info = asset.stat()
            identities.append((role, info.st_size, info.st_mtime_ns, info.st_ctime_ns))
        return (digest, tuple(identities))
    except (KeyError, TypeError, ValueError, OSError):
        return (digest, "invalid_assets")


def finite(value, shape, label):
    array = np.asarray(value, np.float32)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{label}: expected finite {shape}, got {array.shape}")
    return array


def binary(value, shape, label):
    array = np.asarray(value)
    if array.shape != shape or not np.isin(array, (0, 1)).all():
        raise ValueError(f"{label}: expected binary {shape}")
    return array.astype(bool)


@dataclass(frozen=True)
class FaceTopology:
    family: str
    vertex_count: int
    vertex_ids: np.ndarray
    faces: np.ndarray
    uvs: np.ndarray

    @classmethod
    def load(cls, family, path):
        """Physically compact the surface; a boundary triangle must have ALL vertices allowed."""
        count = TOPOLOGIES[family]
        with np.load(path, allow_pickle=False) as data:
            vertices = finite(data["vertices_m"], (count, 3), "template")
            uvs = finite(data["uvs"], (count, 2), "UVs")
            if np.any((uvs < 0) | (uvs > 1)):
                raise ValueError("UVs must be in [0,1]")
            raw_faces = data["faces"]
            if raw_faces.dtype.kind not in "iu":
                raise ValueError("Triangle indices must be integers")
            mesh = Mesh3D(vertices, raw_faces, uvs=uvs)
            allowed = binary(data["face_surface"], (count,), "anatomical face mask")
            # Separate exclusions veto even an erroneous broad face annotation.
            for name in ("scalp_hair", "ears", "neck"):
                allowed &= ~binary(data[name], (count,), name)
            if family == "bfm35709":
                parts = binary(data["semantic_parts"], (8, count), "3DDFA-V3 eight-part annotation")
                allowed &= parts.any(0)
            faces = mesh.faces[allowed[mesh.faces].all(1)]
            if not len(faces):
                raise ValueError("Semantic face mask leaves no facial triangles")
            ids = np.unique(faces)
            remap = np.full(count, -1, np.int32)
            remap[ids] = np.arange(len(ids))
            return cls(family, count, ids, remap[faces], uvs[ids])

    def select(self, vertices):
        return finite(vertices, (self.vertex_count, 3), "native topology vertices")[self.vertex_ids]


@dataclass(frozen=True)
class FittedFace:
    vertices: np.ndarray  # compact canonical vertices, OpenCV model frame, metres
    camera_vertices: np.ndarray
    intrinsics: np.ndarray
    alignment: object
    identity: np.ndarray
    expression: np.ndarray
    mid_offsets: np.ndarray
    displacement_uv: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray
    scale: float
    jaw: np.ndarray | None = None
    eyelids: np.ndarray | None = None
    expression_vertices: np.ndarray | None = None  # expression-only geometry before HRN/detail offsets


def load_geometry_session(catalog, name: str, path, config):
    """Session for one dense-face geometry graph: ``(session, provider_name)``.

    These graphs (coarse pose, SMIRK expression, HRN mid/high detail) were pinned to the CPU, which made the 3D-head
    stage cost 10-15 s per frame on a machine whose GPU sat idle.  On a 12 GB card they now run on the CUDA EP in fp32
    (the same precision, so identical maths); 6 GB and CPU profiles keep the CPU so the neural swapper and enhancer
    still own the VRAM.  Any GPU initialisation failure falls back to the CPU.  ``ROOP3D_GEOMETRY_PROVIDER`` =
    ``cpu`` | ``cuda`` | ``auto`` (default) overrides the profile rule.
    """
    import logging
    import os
    from dataclasses import replace
    from roop_3d.engine.volumetric.swapper_engine import load_local_session
    wanted = os.environ.get("ROOP3D_GEOMETRY_PROVIDER", "auto").strip().lower()
    on_gpu_stack = config.provider in ("TensorrtExecutionProvider", "CUDAExecutionProvider")
    use_gpu = wanted != "cpu" and on_gpu_stack and (wanted == "cuda" or config.hardware_profile == "12gb")
    if use_gpu:
        try:
            session = load_local_session(catalog, name, path, replace(config, provider="CUDAExecutionProvider", precision_policy="fp32"))
            return session, "CUDAExecutionProvider"
        except Exception as error:   # no GPU session -> the proven CPU path, loudly
            logging.getLogger("roop_3d.ort").warning("[ORT] %s could not start on CUDA (%s); using CPU", name, type(error).__name__)
    return load_local_session(catalog, name, path, replace(config, provider="CPUExecutionProvider", precision_policy="fp32")), "CPUExecutionProvider"


class NamedGraph:
    def __init__(self, session):
        self.session = session
        self.inputs = {node.name for node in session.get_inputs()}
        self.outputs = [node.name for node in session.get_outputs()]

    def run(self, values):
        if self.inputs != values.keys():
            raise ValueError(f"Reconstruction export input mismatch: {sorted(self.inputs)}")
        outputs = dict(zip(self.outputs, self.session.run(None, values)))
        if any(not np.isfinite(v).all() for v in outputs.values()):
            raise ValueError("Non-finite reconstruction output")
        return outputs

    def release(self):
        self.session = None


class MorphableBackbone:
    def __init__(self, topology, coarse, expression, detail, *, expression_basis, evidence, detail_size=512,
                 detail_subdivisions=0):
        if expression_basis not in ("arkit52", "smirk_flame50"):
            raise ValueError("Expression basis must be explicitly declared")
        self.topology = topology
        self.surface = DisplacementSurface(topology.faces, topology.uvs, levels=detail_subdivisions)
        self.coarse, self.expression, self.detail = coarse, expression, detail
        self.expression_basis, self.evidence = expression_basis, evidence
        if detail_size not in (512, 1024):
            raise ValueError("HRN requires a declared 512px or 1024px high-resolution crop")
        self.detail_size = detail_size

    @classmethod
    def load(cls, catalog, config, bundle=None):
        path = owned_path(Path(bundle) if bundle else
            application_root() / "roop_3d/.runtime/models/dense_face/bundle.json")
        if not path.is_file():
            raise DomainError("DENSE_FACE_ASSETS_MISSING",
                "Install a compatible 3DDFA-V3 or FLAME 2023 Open + SMIRK + HRN export bundle at "
                "roop_3d/.runtime/models/dense_face/bundle.json; see docs/dense-face-backbone.md", 503)
        graphs = []
        try:
            spec = json.loads(path.read_text(encoding="utf-8"))
            from roop_3d.reconstruction.native_backbone import NATIVE_SCHEMA, NativeMorphableBackbone
            if spec.get("schema") == NATIVE_SCHEMA:
                return NativeMorphableBackbone.load(catalog, config, path, spec)
            if (spec["schema"] != SCHEMA or spec["topology"] not in TOPOLOGIES
                    or spec["coordinates"] != "opencv_metres" or spec["uv_origin"] != "bottom_left"
                    or spec["crop"] != "arcface_128_rgb_0_1"
                    or spec["detail_crop_size"] not in (512, 1024)
                    or not spec["mask_provenance"].strip() or not spec["expression_mapping_evidence"].strip()
                    or spec["detail_topology"] != spec["topology"]):
                raise ValueError("Unsupported topology, crop, detail mapping or provenance")
            verified = {}
            for role in ("topology", "coarse", "expression", "detail"):
                record = spec["assets"][role]
                asset = owned_path(path.parent / record["file"])
                if not asset.is_relative_to(path.parent) or not re.fullmatch("[0-9a-f]{64}", record["sha256"]):
                    raise ValueError("Invalid bundle asset location or digest")
                digest = hashlib.sha256()
                with asset.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1048576), b""):
                        digest.update(chunk)
                if digest.hexdigest() != record["sha256"]:
                    raise ValueError(f"Bundle digest mismatch: {role}")
                verified[role] = asset
            topology = FaceTopology.load(spec["topology"], verified["topology"])
            providers = set()
            for role in ("coarse", "expression", "detail"):
                session, used = load_geometry_session(catalog, f"dense_face_{role}", verified[role], config)
                providers.add(used)
                graphs.append(NamedGraph(session))
            instance = cls(topology, *graphs, expression_basis=spec["expression_basis"], evidence=spec,
                           detail_size=spec["detail_crop_size"],
                           detail_subdivisions=detail_levels(spec.get("detail_subdivisions", 1), config.hardware_profile))
            instance.geometry_provider = providers.pop() if len(providers) == 1 else "mixed"
            return instance
        except Exception as error:
            for graph in graphs:
                graph.release()
            raise DomainError("DENSE_FACE_BUNDLE_INVALID", f"Incompatible dense face bundle: {error}", 503) from error

    def fit(self, image_bgr, landmarks5):
        alignment = aligned_crop(image_bgr, np.asarray(landmarks5), 128)
        # Warp directly from original pixels at encoder resolution, avoiding an
        # unnecessary 128px bottleneck before expression/detail inference.
        rgb = cv2.warpAffine(image_bgr, alignment.matrix * (224/128), (224, 224),
                             borderMode=cv2.BORDER_REPLICATE)[:, :, ::-1]
        blob = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], np.float32) / 255
        n = self.topology.vertex_count
        coarse = self.coarse.run({"rgb": blob})
        neutral = finite(coarse["identity_vertices_m"], (1, n, 3), "coarse identity")[0]
        rotation = finite(coarse["rotation"], (1, 3, 3), "rotation")[0]
        if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4)
                or not np.isclose(np.linalg.det(rotation), 1, atol=1e-4)):
            raise ValueError("Coarse pose rotation must be SO(3)")
        translation = finite(coarse["translation_m"], (1, 3), "translation")[0]
        scale = float(finite(coarse["scale"], (1, 1), "scale")[0, 0])
        if scale <= 0:
            raise ValueError("Camera scale must be positive")
        k = finite(coarse["intrinsics"], (1, 3, 3), "224px intrinsics")[0]
        if not np.allclose(k[2], (0, 0, 1)):
            raise ValueError("Expected pinhole camera intrinsics")
        expr = self.expression.run({"rgb": blob, "identity_vertices_m": neutral[None]})
        expression = finite(expr["coefficients"], (1, 52 if self.expression_basis == "arkit52" else 50), "expression")[0]
        if self.expression_basis == "arkit52" and np.any((expression < 0) | (expression > 1)):
            raise ValueError("ARKit coefficients must be in [0,1]")
        # Export applies actual jaw/eyelid articulation, basis conversion and
        # cross-topology registration. A 50-to-52 reshape is never used.
        expressed = finite(expr["expressed_vertices_m"], (1, n, 3), "articulated expression")[0]
        rgb_hd = cv2.warpAffine(image_bgr, alignment.matrix * (self.detail_size/128),
                               (self.detail_size, self.detail_size), borderMode=cv2.BORDER_REPLICATE)[:, :, ::-1]
        hd_blob = np.ascontiguousarray(rgb_hd.transpose(2, 0, 1)[None], np.float32) / 255
        detail = self.detail.run({"rgb": blob, "rgb_hd": hd_blob, "vertices_m": expressed[None]})
        mid = finite(detail["mid_offsets_m"], (1, n, 3), "HRN mid geometry")[0]
        disp = np.asarray(detail["displacement_uv_m"], np.float32)
        if (disp.ndim != 4 or disp.shape[:2] != (1, 1) or not 64 <= min(disp.shape[2:])
                or max(disp.shape[2:]) > 2048 or not np.isfinite(disp).all()):
            raise ValueError("HRN displacement must be finite [1,1,H,W], 64..2048")
        vertices = self.surface.displace(self.topology.select(expressed + mid), disp[0, 0])
        camera_vertices = vertices @ rotation.T * scale + translation
        project(camera_vertices, k)
        return FittedFace(vertices, camera_vertices, k, alignment,
                          self.surface.interpolate(self.topology.select(neutral)), expression,
                          self.surface.interpolate(self.topology.select(mid)), disp[0, 0], rotation, translation, scale,
                          expression_vertices=self.surface.interpolate(self.topology.select(expressed)))

    def capability(self):
        return {"available": True, "topology": self.topology.family,
                "native_vertex_count": self.topology.vertex_count,
                "active_face_vertices": len(self.topology.vertex_ids),
                "warp_surface_vertices": self.surface.vertex_count, "detail_subdivisions": self.surface.levels,
                "expression_basis": self.expression_basis, "geometry_tiers": 3,
                "detail_crop_size": self.detail_size,
                "geometry_provider": getattr(self, "geometry_provider", "CPUExecutionProvider"), "detail": "hrn_refined_surface_uv_displacement",
                "output": "dense_correspondence_neural_synthesis", "raw_mesh_compositing": False}

    def release(self):
        for graph in (self.coarse, self.expression, self.detail):
            graph.release()
