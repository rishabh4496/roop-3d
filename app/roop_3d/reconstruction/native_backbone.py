"""Native 3DDFA-V3 / SMIRK / HRN encoders with explicit geometry decoding.

Unlike the original custom-graph interface, these ONNX files are exports of
upstream neural networks only. Cropping, FLAME articulation, registered surface
transfer and HRN UV input construction run here and can be tested independently.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re

import cv2
import numpy as np

from roop_3d.core.errors import DomainError
from roop_3d.models.owned_files import owned_path
from roop_3d.reconstruction.dense_correspondence import GPUSurfaceRasterizer, project
from roop_3d.reconstruction.displaced_surface import DisplacementSurface, detail_levels
from roop_3d.reconstruction.morphable_backbone import (
    FaceTopology, FittedFace, NamedGraph, finite,
)
from roop_3d.reconstruction.native_face_geometry import (
    CV_AXES, FlameExpressionRig, NativeCrop, SurfaceTransfer, UVSurfaceMap,
    bfm_camera, deep3d_crop, sample_hrn_mid,
)

NATIVE_SCHEMA = "roop3d.native_dense_face.v1"
NATIVE_ROLES = ("topology", "bfm", "bfm_topology", "flame_rig", "registration", "coarse", "expression", "hrn_mid", "hrn_high")
GRAPH_CONTRACTS = {
    "coarse": ({"rgb": (1, 3, 224, 224)}, {"coefficients": (1, 257)}),
    "expression": ({"rgb": (1, 3, 224, 224)}, {"expression": (1, 50), "jaw": (1, 3), "eyelids": (1, 2)}),
    "hrn_mid": ({"features": (1, 6, 64, 64)}, {"deformation": (1, 3, 64, 64)}),
    "hrn_high": ({"features": (1, 9, 256, 256)}, {"displacement": (1, 1, 256, 256)}),
}


def validate_graph_contract(role, session):
    for nodes, expected in zip((session.get_inputs(), session.get_outputs()), GRAPH_CONTRACTS[role]):
        actual = {node.name: node for node in nodes}
        if actual.keys() != expected.keys():
            raise ValueError(f"Native {role} graph names do not match its encoder contract")
        for name, shape in expected.items():
            if actual[name].type != "tensor(float)" or tuple(actual[name].shape) != shape:
                raise ValueError(f"Native {role}/{name} must be float32 {shape}")


def verify_native_assets(bundle, spec):
    directory = owned_path(Path(bundle)).parent
    if (spec.get("schema") != NATIVE_SCHEMA or spec.get("topology") not in ("bfm35709", "flame2023open")
            or spec.get("expression_basis") != "smirk_flame50"
            or spec.get("bfm_unit_m") != .1 or spec.get("flame_model") != "flame2023_Open"
            or spec.get("smirk_crop") != "projected_bfm68_bbox_1.4"
            or not spec.get("mask_provenance", "").strip()
            or not spec.get("expression_mapping_evidence", "").strip()
            or not spec.get("registration_evidence", "").strip()):
        raise ValueError("Unsupported native model, expression conversion, crop or mask provenance")
    paths = {}
    for role in NATIVE_ROLES:
        record = spec["assets"][role]
        path = owned_path(directory / record["file"])
        if not path.is_relative_to(directory) or not re.fullmatch("[0-9a-f]{64}", record["sha256"]):
            raise ValueError(f"Invalid native asset path/digest: {role}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1048576), b""):
                digest.update(chunk)
        if digest.hexdigest() != record["sha256"]:
            raise ValueError(f"Native asset digest mismatch: {role}")
        paths[role] = path
    for key in ("hrn_alpha", "hrn_beta"):
        value = spec[key]
        if not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"Invalid checkpoint scale: {key}")
    return paths


def _arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


class NativeMorphableBackbone:
    def __init__(self, topology, bfm_topology, bfm, rig, registration, graphs, rasterizer, spec,
                 *, detail_subdivisions=0):
        self.topology, self.bfm_topology = topology, bfm_topology
        self.mean = finite(bfm["mean_shape"], (35709, 3), "BFM mean")
        self.id_basis = finite(bfm["identity_basis"], (35709, 3, 80), "BFM identity basis")
        self.exp_basis = finite(bfm["expression_basis"], (35709, 3, 64), "BFM coarse expression basis")
        self.uvs = finite(bfm["hrn_uvs"], (35709, 2), "HRN UVs")
        if np.any((self.uvs < 0) | (self.uvs > 1)):
            raise ValueError("HRN UVs outside [0,1]")
        self.standard = finite(bfm["standard5"], (5, 3), "Deep3D landmarks")
        self.ldm68 = np.asarray(bfm["landmarks68"])
        if self.ldm68.shape != (68,) or self.ldm68.dtype.kind not in "iu" or np.any((self.ldm68 < 0) | (self.ldm68 >= 35709)):
            raise ValueError("Invalid native BFM landmark indices")
        self.rig = rig
        self.to_flame = SurfaceTransfer(registration["bfm_to_flame_indices"], registration["bfm_to_flame_weights"], 35709, 5023)
        self.to_bfm = SurfaceTransfer(registration["flame_to_bfm_indices"], registration["flame_to_bfm_weights"], 5023, 35709)
        self.reg_rotation = finite(registration["flame_to_bfm_rotation"], (3, 3), "registration rotation")
        self.reg_translation = finite(registration["flame_to_bfm_translation_m"], (3,), "registration translation")
        self.flame_hrn_uvs = finite(registration["flame_hrn_uvs"], (5023, 2), "registered FLAME HRN chart")
        if np.any((self.flame_hrn_uvs < 0) | (self.flame_hrn_uvs > 1)):
            raise ValueError("Registered HRN chart outside [0,1]")
        self.reg_scale = float(np.asarray(registration["flame_to_bfm_scale"]).item())
        detail_uvs = self.uvs if topology.family == "bfm35709" else self.flame_hrn_uvs
        self.surface = DisplacementSurface(topology.faces, detail_uvs[topology.vertex_ids], levels=detail_subdivisions)
        if (not np.isfinite(self.reg_scale) or self.reg_scale <= 0
                or not np.allclose(self.reg_rotation.T @ self.reg_rotation, np.eye(3), atol=1e-4)
                or not np.isclose(np.linalg.det(self.reg_rotation), 1, atol=1e-4)):
            raise ValueError("Surface registration must be a proper similarity")
        self.graphs, self.rasterizer, self.spec = graphs, rasterizer, spec
        # Map the PRUNED BFM topology back to native IDs for HRN surface baking.
        self.bfm_faces = bfm_topology.vertex_ids[bfm_topology.faces]
        self.uv_map = UVSurfaceMap(self.uvs, self.bfm_faces)

    @classmethod
    def load(cls, catalog, config, bundle, spec=None):
        spec = spec or json.loads(Path(bundle).read_text(encoding="utf-8"))
        graphs, rasterizer = {}, None
        try:
            paths = verify_native_assets(bundle, spec)
            topology = FaceTopology.load(spec["topology"], paths["topology"])
            bfm_topology = FaceTopology.load("bfm35709", paths["bfm_topology"])
            bfm, registration = _arrays(paths["bfm"]), _arrays(paths["registration"])
            rig = FlameExpressionRig(_arrays(paths["flame_rig"]))
            from roop_3d.engine.volumetric.swapper_engine import load_local_session
            cpu = replace(config, provider="CPUExecutionProvider", precision_policy="fp32")
            for role in ("coarse", "expression", "hrn_mid", "hrn_high"):
                graphs[role] = NamedGraph(load_local_session(catalog, f"native_face_{role}", paths[role], cpu))
                validate_graph_contract(role, graphs[role].session)
            rasterizer = GPUSurfaceRasterizer()
            return cls(topology, bfm_topology, bfm, rig, registration, graphs, rasterizer, spec,
                       detail_subdivisions=detail_levels(spec.get("detail_subdivisions", 1), config.hardware_profile))
        except Exception as error:
            for graph in graphs.values():
                graph.release()
            if rasterizer is not None:
                rasterizer.release()
            raise DomainError("DENSE_FACE_BUNDLE_INVALID", f"Invalid native face bundle: {error}", 503) from error

    @staticmethod
    def _blob(bgr):
        return np.ascontiguousarray(bgr[:, :, ::-1].transpose(2, 0, 1)[None], np.float32) / 255

    def _smirk_crop(self, image, coarse_camera, k, crop):
        points = project(coarse_camera[self.ldm68], k)
        points = points @ crop.inverse[:, :2].T + crop.inverse[:, 2]
        low, high = points.min(0), points.max(0)
        side = int(np.mean(high-low) * 1.4)
        if side < 8:
            raise ValueError("Degenerate SMIRK face crop")
        centre = (low+high)/2
        scale = 223 / side
        matrix = np.array([[scale, 0, (side/2-centre[0])*scale], [0, scale, (side/2-centre[1])*scale]])
        return cv2.warpAffine(image, matrix, (224, 224), flags=cv2.INTER_LINEAR)

    def _flame_identity(self, bfm_identity):
        # Transfer IDENTITY DELTAS only, so the FLAME neutral/joints remain in
        # the actual Open model's canonical frame before articulation.
        delta = self.to_flame((bfm_identity-self.mean) * .1)
        return self.rig.template + delta @ self.reg_rotation / self.reg_scale

    def _to_bfm_frame(self, flame):
        return flame @ self.reg_rotation.T * self.reg_scale + self.reg_translation

    def _hrn_texture(self, bfm_native, camera_rotation, translation, photograph, k, observed):
        native_cv = bfm_native @ CV_AXES * .1
        camera = native_cv @ camera_rotation.T + translation
        triangles = camera[self.bfm_faces]
        normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
        faces = self.bfm_faces[np.einsum("ij,ij->i", normals, triangles.mean(1)) < 0]
        size = photograph.shape[1]
        kh = np.diag((size/224, size/224, 1)) @ k
        depth = self.rasterizer.rasterize(camera, faces, kh, camera, (size, size)).depth
        uv_camera = self.uv_map.interpolate(camera)
        q = uv_camera @ kh.T
        xy = (q[..., :2] / np.maximum(q[..., 2:3], 1e-8)).astype(np.float32)
        sampled_depth = cv2.remap(depth, xy, None, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        valid = (self.uv_map.valid & (uv_camera[..., 2] > .001) & np.isfinite(sampled_depth)
                 & (np.abs(uv_camera[..., 2]-sampled_depth) < .002)
                 & (xy[..., 0] >= 0) & (xy[..., 0] < size-1)
                 & (xy[..., 1] >= 0) & (xy[..., 1] < size-1))
        valid &= cv2.remap(observed, xy, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT) >= .999
        texture = cv2.remap(photograph, xy, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        texture = texture[:, :, ::-1].astype(np.float32) / 127.5 - 1
        return texture, valid.astype(np.float32)

    def fit(self, image_bgr, landmarks5):
        crop = deep3d_crop(image_bgr, landmarks5, self.standard)
        coefficients = finite(self.graphs["coarse"].run({"rgb": self._blob(crop.image)})["coefficients"],
                              (1, 257), "native coarse coefficients")[0]
        neutral = self.mean + np.einsum("vci,i->vc", self.id_basis, coefficients[:80])
        coarse = neutral + np.einsum("vce,e->vc", self.exp_basis, coefficients[80:144])
        rotation, translation, k = bfm_camera(coefficients)
        coarse_camera = (coarse @ CV_AXES * .1) @ rotation.T + translation
        smirk = self.graphs["expression"].run({"rgb": self._blob(self._smirk_crop(image_bgr, coarse_camera, k, crop))})
        expression = finite(smirk["expression"], (1, 50), "SMIRK expression")[0]
        jaw = finite(smirk["jaw"], (1, 3), "SMIRK jaw")[0]
        eyelids = finite(smirk["eyelids"], (1, 2), "SMIRK eyelids")[0]
        flame_identity = self._flame_identity(neutral)
        flame_expressed = self.rig.express(flame_identity, expression, jaw, eyelids)
        delta_bfm_m = (flame_expressed-flame_identity) @ self.reg_rotation.T * self.reg_scale
        expressed = neutral + self.to_bfm(delta_bfm_m) / .1
        # HRN networks consume native position/photograph maps, not RGB crops.
        position = self.uv_map.interpolate(expressed)
        # Sample high-resolution originals directly in the SAME native crop.
        hd = cv2.warpAffine(image_bgr, crop.matrix * (512/224), (512, 512), flags=cv2.INTER_LINEAR)
        observed = cv2.warpAffine(np.ones(image_bgr.shape[:2], np.float32), crop.matrix * (512/224),
                                 (512, 512), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        texture, valid = self._hrn_texture(expressed, rotation, translation, hd, k, observed)
        def chw(value):
            return np.ascontiguousarray(value.transpose(2, 0, 1)[None], np.float32)
        def resize(value, side):
            # PyTorch interpolate(..., align_corners=False) matches linear resize.
            return cv2.resize(value, (side, side), interpolation=cv2.INTER_LINEAR)
        mid_input = np.concatenate((resize(position, 64), resize(texture, 64)), -1)
        mid_raw = finite(self.graphs["hrn_mid"].run({"features": chw(mid_input)})["deformation"], (1, 3, 64, 64), "HRN mid output")
        mid_map = mid_raw[0].transpose(1, 2, 0) * (.1 * self.spec["hrn_alpha"])
        mid_map *= resize(valid, 64)[..., None]
        mid_native = sample_hrn_mid(mid_map, self.uvs)
        # HRN reprojects appearance AFTER the mid-frequency deformation.
        texture_high, _ = self._hrn_texture(expressed+mid_native, rotation, translation, hd, k, observed)
        high_input = np.concatenate((position, texture_high, resize(mid_map, 256)), -1)
        high_raw = finite(self.graphs["hrn_high"].run({"features": chw(high_input)})["displacement"], (1, 1, 256, 256), "HRN high output")
        displacement_m = high_raw[0, 0] * (.1 * self.spec["hrn_beta"] * .1) * valid
        if self.topology.family == "bfm35709":
            identity_native = neutral * .1
            expression_native = expressed * .1
            vertices_native = (expressed+mid_native) * .1
            mid_m = mid_native * .1
        else:
            identity_native = self._to_bfm_frame(flame_identity)
            expression_native = self._to_bfm_frame(flame_expressed)
            mid_m = self.to_flame(mid_native * .1)
            vertices_native = self._to_bfm_frame(flame_expressed) + mid_m
        # The shared refinement plan samples the registered HRN UV chart for
        # either output family. It never changes the native model or its masks.
        vertices = self.surface.displace(self.topology.select(vertices_native) @ CV_AXES, displacement_m)
        camera_vertices = vertices @ rotation.T + translation
        project(camera_vertices, k)
        # The downstream image warp expects an original -> 128px canonical
        # transform. It need not be ArcFace: the complete affine is carried.
        matrix = crop.matrix * (128/224)
        alignment = NativeCrop(cv2.resize(crop.image, (128, 128)), matrix, cv2.invertAffineTransform(matrix))
        return FittedFace(vertices, camera_vertices, k, alignment,
                          self.surface.interpolate(self.topology.select(identity_native) @ CV_AXES), expression,
                          self.surface.interpolate(self.topology.select(mid_m) @ CV_AXES), displacement_m,
                          rotation, translation, 1., jaw=jaw, eyelids=eyelids,
                          expression_vertices=self.surface.interpolate(
                              self.topology.select(expression_native) @ CV_AXES))

    def capability(self):
        return {"available": True, "topology": self.topology.family,
                "native_vertex_count": self.topology.vertex_count, "active_face_vertices": len(self.topology.vertex_ids),
                "warp_surface_vertices": self.surface.vertex_count, "detail_subdivisions": self.surface.levels,
                "expression_basis": "smirk_flame50", "jaw_articulation": True, "separate_eyelids": True,
                "geometry_tiers": 3, "geometry_provider": "CPUExecutionProvider", "detail_crop_size": 512,
                "detail": "native_hrn_uv_mid_high", "implementation": "native_encoder_decoders",
                "expression_crop": self.spec["smirk_crop"], "geometry_rasterizer": "hardware_opengl",
                "camera_gauge": "bfm_0.1m_per_unit_distance_1m_minus_predicted_tz", "predicted_metric_scale": False,
                "output": "dense_correspondence_neural_synthesis", "raw_mesh_compositing": False}

    def release(self):
        for graph in self.graphs.values():
            graph.release()
        self.rasterizer.release()
