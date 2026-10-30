"""Acceptance tests for Stage 08 validated atomic mesh export (.obj, .mtl, .glb)."""
from __future__ import annotations

import json
from pathlib import Path
import struct
import numpy as np
import pytest

from roop_3d.reconstruction.geometry_types import Mesh3D
from roop_3d.reconstruction.mesh_io import (
    export_mesh_glb,
    export_mesh_obj,
    load_mesh_obj,
)
from roop_3d.reconstruction.tps_fallback import build_canonical_head_mesh


def test_export_mesh_obj_round_trip(tmp_path: Path):
    mesh = build_canonical_head_mesh(landmark_count=68)
    # Synthetic texture image (64x64 BGR)
    tex_img = np.full((64, 64, 3), 150, dtype=np.uint8)

    obj_path, mtl_path, tex_path = export_mesh_obj(
        mesh=mesh,
        filename="test_head.obj",
        texture_image=tex_img,
        output_dir=tmp_path,
    )

    assert obj_path.exists()
    assert mtl_path is not None and mtl_path.exists()
    assert tex_path is not None and tex_path.exists()

    # Ensure no temporary files left
    tmp_files = list(tmp_path.glob(".*.tmp"))
    assert len(tmp_files) == 0

    # Load OBJ and verify geometry round-trip
    loaded = load_mesh_obj(obj_path)
    assert len(loaded.vertices) == len(mesh.vertices)
    assert len(loaded.faces) == len(mesh.faces)
    np.testing.assert_allclose(loaded.vertices, mesh.vertices, atol=1e-5)
    np.testing.assert_allclose(loaded.faces, mesh.faces)


def test_export_mesh_glb_structure(tmp_path: Path):
    mesh = build_canonical_head_mesh(landmark_count=68)
    tex_img = np.full((64, 64, 3), 180, dtype=np.uint8)

    glb_path = export_mesh_glb(
        mesh=mesh,
        filename="test_head.glb",
        texture_image=tex_img,
        output_dir=tmp_path,
    )

    assert glb_path.exists()
    assert len(list(tmp_path.glob(".*.tmp"))) == 0

    # Read and inspect binary glTF header
    data = glb_path.read_bytes()
    assert len(data) >= 20

    magic, version, total_length = struct.unpack_from("<4sII", data, 0)
    assert magic == b"glTF"
    assert version == 2
    assert total_length == len(data)

    # Chunk 0: JSON
    json_len, json_type = struct.unpack_from("<II", data, 12)
    assert json_type == 0x4E4F534A  # "JSON"
    json_bytes = data[20:20 + json_len]
    gltf_json = json.loads(json_bytes.decode("utf-8"))

    assert gltf_json["asset"]["version"] == "2.0"
    assert len(gltf_json["meshes"]) == 1
    assert "primitives" in gltf_json["meshes"][0]

    # Chunk 1: BIN
    bin_offset = 20 + json_len
    bin_len, bin_type = struct.unpack_from("<II", data, bin_offset)
    assert bin_type == 0x004E4942  # "BIN"
    assert bin_offset + 8 + bin_len == total_length
