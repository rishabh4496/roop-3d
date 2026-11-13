"""Stage 12: binary glTF serialization (container, buffers, textures, shape keys, head track), the `/mesh.glb` route, and the
GPL-isolated Blender bridge.

Container checks read the bytes back with an independent strict parser (``parse_glb`` plus ``struct``), never with the writer's
own bookkeeping alone.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
import socket
import struct
import sys
import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from scipy.spatial.transform import Rotation

import roop_3d
from roop_3d.core.errors import DomainError
from roop_3d.io import glb_export
from roop_3d.io.blender_bridge import EXPORTER_SCRIPT, BlenderBridgeError, BlenderMCPBridge
from roop_3d.io.glb_export import (CHUNK_BIN, CHUNK_JSON, GLB_MAGIC, GLBBinarySerializer, HeadTrack, MorphTarget, compute_tangents,
                                   encode_png, normal_map_from_height, parse_glb, read_accessor)
from roop_3d.reconstruction.geometry_types import Mesh3D
from roop_3d.reconstruction.identity_asset import to_gltf_frame


# --------------------------------------------------------------------------------------------------------- helpers
def grid_mesh(nu: int = 24, nv: int = 16, radius: float = 0.1) -> Mesh3D:
    """A UV sphere in the project's y-down mesh frame with one UV per vertex (OBJ convention, v up)."""
    u, v = np.meshgrid(np.linspace(0, 1, nu), np.linspace(0, 1, nv))
    lon, lat = u * 2 * np.pi, (v - 0.5) * np.pi * 0.98
    vertices = np.stack([np.cos(lat) * np.sin(lon), -np.sin(lat), np.cos(lat) * np.cos(lon)], -1).reshape(-1, 3) * radius
    index = np.arange(nu * nv).reshape(nv, nu)
    a, b, c, d = index[:-1, :-1], index[:-1, 1:], index[1:, :-1], index[1:, 1:]
    faces = np.concatenate([np.stack([a, c, b], -1).reshape(-1, 3), np.stack([b, c, d], -1).reshape(-1, 3)])
    return Mesh3D(vertices, faces, uvs=np.stack([u, v], -1).reshape(-1, 2))


def texture(size: int = 37, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (size, size, 3), dtype=np.uint8)


def chunks(data: bytes) -> list[tuple[int, int, bytes]]:
    """(type, declared length, payload) straight from the bytes with ``struct``."""
    out, pos = [], 12
    while pos < len(data):
        size, kind = struct.unpack_from("<II", data, pos)
        out.append((kind, size, data[pos + 8:pos + 8 + size]))
        pos += 8 + size
    assert pos == len(data), "chunks must tile the file exactly"
    return out


def image_bytes(doc: dict, binary: bytes, index: int) -> bytes:
    view = doc["bufferViews"][doc["images"][index]["bufferView"]]
    return binary[view["byteOffset"]:view["byteOffset"] + view["byteLength"]]


def primitive(doc: dict) -> dict:
    return doc["meshes"][0]["primitives"][0]


# ------------------------------------------------------------------------------------- 12-byte header, chunk alignment
def test_header_is_the_12_byte_gltf_container_header():
    data = GLBBinarySerializer().serialize(grid_mesh(), frame="mesh", albedo=texture())
    assert data[:4] == b"glTF" == GLB_MAGIC
    magic, version, length = struct.unpack_from("<4sII", data, 0)
    assert (magic, version) == (b"glTF", 2)
    assert length == len(data), "header length must equal the whole stream"
    kinds = [k for k, _, _ in chunks(data)]
    assert kinds == [CHUNK_JSON, CHUNK_BIN] and CHUNK_JSON == 0x4E4F534A and CHUNK_BIN == 0x004E4942


@pytest.mark.parametrize("name_length", range(0, 9))
def test_json_chunk_is_4_byte_aligned_and_space_padded(name_length):
    data = GLBBinarySerializer().serialize(grid_mesh(6, 5), frame="mesh", name="n" * (name_length + 1), extras={"x": "y" * name_length})
    json_kind, json_len, payload = chunks(data)[0]
    assert json_kind == CHUNK_JSON and json_len % 4 == 0 and json_len == len(payload)
    text = payload.decode("utf-8")
    assert json.loads(text)["asset"]["version"] == "2.0"
    stripped = text.rstrip(" ")
    assert set(text[len(stripped):]) <= {" "} and len(text) - len(stripped) < 4, "padding is spaces, at most 3"


@pytest.mark.parametrize("size", [1, 2, 3, 5, 31, 64])
def test_bin_chunk_is_4_byte_aligned_for_odd_sized_textures(size):
    data = GLBBinarySerializer().serialize(grid_mesh(5, 4), frame="mesh", albedo=texture(size), normal_map=texture(size, 1))
    doc, binary = parse_glb(data)
    bin_kind, bin_len, payload = chunks(data)[1]
    assert bin_kind == CHUNK_BIN and bin_len % 4 == 0 == len(payload) % 4
    assert doc["buffers"][0]["byteLength"] == bin_len == len(binary)
    for view in doc["bufferViews"]:
        assert view["byteOffset"] % 4 == 0 and view["byteOffset"] + view["byteLength"] <= bin_len
    end = max(v["byteOffset"] + v["byteLength"] for v in doc["bufferViews"])
    assert bin_len - end < 4 and set(payload[end:]) <= {0}, "trailing BIN padding is at most 3 zero bytes"
    for earlier, later in zip(doc["bufferViews"], doc["bufferViews"][1:]):
        gap = payload[earlier["byteOffset"] + earlier["byteLength"]:later["byteOffset"]]
        assert len(gap) < 4 and set(gap) <= {0}, "alignment gaps between views are zero bytes"


def test_parse_glb_rejects_a_corrupt_container():
    good = GLBBinarySerializer().serialize(grid_mesh(5, 4), frame="mesh")
    for bad, code in [(b"nope" + good[4:], "GLB_BAD_MAGIC"), (good[:-4], "GLB_BAD_LENGTH"),
                      (good[:4] + struct.pack("<I", 1) + good[8:], "GLB_BAD_VERSION"),
                      (good[:12] + struct.pack("<I", 7) + good[16:], "GLB_BAD_CHUNK")]:
        with pytest.raises(DomainError) as err:
            parse_glb(bad)
        assert err.value.code == code


# --------------------------------------------------------------------------------------------- BIN buffer integrity
def test_vertex_buffers_round_trip_bit_for_bit():
    mesh = grid_mesh()
    doc, binary = parse_glb(GLBBinarySerializer().serialize(mesh, frame="gltf", albedo=texture()))
    prim = primitive(doc)
    positions = read_accessor(doc, binary, prim["attributes"]["POSITION"])
    assert np.array_equal(positions, mesh.vertices.astype("<f4"))
    assert np.array_equal(read_accessor(doc, binary, prim["attributes"]["NORMAL"]), mesh.normals.astype("<f4"))
    uv = read_accessor(doc, binary, prim["attributes"]["TEXCOORD_0"])
    assert np.array_equal(uv[:, 0], mesh.uvs[:, 0]) and np.allclose(uv[:, 1], 1.0 - mesh.uvs[:, 1], atol=0), "glTF v origin is the top"
    indices = read_accessor(doc, binary, prim["indices"])
    assert indices.dtype == np.uint16 and np.array_equal(indices.reshape(-1, 3), mesh.faces)
    pos_acc = doc["accessors"][prim["attributes"]["POSITION"]]
    assert pos_acc["min"] == mesh.vertices.min(axis=0).astype(float).tolist() and pos_acc["max"] == mesh.vertices.max(axis=0).astype(float).tolist()
    assert pos_acc["count"] == len(mesh.vertices) and pos_acc["type"] == "VEC3" and pos_acc["componentType"] == 5126


def test_large_meshes_use_32_bit_indices():
    mesh = grid_mesh(300, 230)                                   # 69,000 vertices
    doc, binary = parse_glb(GLBBinarySerializer().serialize(mesh, frame="gltf"))
    acc = doc["accessors"][primitive(doc)["indices"]]
    assert acc["componentType"] == 5125 and acc["count"] == mesh.faces.size
    assert np.array_equal(read_accessor(doc, binary, primitive(doc)["indices"]).reshape(-1, 3), mesh.faces)


def test_mesh_frame_matches_the_identity_asset_converter():
    mesh = grid_mesh()
    doc, binary = parse_glb(GLBBinarySerializer().serialize(mesh, frame="mesh"))
    expected = to_gltf_frame(mesh)
    prim = primitive(doc)
    assert np.array_equal(read_accessor(doc, binary, prim["attributes"]["POSITION"]), expected.vertices)
    assert np.array_equal(read_accessor(doc, binary, prim["attributes"]["NORMAL"]), expected.normals)
    assert np.array_equal(read_accessor(doc, binary, prim["indices"]).reshape(-1, 3), expected.faces)


def test_embedded_pngs_decode_to_the_original_pixels():
    albedo, normal = texture(37, 3), texture(20, 4)
    doc, binary = parse_glb(GLBBinarySerializer().serialize(grid_mesh(), frame="mesh", albedo=albedo, normal_map=normal))
    assert [i["mimeType"] for i in doc["images"]] == ["image/png", "image/png"]
    decoded = [cv2.imdecode(np.frombuffer(image_bytes(doc, binary, n), np.uint8), cv2.IMREAD_UNCHANGED) for n in (0, 1)]
    assert np.array_equal(decoded[0], albedo) and np.array_equal(decoded[1], normal), "BGR in, BGR out"
    material = doc["materials"][0]
    assert doc["textures"][material["pbrMetallicRoughness"]["baseColorTexture"]["index"]]["source"] == 0
    assert doc["textures"][material["normalTexture"]["index"]]["source"] == 1
    assert "TANGENT" in primitive(doc)["attributes"], "a normal map ships with tangents"


def test_png_bytes_pass_through_untouched_and_alpha_and_grey_encode():
    png = encode_png(texture(9))
    doc, binary = parse_glb(GLBBinarySerializer().serialize(grid_mesh(5, 4), frame="mesh", albedo=png))
    assert image_bytes(doc, binary, 0) == png
    for arr in (np.full((4, 6), 77, np.uint8), np.dstack([texture(5), np.full((5, 5), 200, np.uint8)])):
        back = cv2.imdecode(np.frombuffer(encode_png(arr), np.uint8), cv2.IMREAD_UNCHANGED)
        assert np.array_equal(back, arr)


def test_unchanged_texture_is_not_re_encoded(monkeypatch):
    calls = []
    real = glb_export.encode_png
    monkeypatch.setattr(glb_export, "encode_png", lambda *a, **k: calls.append(1) or real(*a, **k))
    serializer, mesh, atlas = GLBBinarySerializer(), grid_mesh(5, 4), texture(64)
    first = serializer.serialize(mesh, frame="mesh", albedo=atlas)
    second = serializer.serialize(mesh, frame="mesh", albedo=atlas.copy())
    assert first == second and len(calls) == 1
    serializer.serialize(mesh, frame="mesh", albedo=atlas + 1)
    assert len(calls) == 2


def test_serialization_meets_the_15ms_budget_with_a_cached_atlas():
    mesh, atlas, serializer = grid_mesh(90, 50), texture(1024), GLBBinarySerializer()      # 4,500 vertices like the MakeHuman head
    morphs = [MorphTarget(f"k{i}", np.random.default_rng(i).normal(0, 1e-3, mesh.vertices.shape)) for i in range(6)]
    serializer.serialize(mesh, frame="mesh", albedo=atlas, morph_targets=morphs)                # warm the PNG cache
    times = []
    for _ in range(15):
        start = time.perf_counter()
        serializer.serialize(mesh, frame="mesh", albedo=atlas, morph_targets=morphs)
        times.append((time.perf_counter() - start) * 1e3)
    assert float(np.median(times)) < 15.0, f"median {np.median(times):.1f} ms"


# ------------------------------------------------------------------------------------------------ shape keys
def test_morph_targets_weights_and_names():
    mesh = grid_mesh()
    rng = np.random.default_rng(2)
    deltas = [rng.normal(0, 1e-3, mesh.vertices.shape).astype(np.float32) for _ in range(3)]
    morphs = [MorphTarget(name, d) for name, d in zip(("jaw_width", "chin", "brow"), deltas)]
    doc, binary = parse_glb(GLBBinarySerializer().serialize(mesh, frame="mesh", morph_targets=morphs, morph_weights=[0.5, 0.0, -1.0]))
    assert doc["meshes"][0]["weights"] == [0.5, 0.0, -1.0] and doc["meshes"][0]["extras"]["targetNames"] == ["jaw_width", "chin", "brow"]
    flip = np.array([1, -1, 1], np.float32)
    for target, delta in zip(primitive(doc)["targets"], deltas):
        assert np.array_equal(read_accessor(doc, binary, target["POSITION"]), delta * flip), "deltas follow the vertex frame"
    with pytest.raises(DomainError) as err:
        GLBBinarySerializer().serialize(mesh, frame="mesh", morph_targets=morphs, morph_weights=[0.0])
    assert err.value.code == "GLB_MORPH_INVALID"
    with pytest.raises(DomainError):
        GLBBinarySerializer().serialize(mesh, morph_targets=[MorphTarget("bad", deltas[0][:-1])])


# ------------------------------------------------------------------------------------------------- head track
def fake_frames(count: int = 5, seed: int = 0):
    rotations = Rotation.random(count, random_state=seed)
    frames = [SimpleNamespace(timestamp=10.0 + i / 25, frame_index=100 + i, confidence=0.9,
                              camera=SimpleNamespace(rotation=rotations[i].as_matrix(), tx=500.0 + 3 * i, ty=300.0 - 2 * i, scale=2000.0))
              for i in range(count)]
    frames.insert(2, SimpleNamespace(timestamp=10.05, frame_index=999, confidence=0.0, camera=None))      # tracker lost the face
    return frames, rotations


def test_head_track_becomes_a_glTF_animation_with_the_right_maths():
    frames, rotations = fake_frames()
    track = HeadTrack.from_tracked_frames(frames)
    assert len(track.times) == 5 and track.times[0] == 0.0, "frames without a camera are skipped; times start at 0"
    doc, binary = parse_glb(GLBBinarySerializer().serialize(grid_mesh(), frame="mesh", track=track))
    animation = doc["animations"][0]
    assert {c["target"]["path"] for c in animation["channels"]} == {"translation", "rotation"}
    assert all(c["target"]["node"] == 0 and doc["nodes"][0].get("mesh") == 0 for c in animation["channels"])
    time_acc = doc["accessors"][animation["samplers"][0]["input"]]
    times = read_accessor(doc, binary, animation["samplers"][0]["input"])
    assert time_acc["min"] == [float(times.min())] and time_acc["max"] == [float(times.max())] and np.all(np.diff(times) > 0)
    quats = read_accessor(doc, binary, animation["samplers"][1]["output"])
    assert np.allclose(np.linalg.norm(quats, axis=1), 1.0, atol=1e-6)
    translations = read_accessor(doc, binary, animation["samplers"][0]["output"])
    assert np.allclose(translations[0], 0) and np.allclose(translations[1], [3 / 2000, 2 / 2000, 0], atol=1e-7), "y is flipped to y-up; z is unknown"
    # The rotated glTF-frame point must equal the y-flip of the tracker's mesh-frame rotation of the mesh-frame point.
    flip = np.diag([1.0, -1.0, 1.0])
    point_mesh = np.array([0.03, -0.05, 0.08])
    for q, rotation in zip(quats, rotations):
        assert np.allclose(Rotation.from_quat(q).as_matrix() @ (flip @ point_mesh), flip @ (rotation.as_matrix() @ point_mesh), atol=1e-6)
    assert animation["extras"]["scale_px_per_m"] == [2000.0] * 5 and "depth not metric" in animation["extras"]["dof"]


def test_track_quaternions_stay_in_one_hemisphere_and_times_must_increase():
    near_pi = [SimpleNamespace(timestamp=i / 30, frame_index=i, confidence=1.0,
                               camera=SimpleNamespace(rotation=Rotation.from_euler("y", 178 + 2 * i, degrees=True).as_matrix(),
                                                      tx=0.0, ty=0.0, scale=1000.0)) for i in range(4)]
    quats = HeadTrack.from_tracked_frames(near_pi).rotations
    assert all(float(quats[i] @ quats[i - 1]) > 0 for i in range(1, len(quats)))
    bad = HeadTrack(np.array([0.0, 0.0]), np.zeros((2, 3)), np.tile([0, 0, 0, 1.0], (2, 1)))
    with pytest.raises(DomainError) as err:
        GLBBinarySerializer().serialize(grid_mesh(5, 4), track=bad)
    assert err.value.code == "GLB_TRACK_INVALID"
    with pytest.raises(DomainError) as err:
        HeadTrack.from_tracked_frames([SimpleNamespace(camera=None)])
    assert err.value.code == "GLB_TRACK_EMPTY"


def test_track_can_drive_morph_weights():
    mesh = grid_mesh()
    morphs = [MorphTarget("a", np.zeros_like(mesh.vertices)), MorphTarget("b", np.zeros_like(mesh.vertices))]
    track = HeadTrack(np.array([0.0, 0.5, 1.0]), np.zeros((3, 3)), np.tile([0, 0, 0, 1.0], (3, 1)), np.array([[0, 1], [0.5, 0.5], [1, 0.0]]))
    doc, binary = parse_glb(GLBBinarySerializer().serialize(mesh, morph_targets=morphs, track=track))
    sampler = doc["animations"][0]["samplers"][2]
    assert doc["accessors"][sampler["output"]]["count"] == 6, "glTF weights output holds N x K scalars"
    assert read_accessor(doc, binary, sampler["output"]).tolist() == [0, 1, 0.5, 0.5, 1, 0]


# ---------------------------------------------------------------------------------------- normal map / tangents
def plane(flip_u: bool = False) -> Mesh3D:
    xs = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], np.float64) * 0.1
    uv = xs / 0.1
    if flip_u:
        uv[:, 0] = 1 - uv[:, 0]
    return Mesh3D(np.c_[xs, np.zeros(4)], np.array([[0, 1, 3], [0, 3, 2]]), uvs=uv, normals=np.tile([0, 0, 1.0], (4, 1)))


@pytest.mark.parametrize("flip_u", [False, True])
@pytest.mark.parametrize("axis", ["u", "v"])
def test_normal_map_and_tangents_reproduce_the_displaced_surface_normal(flip_u, axis):
    size, slope = 32, 0.4
    mesh = plane(flip_u)
    ramp = np.linspace(0, 1, size)
    # height rises toward +x (u, or -u when the UVs are mirrored) or toward +y (v, which is image-up)
    coord = np.tile(ramp, (size, 1)) if axis == "u" else np.tile(ramp[::-1, None], (1, size))
    per_texel = 0.1 / (size - 1)
    height = slope * per_texel * (size - 1) * coord / 1.0                     # dh/d(texel) == slope * per_texel  ->  dh/dx = slope
    normal_bgr = normal_map_from_height(height, per_texel)
    rgb = normal_bgr[size // 2, size // 2][::-1].astype(np.float64) / 255 * 2 - 1
    tangents = compute_tangents(mesh.vertices, mesh.normals, mesh.uvs, mesh.faces)
    n, t, w = mesh.normals[0].astype(np.float64), tangents[0, :3].astype(np.float64), float(tangents[0, 3])
    world = t * rgb[0] + np.cross(n, t) * w * rgb[1] + n * rgb[2]
    # the heightfield is z = slope * x (axis u, unmirrored) / slope * (1-x) (mirrored) / slope * y (axis v)
    if axis == "u":
        expected = np.array([-slope * (-1 if flip_u else 1), 0, 1])
        # mirrored UVs: texture u runs toward -x, so the height gradient w.r.t. x flips sign
    else:
        expected = np.array([0, -slope, 1])
    expected /= np.linalg.norm(expected)
    assert np.allclose(world / np.linalg.norm(world), expected, atol=0.03), (world, expected, w)
    assert abs(float(np.dot(t, n))) < 1e-6 and abs(np.linalg.norm(t) - 1) < 1e-6 and w in (-1.0, 1.0)


def test_flat_height_is_the_neutral_normal_map():
    flat = normal_map_from_height(np.zeros((4, 4)), 1e-3)
    assert (flat == np.array([255, 128, 128], np.uint8)).all(), "BGR of (0.5, 0.5, 1.0)"


# ------------------------------------------------------------------------------------------------- input errors
@pytest.mark.parametrize("kwargs, code", [
    ({"albedo": np.zeros((4, 4, 3), np.float32)}, "GLB_TEXTURE_DTYPE"),
    ({"albedo": b"definitely not a png"}, "GLB_TEXTURE_NOT_PNG"),
    ({"vertex_attributes": {"CONFIDENCE": np.zeros(20)}}, "GLB_ATTRIBUTE_INVALID"),
    ({"vertex_attributes": {"_lower": np.zeros(20)}}, "GLB_ATTRIBUTE_INVALID"),
    ({"frame": "unity"}, "GLB_FRAME_INVALID"),
])
def test_bad_inputs_are_422_domain_errors(kwargs, code):
    with pytest.raises(DomainError) as err:
        GLBBinarySerializer().serialize(grid_mesh(5, 4), **kwargs)
    assert err.value.code == code and err.value.status == 422


def test_textures_need_uvs_and_one_uv_per_vertex():
    bare = Mesh3D(np.eye(3, dtype=np.float32), np.array([[0, 1, 2]]))
    with pytest.raises(DomainError) as err:
        GLBBinarySerializer().serialize(bare, albedo=texture())
    assert err.value.code == "GLB_TEXTURE_NEEDS_UVS"
    corner_uvs = Mesh3D(np.eye(3, dtype=np.float32), np.array([[0, 1, 2]]), uvs=np.zeros((5, 2)))
    with pytest.raises(DomainError) as err:
        GLBBinarySerializer().serialize(corner_uvs)
    assert err.value.code == "GLB_UV_COUNT_MISMATCH"
    assert parse_glb(GLBBinarySerializer().serialize(bare))[0]["materials"][0]["pbrMetallicRoughness"]["baseColorFactor"]


# ======================================================================================== Blender bridge / GPL
class FakeBlender:
    """A localhost server that speaks the assumed BlenderMCP framing: a message is complete once it parses as JSON."""

    def __init__(self, handler=None, split_reply: bool = False, newline: bool = False, jsonrpc: bool = False):
        self.requests: list[dict] = []
        self.handler, self.split, self.newline, self.jsonrpc = handler, split_reply, newline, jsonrpc
        self.server = socket.create_server(("127.0.0.1", 0))
        self.port = self.server.getsockname()[1]
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self):
        while True:
            try:
                conn, _ = self.server.accept()
            except OSError:
                return
            with conn:
                buffer = b""
                while True:
                    chunk = conn.recv(65536)
                    if not chunk:
                        break
                    buffer += chunk
                    try:
                        request = json.loads(buffer)
                        break
                    except json.JSONDecodeError:
                        continue
                else:
                    continue
                self.requests.append(request)
                method = request.get("method") or request.get("type")
                if self.handler:
                    reply = self.handler(request)
                elif method == "execute_code":
                    path = request["params"]["code"].split("_path = ", 1)[1].splitlines()[0]
                    reply = {"status": "success", "result": {"executed": True, "result": json.dumps({"imported": ["Head", "Head.001"], "path": json.loads(path)}) + "\n"}}
                else:
                    reply = {"status": "success", "result": {"name": "Scene", "object_count": 3}}
                if self.jsonrpc:
                    reply = {"jsonrpc": "2.0", "id": request["id"], "result": reply.get("result")}
                body = json.dumps(reply).encode() + (b"\n" if self.newline else b"")
                if self.split:
                    conn.sendall(body[: len(body) // 2])
                    time.sleep(0.05)
                    conn.sendall(body[len(body) // 2:])
                else:
                    conn.sendall(body)

    def close(self):
        self.server.close()


@pytest.fixture()
def blender():
    servers = []

    def make(**kwargs):
        servers.append(FakeBlender(**kwargs))
        return servers[-1]
    yield make
    for server in servers:
        server.close()


def test_bridge_round_trip_and_split_replies(blender):
    for kwargs in ({}, {"split_reply": True}, {"newline": True}):
        fake = blender(**kwargs)
        bridge = BlenderMCPBridge(port=fake.port, timeout=3)
        assert bridge.scene_info() == {"name": "Scene", "object_count": 3} and bridge.is_available()
        assert fake.requests[0] == {"type": "get_scene_info", "params": {}}


def test_bridge_jsonrpc_dialect_frames_with_id_and_newline(blender):
    fake = blender(jsonrpc=True, newline=True)
    bridge = BlenderMCPBridge(port=fake.port, dialect="jsonrpc", timeout=3)
    assert bridge.call("scene.info", {"a": 1}) == {"name": "Scene", "object_count": 3}
    assert fake.requests[0]["jsonrpc"] == "2.0" and fake.requests[0]["method"] == "scene.info" and fake.requests[0]["id"] == 1


def test_bridge_errors_are_typed(blender):
    fake = blender(handler=lambda request: {"status": "error", "message": "boom"})
    with pytest.raises(BlenderBridgeError) as err:
        BlenderMCPBridge(port=fake.port, timeout=3).scene_info()
    assert err.value.code == "BLENDER_BRIDGE_REMOTE_ERROR" and "boom" in str(err.value)
    assert BlenderMCPBridge(port=fake.port, timeout=3).is_available() is False
    closed = socket.create_server(("127.0.0.1", 0))
    port = closed.getsockname()[1]
    closed.close()
    with pytest.raises(BlenderBridgeError) as err:
        BlenderMCPBridge(port=port, timeout=1).scene_info()
    assert err.value.code == "BLENDER_BRIDGE_UNREACHABLE"
    silent = socket.create_server(("127.0.0.1", 0))
    with pytest.raises(BlenderBridgeError) as err:
        BlenderMCPBridge(port=silent.getsockname()[1], timeout=0.3).scene_info()
    silent.close()
    assert err.value.code == "BLENDER_BRIDGE_TIMEOUT"


@pytest.mark.parametrize("host", ["192.168.1.20", "example.com", "0.0.0.0", "10.0.0.1"])
def test_bridge_refuses_non_loopback_hosts(host):
    with pytest.raises(BlenderBridgeError) as err:
        BlenderMCPBridge(host=host)
    assert err.value.code == "BLENDER_BRIDGE_NOT_LOCAL"
    for ok in ("127.0.0.1", "localhost", "::1"):
        BlenderMCPBridge(host=ok)


def test_send_glb_stages_the_file_and_asks_blender_to_import_it(blender, tmp_path):
    fake = blender()
    glb = GLBBinarySerializer().serialize(grid_mesh(), frame="mesh")
    report = BlenderMCPBridge(port=fake.port, timeout=3).send_glb(glb, tmp_path / "stage", "../evil name?")
    staged = Path(report["path"])
    assert staged.parent == (tmp_path / "stage").resolve() and staged.read_bytes() == glb and staged.name == "evil_name.glb"
    assert list((tmp_path / "stage").glob("*.part")) == []
    code = fake.requests[0]["params"]["code"]
    assert "bpy.ops.import_scene.gltf" in code and json.dumps(str(staged)) in code
    assert report["imported_objects"] == ["Head", "Head.001"] and report["bytes"] == len(glb)


def test_a_hostile_path_cannot_break_out_of_the_snippet_literal(blender, tmp_path):
    fake = blender()
    nasty = tmp_path / 'a"; import os; os.system("calc"); x="b.glb'
    BlenderMCPBridge(port=fake.port, timeout=3).import_glb(nasty)
    line = fake.requests[0]["params"]["code"].split("_path = ", 1)[1].splitlines()[0]
    assert json.loads(line) == str(nasty), "the literal round-trips; the injected text stays inside the string"
    tree = ast.parse(fake.requests[0]["params"]["code"])
    assert [type(node).__name__ for node in tree.body].count("Expr") == 2 and not any(
        isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "system" for n in ast.walk(tree)), "no os.system call exists in the AST"


def test_exporter_script_is_valid_python_gpl_and_loopback_only():
    compile(EXPORTER_SCRIPT, "roop3d_blender_import.py", "exec")
    assert EXPORTER_SCRIPT.startswith("# SPDX-License-Identifier: GPL-3.0-or-later")
    for fragment in ("bpy.ops.import_scene.gltf", "bpy.ops.export_scene.gltf", "export_morph=True", '"127.0.0.1", "localhost", "::1"'):
        assert fragment in EXPORTER_SCRIPT


def _package_sources() -> list[Path]:
    root = Path(roop_3d.__file__).parent
    skip = {".venv", ".runtime", "__pycache__", "node_modules", ".pytest_cache"}
    return [p for p in root.rglob("*.py") if not skip & set(p.relative_to(root).parts)]


def test_roop_3d_never_imports_bpy():
    """100% GPL isolation: no module in the package imports Blender's API, statically or by name."""
    offenders = []
    for path in _package_sources():
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            elif isinstance(node, ast.Call) and getattr(node.func, "id", getattr(node.func, "attr", "")) in ("__import__", "import_module"):
                names = [a.value for a in node.args[:1] if isinstance(a, ast.Constant) and isinstance(a.value, str)]
            if any(n == "bpy" or n.startswith("bpy.") or n in ("bmesh", "mathutils") for n in names):
                offenders.append(f"{path}:{node.lineno}")
    assert not offenders, offenders
    assert "bpy" not in sys.modules


# ============================================================================================== HTTP routes
from roop_3d.core.settings import Settings                          # noqa: E402
from roop_3d.reconstruction import identity_asset as ia             # noqa: E402
from roop_3d.reconstruction import topology as topo                 # noqa: E402
from roop_3d.reconstruction.makehuman_build import ASSET_NAME, DEFAULT_ASSET_DIR   # noqa: E402
from roop_3d.reconstruction.semantic_shape import SLIDER_KEYS       # noqa: E402
from roop_3d.server import identity_assets                          # noqa: E402
from roop_3d.server.app import create_app                           # noqa: E402

needs_makehuman = pytest.mark.skipif(not (DEFAULT_ASSET_DIR / ASSET_NAME).exists(), reason="build the MakeHuman asset first")
FAKE_ID, TEX = "stand_in", 128
GLB_URL = f"/api/v1/facesets/{FAKE_ID}/identity-asset/mesh.glb"
JSON_URL = f"/api/v1/facesets/{FAKE_ID}/identity-asset/mesh"


@pytest.fixture()
def api(tmp_path, monkeypatch):
    registry = topo.MorphableTopologyRegistry()
    base = registry.load("makehuman_cc0")
    yy, xx = np.mgrid[0:TEX, 0:TEX]
    atlas = np.stack([xx * 255 // TEX, yy * 255 // TEX, np.full_like(xx, 140)], -1).astype(np.uint8)
    state = np.full((TEX, TEX), 2, np.uint8)
    state[:, : TEX // 8] = 0
    mesh = Mesh3D(base.vertices, base.faces, uvs=base.uvs)
    asset = SimpleNamespace(vertices=base.vertices.astype(np.float32), faces=base.faces, texture=atlas, texel_state=state,
                            vertex_confidence=np.linspace(0, 1, len(base.vertices), dtype=np.float32),
                            vertex_state=np.full(len(base.vertices), 2, np.uint8),
                            manifest=SimpleNamespace(build_key="k" * 64, reconstruction_level="template_prior", landmark_layout="5pt",
                                                     shape={}, expression={}, canonical_head_pose={}, statistics={}, settings={}),
                            mesh=lambda: mesh)
    monkeypatch.setattr(identity_assets, "load_document", lambda request, faceset_id: SimpleNamespace(faceset_id=faceset_id))
    monkeypatch.setattr("roop_3d.server.blender_routes.load_document", identity_assets.load_document)
    monkeypatch.setattr(identity_assets, "_store", lambda: SimpleNamespace(load=lambda faceset_id, key: asset))
    monkeypatch.setattr(ia, "compute_build_key", lambda document, settings: "k" * 64)
    monkeypatch.setattr(topo, "_DEFAULT", registry)
    identity_assets._CONTROLLERS.clear()
    client = TestClient(create_app(Settings(), repo_root=tmp_path), base_url="http://127.0.0.1:8030")
    client.asset, client.root = asset, tmp_path
    return client


@needs_makehuman
def test_mesh_glb_route_serves_the_same_mesh_as_the_json_route(api):
    with api as client:
        response = client.get(GLB_URL, params={"template": "makehuman"})
        body = client.get(JSON_URL, params={"template": "makehuman"}).json()
    assert response.status_code == 200 and response.headers["content-type"] == "model/gltf-binary"
    assert response.headers["content-disposition"].startswith("inline")
    doc, binary = parse_glb(response.content)
    prim = primitive(doc)
    assert np.allclose(read_accessor(doc, binary, prim["attributes"]["POSITION"]), np.asarray(body["vertices"]), atol=2e-6)
    assert np.array_equal(read_accessor(doc, binary, prim["indices"]).reshape(-1, 3), np.asarray(body["faces"]))
    json_uv = np.asarray(body["uvs"])
    glb_uv = read_accessor(doc, binary, prim["attributes"]["TEXCOORD_0"])
    assert np.allclose(glb_uv[:, 0], json_uv[:, 0], atol=2e-6) and np.allclose(glb_uv[:, 1], 1 - json_uv[:, 1], atol=2e-6)
    json_png = cv2.imdecode(np.frombuffer(__import__("base64").b64decode(body["texture"].split(",", 1)[1]), np.uint8), cv2.IMREAD_UNCHANGED)
    assert np.array_equal(cv2.imdecode(np.frombuffer(image_bytes(doc, binary, 0), np.uint8), cv2.IMREAD_UNCHANGED), json_png)
    assert "_CONFIDENCE" in prim["attributes"] and doc["extras"]["roop3d_identity_asset"]["normal_map"] == "asset carries no UV displacement"
    assert client.get(GLB_URL, params={"download": True, "template": "makehuman"}).headers["content-disposition"].startswith("attachment")


@needs_makehuman
def test_shape_keys_reproduce_the_json_route_when_weighted(api):
    sliders = {"jaw_width": 1.5, "chin_projection": -1.0, "brow_arch": 0.5}
    with api as client:
        glb = client.get(GLB_URL, params={"template": "makehuman", "shape_keys": True, **sliders})
        baked = client.get(JSON_URL, params={"template": "makehuman", **sliders}).json()
        unmorphed = client.get(JSON_URL, params={"template": "makehuman"}).json()
    doc, binary = parse_glb(glb.content)
    prim, weights = primitive(doc), doc["meshes"][0]["weights"]
    assert doc["meshes"][0]["extras"]["targetNames"] == list(SLIDER_KEYS) and len(prim["targets"]) == 6
    assert np.allclose(weights, [sliders.get(k, 0.0) for k in SLIDER_KEYS]), "unclamped request: weights are the slider values"
    base = read_accessor(doc, binary, prim["attributes"]["POSITION"])
    assert np.allclose(base, np.asarray(unmorphed["vertices"]), atol=2e-6), "the base mesh is the un-slid one"
    shaped = base + sum(w * read_accessor(doc, binary, t["POSITION"]) for w, t in zip(weights, prim["targets"]))
    assert np.allclose(shaped, np.asarray(baked["vertices"]), atol=2e-5), "base + sum(weight * target) == baked sliders"
    assert float(np.abs(shaped - base).max()) > 1e-4, "the sliders really moved the mesh"


@needs_makehuman
def test_shape_keys_need_the_makehuman_template(api):
    with api as client:
        response = client.get(GLB_URL, params={"shape_keys": True})
    assert response.status_code == 422 and response.json()["error"]["code"] == "SHAPE_KEYS_REQUIRE_MAKEHUMAN_TEMPLATE"


@needs_makehuman
def test_measured_relief_becomes_a_normal_map_with_tangents(api):
    height = np.zeros((TEX, TEX), np.float32)
    height[:, TEX // 2:] = 2e-4
    api.asset.displacement_uv_m = height
    with api as client:
        response = client.get(GLB_URL, params={"template": "makehuman"})
    doc, binary = parse_glb(response.content)
    assert "TANGENT" in primitive(doc)["attributes"] and len(doc["images"]) == 2
    normal = cv2.imdecode(np.frombuffer(image_bytes(doc, binary, 1), np.uint8), cv2.IMREAD_UNCHANGED)
    assert normal.shape[:2] == (TEX, TEX) and not (normal == [255, 128, 128]).all(), "the step edge tilts the normals"
    assert doc["extras"]["roop3d_identity_asset"]["normal_map"].startswith("baked")
    with api as client:
        off = parse_glb(client.get(GLB_URL, params={"template": "makehuman", "normal_map": False}).content)[0]
    assert len(off["images"]) == 1


@needs_makehuman
def test_glb_route_errors_match_the_json_route(api):
    with api as client:
        assert client.get(GLB_URL, params={"template": "makehuman", "jaw_width": 99}).status_code == 422
        assert client.get(GLB_URL, params={"jaw_width": 1}).status_code == 422
        assert client.get(GLB_URL, params={"topology": "nope", "template": "makehuman"}).status_code == 422


@needs_makehuman
def test_blender_routes(api, blender):
    fake = blender()
    with api as client:
        script = client.get("/api/v1/blender/exporter-script")
        assert script.status_code == 200 and script.text == EXPORTER_SCRIPT
        assert "roop3d_blender_import.py" in script.headers["content-disposition"]
        assert client.get("/api/v1/blender/status", params={"port": fake.port}).json()["available"] is True
        gone = socket.create_server(("127.0.0.1", 0))
        dead = gone.getsockname()[1]
        gone.close()
        status = client.get("/api/v1/blender/status", params={"port": dead}).json()
        assert status["available"] is False and status["host"] == "127.0.0.1"
        sent = client.post(f"/api/v1/facesets/{FAKE_ID}/identity-asset/send-to-blender", params={"template": "makehuman", "port": fake.port, "shape_keys": True})
        assert sent.status_code == 200, sent.text
        body = sent.json()
        staged = Path(body["path"])
        assert body["sent"] and body["imported_objects"] == ["Head", "Head.001"] and staged.is_file()
        assert parse_glb(staged.read_bytes())[0]["meshes"][0]["extras"]["targetNames"] == list(SLIDER_KEYS)
        assert (api.root / "roop_3d/.runtime/exports/blender").resolve() == staged.parent
        offline = client.post(f"/api/v1/facesets/{FAKE_ID}/identity-asset/send-to-blender", params={"template": "makehuman", "port": dead})
        assert offline.status_code == 503 and offline.json()["error"]["code"] == "BLENDER_BRIDGE_UNREACHABLE"


# --------------------------------------------------------------------------- a real built asset (nothing mocked)
def test_mesh_glb_of_a_really_built_identity_asset(tmp_path, monkeypatch):
    from roop_3d.reconstruction.identity_asset import IdentityAssetStore
    from roop_3d.tests import identity_world as W
    document = W.full_set(W.make_subject("glb_real", W.LAYOUT_68, 81), (0, -40, 40))
    runtime = tmp_path / "roop_3d/.runtime/facesets"
    runtime.mkdir(parents=True)
    (runtime / f"{document.faceset_id}.json").write_text(json.dumps({"faceset_id": document.faceset_id, "name": document.name,
                                                                      "item_count": len(document.views), "schema_version": 1}))
    (runtime / f"{document.faceset_id}.fsz").write_bytes(b"placeholder")

    class Reader:
        def __init__(self, *args, **kwargs):
            pass

        def read(self, archive, faceset_id):
            return document

    store_root = tmp_path / "identity_store"
    monkeypatch.setattr(identity_assets, "FaceSetReader", Reader)
    monkeypatch.setattr(identity_assets, "_store", lambda: IdentityAssetStore(root=store_root))
    url = f"/api/v1/facesets/{document.faceset_id}/identity-asset"
    with TestClient(create_app(Settings(), repo_root=tmp_path), base_url="http://127.0.0.1:8030") as client:
        assert client.get(f"{url}/mesh.glb").status_code == 404, "never built implicitly"
        assert client.post(url).status_code == 200
        glb = client.get(f"{url}/mesh.glb")
        body = client.get(f"{url}/mesh").json()
    assert glb.status_code == 200
    doc, binary = parse_glb(glb.content)
    prim = primitive(doc)
    assert np.allclose(read_accessor(doc, binary, prim["attributes"]["POSITION"]), np.asarray(body["vertices"]), atol=2e-6)
    assert np.array_equal(read_accessor(doc, binary, prim["indices"]).reshape(-1, 3), np.asarray(body["faces"]))
    assert len(doc["images"]) >= 1 and doc["extras"]["roop3d_identity_asset"]["reconstruction_level"]
    print("real asset:", len(body["vertices"]), "vertices;", len(glb.content), "bytes; normal map:", doc["extras"]["roop3d_identity_asset"]["normal_map"])
