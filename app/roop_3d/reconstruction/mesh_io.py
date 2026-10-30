"""Validated atomic 3D mesh and UV export to roop_3d/outputs/meshes/.

Supports Wavefront OBJ with MTL/PNG textures and binary glTF 2.0 (GLB)
for inspection in Stage 14's Three.js viewport.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import struct
import uuid
import cv2
import numpy as np

from roop_3d.audit.isolation import safe_path
from roop_3d.audit.project_paths import application_root
from roop_3d.core.errors import DomainError
from roop_3d.reconstruction.geometry_types import Mesh3D, UVOrigin, uv_to_pixel


def _get_meshes_dir(root: Path | None = None) -> Path:
    """Resolve and ensure the authorized meshes output directory."""
    app_root = root or application_root()
    # Path relative to application_root is roop_3d/outputs/meshes
    target = safe_path(app_root, "roop_3d/outputs/meshes", write=True)
    target.mkdir(parents=True, exist_ok=True)
    return target


def export_mesh_obj(
    mesh: Mesh3D,
    filename: str,
    texture_image: np.ndarray | None = None,
    output_dir: Path | None = None,
) -> tuple[Path, Path | None, Path | None]:
    """Atomically export Mesh3D to Wavefront OBJ format.

    Args:
        mesh: Validated Mesh3D instance.
        filename: Destination filename (e.g. 'head_mesh.obj').
        texture_image: Optional (H, W, 3) BGR UV texture image.
        output_dir: Optional destination directory (defaults to roop_3d/outputs/meshes).

    Returns:
        Tuple of (obj_path, mtl_path_or_none, texture_path_or_none).
    """
    mesh.validate()

    out_dir = output_dir or _get_meshes_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    base_name = Path(filename).stem
    obj_filename = f"{base_name}.obj"
    mtl_filename = f"{base_name}.mtl"
    tex_filename = f"{base_name}.png"

    final_obj = out_dir / obj_filename
    final_mtl = out_dir / mtl_filename if texture_image is not None else None
    final_tex = out_dir / tex_filename if texture_image is not None else None

    nonce = uuid.uuid4().hex[:8]
    tmp_obj = out_dir / f".{obj_filename}.{nonce}.tmp"
    tmp_mtl = out_dir / f".{mtl_filename}.{nonce}.tmp" if texture_image is not None else None
    tmp_tex = out_dir / f".{tex_filename}.{nonce}.tmp" if texture_image is not None else None

    try:
        # Write texture image if provided
        if texture_image is not None and tmp_tex is not None and final_tex is not None:
            success, enc_bytes = cv2.imencode(".png", texture_image)
            if not success:
                raise DomainError("TEXTURE_ENCODE_FAILED", "Failed to encode UV texture to PNG", 500)
            tmp_tex.write_bytes(enc_bytes.tobytes())

            # Write material file
            mtl_content = (
                f"# Roop 3D Stage 08 Material\n"
                f"newmtl material_0\n"
                f"Ka 1.000 1.000 1.000\n"
                f"Kd 1.000 1.000 1.000\n"
                f"Ks 0.000 0.000 0.000\n"
                f"d 1.0\n"
                f"illum 2\n"
                f"map_Kd {tex_filename}\n"
            )
            tmp_mtl.write_text(mtl_content, encoding="utf-8")

        # Write OBJ content
        lines = [
            f"# Roop 3D Stage 08 Validated Mesh",
            f"# Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}",
        ]
        if texture_image is not None:
            lines.append(f"mtllib {mtl_filename}")
            lines.append("usemtl material_0")

        # Vertices
        for v in mesh.vertices:
            lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")

        # UV texture coordinates
        has_uvs = mesh.uvs is not None and len(mesh.uvs) == len(mesh.vertices)
        if has_uvs and mesh.uvs is not None:
            for uv in mesh.uvs:
                lines.append(f"vt {uv[0]:.6f} {uv[1]:.6f}")

        # Normals
        for n in mesh.normals:
            lines.append(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}")

        # Faces (1-indexed)
        for f in mesh.faces:
            i0, i1, i2 = f[0] + 1, f[1] + 1, f[2] + 1
            if has_uvs:
                lines.append(f"f {i0}/{i0}/{i0} {i1}/{i1}/{i1} {i2}/{i2}/{i2}")
            else:
                lines.append(f"f {i0}//{i0} {i1}//{i1} {i2}//{i2}")

        tmp_obj.write_text("\n".join(lines) + "\n", encoding="utf-8")

        # Atomic commit
        if tmp_tex is not None and final_tex is not None:
            os.replace(tmp_tex, final_tex)
        if tmp_mtl is not None and final_mtl is not None:
            os.replace(tmp_mtl, final_mtl)
        os.replace(tmp_obj, final_obj)

        return final_obj, final_mtl, final_tex

    except Exception:
        for tmp in (tmp_obj, tmp_mtl, tmp_tex):
            if tmp is not None and tmp.exists():
                tmp.unlink(missing_ok=True)
        raise


def load_mesh_obj(path: Path) -> Mesh3D:
    """Read and validate Wavefront OBJ file."""
    if not path.exists():
        raise DomainError("OBJ_NOT_FOUND", f"OBJ file not found: {path}", 404)

    vertices = []
    normals = []
    uvs = []
    faces = []

    for line in path.read_text("utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        tag = parts[0]

        if tag == "v" and len(parts) >= 4:
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif tag == "vn" and len(parts) >= 4:
            normals.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif tag == "vt" and len(parts) >= 3:
            uvs.append([float(parts[1]), float(parts[2])])
        elif tag == "f" and len(parts) >= 4:
            face_verts = []
            for token in parts[1:4]:
                v_idx = int(token.split("/")[0]) - 1
                face_verts.append(v_idx)
            faces.append(face_verts)

    v_arr = np.array(vertices, dtype=np.float32)
    f_arr = np.array(faces, dtype=np.int32)
    uv_arr = np.array(uvs, dtype=np.float32) if uvs else None
    norm_arr = np.array(normals, dtype=np.float32) if normals else None

    return Mesh3D(vertices=v_arr, faces=f_arr, uvs=uv_arr, normals=norm_arr)


def export_mesh_glb(
    mesh: Mesh3D,
    filename: str,
    texture_image: np.ndarray | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Atomically export Mesh3D to binary glTF 2.0 (.glb) format.

    Produces a self-contained, fully compliant glTF 2.0 binary file for Stage 14 ThreeViewport.
    """
    mesh.validate()

    out_dir = output_dir or _get_meshes_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    base_name = Path(filename).stem
    final_glb = out_dir / f"{base_name}.glb"
    nonce = uuid.uuid4().hex[:8]
    tmp_glb = out_dir / f".{base_name}.{nonce}.tmp"

    try:
        n_verts = len(mesh.vertices)
        n_faces = len(mesh.faces)

        # Binary buffers:
        # 1. POSITION: (N, 3) float32
        pos_bytes = mesh.vertices.astype(np.float32).tobytes()
        min_pos = [float(x) for x in np.min(mesh.vertices, axis=0)]
        max_pos = [float(x) for x in np.max(mesh.vertices, axis=0)]

        # 2. NORMAL: (N, 3) float32
        norm_bytes = mesh.normals.astype(np.float32).tobytes()

        # 3. TEXCOORD_0: (N, 2) float32 (in glTF, UV v=0 is top, so v_gltf = 1.0 - v_obj)
        if mesh.uvs is not None and len(mesh.uvs) == n_verts:
            uvs_gltf = mesh.uvs.copy()
            # glTF specification requires UV v origin at top
            uvs_gltf[:, 1] = 1.0 - uvs_gltf[:, 1]
            uv_bytes = uvs_gltf.astype(np.float32).tobytes()
        else:
            uv_bytes = np.zeros((n_verts, 2), dtype=np.float32).tobytes()

        # 4. INDICES: uint16 or uint32
        indices = mesh.faces.flatten()
        if n_verts < 65535:
            idx_bytes = indices.astype(np.uint16).tobytes()
            idx_component_type = 5123  # UNSIGNED_SHORT
        else:
            idx_bytes = indices.astype(np.uint32).tobytes()
            idx_component_type = 5125  # UNSIGNED_INT

        # 5. Optional texture PNG image
        image_bytes = b""
        if texture_image is not None:
            success, enc_png = cv2.imencode(".png", texture_image)
            if success:
                image_bytes = enc_png.tobytes()

        # Assemble buffer views and accessors
        bin_chunks = []
        buffer_views = []
        accessors = []

        def add_buffer_data(data: bytes, target: int | None = None) -> int:
            nonlocal bin_chunks
            # Align offset to 4 bytes
            current_len = sum(len(c) for c in bin_chunks)
            pad_len = (4 - (current_len % 4)) % 4
            if pad_len > 0:
                bin_chunks.append(b"\x00" * pad_len)
                current_len += pad_len

            bv_idx = len(buffer_views)
            bv = {"buffer": 0, "byteOffset": current_len, "byteLength": len(data)}
            if target is not None:
                bv["target"] = target
            buffer_views.append(bv)
            bin_chunks.append(data)
            return bv_idx

        # POSITION (ARRAY_BUFFER = 34962)
        bv_pos = add_buffer_data(pos_bytes, target=34962)
        accessors.append({
            "bufferView": bv_pos,
            "byteOffset": 0,
            "componentType": 5126,  # FLOAT
            "count": n_verts,
            "type": "VEC3",
            "min": min_pos,
            "max": max_pos,
        })

        # NORMAL (ARRAY_BUFFER = 34962)
        bv_norm = add_buffer_data(norm_bytes, target=34962)
        accessors.append({
            "bufferView": bv_norm,
            "byteOffset": 0,
            "componentType": 5126,
            "count": n_verts,
            "type": "VEC3",
        })

        # TEXCOORD_0 (ARRAY_BUFFER = 34962)
        bv_uv = add_buffer_data(uv_bytes, target=34962)
        accessors.append({
            "bufferView": bv_uv,
            "byteOffset": 0,
            "componentType": 5126,
            "count": n_verts,
            "type": "VEC2",
        })

        # INDICES (ELEMENT_ARRAY_BUFFER = 34963)
        bv_idx = add_buffer_data(idx_bytes, target=34963)
        accessors.append({
            "bufferView": bv_idx,
            "byteOffset": 0,
            "componentType": idx_component_type,
            "count": len(indices),
            "type": "SCALAR",
        })

        # Texture and Material setup
        materials = []
        textures = []
        images = []
        samplers = []

        if image_bytes:
            bv_img = add_buffer_data(image_bytes)
            images.append({"bufferView": bv_img, "mimeType": "image/png"})
            samplers.append({"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497})
            textures.append({"sampler": 0, "source": 0})
            materials.append({
                "name": "head_material",
                "pbrMetallicRoughness": {
                    "baseColorTexture": {"index": 0},
                    "roughnessFactor": 0.65,
                    "metallicFactor": 0.0,
                },
            })

        # Concatenate BIN payload and pad to 4-byte alignment
        total_bin_data = b"".join(bin_chunks)
        bin_pad = (4 - (len(total_bin_data) % 4)) % 4
        if bin_pad > 0:
            total_bin_data += b"\x00" * bin_pad

        buffers = [{"byteLength": len(total_bin_data)}]

        primitive = {
            "attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2},
            "indices": 3,
        }
        if materials:
            primitive["material"] = 0

        gltf_dict = {
            "asset": {"version": "2.0", "generator": "Roop 3D Stage 08 Clean-Room Exporter"},
            "scene": 0,
            "scenes": [{"nodes": [0]}],
            "nodes": [{"mesh": 0, "name": "HeadMesh"}],
            "meshes": [{"name": "FaceGeometry", "primitives": [primitive]}],
            "accessors": accessors,
            "bufferViews": buffer_views,
            "buffers": buffers,
        }
        if materials:
            gltf_dict["materials"] = materials
            gltf_dict["textures"] = textures
            gltf_dict["images"] = images
            gltf_dict["samplers"] = samplers

        # Serialize JSON and pad with spaces (0x20) to 4-byte boundary
        json_bytes = json.dumps(gltf_dict, separators=(",", ":")).encode("utf-8")
        json_pad = (4 - (len(json_bytes) % 4)) % 4
        if json_pad > 0:
            json_bytes += b" " * json_pad

        # GLB Header (12 bytes)
        # magic: 0x46546C67 ("glTF")
        # version: 2
        # length: total file length
        total_glb_len = 12 + 8 + len(json_bytes) + 8 + len(total_bin_data)
        header = struct.pack("<4sII", b"glTF", 2, total_glb_len)

        # Chunk 0: JSON (0x4E4F534A)
        json_chunk_header = struct.pack("<II", len(json_bytes), 0x4E4F534A)

        # Chunk 1: BIN (0x004E4942)
        bin_chunk_header = struct.pack("<II", len(total_bin_data), 0x004E4942)

        glb_data = header + json_chunk_header + json_bytes + bin_chunk_header + total_bin_data

        tmp_glb.write_bytes(glb_data)
        os.replace(tmp_glb, final_glb)

        return final_glb

    except Exception:
        if tmp_glb.exists():
            tmp_glb.unlink(missing_ok=True)
        raise
