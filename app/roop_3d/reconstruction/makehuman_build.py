"""Build the open MakeHuman head model (CC0 data) used as a FLAME-free parametric face.

Data source: https://github.com/makehumancommunity/makehuman, ``makehuman/data`` (base mesh ``3dobjs/base.obj`` and
``targets/*.target`` morphs), released CC0-1.0 (``LICENSE.ASSETS.md``). Only DATA is used; the AGPL application code is
not copied. Every upstream file is pinned by commit and SHA-256 in ``roop_3d/models/makehuman_sources.json``.

The builder crops the head and neck from the full body mesh, combines the curated signed morph targets into identity
and expression axes, converts to the project's right-handed OpenCV head frame (metres, x right, y down, z away from
a camera that the head faces) and derives the five detector landmarks (eyes, nose tip, mouth corners) from geometry.

CLI: ``python -m roop_3d.reconstruction.makehuman_build [--pin]`` from ``app/``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path
from typing import Iterable

import numpy as np

from roop_3d.core.errors import DomainError

COMMIT = "a8bc2d54ff0ac92e78ff71431b1023eda42bf482"
RAW_BASE = f"https://raw.githubusercontent.com/makehumancommunity/makehuman/{COMMIT}/makehuman/data/"
ALLOWED_HOST_PREFIX = "https://raw.githubusercontent.com/makehumancommunity/makehuman/"
LICENSE_ID = "CC0-1.0"
SCHEMA = "roop3d.makehuman_head.v1"
BUILD_VERSION = 1

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
LOCK_PATH = MODELS_DIR / "makehuman_sources.json"
DEFAULT_ASSET_DIR = Path(__file__).resolve().parents[1] / ".runtime" / "models" / "makehuman"
ASSET_NAME = f"makehuman_head_v{BUILD_VERSION}.npz"

BASE_OBJ = "3dobjs/base.obj"
MH_UNITS_TO_M = 0.1          # MakeHuman units are decimetres
HEAD_Y_CUT_MH = 5.9          # keep body vertices at/above this MakeHuman height (includes the neck)

# (axis name, [(target path relative to data/targets, weight), ...]); order = fit priority (landmark-relevant first).
_PAIR = lambda stem, inc, dec, w=1.0: [(f"{stem}-{inc}.target", w), (f"{stem}-{dec}.target", -w)]  # noqa: E731


def _lr(stem: str, inc: str, dec: str | None = None) -> list[tuple[str, float]]:
    """Bundle the left and right target so one axis moves both sides symmetrically."""
    out: list[tuple[str, float]] = []
    for side in ("l", "r"):
        out.append((f"{stem.replace('@', side)}-{inc}.target", 1.0))
        if dec:
            out.append((f"{stem.replace('@', side)}-{dec}.target", -1.0))
    return out


SHAPE_AXES: list[tuple[str, list[tuple[str, float]]]] = [
    ("head_scale_horiz", _PAIR("head/head-scale-horiz", "incr", "decr")),
    ("head_scale_vert", _PAIR("head/head-scale-vert", "incr", "decr")),
    ("head_scale_depth", _PAIR("head/head-scale-depth", "incr", "decr")),
    ("eyes_trans_out", _lr("eyes/@-eye-trans", "out", "in")),
    ("eyes_trans_up", _lr("eyes/@-eye-trans", "up", "down")),
    ("mouth_scale_horiz", _PAIR("mouth/mouth-scale-horiz", "incr", "decr")),
    ("mouth_trans_up", _PAIR("mouth/mouth-trans", "up", "down")),
    ("nose_trans_up", _PAIR("nose/nose-trans", "up", "down")),
    ("nose_scale_depth", _PAIR("nose/nose-scale-depth", "incr", "decr")),
    ("nose_scale_horiz", _PAIR("nose/nose-scale-horiz", "incr", "decr")),
    ("nose_scale_vert", _PAIR("nose/nose-scale-vert", "incr", "decr")),
    ("chin_width", _PAIR("chin/chin-width", "incr", "decr")),
    ("chin_height", _PAIR("chin/chin-height", "incr", "decr")),
    ("chin_prominent", _PAIR("chin/chin-prominent", "incr", "decr")),
    ("chin_prognathism", _PAIR("chin/chin-prognathism", "incr", "decr")),
    ("chin_bones", _PAIR("chin/chin-bones", "incr", "decr")),
    ("forehead_scale_vert", _PAIR("forehead/forehead-scale-vert", "incr", "decr")),
    ("forehead_trans_forward", _PAIR("forehead/forehead-trans", "forward", "backward")),
    ("forehead_temple", _PAIR("forehead/forehead-temple", "incr", "decr")),
    ("eyes_scale", _lr("eyes/@-eye-scale", "incr", "decr")),
    ("eyes_height", _lr("eyes/@-eye-height1", "incr", "decr")),
    ("eyes_push", _lr("eyes/@-eye-push1", "out", "in")),
    ("mouth_scale_vert", _PAIR("mouth/mouth-scale-vert", "incr", "decr")),
    ("mouth_scale_depth", _PAIR("mouth/mouth-scale-depth", "incr", "decr")),
    ("mouth_trans_forward", _PAIR("mouth/mouth-trans", "forward", "backward")),
    ("mouth_angles", _PAIR("mouth/mouth-angles", "up", "down")),
    ("lowerlip_volume", _PAIR("mouth/mouth-lowerlip-volume", "incr", "decr")),
    ("upperlip_volume", _PAIR("mouth/mouth-upperlip-volume", "incr", "decr")),
    ("nose_width1", _PAIR("nose/nose-width1", "incr", "decr")),
    ("nose_width2", _PAIR("nose/nose-width2", "incr", "decr")),
    ("nose_width3", _PAIR("nose/nose-width3", "incr", "decr")),
    ("nose_flaring", _PAIR("nose/nose-flaring", "incr", "decr")),
    ("nose_point", _PAIR("nose/nose-point", "up", "down")),
    ("nose_volume", _PAIR("nose/nose-volume", "incr", "decr")),
    ("nose_base", _PAIR("nose/nose-base", "up", "down")),
    ("head_fat", _PAIR("head/head-fat", "incr", "decr")),
    ("head_age", _PAIR("head/head-age", "incr", "decr")),
    ("cheek_bones", _lr("cheek/@-cheek-bones", "incr", "decr")),
    ("cheek_volume", _lr("cheek/@-cheek-volume", "incr", "decr")),
    ("ears_scale", _lr("ears/@-ear-scale", "incr", "decr")),
    ("eyebrows_trans_up", _PAIR("eyebrows/eyebrows-trans", "up", "down")),
    ("eyebrows_angle", _PAIR("eyebrows/eyebrows-angle", "up", "down")),
    ("head_oval", [("head/head-oval.target", 1.0)]),
    ("head_round", [("head/head-round.target", 1.0)]),
    ("head_square", [("head/head-square.target", 1.0)]),
    ("head_rectangular", [("head/head-rectangular.target", 1.0)]),
    ("head_triangular", [("head/head-triangular.target", 1.0)]),
    ("head_diamond", [("head/head-diamond.target", 1.0)]),
    ("head_invertedtriangular", [("head/head-invertedtriangular.target", 1.0)]),
]

_U = "expression/units/caucasian/"
EXPR_AXES: list[tuple[str, list[tuple[str, float]]]] = [
    ("mouth_open", [(_U + "mouth-open.target", 1.0)]),
    ("mouth_corner_puller", [(_U + "mouth-corner-puller.target", 1.0)]),
    ("mouth_pursing", [(_U + "mouth-pursing.target", 1.0)]),
    ("mouth_compression", [(_U + "mouth-compression.target", 1.0)]),
    ("mouth_elevation", [(_U + "mouth-elevation.target", 1.0)]),
    ("mouth_protusion", [(_U + "mouth-protusion.target", 1.0)]),
    ("brows_up", [(_U + "eyebrows-left-up.target", 1.0), (_U + "eyebrows-right-up.target", 1.0)]),
    ("brows_inner_up", [(_U + "eyebrows-left-inner-up.target", 1.0), (_U + "eyebrows-right-inner-up.target", 1.0)]),
    ("brows_down", [(_U + "eyebrows-left-down.target", 1.0), (_U + "eyebrows-right-down.target", 1.0)]),
    ("eyes_closed", [(_U + "eye-left-closure.target", 1.0), (_U + "eye-right-closure.target", 1.0)]),
]

# Neutral = MakeHuman default human: gender 0.5, young, average muscle/weight, equal race mix.
_MACRO = "macrodetails/"
_RACES = ("african", "asian", "caucasian")
NEUTRAL_TERMS: list[tuple[str, float]] = (
    [(f"{_MACRO}universal-{g}-young-averagemuscle-averageweight.target", 0.5) for g in ("male", "female")]
    + [(f"{_MACRO}{r}-{g}-young.target", 0.5 / 3.0) for r in _RACES for g in ("male", "female")]
)
GENDER_AXIS: tuple[str, list[tuple[str, float]]] = ("gender", (
    [(f"{_MACRO}universal-male-young-averagemuscle-averageweight.target", 0.5),
     (f"{_MACRO}universal-female-young-averagemuscle-averageweight.target", -0.5)]
    + [(f"{_MACRO}{r}-male-young.target", 0.5 / 3.0) for r in _RACES]
    + [(f"{_MACRO}{r}-female-young.target", -0.5 / 3.0) for r in _RACES]))

LANDMARK_NAMES = ("right_eye", "left_eye", "nose_tip", "mouth_right", "mouth_left")  # detector order: image-left first
_MOUTH_PROBE = "mouth/mouth-scale-horiz-incr.target"


def _all_paths() -> list[str]:
    paths = {BASE_OBJ}
    for _, terms in [*SHAPE_AXES, *EXPR_AXES, GENDER_AXIS]:
        paths.update(("targets/" + p) for p, _ in terms)
    paths.update(("targets/" + p) for p, _ in NEUTRAL_TERMS)
    paths.add("targets/" + _MOUTH_PROBE)
    return sorted(paths)


# ---------------------------------------------------------------------------------------------------- download / pin
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _download(relpath: str) -> bytes:
    url = RAW_BASE + relpath
    if not url.startswith(ALLOWED_HOST_PREFIX):  # defence in depth: only the pinned upstream repository
        raise DomainError("MAKEHUMAN_SOURCE_FORBIDDEN", f"Refusing non-upstream URL {url}", 403)
    with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 - fixed https host asserted above
        return response.read()


def load_lock() -> dict:
    return json.loads(LOCK_PATH.read_text("utf-8"))


def fetch_sources(cache_dir: Path, *, pin: bool = False) -> dict[str, bytes]:
    """Return {relative path: bytes}, downloading missing files and verifying each against the committed lock."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    lock = None if pin else load_lock()
    files: dict[str, bytes] = {}
    pinned: dict[str, dict] = {}
    for rel in _all_paths():
        local = cache_dir / rel
        data = local.read_bytes() if local.exists() else None
        if data is None:
            data = _download(rel)
        digest = sha256_bytes(data)
        if lock is not None:
            expected = lock["files"].get(rel)
            if expected is None or expected["sha256"] != digest:
                raise DomainError("MAKEHUMAN_HASH_MISMATCH", f"{rel}: sha256 {digest} does not match the pinned value", 422)
        else:
            pinned[rel] = {"sha256": digest, "bytes": len(data)}
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(data)
        files[rel] = data
    if pin:
        lock_doc = {
            "schema": "roop3d.makehuman_sources.v1", "license": LICENSE_ID,
            "license_note": "MakeHuman data assets are CC0-1.0 (LICENSE.ASSETS.md). The AGPL application code is not used.",
            "repository": "https://github.com/makehumancommunity/makehuman", "commit": COMMIT, "files": pinned,
        }
        LOCK_PATH.write_text(json.dumps(lock_doc, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return files


# ---------------------------------------------------------------------------------------------------- parsing
def parse_obj(data: bytes) -> dict:
    """Parse MakeHuman's base.obj: vertices, per-group quad faces (with uv indices) and the uv table."""
    verts: list[list[float]] = []
    uvs: list[list[float]] = []
    faces: dict[str, list[list[tuple[int, int]]]] = {}
    group = ""
    for line in data.decode("utf-8").splitlines():
        if line.startswith("v "):
            verts.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("vt "):
            uvs.append([float(x) for x in line.split()[1:3]])
        elif line.startswith("g "):
            group = line.split()[1]
            faces.setdefault(group, [])
        elif line.startswith("f "):
            corners = []
            for tok in line.split()[1:]:
                parts = tok.split("/")
                corners.append((int(parts[0]) - 1, int(parts[1]) - 1 if len(parts) > 1 and parts[1] else -1))
            faces.setdefault(group, []).append(corners)
    return {"vertices": np.asarray(verts, np.float64), "uvs": np.asarray(uvs, np.float64), "faces": faces}


def parse_target(data: bytes, n_vertices: int) -> tuple[np.ndarray, np.ndarray]:
    idx: list[int] = []
    delta: list[list[float]] = []
    for line in data.decode("utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        i = int(parts[0])
        if 0 <= i < n_vertices:
            idx.append(i)
            delta.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.asarray(idx, np.int64), np.asarray(delta, np.float64).reshape(-1, 3)


def _dense(files: dict[str, bytes], terms: Iterable[tuple[str, float]], n_vertices: int, prefix: str = "targets/") -> np.ndarray:
    out = np.zeros((n_vertices, 3), np.float64)
    for rel, weight in terms:
        idx, delta = parse_target(files[prefix + rel], n_vertices)
        np.add.at(out, idx, weight * delta)
    return out


# ---------------------------------------------------------------------------------------------------- build
def _to_head_frame(points_mh: np.ndarray, origin_m: np.ndarray | None = None) -> np.ndarray:
    """MakeHuman (x character-left, y up, z forward; decimetres) -> OpenCV head frame (x right, y down, z away from the
    camera the face points at; metres). diag(1,-1,-1) is a proper rotation, so handedness is preserved."""
    p = np.asarray(points_mh, np.float64) * MH_UNITS_TO_M * np.array([1.0, -1.0, -1.0])
    return p if origin_m is None else p - origin_m


def build_arrays(files: dict[str, bytes]) -> dict[str, np.ndarray]:
    obj = parse_obj(files[BASE_OBJ])
    verts = obj["vertices"]
    n_all = len(verts)
    body_faces = obj["faces"]["body"]
    n_body = 1 + max(c[0] for f in body_faces for c in f)

    neutral_mh = verts[:n_body] + _dense(files, NEUTRAL_TERMS, n_body)
    keep_mask = neutral_mh[:, 1] >= HEAD_Y_CUT_MH
    # Faces are kept only when all four corners survive the crop.
    quads = [f for f in body_faces if all(keep_mask[c[0]] for c in f)]
    used = np.unique(np.array([c[0] for f in quads for c in f]))
    remap = -np.ones(n_body, np.int64)
    remap[used] = np.arange(len(used))
    uv_used = np.unique(np.array([c[1] for f in quads for c in f]))
    uv_remap = -np.ones(len(obj["uvs"]), np.int64)
    uv_remap[uv_used] = np.arange(len(uv_used))

    tri, tri_uv = [], []
    for f in quads:
        v = [remap[c[0]] for c in f]
        t = [uv_remap[c[1]] for c in f]
        if len(f) == 4:
            tri += [[v[0], v[1], v[2]], [v[0], v[2], v[3]]]
            tri_uv += [[t[0], t[1], t[2]], [t[0], t[2], t[3]]]
        elif len(f) == 3:
            tri.append(v)
            tri_uv.append(t)
    faces = np.asarray(tri, np.int32)
    face_uvs = np.asarray(tri_uv, np.int32)

    head_neutral_mh = neutral_mh[used]

    def axis_matrix(axes: list[tuple[str, list[tuple[str, float]]]]) -> np.ndarray:
        cols = []
        for _, terms in axes:
            d = _dense(files, terms, n_body)[used]
            cols.append(d * MH_UNITS_TO_M * np.array([1.0, -1.0, -1.0]))
        return np.stack(cols, axis=2).astype(np.float32)  # (V, 3, K)

    shape_axes = [GENDER_AXIS, *SHAPE_AXES]
    shape_basis = axis_matrix(shape_axes)
    expr_basis = axis_matrix(EXPR_AXES)

    # ---- landmarks from geometry (MakeHuman frame), chosen among the kept head vertices
    h = head_neutral_mh
    eye_centre_mh = np.array([0.308, 7.284, 1.245])  # joint-l-eye group centroid in base.obj
    front = h[:, 2] > 1.3

    def nearest_front(x: float, y: float) -> int:
        d = np.hypot(h[:, 0] - x, h[:, 1] - y)
        d = np.where(front, d, np.inf)
        cand = np.where(d <= d.min() + 0.04)[0]
        return int(cand[np.argmax(h[cand, 2])])  # most forward vertex within 0.4 cm of the closest

    l_eye, r_eye = nearest_front(eye_centre_mh[0], eye_centre_mh[1]), nearest_front(-eye_centre_mh[0], eye_centre_mh[1])
    nose_pool = np.where((np.abs(h[:, 0]) < 0.15) & (h[:, 1] > 6.6) & (h[:, 1] < 7.3))[0]
    nose = int(nose_pool[np.argmax(h[nose_pool, 2])])
    probe_idx, probe_delta = parse_target(files["targets/" + _MOUTH_PROBE], n_body)
    local = remap[probe_idx]
    ok = local >= 0
    local, probe_dx = local[ok], probe_delta[ok, 0]
    left_pool = local[probe_dx > 0] if np.any(probe_dx > 0) else local
    right_pool = local[probe_dx < 0] if np.any(probe_dx < 0) else local
    dx_by_local = dict(zip(local.tolist(), probe_dx.tolist()))
    mouth_l = int(max(left_pool, key=lambda i: (dx_by_local[int(i)] if h[int(i), 1] < 6.9 else -1)))
    mouth_r = int(min(right_pool, key=lambda i: (dx_by_local[int(i)] if h[int(i), 1] < 6.9 else 1)))
    landmark_indices = np.array([r_eye, l_eye, nose, mouth_r, mouth_l], np.int32)  # image-left (character right) first

    origin = _to_head_frame(h[landmark_indices]).mean(axis=0)
    vertices = _to_head_frame(head_neutral_mh, origin).astype(np.float32)
    uvs = obj["uvs"][uv_used].astype(np.float32)
    return {
        "vertices": vertices, "faces": faces, "uvs": uvs, "face_uvs": face_uvs,
        "shape_basis": shape_basis, "expr_basis": expr_basis, "landmark_indices": landmark_indices,
        "shape_names": np.array([n for n, _ in shape_axes]), "expr_names": np.array([n for n, _ in EXPR_AXES]),
        "landmark_names": np.array(LANDMARK_NAMES), "origin_m": origin.astype(np.float64),
    }


def asset_digest(arrays: dict[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    for key in sorted(arrays):
        a = np.ascontiguousarray(arrays[key])
        h.update(key.encode())
        h.update(str(a.dtype).encode() + str(a.shape).encode())
        h.update(a.tobytes() if a.dtype.kind != "U" else "|".join(a.tolist()).encode())
    return h.hexdigest()


def build_asset(out_dir: Path | None = None, cache_dir: Path | None = None) -> Path:
    """Download (hash-verified), build and write the model ``.npz`` + ``.json`` provenance next to it."""
    out_dir = out_dir or DEFAULT_ASSET_DIR
    cache_dir = cache_dir or (out_dir / "sources")
    files = fetch_sources(cache_dir)
    arrays = build_arrays(files)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / ASSET_NAME
    np.savez_compressed(target, **arrays)
    meta = {
        "schema": SCHEMA, "license": LICENSE_ID, "commit": COMMIT, "build_version": BUILD_VERSION,
        "asset_digest": asset_digest(arrays), "vertices": int(len(arrays["vertices"])), "faces": int(len(arrays["faces"])),
        "shape_axes": int(arrays["shape_basis"].shape[2]), "expr_axes": int(arrays["expr_basis"].shape[2]),
        "frame": "OpenCV head frame, metres: x right, y down, z away from the camera the face points at (R=I = frontal)",
    }
    (out_dir / (ASSET_NAME + ".json")).write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8", newline="\n")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pin", action="store_true", help="(maintainers) re-download and rewrite the SHA-256 lock file")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.pin:
        fetch_sources((args.out or DEFAULT_ASSET_DIR) / "sources", pin=True)
        print(f"pinned {len(_all_paths())} files -> {LOCK_PATH}")
        return 0
    print(build_asset(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
