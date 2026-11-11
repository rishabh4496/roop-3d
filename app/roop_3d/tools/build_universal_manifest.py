"""Maintainer tool: (re)generate ``roop_3d/audit/universal_model_manifest.json``.

Needs network. Runtime never does: the generated JSON is committed and read offline.

Identity rule: a mirror is recorded **only** when its bytes are proven identical to the pinned SHA-256, by one of
  * the Hugging Face Git-LFS object id (the SHA-256 of the file content) in the repository tree at a pinned commit,
  * the ``digest`` GitHub publishes for a release asset, or
  * downloading the (small) file and hashing it here.
Nothing is guessed. A model with no proven public source keeps ``mirrors: []`` and says why in ``acquisition``.

Usage (from ``app/``):  python -m roop_3d.tools.build_universal_manifest [--out PATH] [--offline-check]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

APP = Path(__file__).resolve().parents[2]
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from roop_3d.models.registry import ModelRegistry  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "audit" / "universal_model_manifest.json"
S1, S2, S3 = "STAGE_1_SPARSE_PRESCAN", "STAGE_2_ACTIVE_PREPASS", "STAGE_3_MAIN_SWAP"
UA = {"User-Agent": "roop-3d-manifest-builder"}

# Hugging Face repositories whose trees are indexed for exact-byte mirrors, in trust order.
HF_REPOS = (
    "facefusion/models-3.0.0", "facefusion/models-3.1.0", "facefusion/models-3.2.0", "facefusion/models-3.3.0",
    "facefusion/models-3.4.0", "hacksider/deep-live-cam", "CountFloyd/deepfake", "Aitrepreneur/insightface",
    "ezioruan/inswapper_128.onnx", "netrunner-exe/Face-Upscalers-onnx", "opencv/face_detection_yunet",
    "opencv/face_recognition_sface", "datvtn/face_occluder", "AgentRecon/scrfd-2.5g", "DavidHoa/buffalo_l",
    "karanjakhar/insightface_weights", "immich-app/buffalo_l", "immich-app/buffalo_s",
)
SEARCH_TERMS = ("insightface", "buffalo_l", "inswapper", "hyperswap", "gpen", "codeformer", "xseg", "scrfd", "facefusion")
GITHUB_RELEASES = ("facefusion/facefusion-assets",)

# (manifest id, registry file name or None, category, stages, display name, note)
CORE: tuple[tuple[str, str | None, str, tuple[str, ...], str, str], ...] = (
    ("detector_det_10g", "det_10g.onnx", "detection", (S1, S2, S3), "SCRFD det_10g face detector (buffalo_l)", ""),
    ("detector_scrfd_2_5g", None, "detection", (S1, S2), "SCRFD 2.5g face detector", ""),
    ("detector_yunet_2023mar", "face_detection_yunet_2023mar.onnx", "detection", (S1, S2), "YuNet face detector (OpenCV Zoo)", ""),
    ("landmarks_1k3d68", "1k3d68.onnx", "landmarks_3d", (S2, S3), "1k3d68 3D landmarks (buffalo_l)", ""),
    ("landmarks_2d106det", "2d106det.onnx", "landmarks_2d", (S2, S3), "2d106det landmarks (buffalo_l)", ""),
    ("arcface_w600k_r50", "w600k_r50.onnx", "recognition", (S1, S3), "ArcFace w600k_r50 (buffalo_l)", ""),
    ("arcface_w600k_r50_facefusion", "arcface_w600k_r50.onnx", "recognition", (S1, S3), "ArcFace w600k_r50 (FaceFusion export)", ""),
    ("recognizer_sface_2021dec", "face_recognition_sface_2021dec.onnx", "recognition", (S1,), "SFace recognizer (OpenCV Zoo)", ""),
    ("swapper_inswapper_128", "inswapper_128.onnx", "swapper_2d", (S3,), "InSwapper 128", ""),
    ("swapper_hyperswap_1a_256", "hyperswap_1a_256.onnx", "swapper_2d", (S3,), "HyperSwap 1A 256", ""),
    ("swapper_hyperswap_1b_256", "hyperswap_1b_256.onnx", "swapper_2d", (S3,), "HyperSwap 1B 256", ""),
    ("swapper_hyperswap_1c_256", "hyperswap_1c_256.onnx", "swapper_2d", (S3,), "HyperSwap 1C 256", ""),
    ("restorer_gpen_bfr_512", "gpen_bfr_512.onnx", "restoration", (S3,), "GPEN-BFR 512", ""),
    ("restorer_gpen_bfr_512_realistic", "GPEN-BFR-512.onnx", "restoration", (S3,), "GPEN-BFR 512 (realistic export)", ""),
    ("restorer_codeformer_fp16", "codeformer.fp16.onnx", "restoration", (S3,), "CodeFormer FP16 (UltraMax restorer)", ""),
    ("restorer_gfpgan_1_4", "GFPGANv1.4.onnx", "restoration", (S3,), "GFPGAN 1.4", ""),
    ("occluder_xseg", "xseg.onnx", "occlusion", (S2, S3), "XSeg occlusion mask", ""),
    ("occluder_face_occluder", "face_occluder.onnx", "occlusion", (S2, S3), "Face occluder", ""),
    ("parser_bisenet_resnet_34", "bisenet_resnet_34.onnx", "face_parsing", (S3,), "BiSeNet ResNet-34 face parser", ""),
)
# Entries that exist in core/model_downloader.REQUIRED_MODELS but not (by file) in the registry.
EXTRA_IDENTITY: dict[str, dict[str, Any]] = {
    "detector_scrfd_2_5g": {"filename": "scrfd_2.5g.onnx", "sha256": "2c07342347cef21a619c49dd5664fb8c09570ae9eda5bff3e385c11eafc45ada",
                            "size_bytes": 3295067, "license": "UNKNOWN", "commercial_use": "deny",
                            "license_url": "https://github.com/deepinsight/insightface/tree/master/model_zoo"},
}
POLYHAVEN = "https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/1k/"
MAKEHUMAN_COMMIT = "a8bc2d54ff0ac92e78ff71431b1023eda42bf482"


def lock_digest(files: list[dict[str, Any]]) -> str:
    """Identity of a file set: SHA-256 over sorted ``path NUL sha256 LF`` lines."""
    return hashlib.sha256("".join(f"{f['path']}\0{f['sha256']}\n" for f in sorted(files, key=lambda f: f["path"])).encode()).hexdigest()


def asset_entries() -> list[dict[str, Any]]:
    """CC0 / open assets that are not single ONNX weights."""
    from roop_3d.audit.local_models import OPEN_ARTIFACTS
    from roop_3d.audit.procedural_assets import array_digest
    from roop_3d.rendering.physical_camera import ENVIRONMENT_PROBES
    lock = json.loads((APP / "roop_3d/models/makehuman_sources.json").read_text("utf-8"))
    mh_files = [{"path": path, "size_bytes": row["bytes"], "sha256": row["sha256"]} for path, row in sorted(lock["files"].items())]
    hdri_files = []
    for probe in ENVIRONMENT_PROBES:
        name = f"{probe['asset_id']}_1k.hdr"
        result = sha256_url(POLYHAVEN + name)
        if result is None:
            raise SystemExit(f"cannot hash Poly Haven {name}")
        body = urllib.request.urlopen(urllib.request.Request(POLYHAVEN + name, headers=UA), timeout=60).read()
        if hashlib.md5(body).hexdigest() != probe["md5"]:
            raise SystemExit(f"Poly Haven {name} md5 differs from the committed physical_camera pin")
        hdri_files.append({"path": name, "size_bytes": result[1], "sha256": result[0]})
    bundle_sidecar = json.loads((APP / "roop_3d/rendering/assets/stage10_cc0_probes.npz.json").read_text("utf-8"))
    mh_meta = json.loads((APP / "roop_3d/.runtime/models/makehuman/makehuman_head_v1.npz.json").read_text("utf-8"))
    flame = OPEN_ARTIFACTS["flame2023_open"]
    cc0 = "https://creativecommons.org/publicdomain/zero/1.0/"
    base = {"license_registry_id": None, "license_note": "", "requires_acceptance": False, "optional": False, "registry_ids": []}
    return [
        {**base, "id": "makehuman_cc0_sources", "name": "MakeHuman CC0 source data (pinned commit, 130 files)",
         "kind": "asset_set", "category": "3d_asset", "install_root": "app",
         "install_path": "roop_3d/.runtime/models/makehuman/sources", "files": mh_files,
         "mirrors": [f"https://raw.githubusercontent.com/makehumancommunity/makehuman/{MAKEHUMAN_COMMIT}/makehuman/data/",
                     f"https://cdn.jsdelivr.net/gh/makehumancommunity/makehuman@{MAKEHUMAN_COMMIT}/makehuman/data/"],
         "sha256": lock_digest(mh_files), "size_bytes": sum(f["size_bytes"] for f in mh_files),
         "license": "CC0-1.0", "license_url": cc0, "commercial_use": "allow",
         "required_for_stages": [S3], "acquisition": "zero_auth", "procedural_fallback": None,
         "note": "Data only (CC0, LICENSE.ASSETS.md); the AGPL MakeHuman application code is not used."},
        {**base, "id": "makehuman_cc0_head", "name": "MakeHuman CC0 head model (compiled locally)",
         "kind": "compiled", "category": "3d_asset", "install_root": "app",
         "install_path": "roop_3d/.runtime/models/makehuman/makehuman_head_v1.npz", "mirrors": [],
         "content_digest": mh_meta["asset_digest"], "depends_on": [], "after": ["makehuman_cc0_sources"],
         "procedural_fallback": "makehuman_cc0_head", "license": "CC0-1.0", "license_url": cc0, "commercial_use": "allow",
         "required_for_stages": [S3], "acquisition": "compiled_locally",
         "note": "Deterministic: rebuilt from the hash-locked sources it is bit-identical (asset digest), with no network."},
        {**base, "id": "anatomical_pack_cc0", "name": "MakeHuman CC0 anatomical full-head pack (synthesised locally)",
         "kind": "compiled", "category": "3d_asset", "install_root": "app",
         "install_path": "roop_3d/.runtime/models/anatomical/makehuman_cc0_anatomical_pack.npz", "mirrors": [],
         "depends_on": [], "after": ["makehuman_cc0_head"], "procedural_fallback": "anatomical_pack_cc0", "license": "CC0-1.0", "license_url": cc0,
         "commercial_use": "allow", "required_for_stages": [S3], "acquisition": "compiled_locally",
         "note": "Uses the MakeHuman head when installed, otherwise an analytic CC0 head (its template_source says which)."},
        {**base, "id": "cc0_probe_bundle", "name": "CC0 studio SH lighting probes + micro-pore normal tile",
         "kind": "bundled", "category": "lighting", "install_root": "app",
         "install_path": "roop_3d/rendering/assets/stage10_cc0_probes.npz", "mirrors": [],
         "sha256": bundle_sidecar["bundle_sha256"], "content_digest": array_digest(APP / "roop_3d/rendering/assets/stage10_cc0_probes.npz"),
         "procedural_fallback": "cc0_probe_bundle",
         "build_inputs": {"hdri_dir": "cc0_studio_hdri_sources"}, "license": "CC0-1.0",
         "license_url": "https://polyhaven.com/license", "commercial_use": "allow", "required_for_stages": [S3],
         "acquisition": "ships_with_app",
         "note": "Reduced 32x64 maps from 5 Poly Haven CC0 HDRIs; the pore tile is procedural CC0."},
        {**base, "id": "cc0_studio_hdri_sources", "name": "Poly Haven CC0 1K HDRIs (only to rebuild the probe bundle)",
         "kind": "asset_set", "category": "lighting", "install_root": "app",
         "install_path": "roop_3d/.runtime/models/cc0_hdri", "files": hdri_files, "mirrors": [POLYHAVEN],
         "sha256": lock_digest(hdri_files), "size_bytes": sum(f["size_bytes"] for f in hdri_files), "optional": True,
         "license": "CC0-1.0", "license_url": "https://polyhaven.com/license", "commercial_use": "allow",
         "required_for_stages": [], "acquisition": "zero_auth", "procedural_fallback": None,
         "note": "Not needed at runtime: the committed bundle already holds the reduced probes."},
        {**base, "id": "correspondence_maps", "name": "MakeHuman <-> FLAME 2023 barycentric correspondence maps",
         "kind": "compiled", "category": "3d_asset", "install_root": "app",
         "install_path": "roop_3d/.runtime/models/topology/makehuman_cc0__flame2023_open.npz", "mirrors": [],
         "depends_on": ["makehuman_cc0_head", "flame2023_open"], "procedural_fallback": "correspondence_maps", "optional": True,
         "license": "CC-BY-4.0", "license_url": flame["license_url"], "commercial_use": "allow", "required_for_stages": [S3],
         "acquisition": "compiled_locally",
         "note": "Pure geometry from two local meshes; cannot exist until FLAME 2023 Open is installed."},
        {**base, "id": "flame2023_open", "name": flame["name"], "kind": "weight", "category": "3d_asset",
         "install_root": "repo", "install_path": "flame/flame2023/flame2023_Open.pkl", "filename": "flame2023_Open.pkl",
         "mirrors": [], "sha256": flame["sha256"], "size_bytes": flame["size_bytes"], "optional": True,
         "license": flame["license_id"], "license_url": flame["license_url"], "commercial_use": flame["commercial_use"],
         "attribution": flame["attribution"], "required_for_stages": [S3], "format": "pickle (allow-listed unpickler)",
         "acquisition": "registration_required", "procedural_fallback": None,
         "note": ("The official download is behind a free FLAME account. A search of 144 public Hugging Face repos found no "
                  "mirror with these exact bytes, so none is listed; place the file at install_path (SHA-256 is enforced).")},
        {**base, "id": "smirk_em1_checkpoint", "name": "SMIRK em1 encoder checkpoint (PyTorch)", "kind": "weight",
         "category": "encoder_3d", "install_root": "app",
         "install_path": "roop_3d/.runtime/models/dense_face/upstream/SMIRK_em1.pt", "filename": "SMIRK_em1.pt",
         "mirrors": ["https://drive.usercontent.google.com/download?id=1T65uEd9dVLHgVw5KiUYL66NUee-MCzoE&export=download&confirm=t"],
         "sha256": "26b234e3cc31a5de226bcba4321bb5d7343a2ad48234c028b57a1c2f8c2d22d0", "size_bytes": 140495895,
         "optional": True, "license": "UNVERIFIED", "license_url": "https://github.com/georgeretsi/smirk",
         "license_note": ("SMIRK code is MIT; the checkpoint regresses FLAME parameters, whose terms for these weights were "
                          "not verified. Treated as non-commercial."), "commercial_use": "deny", "requires_acceptance": True,
         "required_for_stages": [S3], "format": "pickle (download-only; this manager never loads it)",
         "acquisition": "zero_auth", "procedural_fallback": None,
         "note": "Single public source (Google Drive), so there is no second mirror."},
    ]


PERMISSIVE = {"CC0-1.0", "MIT", "Apache-2.0", "CC-BY-4.0", "BSD-3-Clause"}


GITHUB_HASH_LIMIT = 120 * 1024 * 1024


def normalize_license(registry_license: str, rights_unpublished: bool, commercial: str) -> tuple[str, str]:
    """Registry licence ids that are not licences (acknowledgements, UNKNOWN) are shown as UNVERIFIED, never as open."""
    if registry_license in PERMISSIVE:
        return registry_license, ""
    if registry_license in {"LOCAL-USE-ACKNOWLEDGED-UNVERIFIED", "UNKNOWN", ""} or rights_unpublished:
        return "UNVERIFIED", ("No upstream licence was verified for these exact bytes. Download needs explicit "
                              "acknowledgement; local personal use only, commercial use and redistribution are denied.")
    return registry_license, "Restricted licence; see license_url. Commercial use: " + commercial


def get_json(url: str) -> Any:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as response:
        return json.load(response)


def sha256_url(url: str, limit: int = 8 * 1024 * 1024) -> tuple[str, int] | None:
    try:
        digest, total = hashlib.sha256(), 0
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as response:
            while block := response.read(1 << 20):
                total += len(block)
                if total > limit:
                    return None
                digest.update(block)
        return digest.hexdigest(), total
    except (urllib.error.URLError, OSError):
        return None


def index_hf(repos: list[str]) -> dict[str, list[dict[str, Any]]]:
    """sha256 -> [{repo, commit, path, size}] for every LFS file; small non-LFS files are hashed on demand later."""
    by_oid: dict[str, list[dict[str, Any]]] = {}
    by_size: dict[int, list[dict[str, Any]]] = {}
    for repo in repos:
        try:
            commit = get_json(f"https://huggingface.co/api/models/{repo}")["sha"]
            tree = get_json(f"https://huggingface.co/api/models/{repo}/tree/{commit}?recursive=true&expand=false")
        except (urllib.error.URLError, OSError, KeyError, ValueError):
            print(f"  skip {repo}", file=sys.stderr)
            continue
        for item in tree:
            if item.get("type") != "file":
                continue
            row = {"repo": repo, "commit": commit, "path": item["path"], "size": item["size"]}
            oid = (item.get("lfs") or {}).get("oid")
            if oid:
                by_oid.setdefault(oid, []).append(row)
            else:
                by_size.setdefault(item["size"], []).append(row)
    by_oid["__small__"] = by_size   # type: ignore[assignment]
    return by_oid


def hf_url(row: dict[str, Any]) -> str:
    return f"https://huggingface.co/{row['repo']}/resolve/{row['commit']}/{urllib.parse.quote(row['path'])}"


def github_assets() -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    """(sha256 -> assets from the ``digest`` GitHub publishes, file name -> every asset); older uploads carry no digest."""
    found: dict[str, list[dict[str, Any]]] = {}
    by_name: dict[str, list[dict[str, Any]]] = {}
    for repo in GITHUB_RELEASES:
        try:
            releases = get_json(f"https://api.github.com/repos/{repo}/releases?per_page=100")
        except (urllib.error.URLError, OSError, ValueError):
            continue
        for release in releases:
            for asset in release.get("assets", []):
                row = {"url": asset["browser_download_url"], "size": asset["size"]}
                by_name.setdefault(asset["name"], []).append(row)
                digest = (asset.get("digest") or "")
                if digest.startswith("sha256:"):
                    found.setdefault(digest[7:], []).append(row)
    return found, by_name


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)

    registry = ModelRegistry()
    by_name: dict[str, Any] = {}
    for record in registry.records():
        if record.upstream_url and record.expected_sha256:
            by_name.setdefault(record.name, record)

    print("indexing Hugging Face repositories ...", file=sys.stderr)
    repos = list(HF_REPOS)
    for term in SEARCH_TERMS:
        try:
            for hit in get_json(f"https://huggingface.co/api/models?search={term}&limit=25"):
                if hit["id"] not in repos and len(repos) < 110:
                    repos.append(hit["id"])
        except (urllib.error.URLError, OSError, ValueError):
            pass
    oid_index = index_hf(repos)
    small = oid_index.pop("__small__")   # type: ignore[assignment]
    github, github_by_name = github_assets()

    models: list[dict[str, Any]] = []
    for model_id, reg_name, category, stages, title, note in CORE:
        if reg_name is not None:
            record = by_name[reg_name]
            identity = {"filename": record.name, "sha256": record.expected_sha256, "size_bytes": record.byte_size,
                        "license": record.license_id, "commercial_use": record.commercial_use,
                        "license_url": record.license_url, "pinned": record.upstream_url,
                        "rights_unpublished": record.rights_unpublished, "registry_ids": [record.model_id]}
        else:
            identity = {**EXTRA_IDENTITY[model_id], "pinned": None, "rights_unpublished": False, "registry_ids": []}
        sha = identity["sha256"]
        mirrors: list[str] = [identity["pinned"]] if identity.get("pinned") else []
        rows = list(oid_index.get(sha, []))
        for row in small.get(identity["size_bytes"], []):       # small non-LFS files: hash them to prove identity
            if identity["size_bytes"] <= 8 * 1024 * 1024:
                result = sha256_url(hf_url(row))
                if result and result[0] == sha:
                    rows.append(row)
        order = {repo: index for index, repo in enumerate(HF_REPOS)}
        rows.sort(key=lambda r: (order.get(r["repo"], 999), r["repo"], r["path"]))
        for row in rows:
            url = hf_url(row)
            if url not in mirrors:
                mirrors.append(url)
        for asset in github.get(sha, []):
            if asset["size"] == identity["size_bytes"] and asset["url"] not in mirrors:
                mirrors.append(asset["url"])
        for asset in github_by_name.get(identity["filename"], []):    # older releases carry no digest: prove by hashing
            if (asset["size"] == identity["size_bytes"] and asset["url"] not in mirrors
                    and asset["size"] <= GITHUB_HASH_LIMIT):
                result = sha256_url(asset["url"], GITHUB_HASH_LIMIT)
                if result and result[0] == sha:
                    mirrors.append(asset["url"])
        license_id, note_license = normalize_license(identity["license"], identity["rights_unpublished"],
                                                    identity["commercial_use"])
        entry = {
            "id": model_id, "name": title, "filename": identity["filename"], "category": category, "kind": "weight",
            "install_root": "app", "install_path": f"models/{identity['filename']}", "mirrors": mirrors[:8],
            "sha256": sha, "size_bytes": identity["size_bytes"],
            "license": license_id, "license_registry_id": identity["license"], "license_url": identity.get("license_url"),
            "license_note": note_license, "commercial_use": identity["commercial_use"],
            "requires_acceptance": license_id not in PERMISSIVE or bool(identity["rights_unpublished"]),
            "required_for_stages": list(stages), "optional": False, "registry_ids": identity["registry_ids"],
            "acquisition": "zero_auth" if mirrors else "no_public_source", "note": note,
        }
        models.append(entry)
        print(f"{model_id:36} mirrors={len(entry['mirrors'])}  {license_id}", file=sys.stderr)

    models.extend(asset_entries())
    document = {
        "schema": "roop3d.universal_model_manifest.v1",
        "evidence": ("Mirrors are listed only when byte identity to the pinned SHA-256 was proven (HF LFS object id at a "
                     "pinned commit, GitHub release digest, or a full hash). required_for_stages is derived from each "
                     "model's role in the three pipeline stages, not from tracing which sessions open at runtime."),
        "models": models,
    }
    args.out.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {args.out} ({len(models)} weight entries)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
