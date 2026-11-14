"""Execute real Pure-3D component forwards in the active application interpreter.

Run from app/: roop_3d/.venv/Scripts/python.exe -B -m roop_3d.tools.verify_stage1
No downloads, fake sessions, render-readiness overrides, or alternate swap engines.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import sys
import time
import zipfile

import cv2
import numpy as np

from roop_3d.audit.project_paths import application_root
from roop_3d.core.storage import atomic_json
from roop_3d.core.ort_session import session_selection
from roop_3d.engine.runtime import RuntimeConfig, session_factory, OrtModel
from roop_3d.models.catalog import ModelCatalog
from roop_3d.models.pure3d import resolve_components


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, help="Optional local test face image; otherwise first image in harjot.fsz")
    parser.add_argument("--provider", choices=("auto", "CPUExecutionProvider", "CUDAExecutionProvider", "DmlExecutionProvider"), default="auto")
    args = parser.parse_args()
    root = application_root()
    catalog = ModelCatalog(root)
    catalog.rescan_local()
    config = RuntimeConfig(provider=args.provider, hardware_profile="6gb" if args.provider == "CUDAExecutionProvider" else "cpu")
    rows = {row["component"]: row for row in resolve_components(catalog)}
    raster_path = Path(__file__).resolve().parents[1] / "rendering/cpu_perspective_raster.py"
    rows["uv_renderer"] = {"path": str(raster_path), "state": "READY", "kind": "implementation"}
    if args.image:
        image_bytes = args.image.read_bytes()
        source = str(args.image.resolve())
    else:
        archive_path = root / "facesets/harjot.fsz"
        with zipfile.ZipFile(archive_path) as archive:
            name = next(name for name in sorted(archive.namelist()) if Path(name).suffix.lower() in (".png", ".jpg", ".jpeg"))
            image_bytes = archive.read(name)
        source = f"{archive_path}!{name}"
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode test image: {source}")
    results = []
    state = {}

    def run(component, forward):
        row = rows[component]
        path = Path(row["path"]) if row.get("path") else None
        result = {"component": component, "path": str(path.resolve()) if path else None,
                  "provider": None, "execution_verified": False, "error": None}
        if path and path.is_file():
            stat = path.stat()
            result.update(size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
        started = time.perf_counter()
        try:
            if row["state"] != "READY":
                raise RuntimeError(json.dumps(row.get("error"), ensure_ascii=True))
            provider, details = forward()
            result.update(provider=provider, execution_verified=True, details=details)
        except Exception as error:
            result["error"] = f"{type(error).__name__}: {error}"
            print(f"FAIL {component}: {result['error']}", flush=True)
        result["seconds"] = round(time.perf_counter() - started, 3)
        results.append(result)
        if result["execution_verified"]:
            print(f"PASS {component}: {result['provider']} {json.dumps(result['details'])}", flush=True)
        gc.collect()

    def model_provider(component):
        diagnostic = session_selection(Path(rows[component]["path"]).name)
        if not diagnostic or not diagnostic["execution_verified"]:
            raise RuntimeError(f"No genuine forward diagnostic for {component}")
        return diagnostic["provider"]

    def detect():
        from roop_3d.media.detector import Detector
        detector = Detector(catalog, session_factory(catalog, config), provider=args.provider)
        try:
            faces, _ = detector.detect_scored(image, .3)
            if not faces:
                raise RuntimeError("Real detector executed but found no face in test image; use --image with a visible face")
            box, landmarks, score = max(faces, key=lambda face: float(np.prod(face[0][2:] - face[0][:2])))
            state.update(box=box, landmarks=landmarks)
            return model_provider("detector"), {"faces": len(faces), "score": float(score)}
        finally:
            detector.release_for_build()

    def embed():
        from roop_3d.faceset.recognizer import Recognizer
        recognizer = Recognizer(catalog, session_factory(catalog, config))
        try:
            embedding = recognizer.embed(image, state["landmarks"])
            if embedding.shape != (512,) or not np.isfinite(embedding).all() or abs(np.linalg.norm(embedding) - 1) > 1e-4:
                raise ValueError("ArcFace forward did not return a finite unit 512-vector")
            return model_provider("recognizer"), {"shape": list(embedding.shape), "norm": float(np.linalg.norm(embedding))}
        finally:
            recognizer.release_for_build()

    def landmarks(component):
        from roop_3d.engine.detection.dense_landmarks import DenseLandmarks
        dense = DenseLandmarks(catalog, config)
        try:
            points = dense.infer68(image, state["box"]) if component == "dense_landmarks_3d" else dense.infer106(image, state["box"])
            expected = (68, 3) if component == "dense_landmarks_3d" else (106, 2)
            if points is None or points.shape != expected or not np.isfinite(points).all() or np.ptp(points[:, :2]) <= 0:
                raise ValueError(f"Invalid measured landmark output: expected {expected}")
            return model_provider(component), {"shape": list(points.shape)}
        finally:
            dense.release()

    def reconstruct():
        from roop_3d.reconstruction.makehuman_head import MakeHumanHeadModel, ViewObservation, fit_multiview
        model = MakeHumanHeadModel.load(rows["head_reconstructor"]["path"], verify=True)
        observation = ViewObservation(state["landmarks"], image.shape[1], image.shape[0], name="stage1_test_image")
        fit = fit_multiview(model, [observation], max_iterations=60, holdout=False)
        vertices = model.forward(fit.shape_params, np.zeros(model.num_expr))
        if len(vertices) < 1500 or vertices.shape[1] != 3 or not np.isfinite(vertices).all() or not np.isfinite(fit.cost):
            raise ValueError("Invalid fitted full-head mesh")
        # Single-view fitting executes real mesh math but does not prove unobserved identity geometry.
        return "CPU (NumPy/SciPy)", {"vertices": len(vertices), "faces": len(model.base_mesh.faces),
                                     "fit_rms_px": fit.views[0].rms_px, "converged": fit.converged,
                                     "limitation": "single-view template fit, not observed back-of-head reconstruction"}

    def synthesize():
        from roop_3d.reconstruction.makehuman_template import makehuman_template
        from roop_3d.reconstruction.geometry_types import PerspectiveCamera
        from roop_3d.reconstruction.uv_synthesizer import UVSynthesizer, ViewObservation
        mesh = makehuman_template(rows["head_reconstructor"]["path"])
        center = mesh.vertices.mean(axis=0)
        scale = min(image.shape[:2]) / max(np.ptp(mesh.vertices, axis=0)) * .7
        camera = PerspectiveCamera.from_scaled_pose(np.eye(3), scale,
            image.shape[1] / 2 - center[0] * scale, image.shape[0] / 2 - center[1] * scale,
            frame_size=(image.shape[1], image.shape[0]))
        synthesizer = UVSynthesizer(mesh, uv_size=128, max_interpolation_fraction=.02)
        atlas = synthesizer.synthesize([ViewObservation(image=image, camera=camera)])
        if atlas.observed_ratio <= 0 or atlas.texture_map.shape != (128, 128, 3) or not np.isfinite(atlas.texture_map).all():
            raise ValueError("UV synthesis produced no observed texels")
        state.update(mesh=mesh, camera=camera, atlas=atlas.texture_map,
                     atlas_mask=(atlas.confidence_mask > 0).astype(np.float32))
        return "CPU (NumPy/OpenCV)", {"shape": list(atlas.texture_map.shape), "observed_ratio": atlas.observed_ratio}

    def render():
        from roop_3d.rendering.cpu_perspective_raster import rasterize
        mesh, camera = state["mesh"], state["camera"]
        atlas = state["atlas"]
        uvs = mesh.uvs.copy()
        uvs[:, 1] = 1 - uvs[:, 1]
        uvs *= atlas.shape[0] - 1
        # Reflection from mesh +Z-toward-viewer to camera +Z-away reverses triangle winding.
        rendered, coverage, visible = rasterize(atlas, state["atlas_mask"], camera.project(mesh.vertices),
            camera.camera_points(mesh.vertices), uvs, mesh.faces[:, ::-1], image.shape[:2])
        if rendered.shape != image.shape or not np.isfinite(rendered).all() or not np.any(coverage > 0) or not len(visible):
            raise ValueError("Perspective UV rasterization produced no visible textured surface")
        return "CPU (NumPy/OpenCV)", {"visible_triangles": len(visible), "covered_pixels": int(np.count_nonzero(coverage))}

    def parse_face():
        from roop_3d.compositor.masks import BisenetFaceParser, extract_forbidden_mask
        semantic = BisenetFaceParser(catalog, config)
        try:
            probabilities = semantic.predict(image)
            forbidden = extract_forbidden_mask(probabilities)
            if probabilities.shape != (*image.shape[:2], 19) or not np.allclose(probabilities.sum(-1), 1, atol=1e-4):
                raise ValueError("Semantic parser returned invalid 19-class probabilities")
            labels = probabilities.argmax(-1)
            allowed = np.isin(labels, (1, 2, 3, 4, 5, 10, 11, 12, 13))
            if np.any(allowed[labels == 17]) or not np.isfinite(forbidden).all():
                raise ValueError("Hair exclusion failed")
            return model_provider("face_parser"), {"classes": 19, "hair_pixels": int(np.count_nonzero(labels == 17)),
                                                  "allowed_face_pixels": int(np.count_nonzero(allowed))}
        finally:
            semantic.release()

    def xseg():
        model = OrtModel(catalog, "xseg.onnx", config)
        try:
            node = model.session.get_inputs()[0]
            resized = cv2.resize(image, (256, 256)).astype(np.float32) / 255
            value = np.ascontiguousarray(resized[None] if node.shape[-1] == 3 else resized.transpose(2, 0, 1)[None])
            output = model.run({node.name: value})
            if not output or any(not np.isfinite(array).all() or array.size == 0 for array in output.values()):
                raise ValueError("XSeg returned invalid mask tensors")
            return model.provider, {name: list(array.shape) for name, array in output.items()}
        finally:
            model.release()

    def enhance():
        from roop_3d.enhancers.uv_detailer import UVTextureDetailer
        detailer = UVTextureDetailer.from_catalog(catalog, config)
        try:
            atlas = cv2.resize(state["atlas"], (1024, 1024))
            detailed = detailer.detail_albedo(atlas)
            changed = int(np.count_nonzero(detailed != atlas))
            if detailed.shape != atlas.shape or not np.isfinite(detailed).all() or changed == 0:
                raise ValueError("UV skin enhancer returned unchanged or invalid pixels")
            return model_provider("uv_enhancer"), {"shape": list(detailed.shape), "changed_channel_values": changed,
                "normalization": "RGB [-1,1] CodeFormer convention; upstream normalization not independently verified"}
        finally:
            detailer.session.model.release()

    print("Stage 1 Verification Report", flush=True)
    print(f"Interpreter: {sys.executable}\nTest image: {source}\nRequested provider: {args.provider}", flush=True)
    for component, forward in (("detector", detect), ("recognizer", embed),
        ("dense_landmarks_3d", lambda: landmarks("dense_landmarks_3d")),
        ("dense_landmarks_2d", lambda: landmarks("dense_landmarks_2d")),
        ("head_reconstructor", reconstruct), ("uv_synthesis", synthesize), ("uv_renderer", render),
        ("face_parser", parse_face), ("occlusion_mask", xseg), ("uv_enhancer", enhance)):
        run(component, forward)
    passed = sum(result["execution_verified"] for result in results)
    report = {"schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
              "interpreter": sys.executable, "test_image": source, "results": results,
              "passed": passed, "total": len(results), "production_renderer_wired": False}
    atomic_json(root, "roop_3d/.runtime/stage1_verification.json", report)
    print(f"Result: {passed}/{len(results)} passed. Production renderer remains NOT WIRED.", flush=True)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
