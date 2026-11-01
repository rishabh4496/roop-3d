"""Stage 7 end-to-end gate: the built single-origin workstation plus the remediation pipeline contracts.

Run from `app/` with the dedicated interpreter after `npm run build` in `roop_3d_ui`:

    roop_3d/.venv/Scripts/python.exe -m roop_3d.tools.verify_e2e_workstation

Exit code 0 only when every stage below passes. Nothing here is mocked at the HTTP boundary: the real FastAPI app serves the
real `dist/` bundle. Browser-console cleanliness is asserted by the UI suite (`npx vitest run`), because no browser driver
is installed; this gate says so in its report instead of claiming it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

APP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(APP))

SCRATCH = APP / "roop_3d/.runtime/e2e"   # inside the app root: the isolation guard rejects other locations


def scratch(prefix: str) -> Path:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=SCRATCH))


REMEDIATION_TESTS = (
    "roop_3d/tests/test_remediation_stage2.py",   # strict TensorRT profile binding, dynamic-shape conversion
    "roop_3d/tests/test_remediation_stage3.py",   # 3-stage Pre-Scan / Range Lock / Render & Mux
    "roop_3d/tests/test_remediation_stage4.py",   # functional live preview swapping
    "roop_3d/tests/test_remediation_stage5.py",   # Use Face From Frame (extract_from_frame), pose, TPS
    "roop_3d/tests/test_remediation_stage7.py",   # Stage 27 linter, stage timings, NPZ checkpoint
)
LINT_IDS = {"3d_pose_coverage", "vram_headroom", "out_of_frame_seams", "output_encoder_storage",
            "trt_dynamic_models", "frontalization_vs_pose_matrix", "ultramax_crop_bypass", "vram_threads"}


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, bool, str]] = []

    def run(self, name: str, function) -> None:
        began = time.perf_counter()
        try:
            detail = function() or ""
            self.rows.append((name, True, f"{detail} ({time.perf_counter() - began:.1f}s)".strip()))
        except Exception as error:  # every stage is reported; the first failure does not hide the rest
            self.rows.append((name, False, f"{type(error).__name__}: {error}\n{traceback.format_exc(limit=3)}"))

    @property
    def ok(self) -> bool:
        return all(passed for _, passed, _ in self.rows)

    def print(self) -> None:
        for name, passed, detail in self.rows:
            print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")


# -- built UI served from a single origin -------------------------------------------------------------------
def check_built_bundle(dist: Path) -> str:
    index = dist / "index.html"
    assert index.is_file(), f"{index} is missing; run `npm run build` in roop_3d_ui first"
    html = index.read_text("utf-8")
    refs = re.findall(r'(?:src|href)="([^"]+)"', html)
    local = []
    for ref in refs:
        assert not re.match(r"^(?:https?:)?//", ref), f"index.html loads a remote resource: {ref}"
        if not ref.startswith(("data:", "#")):
            local.append(ref)
    assert any(ref.endswith(".js") for ref in local), "index.html references no script"
    for ref in local:
        assert (dist / ref.lstrip("/")).is_file(), f"index.html references a missing file: {ref}"
    for asset in (dist / "assets").glob("*.css"):
        remote = re.findall(r"url\(\s*['\"]?(https?://[^)'\"]+)", asset.read_text("utf-8", errors="ignore"))
        assert not remote, f"{asset.name} fetches remote resources: {remote[:3]}"
    return f"{len(local)} local assets, zero remote references"


def check_single_origin_server(dist: Path) -> str:
    from fastapi.testclient import TestClient
    from roop_3d.core.settings import Settings
    from roop_3d.server.app import create_app
    root = scratch("serve-")
    try:
        shutil.copytree(dist, root / "roop_3d_ui/dist")
        app = create_app(Settings(hardware_profile="cpu", preferred_provider="CPUExecutionProvider"), repo_root=root, worker_enabled=False)
        with TestClient(app, base_url="http://127.0.0.1:8030") as client:
            home = client.get("/")
            assert home.status_code == 200 and "text/html" in home.headers["content-type"], home.status_code
            assert '<div id="root"' in home.text
            served = 0
            for ref in re.findall(r'(?:src|href)="(/[^"]+)"', home.text):
                response = client.get(ref)
                assert response.status_code == 200, f"{ref} -> {response.status_code}"
                if ref.endswith(".js"):
                    assert "javascript" in response.headers["content-type"], response.headers["content-type"]
                served += 1
            for route in ("/projects/example", "/workstation/example"):
                fallback = client.get(route)
                assert fallback.status_code == 200 and '<div id="root"' in fallback.text, route
            missing_api = client.get("/api/v1/does-not-exist")
            assert missing_api.status_code == 404 and "text/html" not in missing_api.headers.get("content-type", ""), "API 404 must not be the SPA shell"
            assert client.get("/api/v1/health").status_code == 200
            lint = client.post("/api/v1/system/preflight-lint", json={"target_path": "roop_3d/.runtime/media/none.mp4"})
            assert lint.status_code == 200, lint.text
            ids = {check["id"] for check in lint.json()["checks"]}
            assert LINT_IDS <= ids, f"missing lint checks: {LINT_IDS - ids}"
            return f"{served} assets + SPA fallback + API boundary + {len(ids)} lint checks"
    finally:
        shutil.rmtree(root, ignore_errors=True)


# -- engine contracts exercised directly --------------------------------------------------------------------
def check_strict_trt_profile_binding() -> str:
    import onnx
    from onnx import TensorProto, helper
    from roop_3d.trt_session_builder import BatchProfile, StaticBatchError, validate_profile

    def model(batch):
        graph = helper.make_graph([helper.make_node("Identity", ["x"], ["y"])], "g",
                                  [helper.make_tensor_value_info("x", TensorProto.FLOAT, [batch, 3, 8, 8])],
                                  [helper.make_tensor_value_info("y", TensorProto.FLOAT, [batch, 3, 8, 8])])
        return helper.make_model(graph)
    profile = BatchProfile(batches=(1, 4, 8), optimum=4)
    try:
        validate_profile(model(1), profile)
    except StaticBatchError:
        pass
    else:
        raise AssertionError("a static batch-1 graph must be refused for a (1, 4, 8) TensorRT profile")
    validate_profile(model("N"), profile)  # a dynamic batch axis binds
    validate_profile(model(1), BatchProfile(batches=(1,), optimum=1))
    assert onnx.checker is not None
    return "static batch refused for (1,4,8); dynamic batch and batch-1 profile accepted"


def check_three_stage_render_and_timings() -> str:
    import numpy as np
    from roop_3d.core.stage_timing import STAGES
    from roop_3d.faceset.recognizer import ARCFACE_REFERENCE_5PTS
    from roop_3d.optimized_processor import StreamingProcessor
    from roop_3d.render_policy import RenderPolicy
    from roop_3d.tools.verify_stream_integration import create_clip
    from roop_3d.vectorized_pipeline import CpuBatchPipeline
    root = scratch("render-")
    try:
        source = root / "input.mp4"
        create_clip(source)
        points = ((ARCFACE_REFERENCE_5PTS + [8, 0]) * 0.4).astype(np.float32)
        events: list[dict] = []
        processor = StreamingProcessor(root, "e2e", source, detect=lambda _: [1],
            track=lambda _: [(0, np.array([0, 0, 64, 64]), points)], pipeline=CpuBatchPipeline(lambda value, _: value),
            policy=RenderPolicy(hardware_profile="cpu", execution_threads=1, ring_slots=3, segment_frames=4), emit=events.append)
        result = processor.run(root / "output.mp4", audio=False)
        done = [event["stage"] for event in events if event["state"] == "DONE"]
        assert done == ["STAGE_1_SPARSE_PRESCAN", "STAGE_2_ACTIVE_PREPASS", "STAGE_3_MAIN_SWAP"], done
        assert result["frames"] == 10
        timings = processor.timings.take()
        assert {"decode", "inswapper", "encode"} <= set(timings) <= set(STAGES), timings
        assert all(value >= 0 for value in timings.values())
        return f"stages {', '.join(s.split('_', 2)[2].title() for s in done)}; measured {sorted(timings)}"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def check_npz_checkpoint() -> str:
    from roop_3d.engine.pipeline_runner import PipelineCheckpoint, checkpoint_npz_bytes, read_checkpoint_npz
    value = PipelineCheckpoint(job_id="a" * 32, input_sha256="b" * 64, request_sha256="c" * 64, completed_frame=7,
                               execution_provider="TensorrtExecutionProvider", pose_cache={"track_0000": [1.5, -2.0, 0.25]},
                               inputs={"target_path": "t.mp4"}, pose_settings={"pose_frontalization": 0.4},
                               hardware_state={"probe": "VERIFIED_LOCAL"})
    restored = read_checkpoint_npz(checkpoint_npz_bytes(value))
    assert restored.model_dump(exclude={"updated_at"}) == value.model_dump(exclude={"updated_at"})
    return "JSON and NPZ checkpoints agree"


def run_pytest(selection: tuple[str, ...]) -> str:
    command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *selection]
    done = subprocess.run(command, cwd=APP, capture_output=True, text=True, timeout=1800)
    tail = (done.stdout.strip().splitlines() or [""])[-1]
    assert done.returncode == 0, f"pytest exited {done.returncode}\n{done.stdout[-2500:]}"
    return tail


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--skip-pytest", action="store_true", help="skip the remediation test modules (they take about 25s)")
    args = parser.parse_args(argv)
    dist = APP / "roop_3d_ui/dist"
    report = Report()
    report.run("built bundle is self-contained (zero CDN)", lambda: check_built_bundle(dist))
    report.run("single-origin server serves the built workstation", lambda: check_single_origin_server(dist))
    report.run("strict TensorRT profile binding", check_strict_trt_profile_binding)
    report.run("3-stage render + measured stage timings", check_three_stage_render_and_timings)
    report.run("JSON + NPZ project checkpoint", check_npz_checkpoint)
    if not args.skip_pytest:
        report.run("preview swap, Use Face From Frame, TRT, 3-stage, Stage 27 tests", lambda: run_pytest(REMEDIATION_TESTS))
    report.print()
    print("Browser console errors are asserted by `npx vitest run` (no browser driver is installed here).")
    print("E2E_WORKSTATION_PASSED" if report.ok else "E2E_WORKSTATION_FAILED")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
