"""Headless command line for roop-3d: ``python -m roop_3d.cli {render,diagnostics,queue} ...``.

* ``render INPUT --source FACESET -o OUTPUT`` runs the real Stage-17 job engine (``pipeline/video_job_runner.py``,
  engine id ``photoreal_3d_ffmpeg``) in-process through the application's own job store and worker: the FaceSet (``.fsz`` file or an
  existing FaceSet id) is imported, the video uploaded and inspected, one job submitted, and its progress is shown on a live
  dashboard (``telemetry/tui.py``) until the result is written to ``OUTPUT``. Exit 0 only for a succeeded job.
* ``diagnostics`` (or the global ``--diagnostics``) prints the EXISTING 14-section runtime report plus verified-local model SHA-256
  status and 3D asset integrity (read-only; nothing is built or downloaded).
* ``queue list`` / ``queue resume JOB_ID`` operate on the persisted 12-state queue (``queue_manager.QueueManager``).

Global flags: ``--headless`` (plain lines, no live dashboard), ``--json-logs`` (one JSON object per line on stdout, schema
``roop3d.cli.log/1``, no ANSI), ``--log-level``. Exit codes: 0 ok, 1 failure, 2 usage / unavailable prerequisite.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Sequence, TextIO

from roop_3d.telemetry.json_log import JsonLogger, LEVELS
from roop_3d.telemetry.tui import FrameStats, SystemProbe, make_dashboard

TERMINAL_JOB_STATES = {"succeeded", "failed", "cancelled", "interrupted"}


def _common(suppress: bool) -> argparse.ArgumentParser:
    """Shared flags. Subparsers use SUPPRESS defaults so a flag given before the subcommand is not overwritten."""
    parent = argparse.ArgumentParser(add_help=False)
    default = argparse.SUPPRESS
    parent.add_argument("--headless", action="store_true", default=default if suppress else False,
                        help="plain line output; never start the live dashboard")
    parent.add_argument("--json-logs", action="store_true", default=default if suppress else False,
                        help="one JSON object per line on stdout (schema roop3d.cli.log/1), no ANSI")
    parent.add_argument("--diagnostics", action="store_true", default=default if suppress else False,
                        help="print the 14-section runtime report + model SHA-256 status + 3D asset integrity and exit")
    parent.add_argument("--no-model-hash", action="store_true", default=default if suppress else False,
                        help="with diagnostics: skip re-hashing installed model files (reported as skipped)")
    parent.add_argument("--log-level", choices=tuple(LEVELS), default=default if suppress else "info")
    parent.add_argument("--root", type=Path, default=default if suppress else None,
                        help="application root holding runtime state (default: this checkout's app/ directory)")
    return parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="roop_3d.cli", description=__doc__.split("\n\n")[0], parents=[_common(False)])
    sub = parser.add_subparsers(dest="command")
    render = sub.add_parser("render", parents=[_common(True)], help="swap faces in a video using the Stage-17 pipeline")
    render.add_argument("input", type=Path, help="target video")
    render.add_argument("--source", required=True, help="source FaceSet: path to a .fsz file or an existing faceset id")
    render.add_argument("-o", "--output", required=True, type=Path, help="result .mp4 path")
    render.add_argument("--track", type=int, default=0, help="index of the detected face track to replace (default 0)")
    render.add_argument("--cpu", action="store_true", help="force the CPU provider")
    render.add_argument("--timeout", type=float, default=0.0, help="seconds before giving up on the job (0 = no limit)")
    diagnostics = sub.add_parser("diagnostics", parents=[_common(True)], help="14-section runtime report + model/asset integrity")
    queue = sub.add_parser("queue", parents=[_common(True)], help="inspect or resume the persisted job queue")
    queue_sub = queue.add_subparsers(dest="queue_command", required=True)
    queue_sub.add_parser("list", parents=[_common(True)], help="list queued jobs")
    resume = queue_sub.add_parser("resume", parents=[_common(True)], help="re-queue a PAUSED job from its checkpoint")
    resume.add_argument("job_id")
    return parser


class Output:
    """One emission API for both modes: JSON lines when ``--json-logs``, human text otherwise."""

    def __init__(self, json_logs: bool, level: str, stream: TextIO | None = None):
        self.stream = stream or sys.stdout
        self.json = JsonLogger(self.stream, level) if json_logs else None
        self.level = level

    def event(self, event: str, text: str | None = None, level: str = "info", **fields: Any) -> None:
        if self.json is not None:
            self.json.emit(event, level, **fields)
        elif LEVELS[level] >= LEVELS[self.level] and text is not None:
            print(text, file=self.stream, flush=True)


# ------------------------------------------------------------------------------------------------------ diagnostics
def collect_diagnostics(root: Path, *, hash_models: bool = True) -> dict[str, Any]:
    """The existing 14-section report plus model SHA-256 state and 3D asset integrity. Read-only; one report, not a third one."""
    from roop_3d.audit.local_models import universal_manifest_report, verify_local_cc0_assets
    from roop_3d.telemetry.runtime_report import build_runtime_report
    report = build_runtime_report(root, None, trigger="cli --diagnostics")
    if hash_models:
        try:
            models: dict[str, Any] = universal_manifest_report(root)
        except Exception as error:   # a diagnostic degrades; it must not hide the other sections
            models = {"ok": False, "state": "UNKNOWN", "reason": f"{type(error).__name__}: {error}"}
    else:
        models = {"ok": None, "state": "SKIPPED", "reason": "--no-model-hash"}
    assets = verify_local_cc0_assets(root, build_assets=False)
    return {"runtime_report": report, "model_sha256": models, "asset_integrity": assets,
            "ok": bool(models.get("ok") is not False and assets["ok"])}


def render_diagnostics_text(result: dict[str, Any]) -> str:
    models, assets = result["model_sha256"], result["asset_integrity"]
    lines = [result["runtime_report"]["text"], "", "MODEL SHA-256 VERIFICATION"]
    if models.get("state") in ("SKIPPED", "UNKNOWN"):
        lines.append(f"  {models['state'].lower()}: {models['reason']}")
    else:
        lines.append(f"  entries={models['entries']} by_state={json.dumps(models['by_state'], sort_keys=True)}")
        lines.append(f"  corrupt={models['corrupt'] or 'none'} required_not_installed={models['required_not_installed'] or 'none'}")
    lines.append("3D ASSET INTEGRITY (npz digest vs .npz.json sidecar)")
    for row in assets["assets"]:
        lines.append(f"  OK   {row['path']} sha256={row['sha256'][:16]}... ({row['size_bytes']} bytes)")
    for violation in assets["violations"]:
        lines.append(f"  FAIL {violation['artifact_id']}: {violation['detail']}")
    return "\n".join(lines)


def run_diagnostics(arguments: argparse.Namespace, out: Output, root: Path) -> int:
    result = collect_diagnostics(root, hash_models=not getattr(arguments, "no_model_hash", False))
    if out.json is not None:
        out.json.emit("diagnostics", "info" if result["ok"] else "error", **result)
    else:
        print(render_diagnostics_text(result), file=out.stream, flush=True)
    return 0 if result["ok"] else 1


# ----------------------------------------------------------------------------------------------------------- queue
def run_queue(arguments: argparse.Namespace, out: Output, root: Path) -> int:
    from roop_3d.core.errors import DomainError
    from roop_3d.queue_manager import QueueManager
    manager = QueueManager(root)
    if arguments.queue_command == "list":
        jobs = manager.list_jobs()
        for job in jobs:
            out.event("queue.job", f"{job.job_id}  {job.state.value:<10} {job.lifecycle_state:<14} {job.name}",
                      job_id=job.job_id, state=job.state.value, lifecycle_state=job.lifecycle_state, name=job.name)
        out.event("queue.summary", f"{len(jobs)} job(s)", count=len(jobs))
        return 0
    try:
        job = manager.resume(arguments.job_id)
    except DomainError as error:
        out.event("queue.resume_refused", f"{error.code}: {error.message if hasattr(error, 'message') else error}", "error",
                  job_id=arguments.job_id, code=error.code)
        return 1
    out.event("queue.resumed", f"{job.job_id} -> {job.state.value}", job_id=job.job_id, state=job.state.value)
    return 0


# ---------------------------------------------------------------------------------------------------------- render
def run_render(arguments: argparse.Namespace, out: Output, root: Path) -> int:
    source_video = arguments.input
    if not source_video.is_file():
        out.event("render.error", f"input not found: {source_video}", "error", code="INPUT_MISSING", path=str(source_video))
        return 2
    source_path = Path(arguments.source)
    try:
        from fastapi.testclient import TestClient
        from roop_3d.core.settings import Settings
        from roop_3d.pipeline.video_job_runner import ENGINE_ID
        from roop_3d.server.app import create_app
    except ImportError as error:
        out.event("render.error", f"render needs the server stack: {error}", "error", code="DEPENDENCY_MISSING")
        return 2
    settings = (Settings(hardware_profile="cpu", preferred_provider="CPUExecutionProvider") if arguments.cpu
                else Settings.from_env())
    app = create_app(settings, repo_root=root, worker_enabled=True)
    stats = FrameStats()
    probe = SystemProbe(provider="CPUExecutionProvider" if arguments.cpu else None)
    dashboard = make_dashboard(headless=arguments.headless or out.json is not None)
    began = time.monotonic()
    with TestClient(app, base_url="http://127.0.0.1:8030") as client, dashboard:
        def fail(code: str, detail: str, status: int = 1) -> int:
            out.event("render.error", f"{code}: {detail}", "error", code=code, detail=detail)
            return status

        if source_path.is_file():
            imported = client.post(f"/api/v1/facesets/import?filename={source_path.name}", content=source_path.read_bytes(),
                                   headers={"Content-Type": "application/octet-stream"})
            if imported.status_code != 201:
                return fail("FACESET_IMPORT_FAILED", imported.text[:300])
            faceset_id = imported.json()["faceset_id"]
        else:
            faceset_id = arguments.source
        out.event("render.faceset", f"source faceset {faceset_id}", faceset_id=faceset_id)
        uploaded = client.post(f"/api/v1/media/upload?filename={source_video.name}", content=source_video.read_bytes(),
                               headers={"Content-Type": "application/octet-stream"})
        if uploaded.status_code != 201:
            return fail("MEDIA_UPLOAD_FAILED", uploaded.text[:300])
        inspection = client.post("/api/v1/media/inspect-faces", json={"path": uploaded.json()["path"]})
        if inspection.status_code != 200:
            return fail("INSPECTION_FAILED", inspection.text[:300])
        tracks = inspection.json()["tracks"]
        if not 0 <= arguments.track < len(tracks):
            return fail("TRACK_NOT_FOUND", f"--track {arguments.track}: the detector found {len(tracks)} track(s)", 2)
        created = client.post("/api/v1/jobs", headers={"Idempotency-Key": f"cli-{int(began)}-{uploaded.json()['sha256'][:12]}"}, json={
            "engine_id": ENGINE_ID, "target_path": uploaded.json()["path"], "inspection_id": inspection.json()["inspection_id"],
            "face_mappings": [{"track_id": tracks[arguments.track]["track_id"], "faceset_id": faceset_id}]})
        if created.status_code != 201:
            return fail("JOB_REJECTED", created.text[:300])
        job_id = created.json()["job_id"]
        out.event("render.started", f"job {job_id}", job_id=job_id)
        record: dict[str, Any] = {}
        last_frame, last_time = 0, time.monotonic()
        while True:
            record = client.get(f"/api/v1/jobs/{job_id}").json()
            progress = record.get("progress") or {}
            frame = int(progress.get("frame") or 0)
            if progress.get("total_frames"):
                stats.total_frames = int(progress["total_frames"])
            now = time.monotonic()
            if frame > last_frame:
                per_frame = (now - last_time) * 1000.0 / (frame - last_frame)
                for _ in range(frame - last_frame):
                    stats.record(per_frame)
                last_frame, last_time = frame, now
            snapshot = probe.snapshot()
            timings = progress.get("stage_timings_ms") or {}
            stage = f"{record['state']}/{progress.get('stage', '')}"
            if out.json is not None:
                out.json.emit("render.progress", "info", job_id=job_id, state=record["state"], stage=progress.get("stage"), frame=frame,
                              total_frames=progress.get("total_frames"), fps=stats.fps(), frame_ms=stats.percentiles(),
                              eta_seconds=stats.eta_seconds() if stats.eta_seconds() is not None else progress.get("eta_seconds"),
                              stage_timings_ms=timings, system=snapshot.as_dict())
            else:
                dashboard.update(stats, timings, snapshot, stage)
            if record["state"] in TERMINAL_JOB_STATES:
                break
            if arguments.timeout and now - began > arguments.timeout:
                client.post(f"/api/v1/jobs/{job_id}/cancel")
                return fail("RENDER_TIMEOUT", f"job {job_id} exceeded {arguments.timeout}s and was cancelled")
            time.sleep(0.5)
        if record["state"] != "succeeded":
            return fail("JOB_" + record["state"].upper(), json.dumps(record.get("failure") or {}, default=str))
        download = client.get(f"/api/v1/jobs/{job_id}/download")
        if download.status_code != 200:
            return fail("DOWNLOAD_FAILED", download.text[:200])
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_bytes(download.content)
    out.event("render.completed", f"wrote {arguments.output} ({record['output']['frames']} frames)", job_id=job_id,
              output=str(arguments.output), frames=record["output"]["frames"], sha256=record["output"]["output_sha256"],
              elapsed_seconds=round(time.monotonic() - began, 2))
    return 0


def main(argv: Sequence[str] | None = None, stream: TextIO | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    out = Output(arguments.json_logs, arguments.log_level, stream)
    if arguments.root is None:
        from roop_3d.audit.project_paths import application_root
        root = application_root()
    else:
        root = arguments.root
    if arguments.diagnostics or arguments.command == "diagnostics":
        return run_diagnostics(arguments, out, root)
    if arguments.command == "render":
        return run_render(arguments, out, root)
    if arguments.command == "queue":
        return run_queue(arguments, out, root)
    parser.print_usage(sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
