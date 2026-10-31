# Stage 12: Pipeline Orchestrator, Worker, and VRAM Governor

## Result

Stage 12 implementation and deterministic worker acceptance gates pass. The local
worker processes a video job through validation, checkpointing, frame decode,
compositing, output encode/mux, manifest verification, and job completion. Pause,
resume, cancellation boundaries, progress events, and simulated CUDA OOM retry are
covered by integration tests.

The acceptance fixture uses a deterministic inference session so it can exercise
job orchestration and media handling without claiming a model quality result. A
render using a user supplied FaceSet and target video was not run in this session;
end-to-end model inference on real face media remains `UNKNOWN`.

## Implementation

- `engine/pipeline_runner.py` validates the target, inspection, and read-only
  FaceSets; probes rational media timing; admits resources before model creation;
  runs the Stage 06 reference session and Stage 09 compositor; checkpoints
  completed frames; muxes source audio under the selected policy; verifies the
  output; and publishes an output manifest atomically.
- `jobs/worker.py` runs durable queued jobs on a bounded thread pool, persists
  state and progress, handles pause/resume/cancel at frame boundaries, recovers
  interrupted work, and retries a CUDA OOM once on CPU.
- `engine/vram_governor.py` probes live hardware at admission. The 6 GB profile
  permits one GPU stream; 12 GB+ uses a bounded reservation policy. The 4096 MiB
  per-stream reservation is a policy value, not a measured model requirement.
- `engine/scheduler.py` owns admission leases. `tools/cache_cleanup.py` reports
  stale temporary files only under `roop_3d/cache/` and performs no deletion.
- Existing Stage 06 reference inference remains the selected model path. Per-track
  FaceSet selection, output paths, and runtime manifests are connected to it.

## Artifact and capability evidence

These read-only local artifacts resolved to their registered hashes. This verifies
artifact availability and bytes, not licensing, provenance, or model output
quality; those evidence states remain `UNKNOWN` in the registry.

| Artifact | SHA-256 | Bytes | Evidence |
| --- | --- | ---: | --- |
| `scrfd_10g_bnkps.onnx` | `5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91` | 16,923,827 | `VERIFIED_LOCAL` hash |
| `w600k_r50.onnx` | `4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43` | 174,383,860 | `VERIFIED_LOCAL` hash |
| `inswapper_128.onnx` | `e4a3f08c753cb72d04e10aa0f7dbe3deebbf39567d4ead6dce08e98aa49e16af` | 554,253,681 | `VERIFIED_LOCAL` hash |
| `2d106det.onnx` | `f001b856447c413801ef5c42091ed0cd516fcd21f2d6b79635b1e733a7109dbf` | 5,030,888 | `VERIFIED_LOCAL` hash |
| `liveportrait_landmark.onnx` | `31d22a5041326c31f19b78886939a634a5aedcaa5ab8b9b951a1167595d147db` | 114,666,491 | `VERIFIED_LOCAL` hash |
| `liveportrait_appearance_feature_extractor.onnx` | `d070afccca7f528ffb0ef5052b21588b42225a996661833f7bdede562d1ab921` | 3,355,896 | `VERIFIED_LOCAL` hash |
| `liveportrait_motion_extractor.onnx` | `219a46174297b2b411bb3c5dce48d3a8c8e07a9d82120a0da21f49f58b67fca6` | 112,648,514 | `VERIFIED_LOCAL` hash |
| `liveportrait_warping_spade.onnx` | `b0e7a566db8fba690c23523bcd2faa4f0d13f05418db84a779397a851062ad69` | 421,233,096 | `VERIFIED_LOCAL` hash |
| `liveportrait_stitching.onnx` | `8e33658425e3f1014dc35b28f60ca7459189e8f78f20540262a2fe5ac1dc6ec7` | 182,363 | `VERIFIED_LOCAL` hash |

Optional capability states:

- Stage 08 UV mode: `UNSUPPORTED`; no calibrated camera-pose adapter is wired.
- Stage 10 Wav2Lip: `UNSUPPORTED`; no verified local model contract/adapter exists.
- Stage 11 diffusion VFX: `UNSUPPORTED`; no verified local family loader is configured.
- GFPGAN/CodeFormer selection: `UNSUPPORTED`; the explicit policies fail closed.
  The automatic compositor can use the existing verified GPEN path when its local
  artifact resolves; otherwise it uses the compositor fallback.
- Remote processing: disabled by default and no remote adapter is configured.

## Protected baseline reconciliation

The mismatch is the ignored, untracked runtime file `roop-ultimate/app/config.yaml`.
Its saved Stage 00 identity is 6,659 bytes /
`a2f4307634291364decfd97fabc9878f73a2e6f33949e5065b944d1c14c0d4eb`; an observed
runtime state was 6,660 bytes /
`b96f19f5bfe46e81172b38862f0a15e978ad54d6b13196e007c85532aae1dc5d`. The legacy
file was not written. Both Stage 00 manifests retain their original saved values.

Roop 3D now classifies only the exact ignored `app/config.yaml` path as mutable
runtime configuration and reports its observed file type and size without claiming
byte identity. A non-regular entry at that path fails the audit. `app/default_config.yaml`,
tracked source, model artifacts, and all other unclassified paths remain immutable
hash checks. This lets the protected source/model gate pass while the legacy app
continues to update its own ignored settings.

## Verification record

Commands ran from `G:\pinokio\api\roop-3d\app`.

| Command | Exit | Result |
| --- | ---: | --- |
| `roop_3d\.venv\Scripts\python.exe -B -m pytest -c roop_3d\pyproject.toml --confcutdir=roop_3d roop_3d\tests\test_stage12_governor.py roop_3d\tests\test_stage12_worker.py -q` | 0 | 7 passed; 1 deprecation warning. |
| `roop_3d\.venv\Scripts\python.exe -B -m pytest -c roop_3d\pyproject.toml --confcutdir=roop_3d roop_3d\tests\test_runtime_policy.py roop_3d\tests\test_stage12_governor.py roop_3d\tests\test_stage12_worker.py -q` | 0 | 13 passed; 1 deprecation warning. |
| `roop_3d\.venv\Scripts\python.exe -B -m pytest -c roop_3d\pyproject.toml --confcutdir=roop_3d roop_3d\tests -q` | 0 | 345 passed, 18 subtests passed; 1 deprecation warning. |
| `roop_3d\.venv\Scripts\python.exe -B roop_3d\audit\verify_isolation.py` | 0 | Historical baseline 2,645/2,645; comprehensive immutable snapshot 68,390/68,390; no violations. Runtime activity: 1,530 files, including the exact `app/config.yaml` exception. |

An earlier targeted run exited 1 with one WebSocket test teardown `CancelledError`.
The test now sends an explicit client disconnect and allows the receive loop to
consume it; the targeted rerun and full suite above both passed.

The audit initially exited 1 on the changing ignored runtime config. The first
manifest-only reconciliation was reverted so the Stage 00 hashes remain original;
the exact runtime-policy exception above is the final fix. The final audit then
passed with no immutable-path violations.

## Files changed

- `audit/runtime_policy.py`
- `engine/pipeline_runner.py`
- `engine/scheduler.py`
- `engine/vram_governor.py`
- `engine/volumetric/identity_swapper.py`
- `engine/volumetric/session.py`
- `jobs/schema.py`
- `jobs/store.py`
- `jobs/worker.py`
- `media/video_writer.py`
- `server/app.py`
- `server/contracts.py`
- `server/routes.py`
- `tests/test_stage12_governor.py`
- `tests/test_stage12_worker.py`
- `tests/test_runtime_policy.py`
- `tools/cache_cleanup.py`
- `audit/STAGE12_REPORT.md`

## Next gate

Stage 12 deterministic integration and source/model isolation gates pass. A
real-media render and hardware-specific performance campaign remain separate
evidence gates; no quality, throughput, or VRAM benchmark claim is made here.
