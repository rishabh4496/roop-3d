# Stage 06: tracker, identity swapper and LivePortrait reference pipeline

**Status: passed for the local reference pipeline.** All changes are inside the
standalone application's `app/roop_3d/` package. No launchers, root policy files,
dependency locks, legacy source/model files, or legacy Git state were changed.
Changes are uncommitted; no push was requested.

## Delivered behavior

- `engine/runtime.py`: lazy read-only ONNX sessions, model hashes at preparation,
  file-stat guards during inference, observed graph/name/shape/dtype assertions,
  finite tensor checks, bounded CPU threads and distinct 6 GB/12 GB GPU budgets
  capped by verified free memory. Model load is an explicit session action.
- `engine/detection/dense_landmarks.py`: actual verified 106- and 203-point
  inference. Five-point landmarks come from SCRFD and 512-D unit embeddings from
  ArcFace. Yaw/pitch/roll are explicitly approximate geometric estimates.
- `engine/detection/face_tracker.py`: IoU plus cosine association, global Hungarian
  assignment with unmatched options, monotonic integer IDs, missed-track retention,
  scene-cut reset without ID reuse, and bounded state (32 simultaneous tracks).
- `engine/volumetric/preprocess.py`: deterministic canonical crops at 128/256/512
  and paired double-precision 2x3 affine/inverse matrices. Portrait crops use
  observed landmarks and preserve the crop-to-frame map.
- `identity_swapper.py`: actual local INSwapper inference, verified buff2fs
  projection, canonical FaceSet embedding with bounded pose-matched view blending,
  RGB normalization and BGR output. Only the projection tensor is materialized by
  the lightweight protobuf reader; the ONNX Python package is not required.
- `reference_adapter.py`: actual appearance volume, source/driving motion,
  eye retargeting, stitching, and fused 3D warp/SPADE generation. A track's initial
  identity-swapped portrait supplies its cached appearance; per-frame motion and
  expression deltas drive that reference. Cache entries are evicted with tracks.
- `session.py`: `prepare_session()`, `infer_frame()`, `release_session()`,
  track selection, lifecycle cleanup and per-call detection/swap/retarget/total
  latency. Returns aligned crops and exact inverse matrices for Stage 09.
- The existing `POST /api/v1/media/inspect-faces` now uses real IoU/ArcFace
  tracking and provides selectable integer `persistent_id` values. Existing
  string `track_id` job labels remain compatible. Scene IDs, landmark count,
  recognizer hash and honest pose/continuity limits are included.

The original reference ONNX graphs are preserved. TensorRT graphs/engines and
Stage 07 optimizations were not introduced.

## Verification

All commands ran from this checkout's `app/` directory in its dedicated environment.

| Exact command | Exit | Final result |
| --- | ---: | --- |
| `roop_3d/.venv/Scripts/python.exe -B -m pytest -c roop_3d/pyproject.toml --confcutdir=roop_3d roop_3d/tests/test_runtime_policy.py -q` | 0 | 6 passed |
| `roop_3d/.venv/Scripts/python.exe -B -m pytest -c roop_3d/pyproject.toml --confcutdir=roop_3d roop_3d/tests/test_stage06_pipeline.py -q` | 0 | 12 passed; 63.56 s |
| `roop_3d/.venv/Scripts/python.exe -B -m pytest -c roop_3d/pyproject.toml --confcutdir=roop_3d roop_3d/tests -q` | 0 | **253 passed, 18 subtests passed; 0 failed/skipped/blocked**; 105.50 s |
| `roop_3d/.venv/Scripts/python.exe -B -u -m roop_3d.tests.manual_stage06_acceptance --provider cuda --profile 6gb` | 0 | **100 real frames** plus 3 warmup frames; output/audio/identity/memory assertions passed |
| `roop_3d/.venv/Scripts/python.exe -B roop_3d/audit/verify_isolation.py --report` | 0 | **68,391 comprehensive + 2,646 historical immutable checks; zero violations** |
| `git diff --check` | 0 | No whitespace errors |

The one warning is the existing FastAPI/Starlette TestClient deprecation.
Tests use real installed models and two local FaceSets. Analytic assignment and
matrix tests exercise the mathematical units directly. No core inference model
was replaced by a mock, no core tests were skipped, and no weights/videos were
downloaded. The missing-portrait case uses a real registry without the optional
records and still executes the real identity swapper.

The tests cover two-person crossing, missed detections, scene cuts, exact inverse
roundtrips, tensor failures, identity improvement, optional-weight fallback,
lifecycle cleanup and the actual inspection endpoint. The full run history,
earlier interruption and the resource failure that led to CPU fallback are
retained in `STAGE06_VALIDATION.json`.

## Completed 100-frame evidence

The campaign generates a local FFV1 target clip with non-silent PCM audio, then
runs tracking, swapping and LivePortrait on every decoded frame through Stage 05's
safe writer. Every final encoded frame is decoded again for PTS/count verification;
the audio track is preserved by packet copy. No cached output image substitutes
for frame inference. INSwapper and warp/SPADE each completed 103 successful calls.

- Source cosine: **0.076498373 -> 0.887024585**.
- Process-private memory growth between matched in-flight windows:
  **10,618,880 bytes (10.127 MiB)**; gate <= 32 MiB.
- Retained Python growth: **524,910 bytes
  (0.501 MiB)**; gate <= 2 MiB.
- One cached appearance track throughout; each feature volume is 8,388,608 bytes.
- Memory windows are zero-based frames 20..39 and 80..99. Both windows contain the
  same decoder/encoder frame buffers; a cold pre-encode sample would omit those
  fixed live allocations. The original thresholds were retained.
- The campaign took 832.040 seconds on this shared machine.
  Per-stage observations are recorded; no general speed or fidelity claim is made.

This establishes bounded observed growth over this campaign, not a proof for
arbitrary inputs or durations. Per-process dedicated VRAM is unavailable under
this machine's WDDM reporting; dedicated VRAM leak/peak evidence remains UNKNOWN.
CUDA arena budgets and application caches are bounded separately.

Aligned preview: `roop_3d/outputs/stage06-cuda-6gb-aligned-preview-1793372204625447000.mov`.
Machine-local detailed evidence:
`roop_3d/.runtime/acceptance/stage06-cuda-6gb.json`.
The output sidecar records exact source/output SHA-256, timing, audio decisions
and local synthetic-media provenance. The preview is an aligned portrait clip;
the full-frame Stage 09 compositor is not claimed.

## Hardware and optional capabilities

The 6 GB budget campaign ran on a physical **12 GB RTX 4070**, with an actual
free-memory-aware arena budget of 3133 MiB.
INSwapper, motion, stitching and eye inference used CUDA. Appearance extraction
used CPU after an arena allocation limit; the unchanged warp/SPADE graph used CPU
because pinned ORT's CUDA GridSample rejects 5-D input. Capability reports retain
both requested and actual providers and the explicit fallback reasons.

A separate **12 GB profile one-frame functional check passed**, with source cosine
0.076498373 -> 0.886831376.
It is not a second 100-frame campaign or a physical 6 GB qualification. Its exact
stdin script and results are in the validation JSON.

Verified separate lip-retarget weights were not found. Mouth articulation is
driven by motion-extractor expression deltas; the learned lip component reports
UNKNOWN. Every installed reference component listed above executed successfully.
Artifact hashes, inference, provenance and rights remain separate evidence states;
commercial permission/provenance is not inferred from local presence.

## Authorized runtime policy change

An earlier audit halted Stage 06 when legacy `app/queue.json` changed. The user's
subsequent instruction, ?do what you feel best?, authorized the proposed one-path
runtime exception. Only that exact path was added to `RUNTIME_FILES`. It is now
reported as runtime activity under the existing policy. Neighboring paths and
source/model files remain protected; a new test proves the exception is exact
and does not mutate its input baseline.

Both preserved baseline files remain byte-for-byte identical, independently
rehashed before and after the change:

- `protected_paths.json`: `035e9d2cd27c25df01a0cd02906e4fbdd661da9b0aa2d9b2a60afb584eaf8707`
- `protected_all_paths.json`: `2c1c95d4fd120fd2f96137938a71eec52bf25c4822c05afbf3b3e0b6cc188b96`

The one-file reductions in immutable counts are the explicit queue classification.
The final audit separately observed 1480 runtime files; they are not claimed
byte-identical. No legacy queue contents or legacy application process were changed.

## Session use and next gate

```python
from roop_3d.engine.volumetric.session import prepare_session, infer_frame, release_session

session = prepare_session(faceset_document, track_ids={0}, retarget="auto")
try:
    result = infer_frame(session, upright_bgr_frame)
    # result.replacements carry model crops and their 2x3 crop-to-frame inverse.
    # result.latency carries detection_ms, swap_ms, retarget_ms, total_ms.
finally:
    release_session(session)
```

The Python inference API and inspection REST backend are available. Queue rendering
worker, frontend, calibrated pose reconstruction, and full-frame segmentation/
occlusion composition remain outside this stage. Stage 07 awaits its numbered
specification; the Stage 06 reference path is ready to preserve.

## Changed files

- `app/roop_3d/engine/runtime.py`
- `app/roop_3d/engine/detection/dense_landmarks.py`
- `app/roop_3d/engine/detection/face_tracker.py`
- `app/roop_3d/engine/volumetric/preprocess.py`
- `app/roop_3d/engine/volumetric/identity_swapper.py`
- `app/roop_3d/engine/volumetric/reference_adapter.py`
- `app/roop_3d/engine/volumetric/session.py`
- `app/roop_3d/media/service.py`
- `app/roop_3d/media/schema.py`
- `app/roop_3d/tests/test_stage06_pipeline.py`
- `app/roop_3d/tests/manual_stage06_acceptance.py`
- `app/roop_3d/audit/STAGE06_REPORT.md`
- `app/roop_3d/audit/STAGE06_VALIDATION.json`
- `app/roop_3d/audit/runtime_policy.py`
- `app/roop_3d/tests/test_runtime_policy.py`


## Primary contract references

Implementation was checked against official InsightFace
[INSwapper](https://raw.githubusercontent.com/deepinsight/insightface/master/python-package/insightface/model_zoo/inswapper.py),
[alignment](https://raw.githubusercontent.com/deepinsight/insightface/master/python-package/insightface/utils/face_align.py),
[landmarks](https://raw.githubusercontent.com/deepinsight/insightface/master/python-package/insightface/model_zoo/landmark.py),
and LivePortrait [wrapper](https://raw.githubusercontent.com/KwaiVGI/LivePortrait/main/src/live_portrait_wrapper.py),
[camera](https://raw.githubusercontent.com/KwaiVGI/LivePortrait/main/src/utils/camera.py),
[human landmarks](https://raw.githubusercontent.com/KwaiVGI/LivePortrait/main/src/utils/human_landmark_runner.py),
[crop geometry](https://raw.githubusercontent.com/KwaiVGI/LivePortrait/main/src/utils/crop.py),
and [ratios](https://raw.githubusercontent.com/KwaiVGI/LivePortrait/main/src/utils/retargeting_utils.py).
Installed local hashes/graphs and actual output remain authoritative for these artifacts.
