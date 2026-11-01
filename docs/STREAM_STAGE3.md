# Stage 3 streaming remediation

All implementations are in the standalone `app/roop_3d/` package. The prompt's
`app/roop/*.py` module names map to that package; benchmark routes are mounted
from `roop_3d/server/api.py`. No legacy application modules or lifecycle APIs are
used. External model weights remain hash-verified, read-only registry inputs.

## Execution and selection

`render_backend` accepts `auto`, `stream`, or `reference`. In `auto`, ordinary
identity-swap jobs admitted on NVIDIA use the stream runner. CPU jobs, timeline
edits, trims, manual occlusion masks, LivePortrait/pose settings, VFX/finishing
exports, unsupported timing/color layouts, and audio-transcode requests retain
the existing reference path. Explicit `stream` rejects unsupported features.
The reference path remains available for fidelity comparisons and its existing
frame-checkpoint semantics; it is not silently advertised as using the new ring.

The streaming path requires upright SDR video with square pixels, zero-based
presentation timestamps and an exact CFR packet grid. The complete packet grid
is checked, including gaps and duplicate PTS, rather than trusting a short timing
sample. VFR, HDR, rotations and non-square pixels are sent to the reference path
by automatic selection. Explicit stream selection rejects them.

Relevant settings are `execution_threads`, `stream_batch_size`,
`stream_ring_slots`, `stream_segment_frames`, `sparse_interval`,
`restorer_threads`, `restorer_pool_size`, `ultramax_native_bypass` and
`ultramax_min_pixels`. They have bounded, validated values in the existing
`GET/PUT /api/v1/config` contract. Read the configuration, change selected fields,
and submit the complete resulting object. Settings do not load models at API boot.

## Three stages

1. `STAGE_1_SPARSE_PRESCAN` samples keyframes and fixed sparse intervals for face
   presence, merging detections into padded active ranges. Its skip percentage
   is explicitly a sparse estimate. A face that appears entirely between samples
   can be missed; choose `sparse_interval=1` for exhaustive presence sampling.
2. `STAGE_2_ACTIVE_PREPASS` runs the real tracker inside those ranges and locks
   selected track IDs, bounding boxes and five-point landmarks. The checkpoint is
   `roop_3d/.runtime/prepass/<project_id>.stage2.npz`, containing numeric arrays,
   source/request/settings/model identity, and a content digest. Object arrays,
   invalid NumPy shape/size headers, archive expansion, bounds and mismatched
   identities are rejected. A completed checkpoint restores as `RESTORED` after
   an interrupted render. An interrupted, incomplete prepass is recomputed.
3. `STAGE_3_MAIN_SWAP` batches consecutive verified single-face frames. Multi-face
   frames use face batches. Non-target frames pass through without swapper calls.
   Each worker has its own input/output bindings and buffers and activates the
   CUDA runtime's primary context on its executing thread. Output order remains
   source order even if inference futures complete out of order.

`active_ranges_count` measures candidate ranges. `verified_target_frames` counts
frames with verified selected targets inside those ranges. The final
`skipped_frames_pct` counts frames passed through by this policy; it is not a
proof that unsampled areas contain no face. The API publishes stage transitions
and these measurements on both existing pipeline WebSocket routes. Reference
jobs continue to report their earlier stage-1/2 `SKIPPED` states.

The GPU path uses generated ordinary ONNX `MatMul`/`GridSample`/composite graphs
on CUDA. ORT I/O binding retains equal-shape device allocations and passes
OrtValues between crop, batched swap and composition without downloading those
intermediates. A host frame is uploaded once and the final result is downloaded
for the encoder. The 6 GB and 12 GB affine budgets are separate and divided by
the configured worker count. Unsupported frame-size/thread combinations fail
before affine allocation; reduce thread count rather than overstate capacity.
TensorRT selection invokes Stage 2's strict builder/canaries. The `.batch.onnx`
swapper alias retains the default FP32 overflow guard.
CUDA activation also directs native/JIT caches and temporary files into this
checkout. Registered batched swappers and restorers have explicit arena shares;
they do not inherit the tiny generic auxiliary-model allocation.

## Transport and codecs

`SharedMemoryFrameRing` is a bounded single-producer/single-consumer transport
using `multiprocessing.shared_memory`, semaphores and fixed-width metadata. It
supports actual spawned Python processes. Only synchronization handles are
serialized at process startup; no frame or per-frame metadata is pickled. A
read-only lease owns a slot until released. The render coordinator makes bounded
host copies for asynchronous thread ownership, so this is zero-pickle transport,
not a claim that all host memory copies were eliminated. Frame payloads never
pass through a multiprocessing Queue or a disk image loop.

`NvdecFrameSource` requests CUDA hardware surfaces and contains one explicit
`hwdownload,format=nv12,format=bgr24` boundary in the FFmpeg filter chain.
`ROOP_STRICT_NVDEC=1` fails if that source path cannot execute, including prepass
and output validation. Without strict mode it can fall back before the first
emitted frame to FFmpeg software decode and then OpenCV. A failure after emitting
frames never restarts the video and duplicates its beginning. Python ring and
FFmpeg subprocesses are owned by this job; no other port owner/process is killed.

`NVHardwareVideoWriter` accepts `h264_nvenc` or `hevc_nvenc`. It tests actual
encoder execution at the intended frame dimensions, rather than treating an
advertised encoder name as execution proof. Without strict mode a failed smoke
test selects `libx264` before any frame is submitted. `ROOP_STRICT_NVENC=1`
prohibits that fallback. Small dimensions unsupported by a specific encoder are
therefore rejected in strict mode. Optional `lut_path` and `lut_bake` bake a
bounded, owned `.cube` LUT during encoding.

Segments have a nanosecond run timestamp, UUID and monotonically increasing
index. A run-specific ffconcat list and JSON index record every segment. Concat
uses `-c copy`; final audio remux copies all source audio tracks and the encoded
video with `-c copy`. Incompatible audio/container combinations fail instead of
silently transcoding. Cancellation stops owned producers/workers and removes a
newly published incomplete output. Completed Stage 2 metadata survives render
interruption; main rendering currently restarts into a new unique segment run.

## Projection and restoration

The existing identity model loader caches its verified `buff2fs` projection at
`app/models/inswapper_128_emap.npy`, with a model-hash and array-hash receipt.
Later calls load that numeric array with `allow_pickle=False`, without reparsing
the model protobuf for the projection. Model verification and normal inference
session initialization still occur; this does not bypass weight integrity.

The detached restorer pool gives each thread its own lazily loaded model and
CUDA activation, with bounded queued work. `gpen` and `codeformer` use different
upstream alignment references and RGB normalization. CodeFormer keeps its scalar
DOUBLE fidelity tensor live; the runtime now preserves scalar rank and FLOAT64
contracts. The alignment coordinates reproduce upstream preprocessing, not an
empirical guarantee of production-model fidelity. Existing reference restoration
remains available for comparison. Detached restoration currently uses a declared
crop-sized host bridge; it does not claim a fully device-only enhancer path.

The preserved UltraMax processor was inspected read only, without importing or
executing it. It identifies UltraMax as the existing `codeformer.fp16.onnx`
weights on an FFHQ-512 aligned RGB `[-1,1]` path, with a live FLOAT64 fidelity
input. This standalone adapter uses that exact registered model identity,
type-specific normalization lookup, retained I/O binding/output buffer, and
source-chrominance/network-luminance composition. It does not inherit the legacy
application's cosmetic eye/detail operators or its performance measurements.

Its configurable native-size bypass uses the original detected box clipped to
the native frame, before upscaling: a crop whose smaller side is at most 320
pixels returns before model construction, alignment or queue submission. The
bypass can be disabled. An optional owned
`roop_3d/.runtime/restorers/ultramax.json` can override the documented contract
with `model_name`, five-point `template`, `size`, `normalization` and nonempty
`evidence` fields. Templates use 512-pixel reference coordinates and scale to
`size`. Overrides still require registered weights and the normal identity/rights
checks. No override is needed for the observed CodeFormer-fp16 identity.

The mask watchdog keys baselines by thread, track and mask kind. Geometric,
occluder and restoration masks have independent histories; an occluder is never
compared to a geometric-face baseline.

## Thread benchmark API

`POST /api/settings/benchmark_threads` accepts `{"profile":"quick"}` or
`{"profile":"full"}` and returns HTTP 202. `GET /api/settings/benchmark_status`
returns `IDLE`, `RUNNING`, `DONE`, `FAILED` or `CANCELLED`, measurements and the
best measured thread/pool configuration. Active jobs or another benchmark cause
HTTP 409. No benchmark runs at API boot or changes the saved configuration.

The measured workload is synthetic affine I/O binding and host/device transfers,
using CUDA when local hardware is observed, otherwise CPU. Every worker warms
before timing starts. Results identify the provider and workload scope. They are
not universal recommendations for a particular swapper/restorer combination and
are not end-to-end video FPS. Production tuning still needs representative media,
the actual models, and separate 6 GB/12 GB campaigns.

## Verification

From the checkout root:

```powershell
& app/roop_3d/.venv/Scripts/python.exe -B verify_integration.py
& app/roop_3d/.venv/Scripts/python.exe -B verify_integration.py --hardware
```

The base gate executes a real synthetic clip, actual ORT CPU affine operators,
an actual spawned-process ring, emap cache reuse without a second protobuf read,
multi-frame batching, checkpoint restoration, and uniquely indexed stream-copy
segments. Albumentations UserWarnings are suppressed without suppressing our
own validation errors. It exits 0 only when all base checks pass.

The optional native gate requires real CUDA, NVDEC and NVENC. Locally on
2026-11-01 it passed: three CUDA workers; zero uint8 pixel error against the CPU
synthetic affine reference; a coupled three-stage native 10-frame synthetic job;
LUT baking; byte-identical video and audio packet payloads through remuxing.
This is execution evidence for the synthetic workload, not production neural
fidelity, speedup or video FPS. GPEN/CodeFormer/UltraMax production output fidelity
still needs a model-specific, hardware-specific real-output acceptance campaign.

Additional local verification on the same date:

- Full suite: 532 tests and 18 subtests passed; the reference golden-video test
  remained green.
- Read-only integrity audit: 68,390 protected files matched, zero violations;
  permitted legacy runtime activity was classified separately.
- Registered INSwapper batch model: CUDA outputs finite at batches 1 and 4 on
  synthetic input. Registered GPEN, CodeFormer and UltraMax adapters also produced
  valid synthetic-input outputs on CUDA. These are execution/finite-value checks,
  not visual-fidelity measurements.
- Thread benchmark API: native quick profile completed six configurations on
  `CUDAExecutionProvider`; recommendations remain scoped to its synthetic workload.

Primary implementation references: [NVIDIA FFmpeg hardware pipeline](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/ffmpeg-with-nvidia-gpu/index.html),
[ORT I/O binding](https://onnxruntime.ai/docs/performance/tune-performance/iobinding.html),
[GPEN alignment](https://github.com/yangxy/GPEN/blob/main/align_faces.py),
[GPEN crop configuration](https://github.com/yangxy/GPEN/blob/main/face_enhancement.py),
[CodeFormer alignment](https://github.com/sczhou/CodeFormer/blob/master/facelib/utils/face_restoration_helper.py).
