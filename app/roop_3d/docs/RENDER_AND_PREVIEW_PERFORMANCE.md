# Render and preview performance (Nov 12 2026)

What changed, what it measured, and the switches. All are environment variables read at call time; defaults are the
recommended settings.

## Measured on `jennifer.mp4` with the `harjot` FaceSet (RTX 4070 12 GB, `reference_onnx`)

| | before | after |
|---|---|---|
| async pipeline on hard angles | crashed (CUDA 906 -> access violation) | completes |
| `angle_scan` for a 6 s trim of a 428 s file | ~7 min (whole file) | ~8 s (window only) |
| warm live preview | 43-68 s | 6-9 s (first one after start: ~55-70 s) |
| 3D projection step per frame | 9-15 s (CPU) | ~1 s (CUDA) |
| steady render, frontal 720p | ~1.9 fps | ~2.4 fps |
| GPU utilisation while rendering | ~15-20 % | ~14-23 % mean (p90 44-58 %) |

**GPU utilisation is not above 90 %.** The composite stage (restorer round trip, sharpness-matching and sensor-physics
blurs, temporal detail) is a chain of dependent CPU image operations of ~400-900 ms per frame, and OpenCV here has no
CUDA build (no torch / cupy either). It is the pipeline's limit; the GPU stages (swap ~30 ms, detection ~60 ms, restorer
~120 ms) finish long before it. Reaching 90 % needs several frames composited concurrently (chunked rendering with
per-chunk temporal state) or a GPU image library; neither was attempted.

## Switches

| variable | default | effect |
|---|---|---|
| `ROOP3D_CUDA_GRAPH` | `0` | `1` re-enables TensorRT CUDA-graph replay (unsafe with the multi-threaded pipeline) |
| `ROOP3D_PREVIEW_SESSION_POOL` | `1` | keep ORT sessions warm between live previews; emptied when a job/compile takes the GPU |
| `ROOP3D_PREVIEW_PREROLL` | `6` | frames before the target that run the full swap in a replayed preview (older ones only track); `0` = all |
| `ROOP3D_GEOMETRY_PROVIDER` | `auto` | dense-face geometry graphs: `auto` = CUDA on 12 GB, CPU on 6 GB/CPU; `cpu` / `cuda` force |
| `ROOP3D_AUTO_3D_YAW` / `ROOP3D_AUTO_3D_PITCH` | `30` / `25` | degrees beyond which a face uses the dense 3D path; `180` disables |
| `ROOP3D_PARSER_CACHE` | `1` | cache BiSeNet parses by image content and prefetch them from the swap stage |
| `ROOP3D_ORT_LOG_LEVEL` | `3` | ONNX Runtime global log level (0 verbose .. 4 fatal) |

## Behaviour notes

* A trimmed job analyses and swaps only frames with `start <= t < end`; progress counts the window.
* Dense 3D replacements use a five-point chin line instead of the 106-point submental jaw cutoff (see the second pass
  below); small eye-socket holes in the mesh silhouette are filled and only confident parser hair/neck/ear evidence
  is subtracted.
* The audio sync check now anchors each decoded frame on its own presentation time (a source whose first AAC frame
  reports pts 512 used to fail every trimmed job with `OUTPUT_AV_DESYNC`).
* Known: the muxer still decodes the source from its start up to the trim start (a 194 s offset adds minutes to the
  encode step); a swapped profile silhouette is stair-stepped; frames with extreme pitch plus motion blur still show
  artifacts; harjot's FaceSet has no up/down views.

## Live preview on the CPU (CPU 90 %, GPU 2 %) - Nov 12 2026, second pass

Cause: `VRAMGovernor.try_acquire` runs the **whole** preview on the CPU profile (swapper, 3D geometry and restorer
included) whenever live free VRAM is below `PER_STREAM_RESERVATION_MIB` (4096) plus the 768 MiB safety reserve.  A
long-running server that has loaded its models (LivePortrait, TensorRT engines, warm preview sessions) easily sits under
that line on a 12 GB card, so previews silently fell to `inswapper_128 -> CPU`.

Fixes:
* admission counts VRAM held by idle warm-preview sessions as free (`ort_session.reclaimable_vram_mib()`, measured with
  NVML per pooled session; the pool is released when a job/compile takes the GPU anyway);
* if admission still chooses the CPU, the preview releases its warm sessions and asks once more, and prints
  `[ADMISSION] preview is running on the CPU` with the reason when it really has to;
* `probe_hardware()` no longer spawns ~12 subprocesses per call (nvidia-smi x3, `ffmpeg -encoders/-decoders`, ...):
  static facts are reused for `ROOP3D_HARDWARE_PROBE_TTL` seconds (default 120) while free VRAM is re-read from NVML on
  every call; unchanged model files are not re-hashed for `ROOP3D_VERIFY_MEMO_SECONDS` (default 600; changed size, mtime
  or inode always re-hashes);
* the session pool holds 64 sessions (alternating render modes thrashed 24).

Swap quality in 3D modes (checked on a 5 min clip with harjot):
* dense 3D replacements use a five-point chin line instead of the 106-point submental cutoff at every yaw (the cutoff
  drew a diagonal wedge across the cheek already at 30 degrees);
* heads rolled more than `ROOP3D_3D_MAX_ROLL` (40) degrees use the standard 2D swap: the dense mesh fit produced a
  distorted face on a head lying on a pillow, the compositor masked it away, and nothing was swapped.

Still open: leftover lipstick slivers at the mouth corners; the dense path keeps the source's closed mouth when the
target mouth is open; ~1 s of CPU NumPy per dense frame (`correspondence_field`, `project`).
