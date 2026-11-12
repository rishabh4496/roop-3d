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
* Turned faces (landmark turn ratio < 0.7, about 40 degrees) on the dense 3D path skip the 106-point submental jaw
  cutoff (its contour collapses) and use a five-point chin line instead; small eye-socket holes in the mesh silhouette
  are filled and only confident parser hair/neck/ear evidence is subtracted.
* The audio sync check now anchors each decoded frame on its own presentation time (a source whose first AAC frame
  reports pts 512 used to fail every trimmed job with `OUTPUT_AV_DESYNC`).
* Known: the muxer still decodes the source from its start up to the trim start (a 194 s offset adds minutes to the
  encode step); a swapped profile silhouette is stair-stepped; frames with extreme pitch plus motion blur still show
  artifacts; harjot's FaceSet has no up/down views.
