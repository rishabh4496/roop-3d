# Asynchronous multi-stream render pipeline and binary live transport (Defect #3)

## What was wrong

Measured on the reference render (720p–4K clips, real TensorRT models, RTX 4070), the engine ran one frame at a time:
decode → `infer_frame` → `composite_frame` → PNG write, each waiting for the previous one. Two further things made it
impossible to fix with threads alone:

* `OrtModel.run`, `Detector.detect`, `infer_frame` and friends were wrapped in `@guarded`, a **single process-wide
  `RLock`**. Every ONNX call in the process was serialised, whichever thread or model it came from.
* `infer_frame` and `composite_frame` shared `ReferencePipeline.lock`, and `composite_frame` read
  `frames_processed`, which is whatever frame the swap stage has reached, not the frame being composited.

Baseline (serial, 1080p, 60 frames): **2.07 fps, GPU 51 % mean / 5 % p10, ≥ 80 % in 39 % of 200 ms samples**.
Inside `infer_frame`, LivePortrait retargeting was ~330 ms of ~375 ms.

## Architecture

```
 decode ─▶ track ─▶ swap ─▶ composite ─▶ write ─▶ results()
 NVDEC    stream A  stream B  stream C    PNG       progress / checkpoint / live hub
 (PyAV)   thread    thread    thread      thread    (caller's thread)
```

| Stage | Code | State it owns |
|---|---|---|
| decode | `VideoReader` (PyAV, `HWAccel("cuda")` = NVDEC, software fallback) into a `PinnedFramePool` | reader |
| track (stage 1) | `ReferencePipeline.track_frame` | tracker (`track_lock`) |
| swap (stage 2) | `ReferencePipeline.infer_frame(tracked=…)` | swapper, portrait, 3D rasteriser (`lock`) |
| composite (stage 3) | `ReferencePipeline.composite_frame(frame_index=…, timestamp_seconds=…)` | compositor, lip-sync (`composite_lock`) |
| write | `_save_frame` | PNG checkpoint sequence |

* **Bounded queues** (`BoundedQueue`, default 8, `ROOP3D_PIPELINE_QUEUE`) hold their mutex only for O(1) `deque`
  operations and block producers at capacity. They are not lock-free: that is not expressible in CPython, and the mutex is
  uncontended in this single-producer/single-consumer layout.
* **Pinned host memory**: decoded frames are staged in `PinnedFramePool` (`torch.empty(pin_memory=True)`, i.e.
  `cudaHostAlloc` through PyTorch's host allocator). The pool is also the global in-flight bound: when every buffer is out,
  the decoder blocks.
* **Streams and events**: each compute stage owns a non-default `torch.cuda.Stream`. A stage records an event after its
  last operation; the next stage's stream does `wait_event` (GPU-side) instead of any `cudaDeviceSynchronize`. A host-only
  stage waits on that one event. CUDA tensors crossing streams are `record_stream`-ed. **Honest scope**: the stage bodies
  are mostly ONNX Runtime and OpenCV calls, which run on ORT's own streams and the CPU; the torch streams order the torch
  work (3D pose projection, GPU compositor) and the event hand-off. The overlap that matters is measured, below.
* **Ordering**: one worker per stage + FIFO queues = frames leave in order, which the temporal trackers require.
* **`TrackedFrame`**: everything stage 2 used to read back from the tracker (scene cut, live track ids, frame time) travels
  with the frame, so the tracker can run ahead.
* **`SharedExclusiveFence`** replaces the global `RLock`. Inference (`@inference`) takes the shared side; model
  construction, release, `evict_for_build`, TensorRT builds and every ORT session creation (`SessionManager.create`,
  lazy CPU sessions, `_switch_to_cuda/cpu`) take the exclusive side and still wait for all in-flight inference. It is
  re-entrant like an `RLock`, and a shared→exclusive upgrade parks the caller's shared holds so two upgraders cannot
  deadlock. Each model instance additionally serialises its own `run` (its I/O binding is mutable state); *different*
  models run concurrently. Loading CUDA/TensorRT libraries while another thread is mid-inference crashes the process
  (access violation), which is why session creation is exclusive.

Switches: `ROOP3D_ASYNC_PIPELINE=0` runs the same four stage functions inline on one thread; `ROOP3D_PINNED_FRAMES=0`
disables pinned staging. A session without `track_frame` (a stub, an older engine) always runs inline.

## Live transport

`WS /ws/v1/preview/stream` carries **binary** messages: a 16-byte little-endian header followed by a JPEG.

| bytes | field |
|---|---|
| 0–3 | frame index (u32) |
| 4–7 | media timestamp, ms (u32) |
| 8–13 | stage 1 / 2 / 3 wall latency, 0.1 ms units (3 × u16, saturating) |
| 14–15 | flags: bits 0–3 version (1), 4 JPEG, 5 frames dropped before this one, 6 last, 7 cached, 8–9 source (0 job, 1 preview) |

* Encoding happens on `LiveFrameHub`'s own thread, **never on an inference thread**. `publish()` is a memcpy into a
  double-buffered staging slot; the newest frame wins and drops are counted and flagged. With no viewer connected it
  does nothing.
* JPEG backend order: **nvJPEG** (`torchvision.io.encode_jpeg` on a private CUDA stream, optional GPU downscale to
  `max_edge`) → PyTurboJPEG → OpenCV's bundled libjpeg-turbo. The active backend is reported. Measured 1080p worst-case
  noise: nvJPEG 0.94 ms, libjpeg-turbo 8.7 ms.
* Interactive frames: send `{"op":"render","id":n,"config":<hash>,"frame_index":i}` for a config POSTed to `/api/preview`;
  the reply is a binary frame for that socket only. A newer request supersedes an unfinished one.
* `LivePreviewService._render_one` no longer encodes a PNG while holding the render slot: the cache entry keeps the raw
  frame and encodes PNG lazily, in a worker thread, only for HTTP clients. (`?response=json` still returns base64 for API
  compatibility; the UI does not use it.)
* Browser: `src/liveStream.ts` (`createImageBitmap`, latest-wins decode) → `LiveStreamCanvas` (`bitmaprenderer`
  transfer). It causes **zero React renders per frame** (asserted in `liveStream.test.tsx`); the readout updates at 4 Hz
  through a ref.
  The existing slider/scrub preview still uses `POST /api/preview`; it can adopt the `render` op without server changes.

## Telemetry

`GET /api/v1/system/pipeline` (polled at 1 Hz by the Diagnostics HUD while it is open):
per-stage queue depth / capacity / high-water, wall ms, CUDA-event stream ms, busy %, starved % (waiting for upstream),
blocked % (back-pressure), the bottleneck stage, the fraction of time ≥ 2 / ≥ 3 stages were active, pinned-pool occupancy,
GPU / CPU / NVENC / NVDEC utilisation, PCIe TX/RX MB/s, VRAM, live-stream encoder stats, and the inference-fence state. GPU
numbers come from NVML via `ctypes` (no `pynvml`), with `nvidia-smi` as the fallback; a value the driver does not report
is `null`, never `0`.

## Benchmark

```
roop_3d/.venv/Scripts/python.exe -m roop_3d.tools.benchmark_async_pipeline CLIP.mp4 --frames 60 --mode both --verify 20
```

Runs the production `ReferenceStages` serially and through the pipeline, samples NVML during the steady state, and
compares the first N output frames pixel by pixel. Results on this machine (RTX 4070, TensorRT, real models, 720p source
re-encoded to each size):

| | serial | async | |
|---|---|---|---|
| **1080p** fps | 2.07 | **3.93** | ×1.90 |
| GPU util mean / p10 / samples ≥ 80 % | 50.7 / 5 / 39 % | **91.3 / 78 / 83 %** | target met |
| **2160p** fps | 1.40 | **3.65** | ×2.61 |
| GPU util mean / p10 / samples ≥ 80 % | 35.3 / 5 / 26 % | **83.7 / 56 / 69 %** | mean > 80, not sustained |
| output equivalence | | 30/30 frames bit-identical | max abs diff 0 |

Stages overlapped (≥ 3 active simultaneously) 61 % (1080p) and 75 % (2160p) of the time.

## What this does not do

* **Not real time.** The slowest stage now sets the pace: `swap` (~270–300 ms/frame; LivePortrait was ~330 of 375 ms of `infer_frame` in the 720p serial profile, and its
  convolutions log ORT CUDA-EP "Fallback mode" warnings) and, at 4K, `composite` (~300 ms, CPU-bound OpenCV/NumPy). Throughput
  is ≈ 1 / max(stage), here 3.6–3.9 fps against the 25–30 fps target.
* GPU "utilisation" is the share of time any kernel is running, not SM occupancy. It can fall when a stage gets faster
  and the pipeline becomes CPU-bound.
* The reference renderer still writes a PNG per frame (resumable checkpoints) and muxes afterwards; NVENC is used by that
  later mux and by the fast stream path, not per frame from CUDA buffers. NVDEC decode goes through PyAV to host memory;
  `PyNvVideoCodec` (device-resident decode/encode, already supported by `nv_video_codec.py`) is not installed here.
* The fast stream path (`StreamingProcessor`, inswapper-only) keeps its existing decode-thread / worker-pool / writer
  structure; it gained the live-hub tap and shares the new inference fence, nothing more.
* `torchvision` and `pillow` were added to the venv (`--no-deps`) for nvJPEG; without them the hub falls back to
  libjpeg-turbo automatically.
