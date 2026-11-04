# GPU troubleshooting guide

Roop 3D uses an NVIDIA GPU when it can and the CPU when it cannot. This guide helps you find out which it is using, why, and what to change.

## 1. Find out what you have

1. **Header hardware pill.** *TRT* = TensorRT, *CUDA* = CUDA, *CPU* = processor only; next to it, video memory used / total and *IDLE* or *RUNNING*. Open
   it for the active provider, GPU memory use, throughput and a bottleneck hint.
2. **Settings → Environment & Offline Manager → Re-Run Preflight.** The checks that matter for the GPU are *NVIDIA GPU, NVIDIA Driver, CUDA Compatibility,
   ONNX Runtime, TensorRT, FFmpeg Codecs*. Each shows PASS / WARN / FAIL and a message that names what it found (GPU name, VRAM, driver, the ONNX
   Runtime providers available, the NVENC/NVDEC codecs registered).
3. **The profile badge** (`12GB`, `6GB`, `CPU`) is the hardware profile Roop 3D resolved.
4. **Settings → Workstation Preferences → Compute** shows *Active Profile* and *Resolved*, and explains the active profile in words.

From a terminal: `nvidia-smi` (driver and memory) and `python -B tools/project_admin.py preflight -v` in the project folder.

## 2. Hardware profiles

| Profile | For | What Roop 3D does |
|---|---|---|
| `12gb` | NVIDIA GPUs with **10 GB or more** of video memory (12 GB recommended): RTX 4070/4070 Ti/4080/4090, RTX 3080 12GB, 3090 and similar | Keeps models resident for the fastest previews; batches several faces (up to 8); up to about 9.5 GB of GPU memory before it tidies its caches; high-throughput NVENC encoding (AV1 / HEVC) |
| `6gb` | GPUs with **5 GB or more**, typically laptops: RTX 3060/4050/4060 laptop, RTX 2060 6GB | Limits itself to about 4.6 GB; works one stage at a time; unloads models that are idle; hardware encoding limited to H.264 |
| `cpu` | No usable NVIDIA GPU (or compute capability below 7.0) | Every model on the processor; software encoding; small previews; no TensorRT. Works, many times slower |
| `auto` (default) | — | `cpu` when no GPU is found; `6gb` when the card has 6 GB or less in total **or less than 4 GB free right now**; otherwise `12gb` |

> **Why "free memory" matters.** If a game, browser or another AI tool is holding video memory, Roop 3D may pick `6gb` (or, with very little free,
> a CPU retry) even on a big card. Close those programs (the profile is re-checked when a job starts), or set the profile by hand in Settings → Compute.

> **A fixed defect.** A "12 GB" card reports slightly *less* than 12288 MiB (an RTX 4070 reports 12282 MiB). Earlier builds therefore refused to
> let you choose `12gb` and silently ran such cards on the clamped `6gb` profile at render time, although *Auto* and the preflight said `12gb`.
> The threshold is now the documented minimum (10,240 MiB) everywhere (`MIN_12GB_PROFILE_VRAM_MIB`).

## 3. Symptoms and fixes

| You see | Likely cause | Do this |
|---|---|---|
| The pill says **CPU** but you have an NVIDIA GPU | Driver too old, ONNX Runtime without the CUDA provider, or the profile/provider was set to CPU | Preflight: *NVIDIA Driver* needs **550 or newer** (CUDA 12); *ONNX Runtime* must list `CUDAExecutionProvider` (and `TensorrtExecutionProvider` for TensorRT). In Settings → Compute set *Hardware profile* and *Preferred provider* to `auto` |
| `PROVIDER_UNAVAILABLE` / `TRT_UNAVAILABLE` | The requested accelerator cannot be used right now | Set *Preferred provider* to `auto` (or CUDA). Renders still work without TensorRT, only slower. TensorRT also needs the *stream* render backend and the packaged TensorRT libraries |
| A banner says a TensorRT engine is **compiling** or running its *canary* | First use on this GPU: engines are built and cached | Wait. It can take minutes once; later runs reuse the cache. Do not close Roop 3D meanwhile |
| `PIPELINE_OOM` — *the graphics card ran out of memory* | Not enough free video memory for the job | Roop 3D already retried **once on the CPU** from the checkpoint. Close other GPU programs, lower the profile (12gb → 6gb → cpu) or *Stream batch size*, then **Retry** |
| Previews say *busy* | A render, a model build or a TensorRT compile owns the GPU | Wait; the preview retries by itself |
| A job is *Interrupted* with `ENCODER_TERMINATED` / `FILE_ACCESS_DENIED` | The encoder process or the output file was stopped, often by security software or a program holding the file open | Close programs that have the output open, check your security software, then **Resume** |
| The render is much slower than expected | CPU profile, restoration on, or a 3D render mode | See the [Performance guide](PERFORMANCE_GUIDE.md) |
| `DISK_SPACE_LOW` / `DISK_FULL` | Not enough room for the output | Free space, then Resume/Retry |

## 4. Checks, step by step

1. Run **Re-Run Preflight** and read every WARN/FAIL message (they are specific).
2. Update the NVIDIA driver if *NVIDIA Driver* is below 550; reboot.
3. Close programs that use the GPU; check `nvidia-smi` shows several free GB.
4. In Settings → Compute: *Hardware profile* `auto`, *Preferred provider* `auto`. If the pill still shows CPU, stop and start Roop 3D in Pinokio.
5. Run **Settings → Diagnostics Suite** and **Verify Offline Health**.
6. Still failing? Note the error code and request ID shown in the message and look it up in [Common problems](COMMON_PROBLEMS.md#gpu).

## 5. What is and is not supported

Windows 10/11 64-bit (fully tested) and Linux x86-64 with the NVIDIA proprietary driver 550+ and CUDA 12 (compatible) are supported. macOS on Apple
Silicon is experimental, CPU only. AMD (ROCm) and Intel Arc/OneAPI GPUs are **not** supported. TensorRT and CUDA acceleration need an NVIDIA GPU of
compute capability 7.0 or higher.
