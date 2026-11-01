# Stage 14: optional NVIDIA device pipeline

Status: implemented adapters; **hardware acceptance pending**. This standalone
checkout uses `app/roop_3d`, not the legacy `app/roop` tree. Existing host-frame
rendering remains the reference/default path. These adapters are explicitly
invoked APIs, not a claim that the workstation job runner now runs entirely in
VRAM.

## Interfaces

- `NvdecFrameSource.gpu_frames(device=0, statistics=False)` yields a `DeviceFrame`
  and the existing CFR metadata. `DeviceFrame.tensor` is RGB HWC uint8, borrowed
  through stream-aware DLPack. Consume it on the same CUDA stream before advancing
  the iterator. The decoder retains its surface owner and synchronizes before
  reuse. `frames()` still supplies the existing FFmpeg NVDEC/NVENC host fallback.
  Strict GPU mode never silently downloads pixels or changes tensor residence.
- `NvVideoDecoder` uses NVIDIA `SimpleDecoder` in device-memory mode. H.264,
  HEVC and AV1 decode are delegated to the installed SDK/driver capabilities.
  Decode statistics are optional. `frame.statistics` contains the SDK's parsed
  metadata; `frame.motion` interprets quarter-pixel vectors and QP values.
  An unavailable statistic is unknown, never a static-frame decision. NVIDIA
  documents H.264/HEVC statistics; AV1 statistics are not assumed. Statistics
  parsing produces host metadata, so this is not an assertion of zero host bytes
  for metadata. No motion-based detection skipping has been enabled.
- `decode_image()` uses optional `nvidia.nvimgcodec`, retains image/decoder owners,
  and requires CUDA DLPack output. That image backend has not been installed or
  validated on this host.
- `NVHardwareVideoWriter.from_cuda()` creates `NvVideoEncoder`. It takes
  contiguous RGBA uint8 tensors on its owning stream, supports H.264/HEVC and
  per-frame `sei_messages` in the installed NVIDIA API's representation. NVENC
  consumes device buffers; compressed packets reach host memory and are muxed
  with FFmpeg stream copy into fresh MP4/MOV files. It accepts integer CFR only;
  fractional rates must use the reference writer. Neither audio nor HDR color
  management is implied by this low-level encoder. Call `close()` on success or
  `abort()` on failure. PyNvVideoCodec 2.2 packet dictionaries and older byte
  packets are handled explicitly.
- `FaceOps('auto' | 'cvcuda' | 'torch')` wraps CUDA buffers without staging pixels
  on the CPU. `align()` accepts GPU destination-to-source pixel affine matrices
  and returns NCHW float32 crops. Var-shape affine matrices use the documented
  `N x 6` layout. Affine warp and fused resize/crop/convert/reformat are separate
  launches; they are not one combined kernel. `auto` selects Torch only when the
  optional module cannot load, recording its reason. A native operator failure
  is surfaced. Gaussian, erosion/dilation, Laplacian, joint bilateral skin
  smoothing, and three-band feathering have Torch fallbacks. Current visibility
  is reapplied after feathering to preserve foreground occluders.
- `OptimizedTRTEngine.run_cuda(feeds, outputs)` binds contiguous CUDA tensors
  directly with ORT I/O binding. Callers provide actual model output names,
  shapes and dtypes; the adapter never guesses a restorer's tensor contract.
  It validates names, checked batch sizes, device and compiled shape limits.
  Host build canaries remain outside the frame transport measurement.
- `NvidiaHardwareFlow` requires an OpenCV build with CUDA OF 2.0 and external
  GpuMat memory bindings. It wraps grayscale CUDA pointers, converts native flow
  to pixel units, and explicitly synchronizes across the OpenCV stream boundary.
  `select_flow(..., raft_runner=...)` can use an injected local RAFT-Small Torch
  or TensorRT runner when hardware flow is unavailable. No weights are downloaded
  and no RAFT model is claimed installed. `OpticalFlowTracker` consumes forward
  and backward full-resolution flow, samples previous alpha with backward flow,
  propagates screen-space vertex XY with forward flow, preserves Z, supports
  confidence/scene resets and applies current visibility. Use one tracker per
  face/shot and reset on track discontinuities. This is temporal propagation of
  provided geometry/mattes, not new 3D reconstruction or occlusion inference.

## Optional FP8

`inswapper_128`, including dynamic/batch variants, is locked to FP32 even if the
old `ROOP_SWAP_FP16=1` override is set. Low-precision constants, casts and Q/DQ in
swapper graphs are rejected. The default precision for other models is preserved.

FP8 is allowlisted to `GPEN-BFR-1024`, `ultramax_bfr`,
`RestoreFormer_Plus_Plus`, and `swin2_sr_x4`. `ROOP_TRT_FP8=1` requests FP8 for
these models but fails closed without a separately calibrated artifact.
Alternatively, pass `fp8_model_path=` explicitly to `TensorRTSessionBuilder.build`.
The artifact must be an owned, separate ONNX file. Build also requires an explicit
`fp8_validate(session)` callback returning `True` only after reference-output
parity meets the caller's real acceptance criteria. No default fidelity threshold
or representative calibration distribution is invented.

`trt_fp8_builder.validate_fp8_graph` requires observed SM 8.9+, TensorRT 10,
positive finite constant scales, E4M3FN zero points and matching explicit Q/DQ.
Scale presence is structural evidence, not proof of calibration provenance.
Use an offline NVIDIA ModelOpt calibration/export campaign with representative
local media to produce that artifact. This change does not perform calibration
or manufacture scales. There is no fictional ORT `trt_fp8_enable` setting.
Reference and quantized digests, precision, observed hardware, native versions
and shape profiles enter the engine cache identity. Existing 6/12 GB workspace
and memory reserves still apply. No production restorer has been quantized or
admitted by this change.

## Verification

From the checkout root, using the dedicated interpreter:

```powershell
app/roop_3d/.venv/Scripts/python.exe -B -m pytest app/tests/test_nv_cvcuda_pipeline.py -q
```

The forwarding gate includes the tests in the owned Python package; its pytest
configuration keeps caches and temporary media inside this checkout. The suite
covers CUDA DLPack pointer identity, Torch affine crop geometry, real NVENC/MP4/
NVDEC transport, explicit FP8 policy/graph rejection and moving-mask propagation
over 30 synthetic frames on CPU and CUDA. Synthetic propagation uses known flow
and is not hardware optical-flow accuracy evidence.

Final local results:

- Requested gate: **21 passed, 1 skipped**, exit 0.
- Full owned Python suite: **697 passed, 1 skipped, 18 subtests passed**, exit 0
  (334.22 seconds). The skipped case is the native CV-CUDA/TensorRT transport
  test. Two existing dependency deprecation warnings were reported.
- Strict native acceptance: fails on the missing `cvcuda` prerequisite; this
  result must not be represented as a completed hardware campaign.

The native stack test uses real PyNvVideoCodec, CV-CUDA and TensorRT. It verifies
the decoder/CV-CUDA storage alias and profiles affine crop plus device-bound
TensorRT inference, requiring CUPTI kernel activity and no device-to-host copy
events. Its small generated ONNX graph tests transport, not production face-swap
fidelity. Missing native prerequisites skip this test in the development suite.
Acceptance mode turns missing prerequisites into failures:

```powershell
$env:ROOP_STAGE14_ACCEPTANCE = '1'
app/roop_3d/.venv/Scripts/python.exe -B -m pytest app/tests/test_nv_cvcuda_pipeline.py -q
```

On this Windows RTX 4070 host, PyNvVideoCodec 2.2.3 was installed only in this
checkout's dedicated environment. A real three-frame H.264 NVENC/MP4/NVDEC CUDA
round-trip passed, and decode motion/QP metadata was observed on the generated
static clip. CV-CUDA is absent; its published 0.19.0 wheels are Linux-only.
TensorRT Python bindings and nvImageCodec are also absent. The strict native
gate therefore fails for missing CV-CUDA. Development success with that test
skipped is **not** Stage 14 hardware acceptance. Hardware OF, nvImageCodec,
SEI round-trip, HEVC/AV1 acceptance, production FP8 parity, 6/12 GB acceptance
campaigns and speed/fidelity measurements remain unverified. No >2,000 FPS,
zero-copy production-swap, or FP8 speedup claim is made.

The read-only legacy integrity audit passed with 68,390 immutable comprehensive
files and 2,645 historical files checked, no violations, at
`2026-11-01T21:15:11.437597+00:00`. Its policy separately classified runtime
activity; no baseline was replaced.

## Upstream contracts consulted

- [NVIDIA decoder and statistics API](https://docs.nvidia.com/video-technologies/pynvvideocodec/pynvc-api-reference/decoder.html)
- [NVIDIA encoder API](https://docs.nvidia.com/video-technologies/pynvvideocodec/pynvc-api-reference/encoder.html)
- [CV-CUDA operator bindings](https://github.com/CVCUDA/CV-CUDA/tree/main/python/mod_cvcuda/operators)
- [CV-CUDA affine contract tests](https://github.com/CVCUDA/CV-CUDA/blob/main/tests/cvcuda/python/test_opwarpaffine.py)
- [OpenCV hardware flow](https://docs.opencv.org/4.12.0/db/d70/classcv_1_1cuda_1_1NvidiaOpticalFlow__2__0.html)
- [TensorRT explicit quantization](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/architecture/capabilities.html)
- [ModelOpt ONNX quantization](https://nvidia.github.io/Model-Optimizer/reference/generated/modelopt.onnx.quantization.quantize.html)
