# Remediation Stage 10: optional CUDA compositor

This checkout's implementation is `app/roop_3d/gpu_compositor.py`, integrated
with `GpuBatchPipeline` and the existing Stage 3 `StreamingProcessor`. There is
no `app/roop/` package in this standalone application.

## Activation and contracts

Install the optional `gpu-compositor` dependency using a CUDA PyTorch wheel in
`app/roop_3d/.venv`, then set `ROOP_GPU_COMPOSITOR=1` for a streaming job. The
local development environment was provisioned with `torch==2.7.1+cu128` from
the official PyTorch CUDA 12.8 index. PyTorch remains optional; API startup and
the preserved reference path do not import it or load models.

The local environment also received the pinned `tensorrt-cu12-libs==10.9.0.34`
official NVIDIA Windows wheel and its CUDA runtime dependency for an additional
native canary. No runtime packages were installed in another app's environment.

Each worker clones the validated swapper session onto its own compute stream,
retaining CUDA/TRT provider settings and checking writable engine-cache paths.
I/O binding passes contiguous CUDA tensor pointers directly to ORT. Inputs,
outputs and pinned transfer slots remain owned until their completion events.
The optional `DecoupledRestorer.cuda_adapter` uses the registered model's actual
input/output metadata, declared normalization, GPEN/CodeFormer training template,
fidelity auxiliaries, and device-side luminance composition. A host-only restorer
is rejected before frame processing; it is never silently bridged through NumPy.

Advanced options can be passed explicitly through `GpuBatchPipeline`:

```python
from roop_3d.gpu_compositor import CompositorOptions

pipeline = GpuBatchPipeline(
    validated_swap_session,
    embeddings,
    restorer=restorer_pool,
    compositor_options=CompositorOptions(
        color_strength=0.7,
        regional_color=True,
        laplacian_blend_weight=0.8,
        erosion=2,
        feather_sigma=2.0,
        jaw_falloff=0.1,
    ),
    device_masker_factory=verified_device_masker_factory,
    hardware_profile="6gb",
)
```

The factory receives the worker's compute stream and returns a device callable
`masker(target_rgb_512, face) -> (alpha, inner_skin, visibility)`. All three
masks must be CUDA float32 `1x1x512x512`. Its model preprocessing and class
mapping must be explicit. `SemanticMaskAdapter` translates documented logits
using required face, skin, protected (eyes/teeth), and occluder IDs. It has no
default labels. Color statistics and writes are restricted to the eroded visible
skin mask; an absent semantic adapter makes positive color strength an error.

The repository's BiSeNet labels/preprocessing remain a candidate contract.
This stage does not fabricate XSeg/SAM-2 tensor contracts or claim those models
are shipped adapters. No candidate parser is activated automatically.

## Device processing and transfer ownership

Processing uses NCHW float32 RGB in 0..1. The per-face path samples a 512 crop
when restoration/semantics require it, resizes to the swapper's actual 128 input,
restores at its declared training resolution, obtains target semantics, and
inverse-warps the result into the device frame. Swap-only mode preserves the
128 reference crop geometry. Affine inversion uses a closed-form device formula
to avoid a synchronous CUDA solver-status check.

Color transfer uses floating-point CIE LAB D65 moments. RCT uses regional Gaussian
moments. The three-level frequency blend retains target coarsest illumination
and weights source detail bands. It intentionally differs from the reference
OpenCV uint8 LAB quantization and the reference all-band pyramid policy.
Feathering, square erosion/dilation and smooth jaw falloff use Torch operations.
Visibility is reapplied after feathering, protecting foreground occluders.

Each worker has three non-default streams and three slots: pinned upload of the
next frame, device inference/composition of the current frame, and pinned download
of the previous frame. CUDA events enforce dependencies and prevent slot reuse
before D2H completion. Frame arrays are returned in submission order for the
existing NVENC writer; its host FFmpeg input interface is preserved. This is not
a claim of CUDA-surface decoding or a GPU-pointer NVENC API. The existing shared
memory coordinator still owns ring leases and its bounded host frame copies.

Frame-grid/intermediate admission budgets are 256 MiB and 768 MiB divided by
worker count under the 6 GB and 12 GB policies. They are separate from model
arena budgets. Oversized frames fail explicitly. Final non-finite output is
rejected at the final D2H boundary. Failure paths drain the three owned streams.

## Verification and limits

Run from the repository root:

```powershell
app/roop_3d/.venv/Scripts/python.exe -B verify_trt_fps.py --benchmark-compositor
app/roop_3d/.venv/Scripts/python.exe -B verify_trt_fps.py --benchmark-compositor --hardware-profile 12gb
```

The gate returns exit 0 only for zero CPU affine calls during CUDA work, speedup
at least 1.35x, PSNR at least 42 dB against `composite_cpu`, CPU/CUDA advanced
kernel error below 0.0002, and exact protected-pixel/occluder assertions.
It includes pinned staging, H2D, crop warp, inverse composition, D2H and owned
host results. The deterministic 720p two-face workload uses synthetic replacement
crops and three-frame bundles, with warm-up excluded. It does not run neural
models; its reported FPS must not be presented as end-to-end swap throughput.

Initial 150-frame campaigns on the RTX 4070 (12 GB), CUDA PyTorch 2.7.1:

| Allocation policy | CPU FPS | GPU FPS | Speedup | PSNR | Exit |
| --- | ---: | ---: | ---: | ---: | ---: |
| 6 GB | 12.53 | 268.06 | 21.40x | 70.70 dB | 0 |
| 12 GB | 12.16 | 247.58 | 20.36x | 70.70 dB | 0 |

Both had zero intermediate CPU affine calls. Advanced kernel maximum absolute
error was 0.00000251. The 6 GB policy run on this card is not a physical 6 GB
hardware acceptance campaign. Default settings preserve the reference feature
policy (color/detail matching off); enabling new appearance options requires the
explicit semantic contract described above.

A final 6 GB-policy repeat after replacing affine solver inversion with the
asynchronous closed-form formula passed at 237.13 GPU FPS versus 12.66 CPU FPS
(18.74x), with the same 70.70 dB PSNR and zero CPU affine calls. Timings vary
with warm-up, scheduler load and other local GPU activity; these are individual
measured campaigns rather than a universal throughput promise.

Eleven focused tests cover float LAB against OpenCV, affine boundary geometry,
semantic exclusions, empty masks, illumination ownership, real ORT pointer
binding, worker stream isolation, slot reuse/order, failure recovery and a full
synthetic 512 restoration/masking chain, and stream workspace cleanup. Model-specific real-output restoration,
verified XSeg/SAM-2 occlusion, physical 6 GB GPU, and end-to-end NVDEC/TRT/NVENC
acceptance remain unmeasured. The optimization remains opt-in pending those
campaigns; the synthetic gate does not establish their performance or fidelity.

The full local regression run passed 659 tests and 18 subtests in 542.28 seconds.
After the final changes, the focused CUDA plus remediation Stage 2/3 regression
run passed all 56 tests in 8.83 seconds. Existing FastAPI/HTTPX and Torch resource
enumeration deprecation warnings were observed.

A native TensorRT 10.9 FP32 `Mul` canary compiled an engine and passed the strict
node-placement gate with `TensorrtExecutionProvider` as the sole executing
provider. CUDA pointer I/O binding on the explicit worker compute stream produced
maximum absolute error 0.0 against Torch. This tests a synthetic operator, not a
face model; it does not change the unmeasured model-chain acceptance states.

The final pre-commit full-suite run caught a retained cuBLAS stream workspace in
the coordinate/color transforms. Those tiny GEMM operations were replaced with
pointwise CUDA arithmetic so model-build eviction can reclaim all allocations
without private Torch cleanup APIs or a weakened resource gate. The new workspace
regression and the existing eviction tests passed together (16 tests).

Final commit gate after the workspace fix: the complete local suite passed
664 tests and 18 subtests in 543.51 seconds (two existing deprecation warnings).
The compositor benchmark returned exit 0 with zero CPU affine calls, 23.02x
synthetic speedup (293.68 GPU FPS / 12.76 CPU FPS), and 70.70 dB PSNR. The fresh
read-only integrity audit again passed all 2,645 historical and 68,390
comprehensive protected-file entries with zero violations.

The read-only legacy integrity audit passed: 2,645 historical and 68,390
comprehensive protected files, zero violations. Explicit runtime activity was
classified separately; preserved baselines were not replaced.

Implementation references:
[ORT CUDA user streams and asynchronous execution](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html),
[ORT TensorRT user streams](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html),
[ORT device tensor I/O binding](https://onnxruntime.ai/docs/performance/device-tensor.html),
[official PyTorch installation](https://pytorch.org/get-started/locally/).
