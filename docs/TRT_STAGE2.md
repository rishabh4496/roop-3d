# Stage 2 TensorRT remediation

This implementation lives in the independent `app/roop_3d` package. The prompt's
`app/roop` module names map to `roop_3d/trt_session_builder.py` and
`roop_3d/optimized_trt_engine.py`. No legacy application modules are imported.

## Provisioning and startup

Pinokio `install.js` invokes `torch.js` on NVIDIA Windows/Linux systems. It syncs
`app/roop_3d/requirements.txt`, combining the existing hash-pinned base and TRT
lockfiles: ORT GPU 1.23.2, TensorRT CUDA 12 metapackage, libraries and bindings
10.9.0.34. Startup selects the same dependency set so a base sync cannot remove
TensorRT. Other systems retain the base installation. Optional inference models
are never loaded to boot the API. CUDA/cuDNN prerequisites remain supplied by
Pinokio's AI bundle and the packaged runtime; missing native dependencies fail
session construction explicitly.

Packaged TensorRT/CUDA directories are registered before ORT initialization.
Windows DLL directory handles stay alive. Linux updates `LD_LIBRARY_PATH` and
preloads available shared objects because changing the variable after process
startup alone does not update the loader's search state.

## Verified graph conversion

From the checkout root, using this app's dedicated interpreter:

```powershell
& app/roop_3d/.venv/Scripts/python.exe -B tools/convert_dynamic_shapes.py app/models/inswapper_128.onnx --check-batches 1,4,8
```

The default result is `<name>_dynamic.onnx` plus a JSON verification receipt.
The converter supports inswapper and detector-style reshape/squeeze exports.
It clones reshape targets, removes stale intermediate shapes, and verifies
finite outputs and parity against independent original batch-1 runs. Flattened
detector outputs must preserve concatenated sample order. A failed validation
never publishes a dynamic model. Existing outputs are not overwritten.

Use `--shapes '{"images":[1,3,640,640]}'` for unknown spatial dimensions, using
the actual graph input names and intended sizes. No unknown dimension is guessed.
External model paths must resolve to an exact, hash-verified model registry
locator. Their derived outputs go to this app's `app/models/`; legacy weights
remain read only. External tensor-data exports require separately verified
support and are rejected here.

GPEN 256/512 exports are classified `STATIC_BATCH_1`, with a receipt and clear
summary, rather than relabeled as dynamic. Models that pass batch 1 but fail a
multi-batch internal reshape receive the same classification. Their prebuild
profile remains `(1,)`. Classification exits 0 without producing a dynamic model.

CodeFormer must already expose a connected `w` or `fidelity_weight` graph input.
Conversion retains it as the second runtime tensor, does not batch its scalar
value, and checks multiple fidelity values against the original graph. An export
that already froze or disconnected it requires re-export; its location cannot
be reconstructed safely from an arbitrary constant.

The weights-free gate creates a fresh actual static ONNX reshape graph:

```powershell
& app/roop_3d/.venv/Scripts/python.exe -B tools/convert_dynamic_shapes.py app/roop_3d/.runtime/stage2-example/synthetic.onnx --synthetic --check-batches 1,4,8
```

Choose a fresh fixture path if that example was previously run. This verifies
conversion on ORT CPU; it does not establish production-model or GPU fidelity.

## Strict sessions and telemetry

`TensorRTSessionBuilder.build()` is strict whenever invoked: it requires TRT
selection, disables CPU EP fallback and Python provider retry, checks actual
provider activation, executes every declared canary batch, checks finite output,
and checks node-placement profiling. CUDA partitions may handle unsupported TRT
operators; any CPU execution or absence of a matching nonempty engine fails.
An unrelated model's cached engine cannot satisfy this gate.

The default profile is min=1, opt=4, max=8, with executed canaries `(1,4,8)`.
Static inputs reject a multi-batch profile before a session is constructed.
Nonbatch dimensions remain fixed unless concrete shape overrides are supplied.
Timing and engine caches live in `app/models/trt_cache/`. Engine prefixes include
model hash, precision, ORT/TRT versions, profiles and workspace; ORT additionally
manages hardware-specific engine filenames. Native cache paths reject links and
hardlinks. The default workspace caps are 1 GiB for 6 GB and 2 GiB for 12 GB profiles.
CUDA partition memory is capped against observed free VRAM, the hardware profile,
the workspace and a 768 MiB reserve. These caps do not guarantee a model will fit
a particular GPU; allocation failures remain hard failures.

Precision defaults are FP16 for detectors/enhancers and FP32 for
`inswapper_128.onnx` and `inswapper_128_dynamic.onnx`. The swapper exception guards
against silent FP16 arithmetic overflow. `ROOP_SWAP_FP16=1` opts the swapper into
FP16; `ROOP_TRT_FP16=0` takes precedence and forces global FP32. Finite canaries
cannot alone establish visual parity or detect every color corruption.

The API exposes a metadata-only `app.state.trt_builder` wired to its pipeline
tracker. Explicit builds through it broadcast `COMPILING_TRT`, `RUNNING_CANARY`
and `IDLE` frames on both `/ws/telemetry` and `/ws/v1/pipeline/status`. Frames
include label, model, precision and an epoch-seconds `engine_since`. IDLE clears
those fields, including on build/canary failure. Existing job/status events remain
available. The retained advanced parity builder keeps its existing observer API.

Logs record the high-resolution elapsed cold-build duration across session
creation and the first synchronous inference call, plus each canary's duration.
This is a host-observed build interval including ORT setup and first inference,
not a fabricated isolated GPU compiler measurement. ORT detailed build logging
is enabled for native compiler timing details.

`OptimizedTRTEngine` loads lazily on explicit `prepare()`/`run()` and rejects
inputs outside its executed batch/shape profile. This stage supplies conversion
and session infrastructure; it does not replace the worker's preserved reference
inference adapters or claim those adapters gained dynamic-batch inference.

```python
from roop_3d.optimized_trt_engine import OptimizedTRTEngine

engine = OptimizedTRTEngine("app/models/inswapper_128_dynamic.onnx")
session = engine.prepare()  # strict compilation + canaries, never API boot
# outputs = engine.run(actual_named_input_arrays)
engine.close()
```

Run Python API examples from the root with `app/` on `PYTHONPATH`, or from `app/`
with model paths relative to that directory.

## Prebuild and benchmark

```powershell
& app/roop_3d/.venv/Scripts/python.exe -B tools/prebuild_engines.py app/models/inswapper_128_dynamic.onnx --hardware-profile 6gb
& app/roop_3d/.venv/Scripts/python.exe -B verify_trt_fps.py app/models/inswapper_128_dynamic.onnx --hardware-profile 6gb --batch 4 --iterations 50
```

Prebuild accepts multiple explicit model paths, batch/optimum arguments and JSON
shape overrides. Both commands fail nonzero if a strict gate fails. Standalone
CLI events are printed as JSON; they do not impersonate events from a separately
running API process. The benchmark excludes compilation/canaries and measures
synchronous model throughput with host transfers and synthetic inputs, not
video-pipeline FPS. No FPS result is claimed without actually running it.

Focused tests run from `app/`:

```powershell
& roop_3d/.venv/Scripts/python.exe -B -m pytest -c roop_3d/pyproject.toml --confcutdir=roop_3d roop_3d/tests/test_remediation_stage2.py -q
```

These tests execute real ORT CPU conversion and live WebSocket fanout, with fake
TRT sessions for deterministic failure/precision tests. They do not prove native
GPU compilation. Run the complete suite and the read-only legacy integrity audit
as documented in the root README before completing a stage.

Provider options follow the [ORT TensorRT execution provider documentation](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html).
Packaging follows the [NVIDIA TensorRT installation guide](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html).

## Local verification, 2026-11-01

- Stage 1 scaffold gate: 15 tests passed.
- Synthetic conversion CLI: exit 0; batches 1, 4, 8 finite and independently equivalent.
- Full local suite: exit 0; 514 tests and 18 subtests passed.
- Focused Stage 2 and pipeline suite: exit 0; 49 tests passed.
- Read-only legacy audit: exit 0; 2,645 historical and 68,390 comprehensive protected
  files matched, with zero violations. Explicit runtime activity was classified separately.
- NVIDIA dependency sync dry run: exit 0 with required package hashes resolved.
- Prebuild and benchmark CLIs failed nonzero as intended when packaged TRT libraries
  were absent. Native TRT compilation, production dynamic-model fidelity, and FPS
  have not been measured in this environment.
