# Stage 02: configuration, inspection and persistent API contracts

Implemented in the independent `roop-3d` checkout, under `app/roop_3d/`.
This is the next contract after the recorded Stage 01 scaffold. The task's
module paths are relative to this standalone application's `app/` directory.
The referenced Section 4 was not supplied; the values below are the implemented
catalog choices, not a claim that missing specification text was verified.

## Delivered behavior

- Frozen Pydantic v2 contracts reject extra fields, non-finite numbers, unknown
  enum values and escaped/link-traversing paths. REST validation envelopes never
  echo submitted values. Structured diagnostics redact credentials and secrets.
- Configuration lives in `roop_3d/.runtime/config.json`. Exclusive process/thread
  locks serialize writes. Exclusive temporary files are flushed/fsynced and
  atomically replaced; the prior bytes are atomically saved as `config.json.bak`.
  Restart loads persisted settings, then applies the four `ROOP3D_*` transport
  overrides. Transport changes through PUT return `TRANSPORT_RESTART_REQUIRED`.
- Live hardware queries read GPU name, total/free VRAM, compute capability, driver,
  driver-supported CUDA, available ORT providers, RAM, physical/logical CPU counts,
  and FFmpeg NVENC/NVDEC codec inventories. A discoverable CUDA runtime is queried
  independently; absence is `UNKNOWN`. No inference session is created at boot
  or by the hardware endpoint. Codec/provider registration is not execution proof.
- PyAV probes real local containers, samples timestamp intervals for VFR evidence,
  lists audio tracks, and decodes precise frame-index/timestamp JPEG previews.
  Optional comparison previews decode both supplied local files. Protocol and
  format allowlists prevent a renamed network playlist from opening a connection.
- Target inspection runs the installed SCRFD ONNX detector with the CPU provider.
  Exact registered bytes are rehashed before use; no weights are copied to disk.
  It scans bounded keyframes and returns spatial track IDs, observed frame/time
  ranges, representative boxes, thumbnails, and eye-line roll ranges. Yaw/pitch
  remain explicitly unknown. Inspection receipts bind mappings to a target hash.
- FaceSet listing reads validated local summary manifests. Job submission validates
  track IDs, FaceSet IDs and the current target against the inspection receipt.
  Idempotency is durable; only a hash of the key is stored. A changed request using
  the same key returns `IDEMPOTENCY_CONFLICT`. Exact retries return the prior job.
- State and progress changes are atomically persisted before publication. Active
  state recovery marks work `interrupted`. Revision checks detect stale controls.
  Corrupt job data fails closed and is preserved rather than replaced with an
  empty queue. Thread and separate-process submission races were both exercised.
- Native Uvicorn WebSockets broadcast typed state, progress/FPS/stage-latency and
  live hardware events. Slow subscribers receive `stream.gap`; connecting clients
  receive `stream.ready`. Both require a jobs snapshot refresh. Delivery is bounded,
  process-local and has no replay log. HTTP and WS enforce explicit Host/Origin
  allowlists; duplicate security-sensitive headers are rejected.

## Authoritative settings catalog

The complete machine-readable constraints are exported by OpenAPI and
`GET /api/v1/system/capabilities`. Requested execution modes do not imply that
their renderer is installed; `enabled_modes` and `rendering_worker_available`
describe implemented behavior.

| Key | Default | Allowed values / bounds |
| --- | --- | --- |
| schema_version | 2 | Literal 2; explicit migration only |
| execution_mode | swap | swap, volumetric, hybrid |
| hardware_profile | auto | auto, cpu, 6gb, 12gb |
| preferred_provider | auto | auto, TensorrtExecutionProvider, CUDAExecutionProvider, CPUExecutionProvider |
| precision_policy | auto | auto, fp32, fp16 |
| vram_limit_mb | 0 | Strict integer 0..131072; 0 means dynamic availability |
| memory_safety_reserve_mb | 768 | Strict integer 0..131072 |
| max_concurrent_jobs | 1 | Strict integer 1..8; renderer scheduling is a later gate |
| face_restoration_policy | off | off, auto, gfpgan, codeformer |
| video_timing_policy | preserve | preserve, cfr |
| audio_policy | copy | copy, transcode, drop |
| lip_sync_mode | off | off, wav2lip |
| model_search_paths | empty | Up to 32 owned runtime paths; external weights use exact registry locators |
| output_root | roop_3d/.runtime/outputs | Relative to app/, or absolute within its owned runtime |
| cache_root | roop_3d/.runtime/cache | Same ownership constraints |
| remote_processing_enabled | false | Boolean; does not install a remote adapter |
| synthetic_media_notice | true | Boolean |
| diagnostics_level | info | error, warning, info, debug |

Existing transport keys remain: loopback-only host; API port 1..65535 excluding
5174; request limit 1..64 MiB (default 2 MiB); development CORS disabled by default.
Vite CORS can explicitly enable `http://127.0.0.1:5174` for GET/POST/PUT.

Dynamic VRAM budget, in MiB:

`max(0, min(currently_free, total, profile_cap, nonzero_configured_limit) - reserve)`.

CPU selection returns a zero GPU budget. The 6 GB and 12 GB caps are separate;
their arithmetic tests do not constitute hardware performance qualification.

Versionless Stage 01 transport-only documents are the sole supported migration
source. Other versions are rejected. Migration is deliberately explicit:

```python
from roop_3d.audit.project_paths import application_root
from roop_3d.core.settings import ConfigStore
store = ConfigStore(application_root())
value = store.load(migrate=True)  # validate migration without modifying the file
if value is None:
    raise ValueError("No saved configuration to migrate")
store.save(value)  # atomic v2 save, preserving original bytes in .bak
```

## Routes, state semantics and local data

All requested routes are registered under `/api/v1`: health; hardware;
capabilities; GET/PUT config; media probe/inspect-faces/preview-frame; models;
model download; FaceSets; job create/list/get/cancel/pause/resume; WS events.
Request/response/event schemas and legal parameter bounds are exported locally.

Video inputs must be staged locally under `roop_3d/.runtime/media/`, or the
configured owned output root, with a supported video suffix. There is no archive,
upload, enrollment or remote-transfer endpoint in this stage. FaceSet summaries
live at `.runtime/facesets/<faceset_id>.json`; listing them is not a claim that a
FaceSet enrollment/embedding pipeline has shipped.

Inspection limits: 1..128 sampled keyframes (default 32), 1..10000 decoded frames,
30-second scan budget, 32 detections per sampled frame and 64 spatial tracks.
Track IDs are deterministic only within the observations of an inspection;
crossings, motion and cuts may merge/split tracks. Mappings must be reviewed.
Preview frame indices are 0..9999; timestamps are finite, 0..86400 seconds.
VFR results are sampled evidence; a constant incomplete sample returns UNKNOWN.

Legal job transitions are defined in `jobs/schema.py`:

| From | Allowed next states |
| --- | --- |
| queued | preparing, pausing, cancelling, failed |
| preparing | running, pausing, cancelling, failed, interrupted |
| running | pausing, cancelling, succeeded, failed, interrupted |
| pausing | paused, cancelling, failed, interrupted |
| paused | queued, cancelling |
| cancelling | cancelled, failed, interrupted |
| interrupted | queued, cancelling |
| cancelled / succeeded / failed | none |

Queued controls can finish immediately. Active workers must release resources
and acknowledge `pausing`/`cancelling`. Resume returns paused/interrupted work to
the queue. API clients cannot arbitrarily set state or fabricate progress.

No rendering worker is installed by this contract. Submitted jobs remain queued
until a later worker is implemented. No face-replacement output, volumetric
quality, CUDA/TRT execution or GPU throughput is claimed. The progress test's
explicit fixture values test event transport, not measured inference performance.
Remote jobs require per-job consent and global enablement, then explicitly fail
with `REMOTE_ADAPTER_UNSUPPORTED` because no provider is installed. No data is sent.
The download route rehashes registered existing artifacts and returns
`downloaded=false`; no remote model URL or permission is invented.

## Final validation

Commands below ran from this repository's `app/` directory. See
`STAGE02_VALIDATION.json` for the exact changed-file list and machine-readable counts.

| Exact command | Exit | Final result |
| --- | --- | --- |
| `roop_3d/.venv/Scripts/python.exe -B -m pytest -c roop_3d/pyproject.toml --confcutdir=roop_3d roop_3d/tests -q` | 0 | 108 passed, 0 failed, 0 skipped; 18 additional subtests passed |
| `roop_3d/.venv/Scripts/python.exe -B -m roop_3d.tests.manual_stage02_acceptance` | 0 | Real SCRFD, two spatial tracks over five frames, crop thumbnails, comparison preview, mappings and persistent controls |
| `roop_3d/.venv/Scripts/python.exe -B roop_3d/audit/local_models.py --verify` | 0 | 95/95 hashes matched, 0 violations |
| `roop_3d/.venv/Scripts/python.exe -B roop_3d/audit/verify_isolation.py` | 0 | 2647 historical and 68392 comprehensive immutable checks passed; 0 violations |
| `uv pip compile roop_3d/pyproject.toml --extra test --python-version 3.10 --generate-hashes --output-file roop_3d/requirements.lock --quiet` | 0 | Existing pins preserved; websockets 15.0.1 added with genuine hashes |
| `uv pip sync --python roop_3d/.venv/Scripts/python.exe --require-hashes roop_3d/requirements.lock` | 0 | Only websockets 15.0.1 added in the dedicated environment |
| `uv pip check --python roop_3d/.venv/Scripts/python.exe` | 0 | 39 distributions compatible |
| `git diff --check` | 0 | No whitespace errors |

uv used `UV_LINK_MODE=copy` and `UV_CACHE_DIR` inside this app's `.runtime/uv`.
The only suite warning is the existing upstream Starlette TestClient/Httpx
deprecation, left visible. No skipped or mocked core inference test is counted
as a pass. The offline suite includes actual blank-frame SCRFD inference; the
separate positive campaign uses the official OpenCV sample and does not require
network access during execution. Its downloaded image and derived video remain
ignored local artifacts, never staged source.

The positive campaign recorded fixture SHA-256
`7de7ed51a1594fff247f4cae2301eceacf5313d6011e37b4a4c8733f7bb72c07`
and installed SCRFD SHA-256
`5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91`.
Both are measured local hashes, not licensing or provenance permissions.

Intermediate failures were corrected and rerun: an audit script helper could not
be package-imported (the catalog now resolves its own read-only scopes); the
positive campaign initially lacked its owned parent directory; an older test
expected CPU cores to stay unknown; and TestClient's relative WebSocket URL used
the disallowed default host. Assertions and Host checks were not weakened.

The original integrity baseline and model registry were not replaced. Explicit
legacy runtime activity (1442 observed files at the final audit) is classified
separately by the existing policy and is not claimed byte-identical. Changes are
confined to the standalone Python package and its lock/metadata. No root launcher,
root lockfile, root ignore policy, sibling code/environment/Git state, or legacy
model artifact was modified. No commit or push was requested or performed.

Next gate: the next numbered worker/inference contract. This stage's acceptance
gates pass; renderer, enrollment, full pose, remote providers and hardware-specific
performance qualification remain separate future work.

References consulted: [Pydantic validators](https://docs.pydantic.dev/latest/concepts/validators/),
[FastAPI WebSockets](https://fastapi.tiangolo.com/advanced/websockets/),
[PyAV container decoding/seeking](https://pyav.basswood.io/docs/stable/api/container.html),
[official SCRFD layout/preprocessing](https://github.com/deepinsight/insightface/blob/master/python-package/insightface/model_zoo/scrfd.py),
[websockets 15.0.1 metadata](https://pypi.org/project/websockets/15.0.1/), and the
[OpenCV acceptance fixture](https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/lena.jpg).
