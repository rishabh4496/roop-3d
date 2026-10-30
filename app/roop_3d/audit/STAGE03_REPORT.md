# Stage 03: model registry, local reuse and downloader

Implemented in the independent Roop 3D checkout, using the established `app/`
path convention. Source changes stay under `app/roop_3d/`; no launchers, sibling
application files, environments, protected baselines or root policy files changed.
The exact changed files and executed commands are in `STAGE03_VALIDATION.json`.

## Registry and evidence

`models/model_manifest.json` and `models/registry.py` are the authoritative model
catalog. The manifest preserves all 95 observed local artifact identities, sizes,
locators and ONNX metadata from Stage 00. SHA-256 bindings to the original discovery
JSON, verified-local JSON and provenance/license CSV detect changes to those source
documents. Model IDs remain compatible with Stage 02.

There are 101 records: 95 `verified_local`, five optional `candidate` records
(Wav2Lip, InstantID, PuLID, SDXL-Turbo and FLUX.1-Schnell), and one `blocked`
FLAME 2020 record. No optional weights, URLs, checksums or tensor contracts were
invented. All 94 historical discovery entries remain represented. Twenty-nine
ancillary artifacts have `role=null` and an explicit unknown-role explanation;
landmark, attribute, frame and feature models are not misrepresented as supported
adapters within the requested role vocabulary. Filename-based role assignments
are explicitly heuristics. Empty provider lists mean execution is unknown, rather
than a claim of incompatibility. The verified SCRFD hash retains its real CPU
execution evidence. No new inference adapter is claimed by registering a model.

Local integrity is separate from licensing and upstream provenance. Installed
artifact licenses and commercial permissions remain UNKNOWN because the discovery
evidence does not identify their exact upstream terms. Acknowledgement never grants
commercial permission. FLAME's academic terms were checked at the official source;
it requires manual acquisition/review and remains blocked even when an API caller
sends `accepted_license=true`.

## Read-only provider and API

`models/local_provider.py` resolves preserved legacy/user-cache locators, this
app's verified download cache, and configured `model_search_paths`. The existing
Stage 02 policy keeps custom search roots under the owned runtime; external
installed models use the exact hashed read-only locators. Searches are bounded,
skip linked directories, and match filenames plus exact sizes and SHA-256 hashes.
A missing source may fall back to configured weights; a present but drifted source
fails closed without falling back to another copy.

`resolve()` returns the original path. `open()` supplies a verified binary
read-only handle and applies activation/license gates. `models/verifier.py`
streams bounded hashes, checks file stability/type, and never deserializes weights.
The existing detector catalog is backed by the new registry and still verifies
the actual bytes handed to ONNX Runtime. Boot loads metadata only, with no model
inference sessions or heavy optional imports.

API contracts:

- `GET /api/v1/models` preserves the existing 95 local summaries and adds notices.
- `GET /api/v1/models/registry` exposes all typed records and evidence states.
- `GET /api/v1/models/{model_id}/license` returns human-readable notices and gates.
- `POST /api/v1/models/{model_id}/download` first reuses verified installed bytes.
  An optional body accepts strict Boolean `accepted_license` and `commercial`
  fields. It cannot accept caller-supplied URLs, paths, checksums or secrets.

Unknown local licenses do not prevent integrity inspection or imply usage rights.
Known manual-acceptance requirements are enforced before activation. Downloads
require a known license and reviewed remote URL/hash/size/provenance evidence.
Commercial requests require an explicit `allow` record. The current real manifest
has no download-eligible upstream identity: local weights work directly, while
unreviewed remote downloads remain gated. The downloader itself is implemented
and exercised through real production code with fixture HTTP transports.

## Download and engine-cache behavior

`models/downloader.py` permits direct HTTPS URLs on the exact allowlisted hosts
`github.com`, `raw.githubusercontent.com` and `huggingface.co`. TLS verification
uses HTTPX defaults; environment proxies are disabled. Redirects, userinfo,
queries, fragments, nonstandard ports, unsafe paths and non-identity HTTP content
encodings are rejected. Consequently signed/CDN redirect workflows need a future
reviewed direct-source policy, rather than being silently followed.

Each artifact uses `models/cache/<model_id>.part`, durable resume metadata and
an OS-backed inter-process lock. Resume requires matching strong ETag or valid
Last-Modified identity, manifest identity, exact Content-Range and registered
total size. Changed identities/HTTP 416 restart; a full HTTP 200 replaces partial
bytes. Missing/weak validators restart. Streamed bytes cannot exceed the exact
expected size or the 64 GiB download ceiling. Incomplete transfers retain fsynced
parts; complete parts can recover after a crash. Final bytes must pass SHA-256
verification before fsync and atomic rename. Corrupt downloaded parts are
quarantined. Source weights are never quarantined, deleted, chmodded or copied.
Symlinks/junctions and hardlinked cache files are rejected. Archives stay opaque;
there is no extraction or code execution.

`models/engine_cache.py` defines metadata under `roop_3d/cache/engines/`. Keys
include source artifact SHA-256, precision, GPU compute capability, driver, ORT
version, TensorRT version and validated min/opt/max shape profiles. Metadata also
binds compiled artifact bytes; incompatible keys miss and corruption fails closed.
This is cache infrastructure only: no TensorRT compilation, GPU throughput,
6 GB/12 GB performance qualification or optimized inference is claimed.

Existing nested `.gitignore` files already cover environments, node_modules,
model caches, quarantines and engines; ignore behavior was checked without
changing protected root files. No dependency additions were needed; the existing
hash-pinned lock remains unchanged and `uv pip check` validates all 39 packages.

## Verification

The final exact commands, exit codes and counts are recorded in
`STAGE03_VALIDATION.json`. The real local-provider campaign rehashed 95 installed
artifacts totaling 15,876,007,985 bytes, returned original paths, preserved all
94 historical discovery entries, and copied/downloaded zero models. The existing
real SCRFD media campaign also passed: two tracks across five frames, thumbnails,
comparison preview, FaceSet mappings and durable controls. It does not render
face-replacement output.

Tests exercise digest/size/read-permission failures, traversal, real Windows
junction rejection, read-only originals, configured matching, source drift,
license gates, interrupted/resumed/restarted transfers, corrupt quarantine,
oversized responses, redirects, opaque archives, thread/process serialization,
cache-to-adapter resolution and all engine key components. Only the HTTP boundary
is simulated for downloader tests; no live external model downloads occur.

Initial downloader validation exposed Windows `fsync` rejecting a read-only
descriptor, plus oversized pytest IDs derived from byte payloads. The owned
partial-file handle now has the required write access for fsync and test IDs are
short; assertions were preserved. Later review added missing-root fallback and
strict acceptance Booleans, followed by targeted and full regression runs.
The existing FastAPI/Starlette HTTPX deprecation warning remains; no dependency
upgrade was introduced as part of this stage.

Final full suite: **177 passed, 18 additional subtests passed, zero failed,
zero skipped, zero blocked tests**, with one existing dependency warning.
All validation commands ran from this checkout's `app/` directory:

| Exact command | Exit | Result |
| --- | --- | --- |
| `roop_3d/.venv/Scripts/python.exe -B -m pytest -c roop_3d/pyproject.toml --confcutdir=roop_3d roop_3d/tests -q` | 0 | 177 tests and 18 subtests passed |
| `roop_3d/.venv/Scripts/python.exe -B -m roop_3d.tests.manual_stage03_acceptance` | 0 | 95 original local weights rehashed; zero copies/downloads |
| `roop_3d/.venv/Scripts/python.exe -B -m roop_3d.tests.manual_stage02_acceptance` | 0 | Real SCRFD/media/control regression passed |
| `roop_3d/.venv/Scripts/python.exe -B roop_3d/audit/verify_isolation.py` | 0 | 2,647 historical and 68,392 immutable hashes checked; zero violations |
| `uv pip check --python roop_3d/.venv/Scripts/python.exe` | 0 | 39 installed packages compatible |

Final integrity verdict: **passed**, timestamp `2026-10-30T12:38:28.694553+00:00`.
The audit observed 1,427 runtime files separately under the existing explicit
policy; these are not claimed byte-identical. Original baselines and the runtime
classification policy were never replaced. `git diff --check` also exits 0.

The next numbered stage requires its own contract. A rendering worker and
end-to-end face-replacement/volumetric output remain later gates; this stage does
not skip those prerequisites. No commit or push was requested or performed.

## Official sources checked

- [FLAME academic model license](https://flame.is.tue.mpg.de/modellicense.html):
  the FLAME 2020 manual/restricted entry and commercial-use denial.
- [InsightFace model zoo](https://github.com/deepinsight/insightface/blob/master/python-package/docs/model_zoo.md):
  model permissions must be considered separately from code licenses. No exact
  local artifact is assigned upstream provenance based on its filename.
- [RFC 9110](https://www.rfc-editor.org/rfc/rfc9110.html): Range/If-Range,
  representation validators and Content-Range semantics.
