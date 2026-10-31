# Stage 13: Local ComfyUI Bridge and Optional Provider Interface

## Scope and files

All Stage 13 source and runtime writes remain under `app/roop_3d/`.
No legacy application, launcher, model artifact, audit baseline, dependency,
or environment was modified. Existing pinned `httpx==0.28.1` and
`websockets==15.0.1` are used without changing the lockfile.

Files added:

- `bridges/comfyui_bridge.py`
- `bridges/provider_protocol.py`
- `bridges/cloud_stub.py`
- `tests/test_stage13_bridges.py`
- `audit/STAGE13_REPORT.md`

## Protocol evidence

`VERIFIED_REMOTE` protocol documentation, reviewed 2026-10-31:

- [Official HTTP and WebSocket routes](https://docs.comfy.org/development/comfyui-server/comms_routes)
- [Official execution messages](https://docs.comfy.org/development/comfyui-server/comms_messages)
- [Official API example](https://github.com/Comfy-Org/ComfyUI/blob/v0.43.0/script_examples/websockets_api_example.py)
- [Tagged server implementation](https://github.com/Comfy-Org/ComfyUI/blob/v0.43.0/server.py)

The reviewed compatibility floor is ComfyUI 0.43.0. This is a conservative floor,
not a claim about the first release implementing each feature. The tagged server
validates client-supplied UUID prompt IDs and supports prompt-specific interrupts.
The bridge rejects older or unrecognized server versions before submitting, and
rechecks the protocol version before cancellation. It never requests a global
interrupt or queue clear.

## Local bridge behavior

`BridgeConfig` defaults to `http://127.0.0.1:8188`. Only loopback IP origins are
accepted; `localhost` is normalized to `127.0.0.1`. Credentials, query strings,
nonlocal hosts, environment proxies, and HTTP/WebSocket redirects are prohibited.
No model sessions are loaded by importing or constructing the bridge.

Submission probes `/system_stats` and `/object_info`. All workflow node classes,
required inputs, exposed combo choices (including checkpoint names), graph links,
and cycles are checked before `/prompt`. Metadata verifies advertised availability,
not checkpoint hashes, provenance, licenses, or inference quality; those remain
`UNKNOWN` unless independently established.

A durable receipt under `roop_3d/.runtime/bridges/comfyui/` reserves one prompt UUID
and client UUID per job and workflow digest. Repeated submission with the same job
returns the same handle; changed workflows fail with an idempotency conflict.
After an ambiguous POST failure the bridge preserves the receipt and never blindly
reposts. The same handle can poll for completion or request cancellation. A truly
unsubmitted ambiguous job requires investigation and a new job ID to retry.

`wait()` correlates JSON events by prompt ID, uses bounded socket reads and an
overall deadline, reconnects with bounded exponential backoff, and polls matching
history to recover completion after missed messages. Legacy binary PNG/JPEG preview
frames are accepted only after a correlated node event on that client connection.
Previews are best effort, are not completion evidence, and are not replayed from
before subscription. Unsupported binary event types are ignored.

Cancellation deletes only the matching queue item and sends a targeted interrupt.
Timeout preserves correlation and does not claim the server stopped executing.
Explicit cancellation remains available after a timeout. Cancellation acknowledgments
mean the requests were accepted, not proof that a noncooperative custom node stopped.

Output retrieval accepts only sanitized descriptors from the completed prompt's
history, then streams `/view` under content-type and byte limits. PNG/JPEG dimensions
are bounded before decoding. `composite_output()` passes the BGR crop and caller's
validated transform/track geometry to the existing Stage 09 compositor. No generated
media is written by the bridge.

## Remote interface and privacy

The typed protocol defines capabilities, consent validation, submit, poll, cancel,
and output fetch. `CloudStub` defaults to `NOT_CONFIGURED` and has no network client,
endpoint, credential, or upload implementation.

Future adapters must be exposed through `ConsentGatedProvider`. Every operation
first requires `remote_processing_enabled is True` and a valid signed per-job
token. HMAC-SHA256 claims bind the job ID, provider ID, exact destination hostname,
ordered filenames, byte sizes, and SHA-256 content identities. Tokens expire in at
most 900 seconds. A trusted local user-consent handler must explicitly acknowledge
the manifest before issuing a token; no public signing endpoint or automatic
consent is provided.

Signing keys come from a supplied OS-store value or `ROOP3D_CONSENT_SECRET`, with a
minimum 32-byte key. Keys and tokens are never stored in bridge receipts, settings,
or logs. Tokens are hidden from object representations. A logging filter redacts
bearer tokens, credential fields, signed URL paths/queries, and exception details;
future adapter logging handlers must use it. The current adapters emit no raw
request or exception logs.

## Verification record

Commands run from `G:\pinokio\api\roop-3d\app`:

| Command | Exit | Result |
| --- | ---: | --- |
| `roop_3d\.venv\Scripts\python.exe -B -m pytest -c roop_3d\pyproject.toml --confcutdir=roop_3d roop_3d\tests\test_stage13_bridges.py -q` | 0 | 30 passed |
| `roop_3d\.venv\Scripts\python.exe -B -m pytest -c roop_3d\pyproject.toml --confcutdir=roop_3d roop_3d\tests -q` | 0 | 375 passed, 18 subtests passed; one existing Starlette TestClient deprecation warning |
| `roop_3d\.venv\Scripts\python.exe -B roop_3d\audit\verify_isolation.py` | 0 | Historical 2,645/2,645; immutable comprehensive 68,390/68,390; zero violations; runtime activity separately classified by the existing policy |

The tests use mock HTTP and WebSocket transports around the actual bridge and
consent code. They cover readiness, missing nodes/checkpoints/inputs, idempotence,
ambiguous submissions, progress/preview correlation, disconnect/reconnect, timeout,
cancellation, protocol incompatibility, unsafe outputs, size/pixel limits, corrupt
receipts, consent tampering/expiry/destination mismatch, and secret redaction. A real
in-process WebSocket handshake verifies redirect refusal, and a real Stage 09
compositor verifies the decoded-crop handoff.

Earlier attempts exposed one invalid compositor test fixture (missing required
landmark geometry) and one overlapping pytest run that reused the active suite's
temporary directory (16 setup errors). Both were corrected without weakening the
assertions; subsequent runs were sequential and passed.

Live default-endpoint probe: `BLOCKED`, `COMFY_UNAVAILABLE`; no responding ComfyUI
server was available at `127.0.0.1:8188`. No actual ComfyUI workflow or model inference
was executed. Cloud execution is `NOT_CONFIGURED`; no external media transfer was
attempted. The mocked acceptance gate is verified locally; live render quality,
custom-node behavior, GPU use, and performance remain `UNKNOWN`.

## Next gate

Required Stage 13 acceptance: pass, zero final failures and zero blocked required
checks. The optional live render campaign is blocked by the absent local service.
Use a configured compatible local ComfyUI server and reviewed workflow for that
separate campaign. No cloud provider is enabled by this stage.
