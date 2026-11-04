# Stage 12 — Model governance and secure model management

## Technical implementation plan

1. Extend the existing `ModelRecord` contract with explicit governance metadata:
   source revision, architecture, required base, redistribution and attribution,
   hardware and precision requirements, dependency compatibility, serialization
   format, and artifact security status. Keep unestablished facts as `UNKNOWN`;
   do not infer rights or provenance from local hashes.
2. Add a compatibility scanner to the existing model hub path. It will combine
   registered dependency constraints, runtime versions, artifact format, hardware
   profile, and integrity checks into one lifecycle state. Only `READY` records
   can be reported as inference-ready. Missing or unknown evidence fails closed.
3. Harden managed downloads while retaining the current downloader: pin every
   download to its registered digest and exact size; reject archives and executable
   payloads; quarantine corrupt managed cache entries; keep resumable partials
   inactive; expose download/verification/failure states; atomically publish only
   verified files. No downloaded installer or model code is executed.
4. Add immutable version pins and rollback metadata to `ModelCatalog`, stored only
   under this checkout's runtime directory and written atomically. A pin may target
   only a registered version that passes the same governance gate; old versions
   remain intact.
5. Add registry, scanner, cache quarantine, lifecycle, cancellation/recovery,
   version pin/rollback, and serialization safety regression tests. Then run the
   full Python suite, production frontend build, local model audit, and the required
   legacy isolation audit.

## Evidence and compatibility policy

The preserved Stage 00 discovery CSV is immutable source evidence: its SHA-256 is
checked by `model_manifest.json`. This stage leaves it unchanged and carries
current governance fields on canonical registry records. Existing values such as
`UNKNOWN`, absent upstream revisions, unreviewed rights, and unverified runtime
compatibility remain explicit. A local SHA-256 proves byte identity against the
local registry only; it does not establish repository origin, license, safety,
compatibility, or inference quality.

Stage 06 adapter records preserve their separate base-model license requirements.
Apache-2.0 redistribution notices are based on the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0.html);
MIT notice requirements are based on the [OSI MIT License](https://opensource.org/license/mit).
The MiniMax H3 and Qwen base-model permissions remain separate from their adapter
licenses.

Downloads are single immutable artifacts. Archive extraction is intentionally not
supported; archive-like formats are rejected before network access. ONNX and
safetensors are treated as data formats, but still require registered hashes and
compatibility review. Pickle-based PyTorch checkpoints are blocked from inference
unless a future adapter uses a restricted loader and a separate security review.

## Acceptance record

Implemented the canonical governance fields on all 121 registry records, including
the Stage 06 and Stage 16 supplemental catalogs. The legacy `filenames` key remains
supported and is validated against `weight_filenames`. Exact source revisions are
accepted only as full commit hashes; mutable tags remain `UNKNOWN`. The API now
reports base-model license terms separately from adapter terms.

The compatibility scanner gates both `LocalWeightProvider.open` and the shared
ONNX runtime session loader. Downloads retain cancellation and resume support,
report `DOWNLOADING`/`VERIFYING`/terminal lifecycle states, reject pickle and archive
formats, verify exact size and SHA-256, quarantine corrupt managed-cache files, and
publish only through an atomic rename. Archive extraction is unsupported, so no
archive can be expanded. PyTorch loading is exposed only through a restricted
`weights_only=True` helper and fails closed on older runtimes. Version updates use
immutable registry IDs; pins and rollback history are atomic and old artifacts are
left intact.

### Evidence status

Current registry evidence is incomplete by design: 18 of 121 records have an exact
full commit revision and identified original repository; five have a model license
identifier; four have reviewed redistribution terms; zero have completed model
security review. Therefore the scanner reports **zero models `READY` for inference**.
The 95 local SHA-256 identities remain valid, but local hashes do not supply the
missing source, license, hardware, compatibility, precision, or security evidence.
The four Stage 06 adapters retain their separate base-model license details and
manual acceptance gates. The legacy CSV was not rewritten because the canonical
manifest checks its digest as preserved discovery evidence.

This gate intentionally prevents the existing real-weight inference tests from
loading records whose license or security state is still unknown. Those tests skip
with the governance reason; synthetic contract tests continue to exercise runtime
behavior without claiming production model execution.

### Verification

- Full Python suite: **941 passed, 38 skipped, 18 subtests passed**, one existing
  Starlette/httpx deprecation warning.
- Frontend `npm run build`: **passed**; typecheck, lint, and Vite build succeeded.
  ESLint reported 80 pre-existing warnings and no errors.
- `audit/local_models.py --verify`: **95/95** local registry hashes verified,
  no violations.
- `audit/verify_isolation.py --report`: **2,645/2,645** historical and
  **68,390/68,390** comprehensive protected entries passed, no violations.

Before commit, a repeated full suite reported 940 passed, 38 skipped and one
worker-test failure after its pytest temporary directory disappeared during a
worker-thread path check. The complete `test_stage12_worker.py` module then passed
on rerun (3 passed); no worker implementation was changed for this transient failure.
