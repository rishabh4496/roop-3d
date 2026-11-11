# Universal model downloader (Stage 11, model-distribution series)

Every model and 3D asset roop-3d uses is described once, installed from zero-auth mirrors, verified by SHA-256, and can
be repaired in one call. Assets that can be built locally have a no-network path.

**Path mapping** (the prompt names `app/roop_3d/audit/local_models.py` and `roop_3d/tests/...`; the real layout is):
`app/roop_3d/audit/local_models.py` (manager + CLI), `audit/procedural_assets.py` (offline builders),
`audit/universal_model_manifest.json` (the manifest), `server/model_manifest_routes.py` (REST),
`server/ws.py` (WebSocket event), `tools/build_universal_manifest.py` (maintainer tool),
`app/roop_3d/tests/test_model_downloader.py`.
New files under `roop_3d/models/` would be silently git-ignored (`.gitignore:31`), which is why nothing new lives there.

## What is and is not "downloadable by anyone"

| Class | Entries | Status |
|---|---|---|
| Open licence (CC0 / MIT / Apache / CC-BY) | MakeHuman CC0 sources, Poly Haven HDRIs, YuNet (MIT), SFace (Apache-2.0), compiled CC0 assets | Fetched with **no acknowledgement** |
| Restricted or **unverified** licence | InsightFace det/ArcFace/landmarks (non-commercial research), InSwapper, HyperSwap, GPEN, CodeFormer, GFPGAN, XSeg, BiSeNet, occluder, SMIRK | Zero-auth URLs, but fetched only with `accepted_license: true` |
| No public source | **FLAME 2023 Open** (CC-BY-4.0) | `mirrors: []`. The official download is behind a free FLAME account, and 144 public Hugging Face repos were searched for the exact bytes without a hit. Place the file at `flame/flame2023/flame2023_Open.pkl`; its SHA-256 is enforced |

The licence column is the registry's truth, not a wish: most core weights are **not** CC0/MIT/Apache. Nothing is
relabelled. A registry value that is not a licence (`LOCAL-USE-ACKNOWLEDGED-UNVERIFIED`, `UNKNOWN`) is shown as
`UNVERIFIED`. `requires_acceptance` is *enforced by the manifest loader*: a non-open licence with
`requires_acceptance: false` fails to load. `commercial: true` requests skip any entry whose `commercial_use` is not `allow`.

## Manifest (`universal_model_manifest.json`)

`id, name, kind, category, install_root (app|repo), install_path, mirrors[], sha256, size_bytes, license, license_url,
license_note, commercial_use, requires_acceptance, required_for_stages[], optional, acquisition, registry_ids[]`, plus
`files[]` (asset sets), `depends_on[]` / `after[]` / `build_inputs{}` (ordering), `content_digest` and
`procedural_fallback` (compiled / bundled assets).

* `kind`: `weight` (one file), `asset_set` (many files under base-URL mirrors), `compiled` (built locally), `bundled` (ships with the app).
* A mirror is listed **only when byte identity to the pinned SHA-256 was proven**: HF Git-LFS object id at a pinned commit,
  a GitHub release `digest`, or a full hash. `hf-mirror.com` was tried and is only a 308 back to `huggingface.co`, so it is not listed.
* `required_for_stages` is derived from each model's *role* in the three pipeline stages. It is not traced from which
  sessions actually open at runtime.
* Regenerate with `python -m roop_3d.tools.build_universal_manifest` (needs network; the runtime never does).

## Downloading

* Range resume into `<file>.part` (+ `.part.json` validator); `If-Range` is sent only to the mirror that issued the validator.
* The final name appears only by atomic `os.replace` **after** the SHA-256 matches. Wrong bytes: `.part` deleted, next mirror.
  Bytes resumed from an earlier/other source that fail the hash are refetched clean from the same mirror once.
* 404/403/410 and range/length violations mark a mirror unusable (no retry); network errors and 429/5xx retry once, resuming.
* Redirects are followed manually, at most 5, only to `*.hf.co`, `*.huggingface.co`, `*.githubusercontent.com`,
  `*.googleusercontent.com`. Every hop is re-checked. Compressed responses and any response larger than the pinned size are refused.
* A file that exists under its final name with wrong bytes is moved to `<dir>/.quarantine/` (never overwritten or deleted).
* The same bytes under another name in `app/models` count as installed (e.g. `ultramax.onnx` is CodeFormer FP16).
* Set members download 4 at a time under one steady status; entries download 3 at a time.
* `trust_env=True`: HTTP(S) proxy environment variables are honoured (the older downloader ignores them).

## REST

| | |
|---|---|
| `GET /api/v1/models/manifest[?refresh=true]` | all entries: licence badge (`open`/`restricted`/`unverified`), `state`, `hash_status`, `installed_bytes`, live `download`, plus `disk` (`installed_bytes`, `download_needed_bytes`, `free_bytes`, `enough_space`) |
| `POST /api/v1/models/download` | `{}` queues every missing entry that needs no acknowledgement. `{"models":[ids],"accepted_license":true}` for specific ones. **202** with a `batch` (`queued`, `skipped[{id,reason,detail}]`). Ids that are registry ids, not manifest ids, keep the older synchronous path |
| `POST /api/v1/models/verify-and-repair` | re-hashes everything (cache bypassed), sets corrupt files aside, queues a repair batch for missing/corrupt entries |
| `GET` / `DELETE /api/v1/models/batches/{id}` | poll / cancel (verified partial bytes are kept) |

**Contract change:** `POST /models/download` with an empty body used to return 422 `MODEL_SELECTION_REQUIRED`
("bulk download is disabled"). It now queues the acknowledgement-free missing entries. `test_stage06_model_hub.py` was updated to match.

## WebSocket (`/ws/v1/pipeline/status`)

`model_download_progress` frames: `model_id, downloaded_bytes, total_bytes, percent, speed_mbps, eta_seconds, mirror_used,
status, batch_id, detail`. `status` is queued, downloading, verifying, compiling, ready, failed or cancelled.
`speed_mbps` is **megabits** per second over ~5 s. `eta_seconds` is `null` while the rate is unknown. A client that connects
mid-download receives the latest frame of every in-flight entry. `bytes_downloaded` is kept equal to `downloaded_bytes` for older producers.

## Offline (zero-network) builds

All four were run with sockets blocked and are deterministic.

* **`makehuman_cc0_head`**: compiled from the 130 hash-locked sources when cached (bit-identical to the pinned asset digest),
  else restored from `reconstruction/assets/makehuman_head_v1.npz` (386 KB, CC0, digest-checked). It was also rebuilt from freshly
  downloaded sources on the real internet and matched the pin.
* **`cc0_probe_bundle`** (5 SH studio probes + procedural pore tile): the committed bundle is pinned by array digest. It was rebuilt
  from freshly downloaded Poly Haven HDRIs and matched. If it is missing/corrupt and the HDRIs are not cached, five **analytic**
  probes are generated (`environment_source` says "Procedural ... not Poly Haven data"; state `PROCEDURAL_FALLBACK`).
* **`anatomical_pack_cc0`**: existing synthesis; `template_source` says whether MakeHuman or the analytic head was used.
* **`correspondence_maps`** (MakeHuman <-> FLAME barycentric): pure geometry from two local meshes (3 s). `NOT_APPLICABLE` until FLAME is installed.

## CLI

`python roop_3d/audit/local_models.py --verify` (unchanged checks + manifest: fails on any corrupt entry; add
`--require-installed` to also fail on missing non-optional entries), `--list`, `--download [ID ...] [--accept-license]`, `--repair`.

## Known limits

* No SMIRK / FLAME public mirror exists to list; SMIRK has a single Google Drive source and is a PyTorch pickle (**downloaded
  and hash-checked only, never loaded by this code**).
* Fully offline *from nothing* works for the MakeHuman head, probes, anatomical pack and (with FLAME) the maps. It does not work
  for ONNX weights: those are only downloadable.
* `core/model_downloader.py` (the older 9-model downloader) is left as is. It lists several `acceptable_sha256s` that share a
  12-hex prefix with the real digest and differ afterwards; those cannot be real hashes of other builds. They are unused by the
  new manager, which pins only the registry-verified digest.
