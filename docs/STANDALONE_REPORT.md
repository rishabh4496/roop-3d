# Independent Pinokio application

The first Stage 01 implementation lived under Roop Ultimate with a separate script.
That did not meet the user's separate-app requirement. This extraction creates
`<PINOKIO_HOME>/api/roop-3d` with its own Pinokio tile/menu, Git repository, environment,
cache, logs and process. Roop Ultimate's source and Git configuration were not edited.

## Verified ownership

- Pinokio app ID/title: `roop-3d` / **Roop 3D**, separate from `roop-ultimate`.
- Source/UI: `app/roop_3d/` and `app/roop_3d_ui/` in this independent checkout.
- Venv: `app/roop_3d/.venv`, freshly installed from the hash-pinned lock with
  `UV_LINK_MODE=copy` and this app's own cache (38 distributions).
- Own Git directory: no worktree, shared Git objects or alternate object database.
- Dedicated remote identity: `rishabh4496/roop-3d`; other push targets are rejected.
- No weights, old venv, caches or legacy application modules were copied. Existing
  model artifacts remain read-only inputs. The copy ledger records the prior source
  revision and original source hashes, not the modified destination hashes.
- The old embedded source stays unchanged as historical material. All next stages
  must use this new app, never the old embedded copy or old launcher.

## Tests and actual runs

Commands from this project's `app/`:

| Command | Exit | Evidence |
| --- | --- | --- |
| `roop_3d/.venv/Scripts/python.exe -B -m pytest -c roop_3d/pyproject.toml --confcutdir=roop_3d roop_3d/tests -q` | 0 | 59 passed, zero skipped; 18 additional unittest subtests passed |
| `roop_3d/.venv/Scripts/python.exe -B roop_3d/audit/local_models.py --verify` | 0 | All 95 original model-artifact hashes matched through read-only locators |
| `roop_3d/.venv/Scripts/python.exe -B roop_3d/audit/verify_isolation.py --report` | 0 | 68,392 protected immutable files matched; zero unauthorized additions |

One upstream Starlette TestClient/Httpx deprecation warning remains visible. Assertions
were not weakened and core behavior was not mocked. The original source/model baseline
was not replaced; allowed legacy runtime activity is reported separately. Detailed local
audit reports are ignored to avoid publishing user media filenames.

Live Pinokio discovery returned distinct app IDs for Roop 3D and Roop Ultimate. The
new app reported its own root, `start.js`, URL `http://127.0.0.1:8030` and `ready=true`.
Live health returned `pinokio_app_id=roop-3d`, `standalone=true`; the Pinokio project
route `/p/roop-3d` returned HTTP 200. Roop Ultimate's original API remained HTTP 200.

All five launcher files passed Node syntax checks. Tests execute the actual menu
export and verify independent Start/Open actions, venv path and capture/index-1
handoff. Real Git-index tests reject disallowed files, and remote tests reject the
HTTPS/SSH Roop Ultimate targets. The actual external legacy root is rejected by the
write guard before mutation. The native Uvicorn socket/readiness/shutdown path is
also exercised by the offline suite.

The optional combined live stop/reinstall check was rejected by automatic execution
review ("blocked by policy") and was not executed. Both apps were left running.
No live Reset deletion or Update pull was performed or counted as passing.

## Workflow and next stage

Pinokio home was resolved from the user's configuration. Destination is its independent
`api/roop-3d` directory. The menu follows the LivePortrait example; daemon/shell/venv
structure follows Mochi. The required URL capture group passes `input.event[1]` directly
to `local.set`, followed by documented `process.wait` health polling. Reset has explicit
owned-path checks before `fs.rm`; update/reset target only this app's `start.js`.

Root AGENTS.md binds future work to this checkout. README documents JavaScript, Python
and Curl API usage, app navigation and local state. The earlier Stage 00/01 records under
the package audit directory are historical evidence, not current ownership instructions.
Hooks are development guards, not an OS filesystem sandbox. The product is still the
Stage 01 scaffold: no video replacement, volumetric quality, GPU inference or performance
qualification is claimed. The next numbered implementation stage belongs in this app.
