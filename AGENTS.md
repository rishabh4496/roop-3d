# Roop 3D — independent application rules

This checkout is the standalone Roop 3D Pinokio app. It is a different Git repository,
Python environment and process from Roop Ultimate. Perform all later stages HERE.

## Hard write boundary

- Write only inside this checkout, resolved from this file's parent directory.
- Never edit, format, move, delete or change permissions in a sibling app, especially
  `roop-ultimate`. Do not modify its launchers, settings, code, environment or Git state.
- Never share code directories, virtual environments or Git metadata through links,
  junctions, worktrees or hardlinks. Reject symlink/junction traversal before writes.
- Installed legacy model artifacts may be read through the hashed model registry.
  They are READ-ONLY inputs, not owned outputs. Do not copy gigabytes of weights.
- Never import legacy application modules or invoke its install, update, start,
  reset, queue, stop or configuration APIs. Its runtime may continue independently.

## Owned layout

Root: Pinokio metadata/launchers, documentation, Git hooks and project policy.
`app/roop_3d/`: typed Python code and tests.
`app/roop_3d_ui/`: future frontend; Vite development port 5174.
`app/roop_3d/.venv/`: dedicated runtime, never another app's interpreter.
`cache/`, `logs/` and application runtime directories: ignored local state.

## Git and stage discipline

- Use ONLY this repository's origin, the dedicated `roop-3d` repository. Never push
  this project to the `roop-ultimate` remote. The Git guard enforces the identity.
- Run the full local Roop 3D test suite before committing and pushing.
- Stage source explicitly. Never stage environments, caches, media, FaceSets,
  embeddings, checkpoints, engines or secrets, including ignored files by force.
- Implement only the requested numbered stage. Existing empty package declarations
  are package structure, not working inference adapters; do not call them shipped.
- Preserve reference inference paths. No speed/fidelity claim without real output
  evidence and the applicable hardware-specific acceptance campaign.

## Models, privacy and runtime

- Model existence/hashes, provenance, inference, rights and performance are separate
  evidence states. Never invent a URL, hash, tensor shape, license or measurement.
- Respect 6 GB and 12 GB GPU profiles separately. Do not load inference models merely
  to boot the API. Optional diffusion/TRT/FLAME dependencies remain optional.
- Keep processing local. Require per-job consent before any remote media/biometric
  transfer. Store secrets only in environment/OS stores, never bundles or logs.
- Validate archive traversal, expansion sizes and formats before adding import APIs.
- Bind the service to 127.0.0.1; API defaults to 8030. Never kill another port owner.
- Before completing a stage, run the read-only legacy integrity audit. The preserved
  source/model baseline is never silently replaced. Runtime activity is separately
  classified by the explicit audit policy; a source/model drift halts the stage.

## Pinokio launchers

Resolve PINOKIO_HOME from the user configuration, then read the local PINOKIO.md
and closest examples before changing scripts. Preserve daemon lifecycle,
relative shell paths and the dedicated venv. Capture `(http:\/\/[0-9.:]+)` and pass
`input.event[1]` directly to `local.set.url`. Poll the actual health endpoint.
No custom app-wide kill operations: Pinokio owns this app's process tree.

Tests from this project's `app/` directory:

    roop_3d/.venv/Scripts/python.exe -B -m pytest -c roop_3d/pyproject.toml --confcutdir=roop_3d roop_3d/tests -q

On Unix use `.venv/bin/python`. Test/cache paths must remain inside this checkout.
