# Roop 3D — standalone application rules

This checkout is the standalone Roop 3D Pinokio app. It is a dedicated Git repository,
Python environment and process.

## Unified Architecture & Direct Modification

- All legacy file protections within this repository are removed. Any file in `app/`, `app/roop/`,
  `app/api.py`, `roop_3d/`, and the frontend can be directly modified and integrated cleanly.
- No artificial file-hash locks or monkey-patching barriers exist inside this repository.
- Sibling applications (such as `roop-ultimate`) remain external: never write to or modify
  sibling folders. All work takes place inside this checkout.

## Layout

Root: Pinokio metadata/launchers, documentation, Git hooks and project configuration.
`app/`: unified backend, `app/api.py`, `app/roop/`, and `app/roop_3d/`.
`app/roop_3d_ui/`: unified React workstation frontend.
`app/roop_3d/.venv/`: dedicated runtime environment.
`cache/`, `logs/` and runtime directories: ignored local state.

## Git and stage discipline

- Use ONLY this repository's origin, the dedicated `roop-3d` repository. Never push
  this project to the `roop-ultimate` remote. The Git guard enforces the identity.
- Stage source explicitly. Never stage virtual environments, caches, media, FaceSets,
  embeddings, checkpoints, engines or secrets.
- Keep processing local. Respect 6 GB and 12 GB GPU profiles separately.

## Pinokio launchers

Resolve PINOKIO_HOME from the user configuration. Capture `(http:\/\/[0-9.:]+)` and pass
`input.event[1]` directly to `local.set.url`. Poll the actual health endpoint.
Pinokio owns this app's process tree.

Tests from this project's `app/` directory:

    roop_3d/.venv/Scripts/python.exe -B -m pytest -c roop_3d/pyproject.toml --confcutdir=roop_3d roop_3d/tests -q

On Unix use `.venv/bin/python`. Test/cache paths must remain inside this checkout.
