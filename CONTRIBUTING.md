# Develop in this app only

Open `<PINOKIO_HOME>/api/roop-3d` in Pinokio Files / Ask AI or your editor.
This is the development workspace for every following Roop 3D stage. Do not return
to the Roop Ultimate checkout to edit or run the old embedded copy.

Use this repository's own branches and `origin` (`rishabh4496/roop-3d`). The `main`
branch here is unrelated to Roop Ultimate's `main`. Never push to its origin.
Git hooks reject foreign remotes, symlinks and generated artifacts. There is no
shared worktree or alternate Git object database.

The app's startup/install tooling sets `core.hooksPath=.githooks` in this repo only.
For a manually created clone, from `app/` run:

    roop_3d/.venv/Scripts/python.exe -B -m roop_3d.tools.install_git_guard

Run the complete local suite and the read-only legacy integrity check before
committing. Tests, temporary repositories, caches and reports stay in this checkout.
Stage explicit source paths; do not force-add ignored data. Do not disable hooks.

All production writes must use checked owned paths. Existing external models are
read-only inputs addressed by a locally computed digest. Neither model reuse nor
the audit permits changing a legacy model, cache, environment, setting or source.
The external-write unit test proves the guard rejects the legacy root before a
write occurs. Git hooks are development guards, not an OS filesystem sandbox.

Root launchers follow Pinokio's LivePortrait menu and Mochi daemon/shell patterns.
Ready URLs use a capture group followed by `local.set` with `input.event[1]` and
actual health polling. The app binds only to loopback; its process belongs to this
Pinokio app ID. Stopping/updating/resetting it must never stop another app.

Preserve the reference inference path when later optimizations are introduced.
Do not label a declared package or optional dependency as a working model adapter.
Any model/provenance/license/performance claim needs evidence from its own gate.
