module.exports = {
  requires: { bundle: "ai" },
  run: [{
    method: "shell.run",
    params: { path: ".", cache: "cache", message: "python -B tools/project_admin.py preflight" }
  }, {
    method: "shell.run",
    params: {
      path: ".", cache: "cache", venv: "app/roop_3d/.venv",
      env: { PYTHONDONTWRITEBYTECODE: "1", UV_CACHE_DIR: "{{path.resolve(cwd, 'cache/uv')}}", UV_LINK_MODE: "copy", UV_PYTHON_DOWNLOADS: "never" },
      message: ["uv pip sync --require-hashes app/roop_3d/requirements.lock"],
      on: [{ event: "/(ROOP3D_GUARD_ERROR|ModuleNotFoundError|error:)/", break: true }]
    }
  }, {
    method: "script.start",
    params: { uri: "torch.js" }
  }, {
    // Reproducible frontend install (autogpt example: shell.run + path + npm). `npm ci` fails loudly on a stale lockfile.
    method: "shell.run",
    params: {
      path: "app/roop_3d_ui",
      message: ["npm ci", "npm run build"],
      on: [{ event: "/(npm error|npm ERR!)/i", break: true }]
    }
  }, {
    // Only after Python and frontend installs both succeeded is the app marked ready.
    method: "shell.run",
    params: {
      path: ".", cache: "cache", venv: "app/roop_3d/.venv",
      env: { PYTHONDONTWRITEBYTECODE: "1" },
      message: ["python -B tools/project_admin.py mark-ready"],
      on: [{ event: "/(ROOP3D_GUARD_ERROR|ModuleNotFoundError|error:)/", break: true }]
    }
  }]
}
