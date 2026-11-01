// Provisioning mirrors prototype/system/examples/facepoke/{install,torch}.js.
// Roop 3D uses ORT; this helper provisions its GPU runtime without PyTorch.
module.exports = {
  run: [{
    when: "{{gpu === 'nvidia' && (platform === 'win32' || platform === 'linux')}}",
    method: "shell.run",
    params: {
      path: ".", cache: "cache", venv: "app/roop_3d/.venv",
      env: { PYTHONDONTWRITEBYTECODE: "1", UV_CACHE_DIR: "{{path.resolve(cwd, 'cache/uv')}}", UV_LINK_MODE: "copy", UV_PYTHON_DOWNLOADS: "never" },
      message: ["uv pip sync --require-hashes app/roop_3d/requirements.txt"],
      on: [{ event: "/(ROOP3D_GUARD_ERROR|ModuleNotFoundError|error:)/", break: true }]
    }
  }]
}
