// Own app, own process tree, own virtual environment. No legacy lifecycle calls.
module.exports = {
  requires: { bundle: "ai" },
  daemon: true,
  run: [
    {
      method: "shell.run",
      params: { path: ".", cache: "cache", message: "python -B tools/project_admin.py preflight" }
    },
    {
      when: "{{!exists('cache/install-ready.json')}}",
      method: "script.start",
      params: { uri: "install.js" }
    },
    {
      when: "{{!exists('app/roop_3d_ui/node_modules') || !exists('app/roop_3d_ui/dist')}}",
      method: "shell.run",
      params: {
        path: "app/roop_3d_ui",
        message: ["npm ci", "npx --no-install vite build"],
        on: [{ event: "/(npm error|npm ERR!)/i", break: true }]
      }
    },
    {
      when: "{{exists('app/roop_3d_ui/node_modules') && exists('app/roop_3d_ui/dist')}}",
      method: "shell.run",
      params: {
        path: "app/roop_3d_ui",
        message: ["npx --no-install vite build"]
      }
    },
    {
      method: "shell.run",
      params: {
        path: ".",
        cache: "cache",
        venv: "app/roop_3d/.venv",
        env: {
          PYTHONDONTWRITEBYTECODE: "1",
          PYTHONUNBUFFERED: "1",
          UV_CACHE_DIR: "{{path.resolve(cwd, 'cache/uv')}}",
          UV_LINK_MODE: "copy",
          UV_PYTHON_DOWNLOADS: "never"
        },
        message: [
          "uv pip sync --require-hashes app/roop_3d/{{gpu === 'nvidia' && (platform === 'win32' || platform === 'linux') ? 'requirements.txt' : 'requirements.lock'}}",
          "python -B app/serve.py"
        ],
        on: [
          {
            event: "/(http:\\/\\/[0-9.:]+)/",
            done: true
          },
          {
            event: "/(PORT_IN_USE|INVALID_SETTINGS|ROOP3D_GUARD_ERROR|ModuleNotFoundError|error:)/",
            break: true
          }
        ]
      }
    },
    {
      method: "local.set",
      params: {
        url: "{{input.event[1]}}"
      }
    },
    {
      method: "process.wait",
      params: {
        url: "{{local.url}}/api/v1/health",
        interval: 1,
        title: "Starting Roop 3D",
        description: "Waiting for this app's loopback health endpoint"
      }
    }
  ]
}
