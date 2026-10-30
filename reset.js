module.exports = {
  run: [{ method: "script.stop", params: { uri: "start.js" } },
    { method: "shell.run", params: { path: ".", message: "python -B tools/project_admin.py check-reset" } },
    { method: "fs.rm", params: { path: "app/roop_3d/.venv" } },
    { method: "fs.rm", params: { path: "cache/install-ready.json" } }]
}
