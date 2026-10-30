module.exports = {
  run: [{ method: "script.stop", params: { uri: "start.js" } },
    { method: "shell.run", params: { path: ".", cache: "cache", message: ["python -B tools/project_admin.py preflight", "git pull --ff-only"] } },
    { method: "script.start", params: { uri: "install.js" } }]
}
