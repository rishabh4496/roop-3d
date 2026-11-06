from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess
import sys

import pytest
from roop_3d.audit.isolation import AuditError, safe_path
from roop_3d.audit.project_paths import application_root, legacy_root, project_root


def test_project_has_its_own_git_directory_and_environment() -> None:
    root = project_root()
    external = legacy_root()
    assert root != external
    assert (root / ".git").is_dir()
    assert not (root / ".git").is_symlink()
    assert not (root / ".git/commondir").exists()
    assert not (root / ".git/objects/info/alternates").exists()
    assert (application_root() / "roop_3d/.venv").is_dir()
    assert Path(sys.prefix).absolute() == application_root() / "roop_3d/.venv"
    assert (root / ".roop3d-project.json").is_file()


def test_unified_app_allows_direct_writes_and_imports() -> None:
    root = project_root()
    assert safe_path(root, "app/api.py") == root / "app" / "api.py"


def test_pinokio_menu_is_independent_and_has_a_real_start_action() -> None:
    root = project_root()
    script = """
const assert = require('node:assert/strict');
const menu = require('./pinokio.js');
(async () => {
 const info = (running, installed, url) => ({running: p => running && p === 'start.js', exists: () => installed, local: () => ({url})});
 let items = await menu.menu({}, info(false, true));
 assert.equal(items[0].text, 'Start Roop 3D'); assert.equal(items[0].href, 'start.js');
 items = await menu.menu({}, info(true, true, 'http://127.0.0.1:8030'));
 assert.equal(items[0].text, 'Open Roop 3D'); assert.equal(items[0].href, 'http://127.0.0.1:8030');
 assert(!JSON.stringify(items).includes('roop-ultimate'));
 const launch = require('./start.js');
 const captured = launch.run.find(s => s.method === 'local.set');
 assert.equal(captured.params.url, '{{input.event[1]}}');
 const server = launch.run.find(s => s.params && s.params.venv);
 assert.equal(server.params.venv, 'app/roop_3d/.venv');
 assert(!JSON.stringify(launch).includes('../roop-ultimate'));
})().catch(e => { console.error(e); process.exit(1) });
"""
    result = subprocess.run(["node", "-e", script], cwd=root, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
