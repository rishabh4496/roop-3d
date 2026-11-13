"""W2 ops G18/G19: fail-closed isolation checks (inject a violation -> failure) and read-only local-model verification."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

AUDIT = Path(__file__).absolute().parents[1] / "audit"
from roop_3d.audit import boundary
from roop_3d.audit import local_models

if shutil.which("git") is None:
    pytest.skip("git is not on PATH", allow_module_level=True)


def git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=str(cwd), check=True, capture_output=True)


@pytest.fixture()
def checkout(tmp_path):
    """tmp/api/roop-3d (git repo, origin roop-3d) with app/, plus sibling tmp/api/roop-ultimate and a fake venv prefix."""
    api = tmp_path / "api"
    repo = api / "roop-3d"
    sibling_name = "roop-" + "ultimate"
    (api / sibling_name / "app").mkdir(parents=True)
    (repo / "app" / "pkg").mkdir(parents=True)
    (repo / "app" / "pkg" / "ok.py").write_text("import os\nx = 1\n", "utf-8")
    venv = repo / "app" / "roop_3d" / ".venv"
    venv.mkdir(parents=True)
    git(repo, "init", "-q")
    git(repo, "remote", "add", "origin", "https://github.com/someone/roop-3d.git")
    git(repo, "add", "-A")
    return type("C", (), {"repo": repo, "app": repo / "app", "venv": venv, "api": api})


def run(c, **kwargs):
    return boundary.run_checks(c.app, prefix=kwargs.pop("prefix", c.venv), **kwargs)


def add(c, relative: str, body: str):
    path = c.app / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, "utf-8")


def test_clean_checkout_passes_all_five_checks_with_computed_counts(checkout):
    result = run(checkout)
    assert result["ok"] is True and set(result["checks"]) == {"sibling_imports", "sibling_writes", "interpreter", "git_origin", "tracked_artefacts"}
    assert result["checks"]["sibling_imports"]["files_scanned"] == 1
    assert result["checks"]["tracked_artefacts"]["checked"] >= 1
    add(checkout, "pkg/two.py", "import json\nimport sys\n")
    again = run(checkout)
    assert again["checks"]["sibling_imports"]["files_scanned"] == 2                      # untracked files count: computed, not constant
    assert again["checks"]["sibling_imports"]["checked"] > result["checks"]["sibling_imports"]["checked"]


@pytest.mark.parametrize("body", [
    "import roop_ultimate.core\n",
    "from roop_ultimate import thing\n",
    "import sys\nsys.path.insert(0, r'C:/pinokio/api/roop-ultimate/app')\n",
    "import importlib.util as u\nspec = u.spec_from_file_location('m', r'C:\\\\pinokio\\\\api\\\\roop-ultimate\\\\app\\\\m.py')\n",
    "import sys\nfrom pathlib import Path\nsys.path.append(str(Path(__file__).resolve().parents[3]))\n",
    "import sys, os\nsys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))\n",
    "from roop_3d.audit.project_paths import legacy_root\nimport sys\nsys.path.insert(0, str(legacy_root()))\n",
])
def test_import_and_sys_path_escapes_into_a_sibling_checkout_fail(checkout, body):
    add(checkout, "pkg/bad.py", body)
    result = run(checkout)
    assert result["ok"] is False and not result["checks"]["sibling_imports"]["ok"]
    assert result["checks"]["sibling_imports"]["violations"][0]["path"] == "pkg/bad.py"


@pytest.mark.parametrize("body", [
    "from roop_3d.audit.project_paths import legacy_root\nroot = legacy_root()\nwith open(root / 'x.json', 'w') as f:\n    f.write('1')\n",
    "from pathlib import Path\nPath(r'C:/api/roop-ultimate/app/x.txt').write_text('boom')\n",
    "import shutil\nshutil.copy('a', r'C:/api/roop-ultimate/out.bin')\n",
    "from roop_3d.audit.project_paths import legacy_root\nlegacy_root().joinpath('d').mkdir()\n",
    "import os\nos.remove(r'C:/api/roop-ultimate/app/f')\n",
])
def test_write_targets_built_from_a_sibling_root_fail(checkout, body):
    add(checkout, "pkg/writer.py", body)
    result = run(checkout)
    assert not result["checks"]["sibling_writes"]["ok"], body


def test_reading_a_sibling_path_or_writing_inside_the_checkout_is_not_a_violation(checkout):
    add(checkout, "pkg/reader.py", "from roop_3d.audit.project_paths import legacy_root\n"
        "data = (legacy_root() / 'models.json').read_text()\nwith open('local.txt', 'w') as f:\n    f.write(data)\n")
    result = run(checkout)
    assert result["checks"]["sibling_writes"]["ok"] and result["checks"]["sibling_imports"]["ok"]


def test_taint_does_not_leak_between_functions(checkout):
    add(checkout, "pkg/scopes.py", "from roop_3d.audit.project_paths import legacy_root\n"
        "def a():\n    path = legacy_root()\n    return path.read_text()\n"
        "def b(path):\n    open(path, 'w').write('x')\n")
    assert run(checkout)["checks"]["sibling_writes"]["ok"]


def test_unparsable_source_fails_closed(checkout):
    add(checkout, "pkg/broken.py", "def (:\n")
    result = run(checkout)
    assert not result["ok"] and "unparsable" in result["checks"]["sibling_imports"]["violations"][0]["detail"]


def test_wrong_interpreter_wrong_origin_and_tracked_artefacts_fail(checkout):
    outside = run(checkout, prefix=checkout.api / "elsewhere")
    assert not outside["checks"]["interpreter"]["ok"] and not outside["ok"]
    git(checkout.repo, "remote", "set-url", "origin", "https://github.com/someone/roop-ultimate.git")
    assert not run(checkout)["checks"]["git_origin"]["ok"]
    git(checkout.repo, "remote", "set-url", "origin", "git@github.com:someone/roop-3d.git")
    assert run(checkout)["checks"]["git_origin"]["ok"]
    add(checkout, "models/w.onnx", "weights")
    add(checkout, "roop_3d/.venv/lib/x.txt", "venv")
    git(checkout.repo, "add", "-f", "-A")
    tracked = run(checkout)["checks"]["tracked_artefacts"]
    assert not tracked["ok"] and {v["path"] for v in tracked["violations"]} == {"app/models/w.onnx", "app/roop_3d/.venv/lib/x.txt"}


def test_missing_origin_remote_fails_closed(checkout):
    git(checkout.repo, "remote", "remove", "origin")
    assert not run(checkout)["checks"]["git_origin"]["ok"]


def test_cli_exit_code_is_nonzero_on_violation_and_json_names_it(checkout):
    audit = checkout.app / "roop_3d" / "audit"
    audit.mkdir(parents=True, exist_ok=True)
    for name in ("isolation.py", "boundary.py", "stage00.py", "verify_isolation.py"):
        shutil.copy(AUDIT / name, audit / name)
    add(checkout, "pkg/bad.py", "import roop_ultimate\n")
    done = subprocess.run([sys.executable, "-B", str(audit / "verify_isolation.py")], capture_output=True, text=True, cwd=str(checkout.app))
    assert done.returncode == 1
    report = json.loads(done.stdout)
    assert report["ok"] is False and report["violations"][0]["check"] in ("sibling_imports", "interpreter")
    assert any(v["path"] == "pkg/bad.py" for v in report["checks"]["sibling_imports"]["violations"])
    assert "historical" not in report and report["checks"]["sibling_imports"]["files_scanned"] >= 2


# ------------------------------------------------------------------------------------------------------- G19
def registry_for(path: Path, scope: str = "repository") -> dict:
    data = path.read_bytes()
    return {"models": [{"locator": {"scope": scope, "path": path.name}, "state": "VERIFIED_LOCAL", "size_bytes": len(data),
                        "sha256": hashlib.sha256(data).hexdigest()}]}


def test_absent_sibling_scope_is_unavailable_with_reason_not_a_crash(tmp_path):
    weight = tmp_path / "m.onnx"
    weight.write_bytes(b"w")
    registry = registry_for(weight)
    result = local_models.verify_registry(registry, {"repository": tmp_path / "no-such-checkout"})
    assert result["ok"] is True and result["checked"] == 0 and result["expected"] == 1
    assert result["unavailable_scopes"][0]["scope"] == "repository" and "absent" in result["unavailable_scopes"][0]["reason"]
    unresolved = local_models.verify_registry(registry, {}, {"repository": "legacy model source cannot be resolved"})
    assert unresolved["unavailable_scopes"] == [{"scope": "repository", "reason": "legacy model source cannot be resolved", "models": 1}]


def test_present_scope_still_detects_drift(tmp_path):
    weight = tmp_path / "m.onnx"
    weight.write_bytes(b"w")
    registry = registry_for(weight)
    weight.write_bytes(b"tampered")
    result = local_models.verify_registry(registry, {"repository": tmp_path})
    assert result["ok"] is False and result["checked"] == 0 and result["unavailable_scopes"] == []


def test_default_scopes_without_a_resolvable_sibling_reports_unavailable(tmp_path):
    (tmp_path / "roop_3d" / "audit").mkdir(parents=True)                      # no model_sources.json at all
    scopes, unavailable = local_models.default_scopes(tmp_path)
    assert "repository" not in scopes and "repository" in unavailable and "user_cache" in scopes


def snapshot(root: Path) -> dict[str, tuple[int, int]]:
    return {p.relative_to(root).as_posix(): (p.stat().st_size, p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()}


def test_cc0_verification_never_builds_a_missing_asset(tmp_path):
    before = snapshot(tmp_path)
    result = local_models.verify_local_cc0_assets(tmp_path)
    assert result["ok"] is False and "--build-assets" in result["violations"][0]["detail"]
    assert snapshot(tmp_path) == before and not (tmp_path / "roop_3d").exists()


def test_cc0_verification_of_a_real_asset_mutates_nothing(tmp_path):
    source = AUDIT.parents[1] / "roop_3d/.runtime/models/anatomical/makehuman_cc0_anatomical_pack.npz"
    if not source.is_file() or not source.with_name(source.name + ".json").is_file():
        pytest.skip(f"{source} (and its .npz.json sidecar) is not present on this machine")
    record = local_models.LOCAL_CC0_ASSETS["makehuman_cc0_anatomical_pack"]
    destination = tmp_path / record["relative_path"]
    destination.parent.mkdir(parents=True)
    shutil.copy(source, destination)
    shutil.copy(source.with_name(source.name + ".json"), destination.with_name(destination.name + ".json"))
    before = snapshot(tmp_path)
    result = local_models.verify_local_cc0_assets(tmp_path)
    assert result["ok"] is True and result["assets"][0]["sha256"] == hashlib.sha256(destination.read_bytes()).hexdigest()
    assert snapshot(tmp_path) == before
    sidecar = destination.with_name(destination.name + ".json")
    sidecar.write_text(json.dumps({**json.loads(sidecar.read_text("utf-8")), "asset_digest": "0" * 64}), "utf-8")
    tampered = local_models.verify_local_cc0_assets(tmp_path)
    assert tampered["ok"] is False and "digest" in tampered["violations"][0]["detail"]
