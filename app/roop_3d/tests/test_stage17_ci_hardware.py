"""Stage 17 Automated Tests: CI/CD Pipeline, Hardware Compatibility & Release System.

Verifies:
1. Current workflow gates, isolated CLI invocation and failure propagation without running nested checks.
2. Hardware profile specifications (RTX 4070 12GB resident, RTX 3060 6GB clamped, CPU fallback).
3. Dependency audit matrix integrity across all 14 required dependencies.
4. Cryptographic HMAC signature and manifest integrity of release artifacts.
5. CycloneDX Software Bill of Materials (SBOM) schema conformance.
6. Model Compatibility Register completeness against registered catalog.
7. SHA-256 release checksums completeness.
8. Project admin CLI ci-gate integration.
"""
from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path
import shlex
import subprocess
import sys

import pytest
import yaml

# Ensure repo root and app directory are in sys.path
TEST_DIR = Path(__file__).resolve().parent
APP_DIR = TEST_DIR.parents[1]
ROOT = APP_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from tools.ci_runner import GATES, verify_hardware_profile, run_pipeline
from tools.release_packager import SECRET_KEY, sha256_file
from roop_3d.models.registry import ModelRegistry


def test_ci_workflow_runs_current_python_gates_and_the_offline_quality_suite() -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text("utf-8"))
    jobs = workflow["jobs"]
    python = jobs["python-tests"]
    assert len(GATES) == 15
    assert python["strategy"]["matrix"]["gate"] == [1, 2, 7, 9, 10, 11, 14]
    assert python.get("continue-on-error", False) is False
    commands = [(step, shlex.split(step["run"])) for step in python["steps"]
                if "run" in step and "tools/ci_runner.py" in step["run"]]
    assert len(commands) == 1
    step, command = commands[0]
    assert command[1:3] == ["-B", "tools/ci_runner.py"]
    assert command[command.index("--gate") + 1] == "${{ matrix.gate }}"
    assert command[command.index("--hardware-profile") + 1] == "cpu"
    assert not step.get("continue-on-error", False)

    quality = jobs["offline-quality"]
    test_steps = [(step, shlex.split(step["run"])) for step in quality["steps"]
                  if "run" in step and "-m pytest" in step["run"]]
    assert len(test_steps) == 1
    step, command = test_steps[0]
    assert step["working-directory"] == "app"
    assert command[1:4] == ["-B", "-m", "pytest"]
    assert command[command.index("-c") + 1] == "roop_3d/pyproject.toml"
    assert "--confcutdir=roop_3d" in command
    assert {
        "roop_3d/tests/test_identity_asset_session.py",
        "roop_3d/tests/test_visual_quality_regressions.py",
        "roop_3d/tests/test_occlusion_compositor.py",
        "roop_3d/tests/test_ultramax_quality_contract.py",
        "roop_3d/tests/test_stage17_ci_hardware.py",
    } <= set(command)
    assert not quality.get("continue-on-error", False) and not step.get("continue-on-error", False)
    assert {"lint-and-typecheck", "python-tests", "offline-quality"} <= set(jobs["release-verification"]["needs"])


def test_frontend_workflow_executes_the_real_package_gate_commands() -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text("utf-8"))
    steps = workflow["jobs"]["lint-and-typecheck"]["steps"]
    commands = {tuple(shlex.split(step["run"])): step for step in steps if "run" in step}
    scripts = json.loads((ROOT / "app/roop_3d_ui/package.json").read_text("utf-8"))["scripts"]
    for command, script in ((("npm", "run", "typecheck"), "typecheck"),
                            (("npm", "run", "lint"), "lint"),
                            (("npm", "test"), "test"),
                            (("npm", "run", "build"), "build")):
        step = commands[command]
        assert step["working-directory"] == "app/roop_3d_ui"
        assert not step.get("continue-on-error", False)
        assert scripts[script].strip()
    assert shlex.split(scripts["typecheck"]) == ["tsc", "--noEmit"]
    assert shlex.split(scripts["lint"])[0] == "eslint"


def test_hardware_profiles_specification() -> None:
    """Verifies RTX 4070 (12GB), RTX 3060 (6GB), and CPU fallback profile constraints."""
    # RTX 4070 Desktop (High-capacity resident profile)
    p_12gb = verify_hardware_profile("12gb")
    assert p_12gb["profile"] == "12gb"
    assert p_12gb["vram_target_mib"] == 12288
    assert p_12gb["resident_engines_allowed"] is True
    assert p_12gb["max_batch_size"] == 8
    assert "TensorrtExecutionProvider" in p_12gb["preferred_execution_providers"]
    assert "CUDAExecutionProvider" in p_12gb["preferred_execution_providers"]

    # RTX 3060 Laptop (Low-VRAM clamped profile)
    p_6gb = verify_hardware_profile("6gb")
    assert p_6gb["profile"] == "6gb"
    assert p_6gb["vram_target_mib"] == 6144
    assert p_6gb["resident_engines_allowed"] is False
    assert p_6gb["max_batch_size"] == 2
    assert "CUDAExecutionProvider" in p_6gb["preferred_execution_providers"]

    # CPU Fallback
    p_cpu = verify_hardware_profile("cpu")
    assert p_cpu["profile"] == "cpu"
    assert p_cpu["vram_target_mib"] == 0
    assert p_cpu["resident_engines_allowed"] is False
    assert p_cpu["max_batch_size"] == 1
    assert p_cpu["preferred_execution_providers"] == ["CPUExecutionProvider"]

    with pytest.raises(ValueError):
        verify_hardware_profile("invalid_profile")


def test_dependency_audit_matrix_completeness() -> None:
    """Verifies all 14 required dependencies are documented in HARDWARE_COMPATIBILITY_MATRIX.md."""
    matrix_file = ROOT / "docs" / "HARDWARE_COMPATIBILITY_MATRIX.md"
    assert matrix_file.is_file(), "HARDWARE_COMPATIBILITY_MATRIX.md is missing"

    content = matrix_file.read_text("utf-8")
    required_deps = [
        "Python",
        "NumPy",
        "OpenCV",
        "ONNX Runtime",
        "PyTorch",
        "CUDA",
        "TensorRT",
        "PyAV",
        "FFmpeg",
        "Node.js",
        "React",
        "TypeScript",
        "Vite",
        "Three.js",
    ]

    for dep in required_deps:
        assert dep in content, f"Dependency '{dep}' missing from HARDWARE_COMPATIBILITY_MATRIX.md"

    # Verify unsupported platforms are explicitly acknowledged
    assert "UNSUPPORTED" in content
    assert "ROCm" in content
    assert "Arc" in content


def test_release_manifest_and_hmac_signature() -> None:
    """Verifies release_manifest.json exists, is valid, and matches HMAC-SHA256 signature."""
    manifest_file = ROOT / "docs" / "releases" / "release_manifest.json"
    assert manifest_file.is_file(), "release_manifest.json is missing"

    manifest = json.loads(manifest_file.read_text("utf-8"))
    assert manifest["application"] == "roop-3d"
    assert manifest["version"] == "0.2.0"
    assert manifest["airgap_compatible"] is True
    assert manifest["ci_gates_required"] == 15
    assert manifest["ci_gates_passed"] == 15

    # Extract signature and verify HMAC
    sig_info = manifest.get("signature", {})
    assert sig_info.get("algorithm") == "HMAC-SHA256"
    recorded_digest = sig_info.get("digest")

    manifest_copy = dict(manifest)
    del manifest_copy["signature"]
    serialized = json.dumps(manifest_copy, sort_keys=True)
    expected_digest = hmac.new(SECRET_KEY, serialized.encode("utf-8"), hashlib.sha256).hexdigest()
    assert recorded_digest == expected_digest, "Release manifest HMAC signature verification failed"


def test_sbom_cyclonedx_compliance() -> None:
    """Verifies sbom_roop_3d.json exists, complies with CycloneDX format, and lists packages."""
    sbom_file = ROOT / "docs" / "releases" / "sbom_roop_3d.json"
    assert sbom_file.is_file(), "sbom_roop_3d.json is missing"

    sbom = json.loads(sbom_file.read_text("utf-8"))
    assert sbom.get("bomFormat") == "CycloneDX"
    assert sbom.get("specVersion") == "1.5"
    assert sbom["metadata"]["component"]["name"] == "roop-3d"
    assert sbom["metadata"]["component"]["version"] == "0.2.0"

    components = sbom.get("components", [])
    assert len(components) > 20, f"Expected >20 SBOM components, found {len(components)}"

    ecosystems = {c.get("ecosystem") for c in components}
    assert "pypi" in ecosystems, "PyPI ecosystem missing from SBOM"
    assert "npm" in ecosystems, "NPM ecosystem missing from SBOM"


def test_model_compatibility_register() -> None:
    """Verifies model_compatibility_register.json maps registered models and hardware profiles."""
    mcr_file = ROOT / "docs" / "releases" / "model_compatibility_register.json"
    assert mcr_file.is_file(), "model_compatibility_register.json is missing"

    mcr = json.loads(mcr_file.read_text("utf-8"))
    assert mcr.get("release_version") == "0.2.0"
    assert "12gb" in mcr.get("profiles", {})
    assert "6gb" in mcr.get("profiles", {})
    assert "cpu" in mcr.get("profiles", {})

    models = mcr.get("models", [])
    registry = ModelRegistry()
    assert len(models) == len(registry.records()), "Model count in register does not match ModelRegistry"
    assert {entry["model_id"] for entry in models} == {record.model_id for record in registry.records()}

    for entry in models:
        assert "model_id" in entry
        assert "name" in entry
        assert "supported_execution_providers" in entry
        assert "supported_hardware_profiles" in entry
        assert len(entry["supported_execution_providers"]) > 0


def test_checksums_sha256_manifest() -> None:
    """Verifies checksums.sha256 includes entries for all core launcher scripts and manifests."""
    chk_file = ROOT / "docs" / "releases" / "checksums.sha256"
    assert chk_file.is_file(), "checksums.sha256 is missing"

    content = chk_file.read_text("utf-8")
    required_files = [
        "pinokio.js",
        "pinokio.json",
        "install.js",
        "start.js",
        "update.js",
        "reset.js",
        "torch.js",
        "docs/releases/sbom_roop_3d.json",
        "docs/releases/model_compatibility_register.json",
        "docs/releases/requirements.lock",
    ]

    for rel_path in required_files:
        assert rel_path in content, f"File '{rel_path}' missing from checksums.sha256"


@pytest.mark.parametrize("exit_code", [0, 2])
def test_project_admin_ci_gate_invocation_is_isolated_and_propagates_failure(monkeypatch, capsys, exit_code) -> None:
    """Exercise the actual gate selector/admin wrapper, substituting only process execution."""
    from tools.project_admin import run_ci_gate
    calls = []

    def execute(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, exit_code, stdout="typecheck observation",
                                           stderr="type error" if exit_code else "")

    monkeypatch.setattr(subprocess, "run", execute)
    if exit_code:
        with pytest.raises(RuntimeError, match="CI gate check failed"):
            run_ci_gate(gate=4, hardware_profile="12gb")
    else:
        run_ci_gate(gate=4, hardware_profile="12gb")
    assert len(calls) == 1
    args, options = calls[0]
    assert args == ["npm.cmd" if sys.platform == "win32" else "npm", "run", "typecheck"]
    assert Path(options["cwd"]) == APP_DIR / "roop_3d_ui"
    assert options["env"]["PYTHONPATH"] == str(APP_DIR)
    output = capsys.readouterr().out
    report = json.loads(output[output.index("{"):])
    assert report["status"] == ("FAIL" if exit_code else "PASS")
    assert report["total_gates"] == 1 and report["failed_count"] == bool(exit_code)
    assert report["hardware_profile"]["profile"] == "12gb"
    gate, = report["gates"]
    assert gate["gate"] == 4 and gate["exit_code"] == exit_code
    assert "typecheck observation" in gate["output"]
    assert gate["passed"] is (exit_code == 0)


def test_pytest_gate_uses_standalone_config_and_creates_a_local_temporary_parent(monkeypatch, tmp_path) -> None:
    from tools import ci_runner
    app = tmp_path / "app"
    python = app / "roop_3d/.venv/Scripts/python.exe"
    monkeypatch.setattr(ci_runner, "APP_DIR", app)
    monkeypatch.setattr(ci_runner, "PYTHON_VENV", python)
    calls = []

    def execute(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 3, stdout="synthetic pytest failure", stderr="")

    monkeypatch.setattr(subprocess, "run", execute)
    selected = ["roop_3d/tests/test_visual_quality_regressions.py"]
    code, output = ci_runner.run_pytest(selected)
    assert code == 3 and output == "synthetic pytest failure"
    args, options = calls[0]
    assert args == [str(python), "-B", "-m", "pytest", "-c", "roop_3d/pyproject.toml",
                    "--confcutdir=roop_3d", "-q", *selected]
    assert Path(options["cwd"]) == app and options["env"]["PYTHONPATH"] == str(app)
    assert (app / "roop_3d/.runtime").is_dir()


def test_ci_runner_cli_exits_nonzero_when_the_selected_gate_fails(monkeypatch, capsys) -> None:
    from tools import ci_runner
    monkeypatch.setattr(sys, "argv", ["ci_runner.py", "--gate", "4", "--hardware-profile", "cpu"])
    monkeypatch.setattr(subprocess, "run", lambda args, **kwargs:
                        subprocess.CompletedProcess(args, 2, stdout="", stderr="typecheck failed"))
    assert ci_runner.main() == 1
    output = capsys.readouterr().out
    assert "Status: FAIL" in output and "Gates Passed: 0 / 1" in output
