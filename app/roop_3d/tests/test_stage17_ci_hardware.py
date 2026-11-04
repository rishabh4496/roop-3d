"""Stage 17 Automated Tests: CI/CD Pipeline, Hardware Compatibility & Release System.

Verifies:
1. 15-Gate CI/CD runner structure, gate interfaces, and execution contracts.
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
import sys

import pytest

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


def test_15_ci_gates_registered_and_contract() -> None:
    """Verifies all 15 required CI gates exist with correct numbering and interfaces."""
    assert len(GATES) == 15, f"Expected exactly 15 CI gates, found {len(GATES)}"

    expected_gate_names = [
        "Python unit tests",
        "Python integration tests",
        "Frontend unit tests",
        "TypeScript type checking",
        "Linting",
        "Production build",
        "API tests",
        "Playwright browser tests",
        "FaceSet archive validation",
        "Model registry validation",
        "Video timing tests",
        "Golden video inference",
        "Model hash verification",
        "Security tests",
        "Legacy isolation audit",
    ]

    for idx, (gate_func, expected_name) in enumerate(zip(GATES, expected_gate_names), 1):
        doc = gate_func.__doc__ or ""
        assert f"Gate {idx}:" in doc or f"Gate {idx:02d}:" in doc, f"Gate {idx} docstring missing numbering"
        assert expected_name.lower() in doc.lower(), f"Gate {idx} does not mention '{expected_name}'"


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


def test_project_admin_ci_gate_invocation() -> None:
    """Verifies project_admin run_ci_gate runs correctly with gate filtering."""
    from tools.project_admin import run_ci_gate

    # Gate 4: TypeScript type checking (fast gate ~1s)
    run_ci_gate(gate=4, hardware_profile="12gb")
