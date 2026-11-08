"""Stage 19: the user documentation must reflect actual application behaviour.

Each check ties a statement in ``docs/`` to the code that makes it true (constants, defaults, registries, launcher scripts, on-screen
labels), so a change to the code fails here until the documentation (and the in-app guidance in ``helpContent.ts`` / ``errorGuide.ts``) is updated.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

from roop_3d.audit.project_paths import project_root
from roop_3d.core.hardware import (LOW_VRAM_CEILING_MIB, MIN_12GB_PROFILE_VRAM_MIB, resolve_hardware_profile)
from roop_3d.core.settings import Settings
from roop_3d.media.export_profiles import PROFILES
from roop_3d.reconstruction import faceset_quality as quality

ROOT = project_root()
DOCS = ROOT / "docs"
UI = ROOT / "app" / "roop_3d_ui" / "src"
PKG = ROOT / "app" / "roop_3d"
NEW_DOCS = ["INSTALLATION_GUIDE", "OFFLINE_MODELS", "USER_MANUAL", "FACESET_GUIDE", "ENGINE_COMPARISON", "GPU_TROUBLESHOOTING",
            "PERFORMANCE_GUIDE", "EXPORT_GUIDE", "COMMON_PROBLEMS", "ARCHITECTURE", "UX_STAGE19"]


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def doc(name: str) -> str:
    return read(DOCS / f"{name}.md")


def ui_source() -> str:
    return "\n".join(read(p) for p in UI.rglob("*") if p.suffix in {".ts", ".tsx", ".jsx", ".js"} and ".test." not in p.name)


def backend_literals() -> set[str]:
    found: set[str] = set()
    for path in PKG.rglob("*.py"):
        parts = set(path.relative_to(PKG).parts)
        if parts & {".venv", ".runtime", "tests", "__pycache__"}:
            continue
        found |= set(re.findall(r"[\"']([A-Z][A-Z0-9_]{3,})[\"']", read(path)))
    return found


def test_every_stage19_document_exists_and_is_indexed():
    index = read(DOCS / "README.md")
    for name in NEW_DOCS:
        text = doc(name)
        assert len(text) > 800, name
        assert f"({name}.md" in index, f"{name} is not linked from docs/README.md"


def test_relative_links_and_anchors_resolve():
    problems = []
    for name in NEW_DOCS + ["README"]:
        text = doc(name)
        for target in re.findall(r"\]\(([^)#]+\.md)(#[^)]*)?\)", text):
            path = DOCS / target[0]
            if not path.exists():
                problems.append(f"{name}: missing {target[0]}")
    for name in NEW_DOCS:
        text = doc(name)
        anchors = set(re.findall(r'<a id="([^"]+)"', text)) | {re.sub(r"[^a-z0-9 -]", "", h.lower()).strip().replace(" ", "-")
                                                                   for h in re.findall(r"^#{1,6} (.+)$", text, re.M)}
        for local in re.findall(r"\]\(#([^)]+)\)", text):
            if local not in anchors:
                problems.append(f"{name}: dead anchor #{local}")
    assert not problems, "\n".join(problems)


def test_error_reference_covers_every_in_app_guide_with_real_codes():
    guide = read(UI / "components/guide/errorGuide.ts")
    codes = re.findall(r"^  ([A-Z][A-Z0-9_]+): g\(", guide, re.M)
    assert len(codes) >= 45
    problems = doc("COMMON_PROBLEMS")
    literals = backend_literals()
    ui_only = {"SERVICE_OFFLINE"}                      # a browser-level failure: the service never answered
    for code in codes:
        assert f"`{code}`" in problems, f"{code} is missing from COMMON_PROBLEMS.md"
        if code not in ui_only:
            assert code in literals, f"{code} is not a code the backend raises: remove it or fix the spelling"
    anchors = set(re.findall(r", '([a-z][a-z-]+)'\),?$", guide, re.M))
    for anchor in anchors:
        assert f'<a id="{anchor}"></a>' in problems, f"anchor {anchor} (used by errorGuide.ts) is not in COMMON_PROBLEMS.md"


def test_documented_hardware_profiles_match_the_resolver():
    assert MIN_12GB_PROFILE_VRAM_MIB == 10240 and "10,240 MiB" in doc("GPU_TROUBLESHOOTING")
    assert LOW_VRAM_CEILING_MIB == 4600 and "about 4.6 GB" in doc("GPU_TROUBLESHOOTING")
    assert resolve_hardware_profile(0, 0, "auto") == "cpu"
    assert resolve_hardware_profile(6144, 5600, "auto") == "6gb"
    assert resolve_hardware_profile(12288, 3999, "auto") == "6gb"
    assert resolve_hardware_profile(12288, 4000, "auto") == "12gb"
    gpu = doc("GPU_TROUBLESHOOTING")
    assert "6 GB or less in total **or less than 4 GB free" in gpu
    helper = read(UI / "components/guide/helpContent.ts")
    assert "12288 MiB total is required" not in helper, "stale 12288 MiB rule in the in-app text"
    assert "10,240 MiB" in helper and "at least 10 GB of video memory" in helper


def test_documented_photo_rules_match_the_quality_analysis():
    assert (quality.MIN_SHARPNESS, quality.MIN_FACE_PIXELS, quality.MAX_CLIPPED_FRACTION, quality.IDENTITY_LINK_SIMILARITY,
            quality.DUPLICATE_POSE_DEG, quality.YAW_TOLERANCE_DEG, quality.PITCH_TOLERANCE_DEG, quality.MAX_ABS_YAW_DEG) == (0.03, 40.0, 0.35, 0.30, 3.0, 20.0, 12.0, 88.0)
    guide, helper = doc("FACESET_GUIDE"), read(UI / "components/guide/helpContent.ts")
    for fact in ("**0.03**", "40 pixels", "35%", "**0.30**", "within 3°", "**20°**", "**12°**", "88°"):
        assert fact in guide, fact
    for fact in ("0.03", "40 pixels", "35%", "0.30", "88°"):
        assert fact in helper, f"in-app text lacks {fact}"
    targets = {name: yaw for name, yaw, _ in quality.YAW_TARGETS}
    assert targets == {"profile_image_left": -75.0, "three_quarter_image_left": -45.0, "frontal": 0.0, "three_quarter_image_right": 45.0, "profile_image_right": 75.0}
    needs = {name: need for name, _, need in quality.YAW_TARGETS}
    assert needs["frontal"] == needs["three_quarter_image_left"] == needs["three_quarter_image_right"] == "required"
    assert needs["profile_image_left"] == needs["profile_image_right"] == "recommended"


def test_every_export_preset_is_documented_with_its_container_and_bit_depth():
    guide = doc("EXPORT_GUIDE")
    for profile_id, profile in PROFILES.items():
        row = next((line for line in guide.splitlines() if f"`{profile_id}`" in line and line.startswith("|")), None)
        assert row, f"{profile_id} has no row in EXPORT_GUIDE.md"
        container = "folder" if profile.container == "sequence" else profile.container
        assert container in row, f"{profile_id}: container {container} not in its row"
        assert f"{profile.container_bit_depth}-bit" in row, f"{profile_id}: bit depth"
    assert [pid for pid, p in PROFILES.items() if p.allow_hardware] == ["standard_mp4"]
    assert "only preset allowed to use hardware" in guide
    helper = read(UI / "components/guide/helpContent.ts")
    for profile_id in PROFILES:
        assert f"{profile_id}:" in helper, f"{profile_id} has no 'use it for' guidance in helpContent.ts"


def test_documented_defaults_match_settings():
    defaults = Settings()
    perf = doc("PERFORMANCE_GUIDE")
    expected = {"*Sparse interval* (25)": defaults.sparse_interval == 25, "*Stream batch size* (4)": defaults.stream_batch_size == 4,
                "*Stream ring slots* (4)": defaults.stream_ring_slots == 4, "*Stream segment frames* (250)": defaults.stream_segment_frames == 250,
                "*Execution threads* (1)": defaults.execution_threads == 1, "*Restorer threads* (1)": defaults.restorer_threads == 1,
                "*Restorer pool size* (2)": defaults.restorer_pool_size == 2, "*Max concurrent jobs* (1)": defaults.max_concurrent_jobs == 1,
                "*Memory safety reserve mb* (768)": defaults.memory_safety_reserve_mb == 768, "*VRAM limit mb* (0 = none)": defaults.vram_limit_mb == 0,
                "*Face restoration policy* (`off`)": defaults.face_restoration_policy == "off", "*Video timing policy* (`preserve`)": defaults.video_timing_policy == "preserve",
                "*Audio policy* (`copy`)": defaults.audio_policy == "copy", "*Hardware profile* (`auto`)": defaults.hardware_profile == "auto"}
    for label, matches in expected.items():
        assert matches, f"{label}: the default changed; update the docs"
        assert label in perf, f"{label} is not documented in PERFORMANCE_GUIDE.md"
    assert defaults.output_root == "roop_3d/.runtime/outputs" and "app/roop_3d/.runtime/outputs" in doc("INSTALLATION_GUIDE")


def test_launcher_and_cli_statements_match_the_scripts():
    install, start, reset, update = (read(ROOT / name) for name in ("install.js", "start.js", "reset.js", "update.js"))
    guide = doc("INSTALLATION_GUIDE")
    for statement, source in (("uv pip sync --require-hashes app/roop_3d/requirements.lock", install), ("npm ci", install), ("vite build", install), ("mark-ready", install),
                              ("torch.js", install), ("python -B app/serve.py", start), ("requirements.txt", start), ("requirements.lock", start),
                              ("git pull --ff-only", update), ("app/roop_3d/.venv", reset), ("cache/install-ready.json", reset)):
        assert statement in source, f"{statement} is no longer in the launcher scripts"
        assert statement.split(" ")[0] in guide
    assert "**8030**" in guide and "ROOP3D_PORT" in guide and '"8030"' in read(ROOT / "tools/project_admin.py")
    admin = read(ROOT / "tools/project_admin.py")
    commands = set(re.findall(r'"([a-z]+(?:-[a-z]+)*)"', re.search(r"choices=\[(.*?)\]", admin, re.S).group(1)))
    for command in re.findall(r"^\| `([a-z-]+)", guide.split("## 7. Command line")[1], re.M):
        assert command in commands, f"{command} is documented but not a project_admin command"
    for command in ("offline-health", "model-import", "model-verify", "preflight"):
        assert command in commands


def test_offline_model_statements_match_the_model_manager():
    from roop_3d.installer import model_manager as manager
    guide = doc("OFFLINE_MODELS")
    assert manager.ALLOWED_MODEL_EXTENSIONS == {".onnx", ".bin", ".json", ".hash", ".safetensors"}
    assert manager.MAX_SINGLE_FILE_BYTES == 5 * 1024 ** 3 and manager.MAX_BUNDLE_UNCOMPRESSED_BYTES == 20 * 1024 ** 3
    for extension in manager.ALLOWED_MODEL_EXTENSIONS | manager.BLOCKED_EXTENSIONS:
        assert f"`{extension}`" in guide or extension in guide, extension
    assert "5 GiB" in guide and "20 GiB" in guide
    offline = read(PKG / "installer/offline_cache.py")
    for name in re.findall(r"[\"']([A-Za-z0-9_.-]+\.onnx)[\"']", offline):
        assert name in guide, f"core model {name} is not listed in OFFLINE_MODELS.md"
    assert "app/roop_3d/models/cache/<model id>.model.<ext>" in guide


def test_cli_model_commands_use_the_server_root():
    sys.path.insert(0, str(ROOT))
    from tools import project_admin
    assert project_admin.APP_ROOT == ROOT / "app"
    assert "app/roop_3d/models/cache" in doc("OFFLINE_MODELS")


def test_shortcut_tables_match_the_defaults():
    hotkeys = read(UI / "components/controls/Hotkeys.tsx")
    defaults = dict(re.findall(r"(\w+): '([^']+)'", re.search(r"DEFAULT_KEYS[^{]*\{(.*?)\n\};", hotkeys, re.S).group(1)))
    manual = doc("USER_MANUAL")
    for action, chord in defaults.items():
        if "+" in chord:
            assert chord in manual, f"{action} ({chord}) is missing from the manual's shortcut table"
    assert "1 2 3 4 5 6 7 8" in manual and sum(1 for key in defaults if key.startswith("buffer")) == 8


def test_screen_labels_quoted_in_the_docs_exist_in_the_interface():
    source = ui_source().lower()          # small buttons are capitalised by CSS: compare what a person reads, not the source casing
    labels = ["Verify Offline Health", "Re-Run Preflight", "Environment & Offline Manager", "Models & Engines Manager", "Diagnostics Suite",
              "Keyboard shortcut remapper", "Workstation Preferences", "Queue render", "Inspect target faces", "Import target video", "Use Face From Frame",
              "Fill Missing Angles from Current Target Video", "Save to FaceSet Library", "Download .Fsz", "Import Source Photos", "Skip to main content",
              "Purge VRAM", "Inspect output", "Export video", "Render manifest", "Verify & open", "Import .r3dp", "Reset Range", "Edit Mask", "Split Comparison Wipe",
              "More export options", "Source photo checklist", "Include selected paused/interrupted checkpoint when saving a project",
              "Build 3D identity asset", "Pre-Flight Render Linter", "Studio layout", "Classic layout", "Save to FaceSet Library", "Local Photo Cosine Distance"]
    missing = [label for label in labels if label.lower() not in source]
    assert not missing, f"interface strings quoted in the docs but absent from the UI source: {missing}"
    manual_text = doc("USER_MANUAL") + doc("INSTALLATION_GUIDE") + doc("OFFLINE_MODELS") + doc("FACESET_GUIDE")
    quoted = [label for label in labels if label in manual_text]
    assert len(quoted) >= 18, "the docs no longer quote the interface labels this test protects"


@pytest.mark.parametrize("name", NEW_DOCS)
def test_no_internal_stage_jargon_in_user_facing_docs(name):
    if name in {"ARCHITECTURE", "UX_STAGE19"}:
        return                                           # developer and report documents may cite stages
    text = doc(name)
    for jargon in ("pre-Stage-15", "Stage 15 default", "Stage 12", "Stage 16"):
        assert jargon not in text, f"{name}: internal wording '{jargon}'"
    for word in ("TODO", "FIXME", "lorem"):
        assert word not in text
