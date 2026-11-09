"""Synthetic contract fixtures ONLY. They are not models or quality evidence."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest

from roop_3d.audit.project_paths import project_root
from roop_3d.tools import dense_face_bundle as tool
from roop_3d.reconstruction.native_backbone import NATIVE_SCHEMA


@pytest.fixture
def local():
    root = project_root() / "cache" / "dense_bundle_tests"
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root) as folder:
        yield Path(folder)


def nodes(contract):
    return [SimpleNamespace(name=name, shape=list(shape), type="tensor(float)")
            for name, shape in contract.items()]


def session(role):
    inputs, outputs = tool.GRAPH_CONTRACTS[role]
    return SimpleNamespace(get_inputs=lambda: nodes(inputs), get_outputs=lambda: nodes(outputs))


@pytest.fixture
def synthetic(local, monkeypatch):
    """Shape-valid but scientifically meaningless arrays, explicitly synthetic."""
    from roop_3d.reconstruction import native_backbone
    def forbidden_gpu(*args, **kwargs):
        raise AssertionError("Bundle tooling must not create a GPU rasterizer")
    monkeypatch.setattr(native_backbone, "GPUSurfaceRasterizer", forbidden_gpu)
    prepared = local / "prepared"
    prepared.mkdir()
    n, f = 35709, 5023
    vertices = np.zeros((n, 3), np.float32)
    vertices[1:3] = [[.01, 0, 0], [0, .01, 0]]
    uv = np.zeros((n, 2), np.float32)
    uv[1:3] = [[1, 0], [0, 1]]
    topology = dict(vertices_m=vertices, faces=np.array([[0, 1, 2]], np.int32), uvs=uv,
                    face_surface=np.ones(n, bool), scalp_hair=np.zeros(n, bool),
                    ears=np.zeros(n, bool), neck=np.zeros(n, bool), semantic_parts=np.ones((8, n), bool))
    np.savez_compressed(prepared / "bfm_topology.npz", **topology)
    np.savez_compressed(prepared / "topology.npz", **topology)
    np.savez_compressed(prepared / "bfm.npz", mean_shape=vertices,
                        identity_basis=np.zeros((n, 3, 80), np.float32),
                        expression_basis=np.zeros((n, 3, 64), np.float32), hrn_uvs=uv,
                        standard5=np.zeros((5, 3), np.float32), landmarks68=np.arange(68))
    np.savez_compressed(prepared / "flame_rig.npz", template_m=np.zeros((f, 3)),
                        expression_dirs_m=np.zeros((f, 3, 50), np.float32),
                        smirk_to_expression=np.eye(50), parents=np.array([-1, 0, 1, 1, 1]),
                        joint_regressor=np.zeros((5, f)), skinning_weights=np.tile([1., 0, 0, 0, 0], (f, 1)),
                        pose_dirs_m=np.zeros((36, f*3)), eyelid_dirs_m=np.zeros((2, f, 3)))
    registration = dict(bfm_to_flame_indices=np.zeros((f, 3), np.int32),
                        bfm_to_flame_weights=np.tile([1., 0, 0], (f, 1)),
                        flame_to_bfm_indices=np.zeros((n, 3), np.int32),
                        flame_to_bfm_weights=np.tile([1., 0, 0], (n, 1)),
                        flame_to_bfm_rotation=np.eye(3), flame_to_bfm_translation_m=np.zeros(3),
                        flame_hrn_uvs=np.zeros((f, 2)), flame_to_bfm_scale=np.array(1.))
    np.savez_compressed(prepared / "registration.npz", **registration)
    for role in tool.GRAPH_CONTRACTS:
        (prepared / (role + ".onnx")).write_bytes(b"synthetic session stub, NOT an ONNX model")
    monkeypatch.setattr(tool, "_cpu_session", lambda path: session(path.stem))
    spec = dict(schema=NATIVE_SCHEMA, topology="bfm35709", expression_basis="smirk_flame50",
                bfm_unit_m=.1, flame_model="flame2023_Open", smirk_crop="projected_bfm68_bbox_1.4",
                mask_provenance="SYNTHETIC TEST ONLY", expression_mapping_evidence="SYNTHETIC TEST ONLY",
                registration_evidence="SYNTHETIC TEST ONLY", hrn_alpha=.01, hrn_beta=.001,
                assets={role: {"file": role + (".onnx" if role in tool.GRAPH_CONTRACTS else ".npz")}
                        for role in tool.NATIVE_ROLES})
    config = prepared / "metadata.json"
    config.write_text(json.dumps(spec), encoding="utf-8")
    return config, local / "bundle.json", spec


def test_publication_and_readiness(synthetic):
    config, output, spec = synthetic
    assembled = tool.assemble(config, output)
    assert json.loads(output.read_text()) == assembled
    assert assembled["mask_provenance"] == spec["mask_provenance"]
    assert all(len(record["sha256"]) == 64 for record in assembled["assets"].values())
    assert tool.readiness(output)["ready"]
    assert not list(output.parent.glob(".bundle-*.part"))
    original = output.read_bytes()
    with pytest.raises(ValueError, match="overwrite"):
        tool.assemble(config, output)
    assert output.read_bytes() == original


@pytest.mark.parametrize("failure", ["missing", "digest", "provenance", "path", "registration", "rig", "anatomy", "graph"])
def test_failure_never_publishes(synthetic, monkeypatch, failure):
    config, output, spec = synthetic
    if failure == "missing":
        (config.parent / "coarse.onnx").unlink()
    elif failure == "digest":
        spec["assets"]["coarse"]["sha256"] = "0" * 64
    elif failure == "provenance":
        del spec["mask_provenance"]
    elif failure == "path":
        spec["assets"]["bfm"]["file"] = "../../escape.npz"
    elif failure in ("registration", "rig", "anatomy"):
        role, key = {"registration": ("registration", "flame_to_bfm_scale"),
                     "rig": ("flame_rig", "parents"), "anatomy": ("topology", "face_surface")}[failure]
        path = config.parent / (role + ".npz")
        arrays = tool._arrays(path)
        arrays[key] = np.zeros_like(arrays[key])
        np.savez_compressed(path, **arrays)
    else:
        monkeypatch.setattr(tool, "_cpu_session", lambda path: session("hrn_high"))
    config.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises((ValueError, KeyError)):
        tool.assemble(config, output)
    assert not output.exists()
    assert not list(output.parent.glob(".bundle-*.part"))


def test_flame_topology_publication(synthetic):
    config, output, spec = synthetic
    path = config.parent / "topology.npz"
    data = tool._arrays(path)
    for key in ("vertices_m", "uvs", "face_surface", "scalp_hair", "ears", "neck"):
        data[key] = data[key][:5023]
    del data["semantic_parts"]
    np.savez_compressed(path, **data)
    spec["topology"] = "flame2023open"
    config.write_text(json.dumps(spec), encoding="utf-8")
    tool.assemble(config, output)
    assert tool.readiness(output)["ready"]


def test_hardlinked_asset_rejected(synthetic):
    config, output, _ = synthetic
    alias = config.parent / "hardlinked.onnx"
    tool.os.link(config.parent / "coarse.onnx", alias)
    try:
        with pytest.raises(ValueError, match="Hardlinked"):
            tool.assemble(config, output)
        assert not output.exists()
    finally:
        alias.unlink()


def test_publication_race_does_not_overwrite(synthetic, monkeypatch):
    config, output, _ = synthetic
    original_link = tool.os.link
    def race(source, destination):
        Path(destination).write_text("concurrent owner", encoding="utf-8")
        return original_link(source, destination)
    monkeypatch.setattr(tool.os, "link", race)
    with pytest.raises(FileExistsError):
        tool.assemble(config, output)
    assert output.read_text() == "concurrent owner"
    assert not list(output.parent.glob(".bundle-*.part"))


def test_status_without_bundle_lists_every_missing_asset_and_evidence(local, capsys):
    prepared = local / "prepared"
    prepared.mkdir()
    (prepared / "coarse.export.json").write_text(json.dumps({"synthetic": True, "exports": {"coarse": {"parity_max_abs_error": 0.}}}))
    (prepared / "bad.export.json").write_text("not json")
    report = tool.readiness(local / "bundle.json")
    assert not report["ready"]
    assert set(report["assets"]) == set(tool.NATIVE_ROLES)
    assert all(not asset["present"] for asset in report["assets"].values())
    assert len(report["export_evidence"]) == 2
    assert "end-to-end" in report["scope"]
    assert tool.main(["status", "--bundle", str(local / "bundle.json")]) == 1
    assert "Missing coarse" in capsys.readouterr().out


def test_nested_prepared_exports_and_ambiguity(local):
    nested = local / "prepared" / "hrn"
    nested.mkdir(parents=True)
    (nested / "hrn_mid.onnx").write_bytes(b"SYNTHETIC INVENTORY ONLY")
    report = tool.readiness(local / "bundle.json")
    assert report["assets"]["hrn_mid"]["present"]
    assert report["assets"]["hrn_mid"]["file"] == str(nested / "hrn_mid.onnx")
    (nested.parent / "hrn_mid.onnx").write_bytes(b"SYNTHETIC AMBIGUITY ONLY")
    report = tool.readiness(local / "bundle.json")
    assert not report["ready"]
    assert "Multiple prepared candidates" in report["assets"]["hrn_mid"]["error"]


def test_status_config_without_bundle_is_not_ready(synthetic):
    config, output, _ = synthetic
    report = tool.readiness(output, config=config)
    assert all(item["present"] for item in report["assets"].values())
    assert not report["ready"]


def test_tampered_published_asset_fails_readiness(synthetic):
    config, output, _ = synthetic
    tool.assemble(config, output)
    (config.parent / "expression.onnx").write_bytes(b"changed")
    report = tool.readiness(output)
    assert not report["ready"]
    assert "digest mismatch" in str(report["errors"])


@pytest.mark.parametrize("role", list(tool.GRAPH_CONTRACTS))
def test_all_available_native_contracts(role):
    tool.validate_graph_contract(role, session(role))
    bad = session(role)
    bad.get_inputs = lambda: [SimpleNamespace(name="wrong", type="tensor(float)", shape=[1])]
    with pytest.raises(ValueError, match="names"):
        tool.validate_graph_contract(role, bad)
    inputs, outputs = tool.GRAPH_CONTRACTS[role]
    bad.get_inputs = lambda: [SimpleNamespace(name=name, type="tensor(double)", shape=shape)
                              for name, shape in inputs.items()]
    with pytest.raises(ValueError, match="float32"):
        tool.validate_graph_contract(role, bad)
    bad.get_inputs = lambda: nodes(inputs)
    bad.get_outputs = lambda: [SimpleNamespace(name=name, type="tensor(float)", shape=[1])
                               for name in outputs]
    with pytest.raises(ValueError, match="float32"):
        tool.validate_graph_contract(role, bad)


def test_external_paths_rejected(local):
    with pytest.raises(ValueError, match="checkout"):
        tool.readiness(project_root().parent / "external-bundle.json")
    with pytest.raises(ValueError, match="directory"):
        tool._asset(local, "../escape.npz")


@pytest.mark.parametrize("placement", ["initializer", "constant", "subgraph"])
def test_external_onnx_data_rejected_before_ort(local, placement):
    onnx = pytest.importorskip("onnx")
    tensor = onnx.TensorProto()
    tensor.name = "SYNTHETIC_EXTERNAL_REFERENCE"
    tensor.data_type = onnx.TensorProto.FLOAT
    tensor.dims.append(1)
    tensor.data_location = onnx.TensorProto.EXTERNAL
    entry = tensor.external_data.add()
    entry.key, entry.value = "location", "../../outside-checkout.bin"
    constant = onnx.helper.make_node("Constant", [], ["x"], value=tensor)
    inner = onnx.helper.make_graph([constant], "SYNTHETIC INNER", [], [])
    if placement == "initializer":
        graph = onnx.helper.make_graph([], "SYNTHETIC", [], [], [tensor])
    elif placement == "constant":
        graph = inner
    else:
        graph = onnx.helper.make_graph([onnx.helper.make_node("If", ["condition"], ["x"], then_branch=inner, else_branch=inner)], "SYNTHETIC", [], [])
    payload = onnx.helper.make_model(graph).SerializeToString()
    with pytest.raises(ValueError, match="External ONNX"):
        tool._embedded_onnx(payload)


@pytest.mark.parametrize("payload", [b"\x00", b"\x3a\x05\x00", b"\x80", b"\x0f"])
def test_malformed_onnx_envelope_rejected(payload):
    with pytest.raises(ValueError, match="ONNX"):
        tool._embedded_onnx(payload)



def test_real_cpu_ort_contracts_when_onnx_available(local):
    """Actual tiny synthetic ONNX graphs, never inference or trained weights."""
    onnx = pytest.importorskip("onnx")
    for role, (inputs, outputs) in tool.GRAPH_CONTRACTS.items():
        ins = [onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape) for name, shape in inputs.items()]
        outs = [onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape) for name, shape in outputs.items()]
        constants = [onnx.helper.make_node("Constant", [], [name], value=onnx.numpy_helper.from_array(np.zeros(shape, np.float32)))
                     for name, shape in outputs.items()]
        graph = onnx.helper.make_graph(constants, "SYNTHETIC CONTRACT ONLY", ins, outs)
        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 17)])
        model.ir_version = 8
        path = local / (role + ".onnx")
        onnx.save(model, path)
        cpu = tool._cpu_session(path)
        assert cpu.get_providers() == ["CPUExecutionProvider"]
        tool.validate_graph_contract(role, cpu)
