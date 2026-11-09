"""Prepare genuine upstream networks and model assets for the dense face runtime.

Run from app/: python -m roop_3d.tools.prepare_dense_face --help.
PyTorch/timm are offline export dependencies only. All paths stay in this repo.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np

from roop_3d.models.owned_files import owned_path


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            result.update(chunk)
    return result.hexdigest()


def source_module(source, relative, namespace):
    """Load only the required architecture files under a private package name.

    Does not add upstream generic ``models``, ``util`` or ``src`` to sys.path.
    Does not run an upstream renderer, CLI, downloader or training initializer.
    """
    source = owned_path(Path(source))
    module_path = owned_path(source / relative)
    package = types.ModuleType(namespace)
    package.__path__ = [str(module_path.parent)]
    sys.modules[namespace] = package
    name = namespace + "." + module_path.stem
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def export_network(model, output, shape, input_name, output_names):
    """Export only after strict state loading; compare ORT against torch twice."""
    import torch
    import onnx
    import onnxruntime as ort
    output = owned_path(Path(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError(f"Refusing to overwrite an existing export: {output}")
    part = owned_path(output.with_suffix(".part"))
    if part.exists():
        raise ValueError(f"An unfinished export already exists: {part}")
    model.eval().cpu()
    generator = torch.Generator().manual_seed(314159)
    example = torch.rand(shape, generator=generator)
    errors = []
    try:
        with torch.inference_mode():
            torch.onnx.export(model, example, str(part), input_names=[input_name], output_names=output_names,
                              opset_version=17, dynamo=False)
        onnx.checker.check_model(str(part))
        options = ort.SessionOptions()
        options.intra_op_num_threads, options.inter_op_num_threads = 2, 1
        session = ort.InferenceSession(str(part), sess_options=options, providers=["CPUExecutionProvider"])
        for sample in (example, torch.rand(shape, generator=generator)):
            with torch.inference_mode():
                expected = model(sample)
            if isinstance(expected, torch.Tensor):
                expected = (expected,)
            actual = session.run(output_names, {input_name: sample.numpy()})
            if len(actual) != len(expected):
                raise ValueError("Export changed the number of outputs")
            for native, converted in zip(expected, actual):
                reference = native.numpy()
                np.testing.assert_allclose(converted, reference, atol=2e-4, rtol=2e-4)
                if not np.isfinite(converted).all():
                    raise ValueError("Export generated nonfinite output")
                errors.append(float(np.max(np.abs(converted-reference))))
        del session
        part.replace(output)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    return {"file": output.name, "sha256": sha256(output), "parity_max_abs_error": max(errors),
            "parity_scope": "two seeded tensors; checkpoint export parity, not reconstruction accuracy"}


def export_component(args):
    import torch
    torch.set_num_threads(2)
    checkpoint = owned_path(Path(args.checkpoint))
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    source = owned_path(Path(args.source))
    output = owned_path(Path(args.output))
    results = {}
    if args.component == "3ddfa":
        module = source_module(source, "model/networks.py", "_roop3d_3ddfa")
        if args.backbone == "mbnetv3":
            model = module.define_net_recon_mobilenetv3("recon_mobilenetv3_large", use_last_fc=False)
        else:
            model = module.define_net_recon("resnet50", use_last_fc=False)
        model.load_state_dict(state["net_recon"], strict=True)
        results["coarse"] = export_network(model, output, (1, 3, 224, 224), "rgb", ["coefficients"])
    elif args.component == "smirk":
        module = source_module(source, "src/smirk_encoder.py", "_roop3d_smirk")
        # Upstream defaults to pretrained=True even when all parameters will be
        # replaced. Override only the architecture factory to prevent downloads.
        def backbone(name, pretrained=True):
            import timm
            network = timm.create_model(name, pretrained=False, features_only=True)
            return network, network.feature_info[-1]["num_chs"]
        module.create_backbone = backbone
        model = module.ExpressionEncoder(n_exp=50)
        prefix = "smirk_encoder.expression_encoder."
        expression_state = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
        model.load_state_dict(expression_state, strict=True)
        class Outputs(torch.nn.Module):
            def __init__(self, encoder):
                super().__init__()
                self.encoder = encoder
            def forward(self, rgb):
                result = self.encoder(rgb)
                return result["expression_params"], result["jaw_params"], result["eyelid_params"]
        results["expression"] = export_network(Outputs(model), output, (1, 3, 224, 224), "rgb", ["expression", "jaw", "eyelids"])
    else:
        module = source_module(source, "pix2pix/networks.py", "_roop3d_hrn")
        output.mkdir(parents=True, exist_ok=True)
        for role, input_channels, output_channels, size, name in (
                ("hrn_mid", 6, 3, 64, "deformation"), ("hrn_high", 9, 1, 256, "displacement")):
            model = module.define_G(input_channels, output_channels, 64, "resnet_9blocks",
                                    norm="instance", use_dropout=True, gpu_ids=[])
            model.load_state_dict(state["mid_net" if role == "hrn_mid" else "high_net"], strict=True)
            results[role] = export_network(model, output / f"{role}.onnx", (1, input_channels, size, size), "features", [name])
        for key in ("alpha", "beta"):
            value = float(state[key].item())
            if not np.isfinite(value):
                raise ValueError(f"Nonfinite HRN checkpoint scale: {key}")
            results["hrn_"+key] = value
    report = {"checkpoint_sha256": sha256(checkpoint), "component": args.component,
              "exports": results, "source_files": {str(path.relative_to(source)): sha256(path)
                for path in source.rglob("*.py") if path.is_file() and "__pycache__" not in path.parts}}
    report_path = output / "hrn.export.json" if args.component == "hrn" else output.with_suffix(".export.json")
    report_path.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def prepare_bfm(args):
    """One-time trusted upstream pickle import; runtime never unpickles assets."""
    from scipy.io import loadmat
    from roop_3d.reconstruction.morphable_backbone import FaceTopology, binary, finite
    source = owned_path(Path(args.model))
    # face_model.npy is an official dictionary-backed NumPy asset. Use this
    # importer only on inspected upstream bytes; do not use on FaceSet content.
    model = np.load(source, allow_pickle=True).item()
    n = 35709
    vertices = finite(np.asarray(model["u"]).reshape(n, 3), (n, 3), "BFM mean")
    annotations = model["annotation"]
    if len(annotations) != 8:
        raise ValueError("3DDFA-V3 must contain eight semantic vertex masks")
    parts = np.zeros((8, n), bool)
    for index, values in enumerate(annotations):
        ids = np.asarray(values)
        if ids.dtype.kind not in "iu" or ids.ndim != 1 or np.any((ids < 0) | (ids >= n)):
            raise ValueError("Invalid 3DDFA semantic indices")
        parts[index, ids] = True
    with np.load(owned_path(Path(args.anatomy)), allow_pickle=False) as masks:
        anatomy = {name: binary(masks[name], (n,), name) for name in ("face_surface", "scalp_hair", "ears", "neck")}
    output = owned_path(Path(args.output))
    output.mkdir(parents=True, exist_ok=True)
    paths = [output / name for name in ("bfm.npz", "bfm_topology.npz")]
    if any(path.exists() for path in paths):
        raise ValueError("BFM outputs already exist; choose an empty output directory")
    uv = finite(np.load(owned_path(Path(args.hrn_uvs)), allow_pickle=False), (n, 2), "native HRN chart")
    if not np.allclose(uv, model["uv_coords"], atol=1e-6):
        raise ValueError("HRN and 3DDFA UV charts differ; supply a validated registered chart")
    lm = loadmat(owned_path(Path(args.standard)))["lm"]
    standard5 = np.stack((lm[[36, 39]].mean(0), lm[[42, 45]].mean(0), lm[30], lm[48], lm[54]))
    np.savez_compressed(paths[0], mean_shape=vertices, identity_basis=np.asarray(model["id"]).reshape(n, 3, 80),
                        expression_basis=np.asarray(model["exp"]).reshape(n, 3, 64), hrn_uvs=uv,
                        standard5=standard5, landmarks68=model["ldm68"])
    np.savez_compressed(paths[1], vertices_m=vertices * (.1, -.1, -.1), faces=model["tri"],
                        uvs=model["uv_coords"], semantic_parts=parts, **anatomy)
    topology = FaceTopology.load("bfm35709", paths[1])
    print(json.dumps({"bfm_sha256": sha256(paths[0]), "topology_sha256": sha256(paths[1]),
                      "upstream_sha256": sha256(source), "active_vertices": len(topology.vertex_ids),
                      "semantic_part_names": ["right_eye", "left_eye", "right_eyebrow", "left_eyebrow", "nose", "up_lip", "down_lip", "skin"]}, indent=2))


def prepare_flame_anatomy(args):
    """Import named official FLAME regions and physically exclude non-face parts."""
    import pickle
    masks_path = owned_path(Path(args.masks))
    with masks_path.open("rb") as stream:
        regions = pickle.load(stream, encoding="latin1")
    def membership(names):
        mask = np.zeros(5023, bool)
        for name in names:
            ids = np.asarray(regions[name])
            if ids.dtype.kind not in "iu" or ids.ndim != 1 or np.any((ids < 0) | (ids >= 5023)):
                raise ValueError(f"Invalid official FLAME region: {name}")
            mask[ids] = True
        return mask
    scalp = membership(("scalp",))
    ears = membership(("left_ear", "right_ear"))
    neck = membership(("neck", "boundary"))
    face = membership(("face", "forehead", "eye_region", "nose", "lips"))
    eyeballs = membership(("left_eyeball", "right_eyeball"))
    face &= ~(scalp | ears | neck | eyeballs)
    output = owned_path(Path(args.output))
    if output.suffix != ".npz" or output.exists():
        raise ValueError("Choose a new .npz anatomy output")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, face_surface=face, scalp_hair=scalp, ears=ears, neck=neck)
    print(json.dumps({"file": output.name, "sha256": sha256(output), "mask_source_sha256": sha256(masks_path),
                      "active_vertices_before_triangle_pruning": int(face.sum())}))


def prepare_flame(args):
    import pickle
    from roop_3d.reconstruction.native_face_geometry import FlameExpressionRig
    source = owned_path(Path(args.model))
    with source.open("rb") as stream:
        model = pickle.load(stream, encoding="latin1")
    def array(value):
        return np.asarray(value.toarray() if hasattr(value, "toarray") else value, np.float32)
    conversion = np.load(owned_path(Path(args.conversion)), allow_pickle=False)
    directions = array(model["shapedirs"])[:, :, 300:]
    parents = np.asarray(model["kintree_table"])[0].astype(np.int64)
    parents[0] = -1
    rig = {"template_m": array(model["v_template"]), "expression_dirs_m": directions,
           "smirk_to_expression": conversion, "pose_dirs_m": array(model["posedirs"]).reshape(5023*3, -1).T,
           "joint_regressor": array(model["J_regressor"]), "parents": parents,
           "skinning_weights": array(model["weights"]),
           "eyelid_dirs_m": np.load(owned_path(Path(args.eyelids)), allow_pickle=False)}
    FlameExpressionRig(rig)
    output = owned_path(Path(args.output))
    if output.suffix != ".npz" or output.exists():
        raise ValueError("Choose a new .npz FLAME rig output")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **rig)
    print(json.dumps({"file": output.name, "sha256": sha256(output), "upstream_sha256": sha256(source)}))


def convert_expression(args):
    """Fit the actual checkpoint basis to Open directions and quantify loss.

    This works on the checkpoint's original model (including FLAME2020), not
    an assumed FLAME2023 basis. A failed residual gate produces no conversion.
    Facial topology and vertex correspondence must be identical in both files.
    """
    import pickle
    from roop_3d.reconstruction.morphable_backbone import binary, finite
    if not all(np.isfinite(value) and value > 0 for value in (args.max_rms_mm, args.max_p95_mm)):
        raise ValueError("Expression conversion residual limits must be positive and finite")
    models = []
    for path in (args.original, args.open_model):
        with owned_path(Path(path)).open("rb") as stream:
            models.append(pickle.load(stream, encoding="latin1"))
    original, opened = models
    if not np.array_equal(original["f"], opened["f"]):
        raise ValueError("FLAME versions do not share registered face topology")
    old = np.asarray(original["shapedirs"], np.float64)[:, :, 300:350]
    new = np.asarray(opened["shapedirs"], np.float64)[:, :, 300:]
    finite(old, (5023, 3, 50), "original SMIRK expression directions")
    with np.load(owned_path(Path(args.anatomy)), allow_pickle=False) as masks:
        active = binary(masks["face_surface"], (5023,), "FLAME anatomical face")
        for name in ("scalp_hair", "ears", "neck"):
            active &= ~binary(masks[name], (5023,), name)
    if active.sum() < 100:
        raise ValueError("Expression conversion needs an anatomical facial surface")
    matrix, _, rank, _ = np.linalg.lstsq(new[active].reshape(-1, new.shape[-1]), old[active].reshape(-1, 50), rcond=1e-8)
    if rank < min(50, new.shape[-1]):
        raise ValueError("Open expression basis is rank deficient on the anatomical face")
    error = np.linalg.norm(np.einsum("vck,ke->vce", new[active], matrix)-old[active], axis=1) * 1000
    rms, p95 = np.sqrt(np.mean(error**2, axis=0)), np.percentile(error, 95, axis=0)
    report = {"original_sha256": sha256(args.original), "open_sha256": sha256(args.open_model),
              "anatomy_sha256": sha256(args.anatomy), "basis_columns": 50, "active_vertices": int(active.sum()),
              "rms_mm_per_unit_coefficient": rms.tolist(), "p95_mm_per_unit_coefficient": p95.tolist(),
              "max_rms_mm": float(rms.max()), "max_p95_mm": float(p95.max()),
              "scope": "linear expression basis only; jaw/eyelid/neutral anatomy require separate validation"}
    if rms.max() > args.max_rms_mm or p95.max() > args.max_p95_mm:
        raise ValueError("Expression conversion exceeds residual limits: " + json.dumps(report))
    output = owned_path(Path(args.output))
    if output.suffix != ".npy" or output.exists():
        raise ValueError("Choose a new .npy conversion output")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, matrix.astype(np.float32), allow_pickle=False)
    output.with_suffix(".conversion.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def assemble(args):
    from roop_3d.tools.dense_face_bundle import assemble as publish_bundle
    publish_bundle(args.config, args.output)
    print(str(args.output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch", help="Fetch pinned public assets; FLAME models remain user-supplied")
    fetch.add_argument("--component", choices=("all", "3ddfa", "smirk", "hrn"), default="all")
    fetch.add_argument("--output", required=True)
    from roop_3d.tools.dense_face_downloads import fetch_public
    fetch.set_defaults(run=fetch_public)
    export = sub.add_parser("export", help="Export upstream encoders and verify torch/ORT parity")
    export.add_argument("--component", choices=("3ddfa", "smirk", "hrn"), required=True)
    export.add_argument("--source", required=True, help="Inspected upstream checkout inside this repository")
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--backbone", choices=("resnet50", "mbnetv3"), default="resnet50")
    export.set_defaults(run=export_component)
    bfm = sub.add_parser("bfm", help="Import official face_model.npy with real anatomical masks and HRN UVs")
    for name in ("model", "standard", "hrn-uvs", "anatomy", "output"):
        bfm.add_argument("--"+name, required=True)
    bfm.set_defaults(run=prepare_bfm)
    anatomy = sub.add_parser("flame-anatomy", help="Convert official named FLAME masks to face-only anatomical masks")
    anatomy.add_argument("--masks", required=True)
    anatomy.add_argument("--output", required=True)
    anatomy.set_defaults(run=prepare_flame_anatomy)
    flame = sub.add_parser("flame", help="Import FLAME Open plus validated SMIRK expression conversion and eyelids")
    for name in ("model", "conversion", "eyelids", "output"):
        flame.add_argument("--"+name, required=True)
    flame.set_defaults(run=prepare_flame)
    conversion = sub.add_parser("convert-expression", help="Fit checkpoint-original FLAME expression directions to Open and measure error")
    for name in ("original", "open-model", "anatomy", "output"):
        conversion.add_argument("--"+name, required=True)
    conversion.add_argument("--max-rms-mm", type=float, default=1.)
    conversion.add_argument("--max-p95-mm", type=float, default=2.)
    conversion.set_defaults(run=convert_expression)
    bundle = sub.add_parser("assemble", help="Assemble prepared native assets into a runtime bundle")
    bundle.add_argument("--config", required=True)
    bundle.add_argument("--output", required=True)
    bundle.set_defaults(run=assemble)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
