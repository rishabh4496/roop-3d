"""Local native bundle tooling. From app/: python -m roop_3d.tools.dense_face_bundle --help.

Assembly config is native bundle metadata plus assets[role].file relative to the
config. Digests are computed, or checked when supplied. Assets must already live
beneath the destination bundle directory, typically in prepared/. Nothing is
copied, downloaded, inferred, or invented. Readiness is structural, not quality.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import tempfile

from roop_3d.models.owned_files import owned_path
from roop_3d.tools.prepare_dense_face import sha256
from roop_3d.reconstruction.native_backbone import (
    GRAPH_CONTRACTS, NATIVE_ROLES, NativeMorphableBackbone, _arrays,
    validate_graph_contract, verify_native_assets,
)
from roop_3d.reconstruction.morphable_backbone import FaceTopology
from roop_3d.reconstruction.native_face_geometry import FlameExpressionRig

SCOPE = ("CPU asset structure and encoder contracts only; no inference, GPU context, "
         "or end-to-end reconstruction/identity/expression/appearance quality validation. "
         "Export parity evidence is not end-to-end quality evidence.")


def _json(path):
    path = owned_path(Path(path))
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _asset(directory, name):
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError("Asset file must be a nonempty relative path")
    path = owned_path(directory / name)
    if not path.is_relative_to(directory):
        raise ValueError(f"Asset leaves its directory: {name}")
    return path


def _embedded_onnx(payload):
    """Reject external TensorProto storage without requiring the onnx package.

    Inspect protobuf envelopes, never tensor contents. Native exports must be
    single files so every weight byte is covered by the manifest digest.
    ORT subsequently validates the actual graph.
    """
    nested = {
        "model": {7: "graph", 20: "training", 25: "function"},
        "training": {1: "graph", 2: "graph"},
        "function": {7: "node", 11: "attribute"},
        "graph": {1: "node", 5: "tensor", 15: "sparse"},
        "node": {5: "attribute"},
        "attribute": {5: "tensor", 6: "graph", 10: "tensor", 11: "graph", 22: "sparse", 23: "sparse"},
        "sparse": {1: "tensor", 2: "tensor"},
        "tensor": {},
    }
    def varint(data, offset):
        value = 0
        for shift in range(0, 70, 7):
            if offset >= len(data):
                raise ValueError("Truncated ONNX protobuf")
            byte = data[offset]
            offset += 1
            value |= (byte & 127) << shift
            if byte < 128:
                return value, offset
        raise ValueError("Invalid ONNX protobuf varint")

    def inspect(data, kind, depth=0):
        if depth > 64:
            raise ValueError("ONNX nesting exceeds inspection limit")
        offset = 0
        while offset < len(data):
            tag, offset = varint(data, offset)
            field, wire = tag >> 3, tag & 7
            if not field:
                raise ValueError("Invalid ONNX protobuf field")
            if kind == "tensor" and field in (13, 14):
                if field == 13:
                    raise ValueError("External ONNX weights are unsupported; prepare a self-contained export")
                value, offset = varint(data, offset)
                if wire != 0 or value != 0:
                    raise ValueError("External ONNX weights are unsupported; prepare a self-contained export")
                continue
            if wire == 0:
                _, offset = varint(data, offset)
            elif wire in (1, 5):
                offset += 8 if wire == 1 else 4
            elif wire == 2:
                size, offset = varint(data, offset)
                end = offset + size
                if end > len(data):
                    raise ValueError("Truncated ONNX protobuf field")
                if field in nested[kind]:
                    inspect(data[offset:end], nested[kind][field], depth + 1)
                offset = end
            else:
                raise ValueError("Unsupported ONNX protobuf wire type")
            if offset > len(data):
                raise ValueError("Truncated ONNX protobuf field")
    inspect(memoryview(payload), "model")
    return payload


def _cpu_session(path):
    import onnxruntime as ort
    payload = _embedded_onnx(owned_path(Path(path)).read_bytes())
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(payload, sess_options=options,
                                providers=["CPUExecutionProvider"])


def validate_bundle(bundle, spec):
    """Use runtime validators directly, but never call load() or a renderer."""
    for key in ("mask_provenance", "expression_mapping_evidence", "registration_evidence"):
        if not isinstance(spec.get(key), str) or not spec[key].strip():
            raise ValueError(f"{key}: supply nonempty genuine preparation evidence, not inferred provenance")
    paths = verify_native_assets(bundle, spec)
    topology = FaceTopology.load(spec["topology"], paths["topology"])
    bfm_topology = FaceTopology.load("bfm35709", paths["bfm_topology"])
    rig = FlameExpressionRig(_arrays(paths["flame_rig"]))
    # Constructor checks BFM dimensions, both surface transfers, registration
    # similarity and UV charts. Its UVSurfaceMap is NumPy-only, not GPU rendering.
    NativeMorphableBackbone(topology, bfm_topology, _arrays(paths["bfm"]), rig,
                           _arrays(paths["registration"]), {}, None, spec)
    for role in GRAPH_CONTRACTS:
        try:
            session = _cpu_session(paths[role])
            validate_graph_contract(role, session)
            del session
        except Exception as error:
            raise ValueError(f"{role}: CPU graph contract validation failed: {error}") from error
    return paths


def assemble(config, output):
    """Publish a validated manifest atomically and exclusively, without copying assets."""
    config, output = owned_path(Path(config)), owned_path(Path(output))
    if output.exists():
        raise ValueError(f"Refusing to overwrite bundle: {output}")
    spec = copy.deepcopy(_json(config))
    for role in NATIVE_ROLES:
        try:
            record = spec["assets"][role]
            path = _asset(config.parent, record["file"])
            if not path.is_file():
                raise ValueError(f"Missing prepared asset: {path}")
            if not path.is_relative_to(output.parent):
                raise ValueError("Prepared assets must be beneath the bundle directory; choose its parent as output")
            digest = sha256(path)
            if "sha256" in record and record["sha256"] != digest:
                raise ValueError("Prepared asset SHA256 mismatch")
            spec["assets"][role] = {"file": path.relative_to(output.parent).as_posix(), "sha256": digest}
        except (KeyError, TypeError, ValueError, OSError) as error:
            raise ValueError(f"{role}: {error}; supply a genuine prepared asset and metadata") from error
    validate_bundle(output, spec)
    # Rehash after validation, catching modifications during CPU/session checks.
    verify_native_assets(output, spec)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".bundle-", suffix=".part", dir=output.parent)
    part = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(spec, indent=2, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        owned_path(output)
        # link is atomic and fails if destination exists, unlike replace/rename
        # on POSIX. Remove the temporary link immediately after publication.
        os.link(part, output)
    finally:
        part.unlink(missing_ok=True)
    return spec


def export_evidence(prepared):
    """Inventory supplied evidence, not an assertion that reports are truthful."""
    prepared = owned_path(Path(prepared))
    found = []
    if prepared.is_dir():
        for candidate in sorted(prepared.rglob("*.json")):
            item = {"file": candidate.relative_to(prepared).as_posix()}
            try:
                path = owned_path(candidate)
                item["sha256"] = sha256(path)
                report = _json(path)
                item["report"] = report
                item["assessment"] = "supplied evidence, not independently certified"
            except (OSError, ValueError) as error:
                item["error"] = str(error)
            found.append(item)
    return found


def readiness(bundle, prepared=None, config=None):
    bundle = owned_path(Path(bundle))
    prepared = owned_path(Path(prepared)) if prepared else owned_path(bundle.parent / "prepared")
    result = {"bundle": str(bundle), "bundle_exists": bundle.is_file(), "ready": False,
              "scope": SCOPE, "errors": [], "assets": {}, "export_evidence": export_evidence(prepared)}
    spec = None
    base = bundle.parent
    try:
        if bundle.is_file():
            spec = _json(bundle)
        elif config:
            config = owned_path(Path(config))
            spec, base = _json(config), config.parent
            result["errors"].append("bundle.json absent; asset presence is not readiness until validated assembly publishes it")
        else:
            result["errors"].append("bundle.json absent; assemble with --config containing genuine assets and provenance")
    except (OSError, ValueError) as error:
        result["errors"].append(str(error))
    for role in NATIVE_ROLES:
        try:
            if spec is not None:
                path = _asset(base, spec["assets"][role]["file"])
            else:
                name = role + (".onnx" if role in GRAPH_CONTRACTS else ".npz")
                candidates = [owned_path(path) for path in prepared.rglob(name)] if prepared.is_dir() else []
                if len(candidates) > 1:
                    raise ValueError(f"Multiple prepared candidates for {role}; use --config to identify the intended asset")
                path = candidates[0] if candidates else owned_path(prepared / name)
            exists = path.is_file()
            result["assets"][role] = {"file": str(path), "present": exists}
            if not exists:
                result["errors"].append(f"Missing {role}: {path}. Prepare this asset locally; no substitute is generated.")
        except (KeyError, TypeError, ValueError, OSError) as error:
            result["assets"][role] = {"present": False, "error": str(error)}
            result["errors"].append(f"{role}: missing/invalid asset record: {error}")
    if bundle.is_file() and spec is not None and not result["errors"]:
        try:
            validate_bundle(bundle, spec)
            result["ready"] = True
        except Exception as error:
            result["errors"].append(f"Bundle validation failed: {error}")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("assemble", help="Validate prepared local assets and exclusively publish bundle.json")
    build.add_argument("--config", required=True)
    build.add_argument("--output", required=True)
    status = sub.add_parser("status", aliases=["readiness"], help="Inspect missing assets, contracts and prepared evidence")
    status.add_argument("--bundle", required=True)
    status.add_argument("--prepared")
    status.add_argument("--config")
    args = parser.parse_args(argv)
    try:
        if args.command == "assemble":
            assemble(args.config, args.output)
            print(json.dumps({"published": str(args.output), "scope": SCOPE}))
            return 0
        report = readiness(args.bundle, args.prepared, args.config)
        print(json.dumps(report, indent=2))
        return 0 if report["ready"] else 1
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"dense-face bundle: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
