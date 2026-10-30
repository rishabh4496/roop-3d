"""Private fixed golden tensors captured from genuine Stage 06 CPU inference."""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import io
import json
import math
import zipfile
from pathlib import Path

import numpy as np

from roop_3d.audit.isolation import sha256_file
from roop_3d.core.errors import DomainError
from roop_3d.core.storage import atomic_bytes, atomic_json, owned
from roop_3d.engine.runtime import OrtModel, RuntimeConfig
from roop_3d.models.export_validation import fixture_digest


@dataclass(frozen=True)
class ModelGolden:
    name: str
    model_sha256: str
    cases: tuple[dict[str, np.ndarray], ...]
    expected: tuple[dict[str, np.ndarray], ...]

    def __post_init__(self) -> None:
        if not 1 <= len(self.cases) == len(self.expected) <= 8 or any(not case for case in (*self.cases, *self.expected)):
            raise ValueError("Golden input/output case counts must match and remain within 1..8")

    def digest(self) -> str:
        return hashlib.sha256((self.model_sha256 + fixture_digest(self.cases) + fixture_digest(self.expected)).encode()).hexdigest()


def frozen_copy(values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    output = {}
    for name, value in values.items():
        if not isinstance(value, np.ndarray) or not np.isfinite(value).all():
            raise DomainError("GOLDEN_TENSOR_INVALID", "Golden tensors must be finite arrays", 422)
        array = np.ascontiguousarray(value).copy()
        array.setflags(write=False)
        output[name] = array
    return output


class GoldenBank:
    def __init__(self, *, cases_per_model: int = 2):
        if not 1 <= cases_per_model <= 8:
            raise ValueError("Golden cases per model must be within 1..8")
        self.limit = cases_per_model
        self.models: dict[str, ModelGolden] = {}

    def record(self, record: dict, inputs: dict[str, np.ndarray], outputs: dict[str, np.ndarray]) -> None:
        old = self.models.get(record["name"])
        if old is not None and old.model_sha256 != record["sha256"]:
            raise DomainError("GOLDEN_MODEL_CHANGED", "Source artifact changed during golden capture", 409)
        if old is not None and (len(old.cases) >= self.limit or any(fixture_digest((case,)) == fixture_digest((inputs,)) for case in old.cases)):
            return
        cases = (*old.cases, frozen_copy(inputs)) if old else (frozen_copy(inputs),)
        expected = (*old.expected, frozen_copy(outputs)) if old else (frozen_copy(outputs),)
        self.models[record["name"]] = ModelGolden(record["name"], record["sha256"], cases, expected)

    def factory(self, catalog, name: str, config: RuntimeConfig):
        reference = OrtModel(catalog, name, replace(config, provider="CPUExecutionProvider", hardware_profile="cpu", model_factory=None))
        bank = self
        class Trace:
            def __getattr__(self, key):
                return getattr(reference, key)
            def run(self, values):
                output = reference.run(values)
                bank.record(reference.record, values, output)
                return output
        return Trace()

    def save(self, root: Path, relative: str = "roop_3d/.runtime/golden/stage07") -> None:
        if not relative.startswith("roop_3d/.runtime/"):
            raise ValueError("Biometric golden tensors must remain in ignored private runtime state")
        import onnxruntime as ort
        records = []
        for name, golden in sorted(self.models.items()):
            values = {}
            for i, (inputs, outputs) in enumerate(zip(golden.cases, golden.expected)):
                for kind, arrays in (("i", inputs), ("o", outputs)):
                    for j, (_, value) in enumerate(sorted(arrays.items())):
                        values[f"{kind}{i}_{j}"] = value
            stream = io.BytesIO()
            np.savez_compressed(stream, **values)
            filename = golden.digest() + ".npz"
            atomic_bytes(root, f"{relative}/{filename}", stream.getvalue())
            records.append({"name": name, "model_sha256": golden.model_sha256, "digest": golden.digest(), "file": filename,
                "sha256": sha256_file(owned(root, f"{relative}/{filename}"))[1],
                "input_names": [sorted(case) for case in golden.cases],
                "output_names": [sorted(case) for case in golden.expected]})
        atomic_json(root, f"{relative}/manifest.json", {"schema_version": 1, "reference_provider": "CPUExecutionProvider",
                    "ort_version": ort.__version__, "cases_per_model": self.limit, "models": records})

    @classmethod
    def load(cls, root: Path, relative: str = "roop_3d/.runtime/golden/stage07") -> "GoldenBank":
        import onnxruntime as ort
        if not relative.startswith("roop_3d/.runtime/"):
            raise ValueError("Golden bank must be inside private runtime state")
        document = json.loads(owned(root, f"{relative}/manifest.json").read_text("utf-8"))
        if document["schema_version"] != 1 or document["reference_provider"] != "CPUExecutionProvider" or document["ort_version"] != ort.__version__:
            raise DomainError("GOLDEN_REFERENCE_MISMATCH", "Recapture reference fixtures for this ORT version", 409)
        bank = cls(cases_per_model=document["cases_per_model"])
        if not 1 <= len(document["models"]) <= 32:
            raise ValueError("Golden model budget exceeded")
        for record in document["models"]:
            if record["file"] != record["digest"] + ".npz":
                raise ValueError("Golden archive name differs")
            path = owned(root, f"{relative}/{record['file']}")
            if sha256_file(path)[1] != record["sha256"]:
                raise DomainError("GOLDEN_DRIFT", "Golden archive hash changed", 409)
            with zipfile.ZipFile(path) as archive:
                entries = archive.infolist()
                if (len(entries) > 256 or len({e.filename for e in entries}) != len(entries)
                        or sum(e.file_size for e in entries) > 256 * 1048576
                        or any(e.file_size > 64 * 1048576 or e.compress_size == 0 and e.file_size
                               or "/" in e.filename or "\\" in e.filename or not e.filename.endswith(".npy")
                               or (e.external_attr >> 16) & 0o170000 == 0o120000 for e in entries)):
                    raise DomainError("GOLDEN_ARCHIVE_INVALID", "Private tensor archive exceeds its format/expansion budget", 422)
                for entry in entries:
                    with archive.open(entry) as data:
                        version = np.lib.format.read_magic(data)
                        if version not in ((1, 0), (2, 0)):
                            raise DomainError("GOLDEN_ARCHIVE_INVALID", "Unsupported NPY header version", 422)
                        reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
                        shape, _, dtype = reader(data)
                        if (not 1 <= len(shape) <= 8 or any(d <= 0 for d in shape) or dtype.hasobject
                                or dtype.kind not in "fiu" or dtype.itemsize > 8
                                or math.prod(shape) * dtype.itemsize != entry.file_size - data.tell()):
                            raise DomainError("GOLDEN_ARCHIVE_INVALID", "NPY allocation/header differs from bounded payload", 422)
            cases, expected = [], []
            with np.load(path, allow_pickle=False) as arrays:
                if len(record["input_names"]) != len(record["output_names"]) or not 1 <= len(record["input_names"]) <= bank.limit:
                    raise ValueError("Golden case count differs")
                for i, (inputs, outputs) in enumerate(zip(record["input_names"], record["output_names"])):
                    cases.append(frozen_copy({name: arrays[f"i{i}_{j}"] for j, name in enumerate(inputs)}))
                    expected.append(frozen_copy({name: arrays[f"o{i}_{j}"] for j, name in enumerate(outputs)}))
            golden = ModelGolden(record["name"], record["model_sha256"], tuple(cases), tuple(expected))
            if golden.digest() != record["digest"]:
                raise DomainError("GOLDEN_DRIFT", "Golden tensor identities changed", 409)
            bank.models[golden.name] = golden
        return bank
