"""Small, hash-bound projection cache; NumPy object arrays are never accepted."""
from __future__ import annotations

from io import BytesIO
import hashlib
import json
from pathlib import Path
import os
import uuid

import numpy as np

from roop_3d.audit.isolation import sha256_file
from roop_3d.audit.project_paths import project_root, application_root
from roop_3d.core.storage import exclusive
from roop_3d.models.owned_files import owned_path


def cached_emap(source: Path, extract, *, source_sha256: str | None = None, destination: Path | None = None) -> np.ndarray:
    path = owned_path(destination or project_root() / "app/models/inswapper_128_emap.npy")
    receipt = owned_path(path.with_suffix(".json"))
    identity = source_sha256 or sha256_file(source)[1]
    with exclusive(application_root(), "roop_3d/.runtime/emap-cache.lock"):
        if path.exists() and receipt.exists():
            data = path.read_bytes()
            info = json.loads(receipt.read_text())
            if info.get("source_sha256") == identity and info.get("emap_sha256") == hashlib.sha256(data).hexdigest():
                value = np.load(BytesIO(data), allow_pickle=False)
                if value.shape != (512, 512) or value.dtype != np.float32 or not np.isfinite(value).all():
                    raise ValueError("Invalid cached emap tensor")
                value.setflags(write=False)
                return value
        value = np.asarray(extract(source), dtype=np.float32)
        if value.shape != (512, 512) or not np.isfinite(value).all():
            raise ValueError("Invalid extracted emap tensor")
        buffer = BytesIO()
        np.save(buffer, value, allow_pickle=False)
        data = buffer.getvalue()
        path.parent.mkdir(parents=True, exist_ok=True)
        for target, content in ((path, data), (receipt, json.dumps({"source_sha256": identity,
                "emap_sha256": hashlib.sha256(data).hexdigest(), "shape": [512, 512], "dtype": "float32"}).encode())):
            temporary = owned_path(target.with_name(target.name + f".{uuid.uuid4().hex}.part"))
            try:
                with temporary.open("xb") as stream:
                    stream.write(content)
                    stream.flush()
                    os.fsync(stream.fileno())
                owned_path(target)
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
        value.setflags(write=False)
        return value
