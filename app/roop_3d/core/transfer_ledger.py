"""Measured host<->device copies at the ONNX Runtime boundary.

What is counted (each is a real memcpy across PCIe, observed at the Python API call that causes it):

* ``OrtValue.ortvalue_from_numpy(array, "cuda", ...)`` and ``OrtValue.update_inplace(array)`` on a device value: host -> device.
* ``OrtValue.numpy()`` on a device value: device -> host.
* ``InferenceSession.run(...)`` on a CUDA / TensorRT session: ONNX Runtime stages every numpy input up and every output
  back down itself; counted per tensor.

What is *not* counted, and therefore never reported as zero: copies made inside torch, CV-CUDA, OpenCV-CUDA or the NVENC / NVDEC
engines, and anything ONNX Runtime does internally between graph partitions. ``scope()`` attributes copies to the pipeline stage
that issued them; ``boundary()`` tags the one frame upload and the one final download a GPU-resident chain is allowed, so a copy
outside a boundary is an *intermediate* copy. Nothing in the current pipeline tags a boundary, so today every copy is intermediate:
the ledger reports that as it is.
"""
from __future__ import annotations

from contextlib import contextmanager
import inspect
import os
import threading
from typing import Any, Callable, Iterator

DIRECTIONS = ("h2d", "d2h")
_BOUNDARY = "boundary"
_INTERMEDIATE = "intermediate"


def _enabled() -> bool:
    return os.environ.get("ROOP3D_TRANSFER_LEDGER", "1").strip().lower() not in {"0", "false", "off", "no"}


class TransferLedger:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._local = threading.local()
        self._total: dict[str, int] = {}
        self._window: dict[str, int] = {}
        self._sites: dict[str, dict[str, int]] = {}
        self._gpu_runs = 0
        self._restore: list[Callable[[], None]] = []
        self._target: Any = None
        self.installed_on: str | None = None
        self.install_error: str | None = None

    # -- recording -----------------------------------------------------------------------------------------
    def _scope(self) -> str:
        return getattr(self._local, "scope", None) or "unscoped"

    def _kind(self) -> str:
        return _BOUNDARY if getattr(self._local, "boundary", 0) else _INTERMEDIATE

    @contextmanager
    def scope(self, name: str) -> Iterator[None]:
        previous = getattr(self._local, "scope", None)
        self._local.scope = name
        try:
            yield
        finally:
            self._local.scope = previous

    @contextmanager
    def boundary(self) -> Iterator[None]:
        """Copies inside this block are the allowed frame upload / final download, not intermediate copies."""
        self._local.boundary = getattr(self._local, "boundary", 0) + 1
        try:
            yield
        finally:
            self._local.boundary -= 1

    def record_gpu_run(self) -> None:
        with self._lock:
            self._gpu_runs += 1

    def record(self, direction: str, nbytes: int, *, count: int = 1) -> None:
        if direction not in DIRECTIONS:
            raise ValueError(f"Unknown transfer direction: {direction}")
        if not isinstance(nbytes, int) or nbytes < 0 or count < 1:
            return
        kind, scope = self._kind(), self._scope()
        keys = (f"{direction}_count", f"{direction}_bytes", f"{kind}_{direction}_count", f"{kind}_{direction}_bytes")
        amounts = (count, nbytes, count, nbytes)
        with self._lock:
            for target in (self._total, self._window):
                for key, amount in zip(keys, amounts):
                    target[key] = target.get(key, 0) + amount
            site = self._sites.setdefault(scope, {})
            site[f"{direction}_count"] = site.get(f"{direction}_count", 0) + count
            site[f"{direction}_bytes"] = site.get(f"{direction}_bytes", 0) + nbytes

    # -- reading -------------------------------------------------------------------------------------------
    def _shape(self, values: dict[str, int]) -> dict[str, Any]:
        out = {key: values.get(key, 0) for key in ("h2d_count", "h2d_bytes", "d2h_count", "d2h_bytes")}
        out["intermediate_copies"] = sum(values.get(f"{_INTERMEDIATE}_{d}_count", 0) for d in DIRECTIONS)
        out["boundary_copies"] = sum(values.get(f"{_BOUNDARY}_{d}_count", 0) for d in DIRECTIONS)
        return out

    def snapshot(self, *, frames: int | None = None) -> dict[str, Any]:
        """Cumulative totals plus the per-scope breakdown. ``frames`` (if known) adds per-frame averages."""
        with self._lock:
            total, sites = dict(self._total), {name: dict(value) for name, value in self._sites.items()}
        result = {"measured": self.installed_on is not None, "gpu_runs": self._gpu_runs, "scope": "onnxruntime-boundary", "installed_on": self.installed_on,
                  "total": self._shape(total), "by_stage": sites, "not_counted": [
                      "torch / CV-CUDA / OpenCV-CUDA copies", "NVENC / NVDEC internal transfers", "copies inside ONNX Runtime between graph partitions"]}
        totals = result["total"]
        if frames:
            result["per_frame"] = {"h2d_tensors": round(totals["h2d_count"] / frames, 3), "d2h_tensors": round(totals["d2h_count"] / frames, 3),
                                   "h2d_bytes": round(totals["h2d_bytes"] / frames), "d2h_bytes": round(totals["d2h_bytes"] / frames),
                                   "intermediate_copies": round(totals["intermediate_copies"] / frames, 3)}
        # None = nothing was observed (ledger off, or no GPU session ran): that is "unknown", never "zero".
        observed = self.installed_on is not None and (self._gpu_runs > 0 or totals["h2d_count"] + totals["d2h_count"] > 0)
        result["zero_intermediate_copies"] = (totals["intermediate_copies"] == 0) if observed else None
        return result

    def take(self) -> dict[str, Any]:
        """Counters since the previous ``take`` (the progress-report window); resets the window."""
        with self._lock:
            window, self._window = self._window, {}
        return self._shape(window)

    def reset(self) -> None:
        with self._lock:
            self._total.clear()
            self._window.clear()
            self._sites.clear()
            self._gpu_runs = 0

    # -- installation --------------------------------------------------------------------------------------
    def install(self, ort: Any = None) -> bool:
        """Wrap the ONNX Runtime transfer points. Returns False (and records nothing) when ORT is unavailable, disabled or shaped
        differently than expected: the ledger only observes, so a failure to install must never reach the code that creates sessions."""
        if not _enabled():
            return False
        if ort is None:
            try:
                import onnxruntime as ort  # type: ignore[no-redef]
            except ImportError:
                return False
        if self.installed_on is not None:
            if self._target is ort:
                return True
            self.uninstall()                 # a different onnxruntime module is in play (an embedded test double, a reload): re-target
        try:
            return self._install(ort)
        except Exception as error:           # noqa: BLE001 - observational tool, never fatal
            self.uninstall()
            self.install_error = f"{type(error).__name__}: {error}"
            return False

    def _install(self, ort: Any) -> bool:
        required = (("OrtValue", ("ortvalue_from_numpy", "update_inplace", "numpy")), ("InferenceSession", ("run",)))
        for owner_name, names in required:
            owner = getattr(ort, owner_name, None)
            if owner is None or not all(hasattr(owner, name) for name in names):
                return False                 # not the ONNX Runtime API this ledger understands: observe nothing rather than half of it
        self.install_error = None
        ledger = self

        def device_of(value) -> str:
            try:
                return str(value.device_name()).casefold()
            except Exception:
                return "cpu"

        ortvalue = ort.OrtValue
        original_from_numpy = inspect.getattr_static(ortvalue, "ortvalue_from_numpy")   # the raw descriptor, restored verbatim on uninstall
        bound_from_numpy = ortvalue.ortvalue_from_numpy                   # works whether ORT declares a classmethod or staticmethod

        def from_numpy(numpy_obj, device_type="cpu", device_id=0, *rest):
            if str(device_type).casefold() != "cpu":
                ledger.record("h2d", int(getattr(numpy_obj, "nbytes", 0)))
            return bound_from_numpy(numpy_obj, device_type, device_id, *rest)

        original_update = ortvalue.update_inplace

        def update_inplace(self, np_arr):
            if device_of(self) != "cpu":
                ledger.record("h2d", int(getattr(np_arr, "nbytes", 0)))
            return original_update(self, np_arr)

        original_numpy = ortvalue.numpy

        def numpy(self):
            array = original_numpy(self)
            if device_of(self) != "cpu":
                ledger.record("d2h", int(getattr(array, "nbytes", 0)))
            return array

        session_class = ort.InferenceSession
        original_run = session_class.run

        def run(self, output_names, input_feed, *rest, **keywords):
            outputs = original_run(self, output_names, input_feed, *rest, **keywords)
            try:
                providers = self.get_providers()
            except Exception:
                providers = ()
            if providers and providers[0] in ("CUDAExecutionProvider", "TensorrtExecutionProvider"):
                for value in input_feed.values():
                    if hasattr(value, "nbytes"):
                        ledger.record("h2d", int(value.nbytes))
                for value in outputs:
                    if hasattr(value, "nbytes"):
                        ledger.record("d2h", int(value.nbytes))
            return outputs

        original_bound = getattr(session_class, "run_with_iobinding", None)

        def run_with_iobinding(self, *args, **keywords):
            try:
                providers = self.get_providers()
            except Exception:
                providers = ()
            if providers and providers[0] in ("CUDAExecutionProvider", "TensorrtExecutionProvider"):
                ledger.record_gpu_run()
            return original_bound(self, *args, **keywords)

        patches = [(ortvalue, "ortvalue_from_numpy", original_from_numpy, staticmethod(from_numpy)),
                   (ortvalue, "update_inplace", original_update, update_inplace),
                   (ortvalue, "numpy", original_numpy, numpy),
                   (session_class, "run", original_run, run)]
        if original_bound is not None:
            patches.append((session_class, "run_with_iobinding", original_bound, run_with_iobinding))
        self._restore = []
        for owner, name, original, replacement in patches:
            own = name in vars(owner)                    # inherited attributes are removed again, not shadowed forever
            setattr(owner, name, replacement)
            self._restore.append(lambda owner=owner, name=name, original=original, own=own:
                                 setattr(owner, name, original) if own else delattr(owner, name))
        self._target = ort
        self.installed_on = getattr(ort, "__version__", "onnxruntime")
        return True

    def uninstall(self) -> None:
        for restore in reversed(self._restore):
            try:
                restore()
            except Exception:                # noqa: BLE001 - best effort; a half-restored double must not raise into the caller
                pass
        self._restore = []
        self._target = None
        self.installed_on = None


TRANSFERS = TransferLedger()
