"""Parity-gated provider selection around the preserved Stage 06 model contract."""
from __future__ import annotations

from dataclasses import asdict, replace
import logging
from pathlib import Path
from typing import Any
import weakref

import numpy as np

from roop_3d.audit.project_paths import application_root
from roop_3d.core.errors import DomainError
from roop_3d.core.storage import atomic_json, owned
from roop_3d.engine.resources import guarded, register
from roop_3d.engine.runtime import RuntimeConfig, record_named, contract_array
from roop_3d.models.benchmark import benchmark
from roop_3d.models.catalog import ModelCatalog
from roop_3d.models.engine_cache import EngineKey
from roop_3d.models.diagnostics import JobDiagnostics, ModelDiagnostic
from roop_3d.models.export_validation import compare_outputs
from roop_3d.models.golden import GoldenBank, ModelGolden
from roop_3d.models.precision_policy import policy_for, require_no_custom_ops
from roop_3d.models.trt_builder import TensorRTBuilder, finish_profile, session_options, trt_availability
from roop_3d.models.registry import authorize

LOGGER = logging.getLogger(__name__)
PROVIDERS = ("TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider")


def run_checked(session, record: dict, values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    inputs = session.get_inputs()
    if set(values) != {node.name for node in inputs}:
        raise DomainError("INFERENCE_INPUT_NAMES", "Input names differ from the model contract", 422)
    for node in inputs:
        contract_array(values[node.name], node.shape, batch_one=True,
                       dtype=np.dtype("float16" if node.type == "tensor(float16)" else "float32"))
    nodes = session.get_outputs()
    expected_outputs = record["graph"]["outputs"]
    names = [output["name"] for output in expected_outputs]
    observed = [node.name for node in nodes]
    if len(names) != len(set(names)) or len(observed) != len(set(observed)) or set(names) != set(observed):
        raise DomainError("INFERENCE_OUTPUT_CONTRACT", f"Expected outputs {names}; observed {observed}", 503)
    try:
        # EPContext can reorder graph outputs. Bind by semantic name and request
        # the source contract's order explicitly; every name remains mandatory.
        arrays = session.run(names, values)
    except Exception as error:
        raise DomainError("PROVIDER_EXECUTION_FAILED", str(error)[:4096], 503) from None
    result = {}
    if len(arrays) != len(names):
        raise DomainError("INFERENCE_OUTPUT_CONTRACT", "Output count differs", 503)
    for expected, array in zip(expected_outputs, arrays):
        result[expected["name"]] = contract_array(array, expected["shape"],
            dtype=np.dtype("float16" if expected["dtype"] == "FLOAT16" else "float32"))
    return result


class AcceleratorFactory:
    """Qualify offline before creating a pipeline; compilation evicts prior sessions.

    Golden tensors remain private runtime data. Every newly opened accelerated
    session replays the fixed cases before becoming visible to frame inference.
    """
    def __init__(self, bank: GoldenBank, catalog: ModelCatalog | None = None, *, root: Path | None = None,
                 profile: str = "6gb", preferred: str = "TensorrtExecutionProvider", reserve_mib: int = 768,
                 cpu_threads: int = 2, allow_fp16: bool = True, custom_ops: bool = False):
        require_no_custom_ops(custom_ops)
        if preferred not in PROVIDERS or profile not in ("6gb", "12gb"):
            raise ValueError("Invalid acceleration provider/profile")
        self.bank, self.catalog, self.root = bank, catalog or ModelCatalog(), root or application_root()
        self.profile, self.preferred, self.reserve_mib, self.cpu_threads = profile, preferred, reserve_mib, cpu_threads
        self.allow_fp16 = allow_fp16
        self.config = RuntimeConfig(provider="CUDAExecutionProvider", hardware_profile=profile,
                                    reserve_mib=reserve_mib, cpu_threads=cpu_threads)
        self.builder = TensorRTBuilder(self.catalog, root=self.root, profile=profile, reserve_mib=reserve_mib, cpu_threads=cpu_threads)
        self.receipts: dict[str, dict] = {}
        self._models: weakref.WeakSet = weakref.WeakSet()
        self._rejections: dict[str, dict[str, list[dict]]] = {}

    def _open(self, record: dict, provider: str, *, engine_key: EngineKey | None = None, online: bool = False):
        import onnxruntime as ort
        source = self.catalog._path(record)
        options = session_options(self.root, self.cpu_threads)
        if provider == "TensorrtExecutionProvider":
            if engine_key is None:
                raise DomainError("TRT_CACHE_REQUIRED", "TRT inference requires a validated embedded engine", 409)
            bundle = self.builder.cache.lookup_bundle(engine_key)
            if bundle is None:
                raise DomainError("TRT_CACHE_MISSING", "Qualified engine bundle is missing; rebuild offline", 409)
            source = owned(self.root, (self.builder.cache.bundle_path(engine_key) / bundle.receipt["context_model"]).relative_to(self.root).as_posix())
            providers = ["TensorrtExecutionProvider", "CPUExecutionProvider"]
        elif provider == "CUDAExecutionProvider":
            if provider not in ort.get_available_providers():
                raise DomainError("PROVIDER_UNAVAILABLE", "ORT does not advertise CUDA EP", 503)
            ort.preload_dlls()
            budget = self.config.gpu_budget_bytes
            if online:
                # Detector/recognizer now join the accelerated chain. Fractions
                # reserve the largest share for the identity model; warp may
                # retain its CPU reference independently.
                divisor = {"identity_swapper": 3, "recognizer": 8, "detector": 16,
                           "liveportrait_warp": 4, "liveportrait_motion": 16,
                           "liveportrait_appearance": 32}.get(record.get("role"), 32)
                budget //= divisor
            providers = [(provider, {"gpu_mem_limit": budget, "arena_extend_strategy": "kSameAsRequested",
                "use_tf32": False, "cudnn_conv_use_max_workspace": False, "cudnn_conv_algo_search": "HEURISTIC"}), "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        try:
            session = ort.InferenceSession(str(source), sess_options=options, providers=providers)
        except Exception as error:
            raise DomainError("PROVIDER_SESSION_FAILED", str(error)[:4096], 503) from None
        if session.get_providers()[0] != provider:
            actual = session.get_providers()
            session = None
            raise DomainError("PROVIDER_ACTIVATION_FAILED", f"Requested {provider}; ORT activated {actual}", 503)
        session.disable_fallback()
        self.catalog.verify(record)
        return session

    def _validate(self, session, record: dict, golden: ModelGolden, precision: str):
        policy = policy_for(record["name"], record.get("role"))
        reports = tuple(compare_outputs(expected, run_checked(session, record, values), policy.tolerance(precision))
                        for values, expected in zip(golden.cases, golden.expected))
        try:
            policy.admit(precision, reports)
        except DomainError:
            rejected = self._rejections.setdefault(record["name"], {})
            rejected[session.get_providers()[0] + ":" + precision] = [asdict(report) for report in reports]
            raise
        return reports

    def _failure(self, history: list[dict], provider: str, precision: str, error: Exception) -> None:
        reason = f"{getattr(error, 'code', type(error).__name__)}: {error}"[:4096]
        history.append({"provider": provider, "precision": precision, "reason": reason})
        LOGGER.warning("Model provider fallback from %s (%s): %s", provider, precision, reason)

    @guarded
    def qualify(self, name: str, *, iterations: int = 5, warmups: int = 1) -> dict:
        record = record_named(self.catalog, name)
        authorize(self.catalog.registry.get(record["model_id"]), action="activate")
        golden = self.bank.models.get(name)
        if golden is None or golden.model_sha256 != record["sha256"]:
            raise DomainError("GOLDEN_MODEL_MISMATCH", "Golden fixture is absent or belongs to a different source artifact", 409)
        policy, history, engine_key = policy_for(name, record.get("role")), [], None
        self.builder.last_diagnostics = {"state": "UNKNOWN", "model_sha256": record["sha256"], "build_requested": False}
        self._rejections.pop(name, None)
        selected = None
        if self.preferred == "TensorrtExecutionProvider":
            availability = trt_availability()
            if not availability.available:
                self._failure(history, PROVIDERS[0], "fp32", DomainError("TRT_UNAVAILABLE", availability.reason or "TensorRT unavailable"))
            else:
                precisions = ("fp16", "fp32") if self.allow_fp16 and policy.fp16_candidate else ("fp32",)
                for precision in precisions:
                    try:
                        engine_key = self.builder.build(record, policy, precision, golden.digest(), golden.cases,
                            lambda session: self._validate(session, record, golden, precision))
                        selected = (PROVIDERS[0], precision)
                        break
                    except DomainError as error:
                        self._failure(history, PROVIDERS[0], precision, error)
        session = None
        reports, nodes = (), {}
        candidates = [selected] if selected else []
        if self.preferred != "CPUExecutionProvider":
            candidates.append(("CUDAExecutionProvider", "fp32"))
        candidates.append(("CPUExecutionProvider", "fp32"))
        try:
            for provider, precision in candidates:
                try:
                    session = self._open(record, provider, engine_key=engine_key)
                    reports = self._validate(session, record, golden, precision)
                    nodes = finish_profile(session, self.root)
                    if not nodes.get(provider):
                        raise DomainError("PROVIDER_NOT_EXECUTED", f"Profile did not execute {provider}; observed {nodes}", 503)
                    selected = provider, precision
                    break
                except (DomainError, RuntimeError) as error:
                    session = None
                    self._failure(history, provider, precision, error)
            if session is None or selected is None:
                raise DomainError("REFERENCE_INFERENCE_FAILED", "No provider passed reference parity", 503)
            provider, precision = selected
            candidate_time = benchmark(lambda: run_checked(session, record, golden.cases[0]), warmups=warmups, iterations=iterations)
        finally:
            session = None
        # Measure the untouched CPU reference independently after candidate release.
        reference = self._open(record, "CPUExecutionProvider")
        try:
            self._validate(reference, record, golden, "fp32")
            finish_profile(reference, self.root)
            reference_time = benchmark(lambda: run_checked(reference, record, golden.cases[0]), warmups=warmups, iterations=iterations)
        finally:
            reference = None
        receipt = {"schema_version": 1, "name": name, "model_sha256": record["sha256"],
            "golden_digest": golden.digest(), "policy_digest": policy.digest(), "profile": self.profile,
            "provider": provider, "requested_provider": self.preferred, "precision": precision,
            "engine_key": engine_key.model_dump(mode="json") if engine_key and provider == PROVIDERS[0] else None,
            "shape_profile": {name: list(value.shape) for name, value in golden.cases[0].items()},
            "fallback_history": history, "node_providers": nodes, "parity": [asdict(r) for r in reports],
            "rejected_parity": self._rejections.get(name, {}),
            "reference_benchmark": asdict(reference_time), "selected_benchmark": asdict(candidate_time),
            "fp16_state": ("VERIFIED_LOCAL" if precision == "fp16" else "UNSUPPORTED"
                if not policy.fp16_candidate or PROVIDERS[0] + ":fp16" in self._rejections.get(name, {})
                else "BLOCKED" if any(h["precision"] == "fp16" for h in history) else "UNKNOWN"),
            "sensitive_reason": policy.sensitive_reason, "build_diagnostics": self.builder.last_diagnostics,
            "acceleration_state": "VERIFIED_LOCAL" if provider != "CPUExecutionProvider" else "UNSUPPORTED",
            "selection": "accelerated" if provider != "CPUExecutionProvider" else "reference_fallback"}
        self.receipts[name] = receipt
        atomic_json(self.root, f"roop_3d/.runtime/acceleration/{self.profile}/{record['sha256']}.json", receipt)
        return receipt

    def qualify_all(self, *, iterations: int = 5, warmups: int = 1) -> list[dict]:
        return [self.qualify(name, iterations=iterations, warmups=warmups) for name in sorted(self.bank.models)]

    def __call__(self, catalog, name: str, config: RuntimeConfig):
        return AcceleratedModel(self, catalog, name)

    def runtime_config(self) -> RuntimeConfig:
        return replace(self.config, model_factory=self)

    @guarded
    def job_diagnostics(self) -> JobDiagnostics:
        models = {}
        for model in list(self._models):
            if model.session is not None:
                capability = model.capability()
                models[model.record["name"]] = ModelDiagnostic.model_validate(
                    {name: capability[name] for name in ModelDiagnostic.model_fields})
        return JobDiagnostics(models=models)

    def record_job_diagnostics(self, store, job_id: str):
        return store.record_diagnostics(job_id, self.job_diagnostics())


class AcceleratedModel:
    @guarded
    def __init__(self, factory: AcceleratorFactory, catalog: ModelCatalog, name: str):
        self.factory, self.catalog = factory, catalog
        self.record = record_named(catalog, name)
        self.path = catalog._path(self.record)
        self.identity = self._stat()
        self.executions = 0
        self.session = None
        self.history: list[dict] = []
        self.receipt = factory.receipts.get(name)
        golden, policy = factory.bank.models.get(name), policy_for(name, self.record.get("role"))
        if (self.receipt is None or golden is None or self.receipt["model_sha256"] != self.record["sha256"]
                or self.receipt["golden_digest"] != golden.digest() or self.receipt["policy_digest"] != policy.digest()):
            raise DomainError("MODEL_NOT_QUALIFIED", "Qualify this model offline on the fixed reference fixture first", 409)
        self.history = list(self.receipt["fallback_history"])
        self._activate(self.receipt["provider"])
        register(self, "release")
        factory._models.add(self)

    def _stat(self):
        value = self.path.stat()
        return value.st_size, value.st_mtime_ns, value.st_ino

    def _activate(self, start: str) -> None:
        order = PROVIDERS[PROVIDERS.index(start):]
        for provider in order:
            try:
                precision = self.receipt["precision"] if provider == self.receipt["provider"] else "fp32"
                key = EngineKey.model_validate(self.receipt["engine_key"]) if provider == PROVIDERS[0] else None
                self.session = self.factory._open(self.record, provider, engine_key=key, online=True)
                self.factory._validate(self.session, self.record, self.factory.bank.models[self.record["name"]], precision)
                nodes = finish_profile(self.session, self.factory.root)
                if not nodes.get(provider):
                    raise DomainError("PROVIDER_NOT_EXECUTED", f"Profile executed {nodes} instead of {provider}", 503)
                self.provider, self.precision, self.node_providers = provider, precision, nodes
                return
            except (DomainError, RuntimeError) as error:
                self.session = None
                self.factory._failure(self.history, provider, "fp32", error)
        raise DomainError("REFERENCE_INFERENCE_FAILED", "No runtime provider passed golden parity", 503)

    @guarded
    def run(self, values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if self.session is None:
            raise DomainError("SESSION_RELEASED", "Accelerated session was released; prepare again after a build", 409)
        if self._stat() != self.identity:
            self.catalog.verify(self.record)
            raise DomainError("MODEL_CHANGED", "Source artifact changed after preparation", 409)
        if set(values) != set(self.receipt["shape_profile"]) or any(
                not isinstance(value, np.ndarray) or list(value.shape) != self.receipt["shape_profile"][name]
                for name, value in values.items()):
            raise DomainError("INFERENCE_PROFILE_MISMATCH", "Frame tensor differs from its qualified fixed shape profile", 422)
        failed = None
        try:
            result = run_checked(self.session, self.record, values)
        except DomainError as error:
            if error.code != "PROVIDER_EXECUTION_FAILED":
                raise  # Input/output assertions are never hidden by a retry.
            if self.provider == "CPUExecutionProvider":
                raise DomainError("MODEL_INFERENCE_FAILED", str(error)[:4096], 503) from None
            failed = self.provider
            self.session = None
            self.factory._failure(self.history, failed, self.precision, error)
        # Leave the exception scope so its traceback cannot retain the failed
        # native session/arena while the lower provider allocates a replacement.
        if failed is not None:
            self._activate(PROVIDERS[PROVIDERS.index(failed) + 1])
            result = run_checked(self.session, self.record, values)
        self.executions += 1
        return result

    @guarded
    def release(self) -> None:
        self.session = None

    def capability(self) -> dict[str, Any]:
        return {"model_id": self.record["model_id"], "sha256": self.record["sha256"], "artifact_state": "VERIFIED_LOCAL",
            "inference_state": "VERIFIED_LOCAL" if self.executions else "UNKNOWN", "provider": self.provider,
            "requested_provider": self.factory.preferred, "precision": self.precision, "node_providers": self.node_providers,
            "executions": self.executions, "golden_digest": self.receipt["golden_digest"],
            "fallback_reason": self.history[-1]["reason"] if self.history else None, "fallback_history": self.history,
            "license_state": self.record["license_state"], "commercial_permission": self.record["commercial_permission"]}
