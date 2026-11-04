from __future__ import annotations

import hashlib
import numpy as np
import pytest

from roop_3d.core.errors import DomainError
from roop_3d.core.storage import owned
from roop_3d.engine.resources import evict_for_build
from roop_3d.engine.runtime import OrtModel, RuntimeConfig, SessionFacade
from roop_3d.models.accelerated_runtime import AcceleratorFactory, run_checked
from roop_3d.models.catalog import ModelCatalog
from roop_3d.models.golden import GoldenBank
from roop_3d.models.local_provider import LocalWeightProvider
from roop_3d.models.registry import Locator, Manifest, ModelRecord, ModelRegistry
from roop_3d.models.trt_builder import trt_availability
from roop_3d.models.precision_policy import policy_for


@pytest.fixture
def graph(tmp_path):
    import onnx
    from onnx import TensorProto, helper
    path = owned(tmp_path,"roop_3d/models/recognizer-fixture.onnx")
    path.parent.mkdir(parents=True)
    model = helper.make_model(helper.make_graph([helper.make_node("Identity",["input"],["output"]),
          helper.make_node("Neg",["input"],["negated"])], "unit-graph",
          [helper.make_tensor_value_info("input",TensorProto.FLOAT,[1,512])],
          [helper.make_tensor_value_info("output",TensorProto.FLOAT,[1,512]),
           helper.make_tensor_value_info("negated",TensorProto.FLOAT,[1,512])]), opset_imports=[helper.make_opsetid("",12)])
    model.ir_version = 9
    onnx.save(model,str(path))
    data=path.read_bytes()
    record=ModelRecord(model_id="stage07-unit",name=path.name,role="recognizer",role_evidence="Generated identity graph for session-control unit tests",
        filenames=(path.name,),expected_sha256=hashlib.sha256(data).hexdigest(),byte_size=len(data),
        input_output_contract={"inputs":[{"name":"input","dtype":"FLOAT","shape":[1,512]}],
                               "outputs":[{"name":"output","dtype":"FLOAT","shape":[1,512]},
                                          {"name":"negated","dtype":"FLOAT","shape":[1,512]}]},
        status="verified_local",locator=Locator(scope="fixture",path=path.name),evidence="Bytes generated and hashed in this test",
        license_notice="Generated unit fixture; no third-party model rights are asserted",
        license_id="LicenseRef-Local-Test",license_url="https://roop3d.invalid/test-fixture-terms",
        commercial_use="allow",redistribution_permission="allow",attribution_requirements="No third-party attribution",
        original_repository="LOCAL_TEST_FIXTURE",source_revision="LOCAL_GENERATED_FIXTURE",architecture="Identity",
        required_base_model="NONE",hardware_requirements={"state":"VERIFIED"},precision_requirements=("FP32",),
        dependency_compatibility={"state":"COMPATIBLE","requirements":[]},security_status="VERIFIED",
        serialization_format="ONNX")
    registry=ModelRegistry(manifest=Manifest(schema_version=1,evidence_scope="generated unit fixture",models=(record,)))
    catalog=ModelCatalog(registry=registry)
    catalog.provider=LocalWeightProvider(registry,root=tmp_path,scopes={"fixture":path.parent})
    return catalog,path.name


def test_named_outputs_reorder_without_swapping_values_or_accepting_missing_names(graph):
    from copy import deepcopy
    catalog,name=graph
    model=OrtModel(catalog,name)
    try:
        contract=deepcopy(model.record)
        contract["graph"]["outputs"].reverse()
        value=np.arange(512,dtype=np.float32).reshape(1,512)
        result=run_checked(model.session,contract,{"input":value})
        assert list(result)==["negated","output"]
        np.testing.assert_array_equal(result["output"],value)
        np.testing.assert_array_equal(result["negated"],-value)
        model.record=deepcopy(contract)
        facade=SessionFacade(model)
        assert [node.name for node in facade.get_outputs()]==["negated","output"]
        positional=facade.run(None,{"input":value})
        np.testing.assert_array_equal(positional[0],-value)
        np.testing.assert_array_equal(positional[1],value)
        contract["graph"]["outputs"][0]["name"]="missing"
        with pytest.raises(DomainError) as error:
            run_checked(model.session,contract,{"input":value})
        assert error.value.code=="INFERENCE_OUTPUT_CONTRACT"
    finally:
        model.release()


def test_actual_ort_sessions_are_evicted_and_aliases_refuse_compilation(graph):
    catalog,name=graph
    model=OrtModel(catalog,name)
    value=np.ones((1,512),np.float32)
    np.testing.assert_equal(model.run({"input":value})["output"],value)
    diagnostics=evict_for_build()
    assert model.session is None and diagnostics["remaining_ort_sessions"] == 0
    assert diagnostics["torch_cuda_empty_cache"] in ("NOT_LOADED","CALLED","CUDA_UNAVAILABLE")
    model=OrtModel(catalog,name)
    alias=model.session
    with pytest.raises(DomainError) as failure:
        evict_for_build()
    assert failure.value.code == "BUILD_LIVE_ORT_SESSIONS"
    del alias
    assert evict_for_build()["remaining_ort_sessions"] == 0


def make_bank(catalog,name):
    bank=GoldenBank()
    reference=OrtModel(catalog,name)
    try:
        for offset in (0,1):
            inputs={"input":(np.arange(512,dtype=np.float32)+offset).reshape(1,512)/512}
            bank.record(reference.record,inputs,reference.run(inputs))
    finally:
        reference.release()
    return bank


def test_actual_ort_qualification_and_fixed_profile_enforcement(tmp_path,graph):
    catalog,name=graph
    bank=make_bank(catalog,name)
    factory=AcceleratorFactory(bank,catalog,root=tmp_path,preferred="CPUExecutionProvider")
    receipt=factory.qualify(name,iterations=3,warmups=1)
    assert receipt["provider"] == "CPUExecutionProvider"
    assert all(report["passed"] for report in receipt["parity"])
    assert receipt["node_providers"]["CPUExecutionProvider"] > 0
    model=OrtModel(catalog,name,factory.runtime_config())
    try:
        expected=bank.models[name].expected[0]["output"]
        np.testing.assert_array_equal(model.run(bank.models[name].cases[0])["output"],expected)
        diagnostic = factory.job_diagnostics().models[name]
        assert diagnostic.provider == "CPUExecutionProvider" and diagnostic.executions == 1
        assert diagnostic.node_providers["CPUExecutionProvider"] > 0
        with pytest.raises(DomainError) as failure:
            model.run({"input":np.ones((2,512),np.float32)})
        assert failure.value.code == "INFERENCE_PROFILE_MISMATCH"
    finally:model.release()
    assert not factory.job_diagnostics().models


def test_trt_unavailable_uses_real_cuda_or_cpu_and_records_reason(tmp_path,graph):
    catalog,name=graph
    factory=AcceleratorFactory(make_bank(catalog,name),catalog,root=tmp_path)
    availability=trt_availability()
    live_reference=OrtModel(catalog,name)
    try:
        receipt=factory.qualify(name,iterations=2,warmups=0)
        if availability.available:
            assert live_reference.session is None
    finally:
        live_reference.release()
    assert receipt["provider"] in ("TensorrtExecutionProvider","CUDAExecutionProvider","CPUExecutionProvider")
    if not availability.available:
        assert receipt["provider"] != "TensorrtExecutionProvider"
        assert any(h["provider"]=="TensorrtExecutionProvider" and "TRT_UNAVAILABLE" in h["reason"] for h in receipt["fallback_history"])
    assert receipt["selected_benchmark"]["p95_ms"] >= receipt["selected_benchmark"]["p50_ms"]


def test_excluded_sensitive_ops_refuse_full_trt_context_before_compilation(tmp_path,graph):
    catalog,name=graph
    factory=AcceleratorFactory(make_bank(catalog,name),catalog,root=tmp_path)
    record=catalog.records()[0]
    golden=factory.bank.models[name]
    with pytest.raises(DomainError) as error:
        factory.builder.build(record,policy_for("warp.onnx","liveportrait_warp"),"fp32",
                              golden.digest(),golden.cases,lambda _: pytest.fail("Excluded graph reached compilation"))
    assert error.value.code == "TRT_CONTEXT_UNSUPPORTED"
    assert factory.builder.last_diagnostics["state"] == "UNSUPPORTED"
    assert factory.builder.last_diagnostics["model_sha256"] == record["sha256"]
    assert not (tmp_path/"roop_3d/cache/engines").exists()
