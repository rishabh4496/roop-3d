import pytest

from roop_3d.core.errors import DomainError
from roop_3d.jobs.schema import JobRequest, JobState
from roop_3d.jobs.store import JobStore
from roop_3d.models.diagnostics import JobDiagnostics, ModelDiagnostic, ProviderFailure


def test_active_job_preserves_exact_provider_fallback_through_restart(tmp_path):
    store = JobStore(tmp_path)
    request = JobRequest(target_path="unit-video", inspection_id="a" * 32,
                         face_mappings=({"track_id": "track_0000", "faceset_id": "unit-source"},))
    job, _ = store.create(request, "stage07-diagnostics")
    value = JobDiagnostics(models={"storage-unit": ModelDiagnostic(sha256="b" * 64,
        golden_digest="c" * 64, provider="CPUExecutionProvider", requested_provider="TensorrtExecutionProvider",
        precision="fp32", executions=3, node_providers={"CPUExecutionProvider": 12},
        fallback_history=(ProviderFailure(provider="TensorrtExecutionProvider", precision="fp32",
                                          reason="TRT_UNAVAILABLE: storage-unit reason"),))})
    with pytest.raises(DomainError) as error:
        store.record_diagnostics(job.job_id, value)
    assert error.value.code == "JOB_NOT_ACTIVE"
    store.transition(job.job_id, JobState.preparing)
    persisted = store.record_diagnostics(job.job_id, value)
    assert persisted.revision == 2
    restarted = JobStore(tmp_path).get(job.job_id)
    assert restarted.diagnostics == value
    assert restarted.model_dump(mode="json")["diagnostics"]["models"]["storage-unit"]["provider"] == "CPUExecutionProvider"


def test_capabilities_and_diagnostics_report_expose_hardware_profiles(tmp_path):
    from fastapi.testclient import TestClient
    from roop_3d.server.app import create_app
    from roop_3d.core.settings import Settings

    settings = Settings(hardware_profile="6gb", preferred_provider="CPUExecutionProvider")
    app = create_app(settings, repo_root=tmp_path)
    with TestClient(app, base_url="http://127.0.0.1:8030") as client:
        cap_resp = client.get("/api/v1/system/capabilities")
        assert cap_resp.status_code == 200
        cap = cap_resp.json()
        assert cap["workstation"]["active_hardware_profile"] == "6gb"
        assert cap["workstation"]["resolved_hardware_profile"] in ("cpu", "6gb", "12gb")

        diag_resp = client.get("/api/v1/system/diagnostics_report")
        assert diag_resp.status_code == 200
        diag = diag_resp.json()
        s2 = next(s for s in diag["sections"] if s["section"] == 2)
        assert s2["configured_hardware_profile"] == "6gb"
        assert s2["resolved_hardware_profile"] in ("cpu", "6gb", "12gb")
