"""Stage 13: ``GET /api/v1/rendering/studio-environments`` serves the five CC0 lighting probes in a browser-ready form."""
from __future__ import annotations

import base64

import numpy as np
import pytest
from fastapi.testclient import TestClient

from roop_3d.core.settings import Settings
from roop_3d.rendering.physical_buffers import evaluate_sh_basis
from roop_3d.rendering.physical_camera import ENVIRONMENT_PROBES, load_cc0_preview_assets
from roop_3d.server import studio_environment_routes as routes
from roop_3d.server.app import create_app


@pytest.fixture()
def client(tmp_path):
    routes._catalog.cache_clear()
    with TestClient(create_app(Settings(), repo_root=tmp_path, worker_enabled=False), base_url="http://127.0.0.1:8030") as test_client:
        yield test_client
    routes._catalog.cache_clear()


def test_lists_the_five_cc0_probes_with_licence_and_source(client):
    body = client.get("/api/v1/rendering/studio-environments").json()
    assert body["count"] == 5 and len(body["environments"]) == 5
    assert [e["id"] for e in body["environments"]] == [p["asset_id"] for p in ENVIRONMENT_PROBES]
    for environment in body["environments"]:
        assert environment["license"] == "CC0-1.0"
        assert environment["page_url"].startswith("https://polyhaven.com/a/")
        assert 2000 < environment["kelvin"] < 12000
        assert len(environment["sh_coefficients"]) == 9 and all(len(row) == 3 for row in environment["sh_coefficients"])


def test_coefficients_are_rgb_not_bgr(client):
    """The bundle stores BGR; the route must hand the browser RGB, or every environment would look blue/red swapped."""
    stored = load_cc0_preview_assets().probes[3]                                  # Golden Hour: measured frontal R 1.55 vs B 0.28
    served = client.get("/api/v1/rendering/studio-environments").json()["environments"][3]
    assert served["id"] == "signal_hill_sunrise"
    np.testing.assert_allclose(np.asarray(served["sh_coefficients"]), np.asarray(stored.sh_coefficients)[:, ::-1], atol=1e-5)
    frontal = evaluate_sh_basis(np.array([[0.0, 0.0, 1.0]]))[0] @ np.asarray(served["sh_coefficients"])
    assert frontal[0] > 3 * frontal[2], "a sunrise must be far redder than blue in RGB order"


def test_thumbnail_is_a_png_data_url_of_the_expected_size(client):
    import cv2
    for environment in client.get("/api/v1/rendering/studio-environments").json()["environments"]:
        prefix, data = environment["thumbnail"].split(",", 1)
        assert prefix == "data:image/png;base64"
        image = cv2.imdecode(np.frombuffer(base64.b64decode(data), np.uint8), cv2.IMREAD_COLOR)
        assert image.shape == (64, 128, 3)


def test_missing_bundle_is_a_503_with_a_code_not_a_500(client, monkeypatch):
    import roop_3d.rendering.physical_camera as camera

    def missing(*_args, **_kwargs):
        raise FileNotFoundError("CC0 preview bundle not found: nowhere.npz")
    monkeypatch.setattr(camera, "load_cc0_preview_assets", missing)
    routes._catalog.cache_clear()
    response = client.get("/api/v1/rendering/studio-environments")
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "STUDIO_ENVIRONMENTS_UNAVAILABLE"
