"""Helpers for real-weight tests gated by current governance evidence."""
from __future__ import annotations

import pytest

from roop_3d.core.errors import DomainError
from roop_3d.models.catalog import ModelCatalog


def require_governed_model(catalog: ModelCatalog, model_id: str) -> None:
    try:
        catalog.resolve_for_inference(model_id)
    except DomainError as error:
        pytest.skip(f"registered model is not inference-ready under Stage 12 governance: {error.code}")


def require_reference_models(catalog: ModelCatalog | None = None) -> ModelCatalog:
    from roop_3d.engine.runtime import record_named
    catalog = catalog or ModelCatalog()
    records = (catalog.registry.get(catalog.detector()["model_id"]),
               catalog.registry.get(catalog.recognizer()["model_id"]),
               catalog.registry.get(record_named(catalog, "inswapper_128.onnx")["model_id"]))
    for record in records:
        require_governed_model(catalog, record.model_id)
    return catalog


def require_model_named(catalog: ModelCatalog, name: str) -> None:
    record = next((item for item in catalog.registry.records() if item.name == name), None)
    if record is None:
        pytest.skip(f"registered model is unavailable: {name}")
    require_governed_model(catalog, record.model_id)
