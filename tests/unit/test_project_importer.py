from __future__ import annotations

import pytest

from packages.schemas.chunk import MetadataError
from services.projects.importer import material_chunk


def test_project_material_is_traceable_and_carries_the_cloud_boundary():
    chunk = material_chunk(
        project_id="orders-prod", version="v1", module="orders",
        path="services/orders/checkout.py", symbol="reserve_stock", text="def reserve_stock(): pass",
        cloud_generation_allowed=False,
    )
    chunk.validate()
    assert chunk.source_url == "project://orders-prod/services/orders/checkout.py#reserve_stock"
    assert chunk.license == "Proprietary" and chunk.cloud_generation_allowed is False


def test_project_url_without_project_id_is_rejected():
    chunk = material_chunk(
        project_id="orders-prod", version="v1", module=None, path="a.py", symbol=None,
        text="x", cloud_generation_allowed=True,
    )
    chunk.project_id = None
    with pytest.raises(MetadataError, match="project_id"):
        chunk.validate()


def test_project_import_refuses_paths_that_escape_the_registered_root():
    with pytest.raises(ValueError, match="相对路径"):
        material_chunk(
            project_id="orders-prod", version="v1", module=None, path="../.env", symbol=None,
            text="x", cloud_generation_allowed=False,
        )
