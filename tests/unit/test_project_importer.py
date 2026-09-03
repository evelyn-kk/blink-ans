from __future__ import annotations

import pytest

from packages.schemas.chunk import MetadataError
from services.projects.importer import material_chunk
from services.projects.importer import Material, build_material_chunks, read_materials
from services.projects.registry import Project


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


def test_batch_inherits_the_registered_project_cloud_boundary(tmp_path):
    root = tmp_path / "orders"
    root.mkdir()
    project = Project("orders-prod", "v1", root, False)
    chunks = build_material_chunks(project, [
        Material("services/orders/app.py", "class CheckoutService: pass", module="orders", symbol="CheckoutService"),
    ])
    assert chunks[0].project_id == "orders-prod"
    assert chunks[0].cloud_generation_allowed is False


def test_batch_rejects_empty_material_before_indexing(tmp_path):
    root = tmp_path / "orders"
    root.mkdir()
    with pytest.raises(ValueError, match="正文为空"):
        build_material_chunks(Project("orders", "v1", root, True), [Material("a.py", "")])


def test_explicit_file_read_stays_inside_registered_root_and_does_not_scan(tmp_path):
    root = tmp_path / "orders"
    (root / "services").mkdir(parents=True)
    (root / "services" / "overview.md").write_text("# App\n\n```java\nclass App {}\n```", encoding="utf-8")
    (root / "ignored.md").write_text("not selected", encoding="utf-8")
    project = Project("orders", "v1", root, False)

    materials = read_materials(project, ["services/overview.md"])

    assert [(m.path, m.module, m.content_type) for m in materials] == [
        ("services/overview.md", "services", "mixed")
    ]
    with pytest.raises(ValueError, match="至少需要一个"):
        read_materials(project, [])
    with pytest.raises(ValueError, match="相对路径"):
        read_materials(project, ["../ignored.py"])


def test_project_cli_reader_refuses_full_source_and_configuration_files(tmp_path):
    root = tmp_path / "orders"
    root.mkdir()
    (root / "App.java").write_text("class App {}", encoding="utf-8")
    (root / "application.yml").write_text("password: do-not-index", encoding="utf-8")
    project = Project("orders", "v1", root, False)
    for path in ("App.java", "application.yml"):
        with pytest.raises(ValueError, match="不接纳源码或配置"):
            read_materials(project, [path])


def test_large_project_material_is_split_for_context_budget(tmp_path):
    root = tmp_path / "orders"
    root.mkdir()
    project = Project("orders", "v1", root, False)
    chunks = build_material_chunks(project, [Material("a.txt", "word " * 2_500)])
    assert len(chunks) > 1
    assert all(c.token_estimate <= 400 for c in chunks)
