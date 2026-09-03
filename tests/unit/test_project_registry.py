from __future__ import annotations

import pytest

from services.projects.registry import ProjectRegistryError, load_projects


def write_manifest(tmp_path, body: str):
    path = tmp_path / "projects.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_registry_loads_explicit_root_version_and_cloud_boundary(tmp_path):
    root = tmp_path / "orders"
    root.mkdir()
    manifest = write_manifest(tmp_path, f"""projects:
  - id: orders-prod
    version: v1.2.3
    root: {root}
    cloud_generation_allowed: false
""")
    projects = load_projects(manifest)
    assert projects[0].id == "orders-prod"
    assert projects[0].root == root and projects[0].cloud_generation_allowed is False


@pytest.mark.parametrize("body, error", [
    ("projects:\n  - id: p\n    version: v1\n    root: /missing\n", "cloud_generation_allowed"),
    ("projects:\n  - id: ../p\n    version: v1\n    root: /tmp\n    cloud_generation_allowed: false\n", "非法项目"),
    ("projects: {}\n", "必须是列表"),
])
def test_registry_rejects_ambiguous_or_invalid_boundaries(tmp_path, body, error):
    with pytest.raises(ProjectRegistryError, match=error):
        load_projects(write_manifest(tmp_path, body))
