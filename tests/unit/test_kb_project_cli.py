from __future__ import annotations

from apps.cli import kb


def test_project_import_cli_requires_explicit_files_and_routes_arguments(monkeypatch, tmp_path):
    seen = {}

    def fake(args):
        seen.update(vars(args))
        return 7

    monkeypatch.setattr(kb, "cmd_project_import", fake)
    rc = kb.main([
        "project-import", "--manifest", str(tmp_path / "projects.yaml"),
        "--project-id", "orders", "--file", "services/app.py", "--file", "pom.xml",
    ])
    assert rc == 7
    assert seen["files"] == ["services/app.py", "pom.xml"]
    assert seen["activate_current"] is False


def test_search_cli_exposes_project_boundary_arguments(monkeypatch):
    seen = {}

    def fake(args):
        seen.update(vars(args))
        return 0

    monkeypatch.setattr(kb, "cmd_search", fake)
    assert kb.main([
        "search", "reserve", "--project-id", "orders", "--module", "services", "--symbol", "reserve",
    ]) == 0
    assert (seen["project_id"], seen["module"], seen["symbol"]) == ("orders", "services", "reserve")
