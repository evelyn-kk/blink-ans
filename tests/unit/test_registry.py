"""来源注册表与许可校验。

许可证是这里最要紧的字段：注册表的声明不被信任，
同步时必须从仓库内的许可文件实测校验（上游换许可是真实发生过的事）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from services.sync.fetch import LicenseError, verify_license  # noqa: E402
from services.sync.registry import (  # noqa: E402
    REGISTRY_PATH, RegistryError, Source, get, ingestible, load_registry,
)

APACHE = "Apache License\nVersion 2.0, January 2004\nhttp://www.apache.org/licenses/"
CC_NC = "Attribution-NonCommercial-ShareAlike 4.0 International Public License"


def src(**over) -> Source:
    base = dict(
        id="x", project="p", technology="t", repo="https://github.com/a/b", ref="main",
        license="Apache-2.0", license_file="LICENSE", format="markdown",
        locale="en", base_url="https://example.com/", paths=("docs",),
    )
    base.update(over)
    return Source(**base)


# ---------- 注册表 ----------

def test_real_registry_loads():
    srcs = load_registry()
    assert len(srcs) >= 5
    assert len(ingestible(srcs)) >= 4


def test_restricted_source_is_marked_link_only():
    """Redis 官方文档为 CC-BY-NC-SA-4.0，必须标记为不入库。"""
    s = get(load_registry(), "redis-official")
    assert s.ingest is False
    assert s.ingest_blocked_reason and "NC" in s.ingest_blocked_reason.upper()


def test_blocked_source_must_state_reason(tmp_path):
    """不入库的来源必须写明原因，否则无法审计。"""
    doc = {"version": 1, "sources": [{
        "id": "x", "project": "p", "technology": "t",
        "repo": "https://github.com/a/b", "ref": "main", "license": "MIT",
        "license_file": "LICENSE", "format": "markdown", "locale": "en",
        "base_url": "https://e.com/", "paths": ["d"], "ingest": False,
    }]}
    p = tmp_path / "s.yaml"
    p.write_text(yaml.safe_dump(doc), encoding="utf-8")
    with pytest.raises(RegistryError, match="ingest_blocked_reason"):
        load_registry(p)


def test_duplicate_ids_rejected(tmp_path):
    one = {"id": "dup", "project": "p", "technology": "t", "repo": "https://github.com/a/b",
           "ref": "main", "license": "MIT", "license_file": "LICENSE", "format": "markdown",
           "locale": "en", "base_url": "https://e.com/", "paths": ["d"]}
    p = tmp_path / "s.yaml"
    p.write_text(yaml.safe_dump({"version": 1, "sources": [one, dict(one)]}), encoding="utf-8")
    with pytest.raises(RegistryError, match="重复"):
        load_registry(p)


def test_unknown_format_rejected(tmp_path):
    doc = {"version": 1, "sources": [{
        "id": "x", "project": "p", "technology": "t", "repo": "https://github.com/a/b",
        "ref": "main", "license": "MIT", "license_file": "LICENSE", "format": "pdf",
        "locale": "en", "base_url": "https://e.com/", "paths": ["d"]}]}
    p = tmp_path / "s.yaml"
    p.write_text(yaml.safe_dump(doc), encoding="utf-8")
    with pytest.raises(RegistryError, match="format"):
        load_registry(p)


def test_unknown_markdown_anchor_style_rejected(tmp_path):
    doc = {"version": 1, "sources": [{
        "id": "x", "project": "p", "technology": "t", "repo": "https://github.com/a/b",
        "ref": "main", "license": "MIT", "license_file": "LICENSE", "format": "markdown",
        "locale": "en", "base_url": "https://e.com/", "paths": ["d"],
        "markdown_anchor_style": "imagined-generator",
    }]}
    p = tmp_path / "s.yaml"
    p.write_text(yaml.safe_dump(doc), encoding="utf-8")
    with pytest.raises(RegistryError, match="markdown_anchor_style"):
        load_registry(p)


# ---------- 许可实测校验 ----------

def test_matching_license_passes(tmp_path):
    (tmp_path / "LICENSE").write_text(APACHE, encoding="utf-8")
    assert verify_license(src(), tmp_path) == "Apache-2.0"


def test_declared_license_mismatch_rejected(tmp_path):
    """上游换了许可而注册表未同步时必须失败，不能静默入库。"""
    (tmp_path / "LICENSE").write_text(CC_NC, encoding="utf-8")
    with pytest.raises(LicenseError, match="实测为"):
        verify_license(src(license="Apache-2.0"), tmp_path)


def test_noncommercial_blocked_even_when_declared(tmp_path):
    """即便注册表如实声明了 NC 许可，只要标记为入库就必须拒绝。"""
    (tmp_path / "LICENSE").write_text(CC_NC, encoding="utf-8")
    with pytest.raises(LicenseError, match="受限"):
        verify_license(src(license="CC-BY-NC-SA-4.0", ingest=True), tmp_path)


def test_noncommercial_allowed_when_link_only(tmp_path):
    """标记为仅链接检索时，NC 许可可以通过校验——它本来就不入库。"""
    (tmp_path / "LICENSE").write_text(CC_NC, encoding="utf-8")
    s = src(license="CC-BY-NC-SA-4.0", ingest=False, ingest_blocked_reason="NC")
    assert verify_license(s, tmp_path) == "CC-BY-NC-SA-4.0"


def test_missing_license_file_rejected(tmp_path):
    with pytest.raises(LicenseError, match="不存在"):
        verify_license(src(), tmp_path)


def test_unrecognized_license_rejected(tmp_path):
    (tmp_path / "LICENSE").write_text("本软件归本人所有，禁止一切使用。", encoding="utf-8")
    with pytest.raises(LicenseError, match="无法.*识别"):
        verify_license(src(), tmp_path)
