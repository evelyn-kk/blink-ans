"""元数据契约的硬门测试。

这些断言存在的理由：回答必须能回链到 source_url 并显示版本与抓取日期，
任何一个必填字段缺失都会让引用失去可追溯性，因此宁可拒绝入库也不能放行。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from packages.schemas.chunk import Chunk, MetadataError, estimate_tokens, utc_now  # noqa: E402


def make(**over) -> Chunk:
    base = dict(
        source_url="https://docs.spring.io/spring-boot/reference/web.html#servlet",
        source_project="spring-boot", version_or_commit="v3.4.1",
        license="Apache-2.0", retrieved_at=utc_now(),
        title_path=["Spring Boot", "Web", "Servlet"], technology="spring",
        content_type="prose", locale="en", text="嵌入式容器的默认端口是 8080。",
    )
    base.update(over)
    return Chunk(**base)


def test_valid_chunk_passes():
    make().validate()


@pytest.mark.parametrize(
    "field", ["source_url", "source_project", "version_or_commit",
              "license", "technology", "content_type", "locale", "text"],
)
def test_missing_required_field_rejected(field):
    with pytest.raises(MetadataError) as e:
        make(**{field: ""}).validate()
    assert e.value.field_name == field


def test_non_url_source_rejected():
    with pytest.raises(MetadataError, match="source_url"):
        make(source_url="docs/web.adoc").validate()


def test_restricted_license_rejected():
    """许可受限的资料只做实时链接检索，不得入核心语料（architecture.md 5.1）。"""
    with pytest.raises(MetadataError, match="不在允许清单"):
        make(license="CC-BY-NC-SA-4.0").validate()


def test_empty_title_path_rejected():
    with pytest.raises(MetadataError, match="title_path"):
        make(title_path=[]).validate()


def test_bad_timestamp_rejected():
    with pytest.raises(MetadataError, match="retrieved_at"):
        make(retrieved_at="2026年8月31日").validate()


def test_checksum_detects_tampering():
    """正文在算完校验和后被改写时必须被发现，否则增量同步会漏掉变更。"""
    c = make()
    c.text = "改写后的正文"
    with pytest.raises(MetadataError, match="checksum"):
        c.validate()


def test_checksum_and_tokens_autofilled():
    c = make()
    assert len(c.checksum) == 64
    assert c.token_estimate > 0


def test_citation_shows_version_and_date():
    c = make(version_or_commit="v3.4.1", retrieved_at="2026-08-31T12:00:00+00:00")
    assert "v3.4.1" in c.citation and "2026-08-31" in c.citation


def test_token_estimate_handles_cjk_and_ascii():
    """中文约 1 字 1 token，英文约 4 字符 1 token；估算用于控制上下文预算。"""
    assert estimate_tokens("慢查询") == 3
    assert estimate_tokens("a" * 40) == 10
