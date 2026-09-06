"""来源注册表的加载与校验。

许可证是这里最重要的字段：`architecture.md` 5.1 要求许可受限的资料只做实时链接检索，
不进核心语料。因此注册表里的 license 声明**不被信任**——同步时必须从仓库内的许可文件
实测校验（见 fetch.verify_license），声明与实测不符即失败。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REGISTRY_PATH = Path(__file__).resolve().parents[2] / "knowledge" / "sources.yaml"

# 通用必填字段，两类来源都要有。
_REQUIRED = ("id", "project", "technology", "format", "locale", "paths")
# 只对"从远程仓库拉取"的来源才适用；authored 来源（人工撰写的场景卡片）
# 没有上游仓库、没有独立许可，这些字段对它没有意义。
_FETCHED_REQUIRED = ("repo", "ref", "license", "license_file", "base_url")
_FORMATS = {"markdown", "asciidoc", "html", "docbook", "authored"}
_MARKDOWN_ANCHOR_STYLES = {"generic", "kafka", "kubernetes"}


class RegistryError(ValueError):
    pass


@dataclass(frozen=True)
class Source:
    id: str
    project: str
    technology: str
    format: str
    locale: str
    paths: tuple[str, ...]
    # 仅"从远程仓库拉取"的来源会填这些；authored 来源留 None。
    repo: str | None = None
    ref: str | None = None
    license: str | None = None
    license_file: str | None = None
    base_url: str | None = None
    url_template: str | None = None   # 可用 {path} 与 {anchor} 占位
    url_strip_prefix: str | None = None
    # Hugo/Docsy 并没有统一的标题 id 规则；这里明确记录已实测的发布器规则。
    markdown_anchor_style: str = "generic"
    # 少数上游文件名含发布路径不保留的字符，例如 Kafka 的括号。
    url_path_drop_chars: str = ""
    ingest: bool = True
    ingest_blocked_reason: str | None = None
    sync_frequency: str = "monthly"
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def slug(self) -> str:
        """本地缓存目录名。"""
        return self.id


def _one(raw: dict[str, Any]) -> Source:
    missing = [k for k in _REQUIRED if not raw.get(k)]
    if missing:
        raise RegistryError(f"来源 {raw.get('id', '?')!r} 缺少必填字段: {', '.join(missing)}")

    if raw["format"] not in _FORMATS:
        raise RegistryError(f"来源 {raw['id']!r} 的 format {raw['format']!r} 不受支持")

    is_authored = raw["format"] == "authored"

    if is_authored:
        # authored 来源（人工撰写的场景卡片）没有上游仓库、没有独立许可——
        # 每个小节的许可/版本在构建时从它引用的真实来源继承（见 services/sync/cards.py）。
        # 写了这些字段就是伪造"这是从某仓库拉取、有独立许可"的元数据，必须拒绝而非静默忽略。
        present = [k for k in _FETCHED_REQUIRED if raw.get(k)]
        if present:
            raise RegistryError(
                f"来源 {raw['id']!r} 是 authored 格式，不适用 {', '.join(present)}；"
                f"这些字段只对拉取式来源有意义"
            )
    else:
        missing_fetched = [k for k in _FETCHED_REQUIRED if not raw.get(k)]
        if missing_fetched:
            raise RegistryError(
                f"来源 {raw.get('id', '?')!r} 缺少必填字段: {', '.join(missing_fetched)}"
            )
        if not str(raw["repo"]).startswith("https://"):
            raise RegistryError(f"来源 {raw['id']!r} 的 repo 必须是 https 地址")

    anchor_style = raw.get("markdown_anchor_style", "generic")
    if anchor_style not in _MARKDOWN_ANCHOR_STYLES:
        raise RegistryError(
            f"来源 {raw['id']!r} 的 markdown_anchor_style {anchor_style!r} 不受支持"
        )
    if anchor_style != "generic" and raw["format"] != "markdown":
        raise RegistryError(f"来源 {raw['id']!r} 的 markdown_anchor_style 只适用于 markdown")

    ingest = bool(raw.get("ingest", True))
    if not ingest and not raw.get("ingest_blocked_reason"):
        raise RegistryError(
            f"来源 {raw['id']!r} 标记为不入库，必须写明 ingest_blocked_reason 以便审计"
        )

    known = set(_REQUIRED) | set(_FETCHED_REQUIRED) | {
        "ingest", "ingest_blocked_reason", "sync_frequency",
        "url_template", "url_strip_prefix", "markdown_anchor_style", "url_path_drop_chars",
    }
    return Source(
        id=raw["id"],
        project=raw["project"],
        technology=raw["technology"],
        format=raw["format"],
        locale=raw["locale"],
        paths=tuple(raw["paths"]),
        repo=(raw["repo"].rstrip("/") if not is_authored else None),
        ref=raw.get("ref") if not is_authored else None,
        license=raw.get("license") if not is_authored else None,
        license_file=raw.get("license_file") if not is_authored else None,
        base_url=raw.get("base_url") if not is_authored else None,
        url_template=raw.get("url_template"),
        url_strip_prefix=raw.get("url_strip_prefix"),
        markdown_anchor_style=anchor_style,
        url_path_drop_chars=raw.get("url_path_drop_chars", ""),
        ingest=ingest,
        ingest_blocked_reason=raw.get("ingest_blocked_reason"),
        sync_frequency=raw.get("sync_frequency", "monthly"),
        extra={k: v for k, v in raw.items() if k not in known},
    )


def load_registry(path: Path | None = None) -> list[Source]:
    path = path or REGISTRY_PATH
    if not path.exists():
        raise RegistryError(f"注册表不存在: {path}")

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw_sources = data.get("sources") or []
    if not raw_sources:
        raise RegistryError("注册表中没有任何来源")

    sources = [_one(r) for r in raw_sources]

    ids = [s.id for s in sources]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise RegistryError(f"来源 id 重复: {', '.join(sorted(dupes))}")

    return sources


def ingestible(sources: list[Source]) -> list[Source]:
    return [s for s in sources if s.ingest]


def get(sources: list[Source], source_id: str) -> Source:
    for s in sources:
        if s.id == source_id:
            return s
    raise RegistryError(f"未登记的来源: {source_id!r}")
