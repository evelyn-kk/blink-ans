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

_REQUIRED = (
    "id", "project", "technology", "repo", "ref",
    "license", "license_file", "format", "locale", "base_url", "paths",
)
_FORMATS = {"markdown", "asciidoc", "html", "docbook"}


class RegistryError(ValueError):
    pass


@dataclass(frozen=True)
class Source:
    id: str
    project: str
    technology: str
    repo: str
    ref: str
    license: str
    license_file: str
    format: str
    locale: str
    base_url: str
    paths: tuple[str, ...]
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

    if not str(raw["repo"]).startswith("https://"):
        raise RegistryError(f"来源 {raw['id']!r} 的 repo 必须是 https 地址")

    ingest = bool(raw.get("ingest", True))
    if not ingest and not raw.get("ingest_blocked_reason"):
        raise RegistryError(
            f"来源 {raw['id']!r} 标记为不入库，必须写明 ingest_blocked_reason 以便审计"
        )

    known = set(_REQUIRED) | {"ingest", "ingest_blocked_reason", "sync_frequency"}
    return Source(
        id=raw["id"],
        project=raw["project"],
        technology=raw["technology"],
        repo=raw["repo"].rstrip("/"),
        ref=raw["ref"],
        license=raw["license"],
        license_file=raw["license_file"],
        format=raw["format"],
        locale=raw["locale"],
        base_url=raw["base_url"],
        paths=tuple(raw["paths"]),
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
