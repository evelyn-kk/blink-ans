"""从已验证的项目块构建确定性的摘要卡片和符号别名，不调用模型。"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from packages.schemas.chunk import Chunk


@dataclass(frozen=True)
class ProjectAssets:
    summary: str
    aliases: dict[str, tuple[str, ...]]


def build_assets(chunks: list[Chunk]) -> ProjectAssets:
    """按现有元数据汇总，宁缺勿猜；别名只指向已存在的 project:// 证据。"""
    if not chunks or not all(c.project_id for c in chunks):
        raise ValueError("摘要资产只能由同一项目的非空项目块构建")
    project_ids = {c.project_id for c in chunks}
    if len(project_ids) != 1:
        raise ValueError("摘要资产不得混合多个项目")
    modules = sorted({c.module for c in chunks if c.module})
    paths = sorted({c.source_path for c in chunks if c.source_path})
    aliases: defaultdict[str, set[str]] = defaultdict(set)
    for c in chunks:
        if c.symbol:
            aliases[c.symbol].add(c.source_url)
        if c.source_path:
            aliases[c.source_path.rsplit("/", 1)[-1]].add(c.source_url)
    project_id = next(iter(project_ids))
    summary = (
        f"项目 {project_id}：{len(chunks)} 个证据块；"
        f"模块：{', '.join(modules) if modules else '未标注'}；"
        f"文件：{', '.join(paths) if paths else '未标注'}。"
    )
    return ProjectAssets(summary, {k: tuple(sorted(v)) for k, v in sorted(aliases.items())})
