"""把用户项目材料规范化为可索引的 Chunk。

本模块刻意不读取工作区目录：调用方必须显式传入已获授权的文本和路径，
避免“导入项目”变成不受控的全盘扫描。
"""

from __future__ import annotations

from pathlib import PurePosixPath
from dataclasses import dataclass
from urllib.parse import quote

from packages.schemas.chunk import Chunk, PROJECT_LICENSE, utc_now
from services.projects.registry import Project


@dataclass(frozen=True)
class Material:
    """调用方已获授权读取的一份项目材料；不接受绝对路径或隐式 glob。"""

    path: str
    text: str
    module: str | None = None
    symbol: str | None = None
    technology: str = "project"
    locale: str = "en"
    content_type: str = "mixed"


def material_chunk(*, project_id: str, version: str, module: str | None,
                   path: str, symbol: str | None, text: str,
                   cloud_generation_allowed: bool, technology: str = "project",
                   locale: str = "en", content_type: str = "mixed") -> Chunk:
    """创建一个项目材料块；路径和符号组成稳定、可回链的 project:// 标识。"""
    relative = PurePosixPath(path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("项目材料路径必须是相对路径且不得包含 '..'")
    fragment = f"#{quote(symbol, safe='._-')}" if symbol else ""
    source_url = f"project://{quote(project_id, safe='._-')}/{quote(relative.as_posix(), safe='/._-')}{fragment}"
    return Chunk(
        source_url=source_url, source_project=f"project:{project_id}",
        version_or_commit=version, license=PROJECT_LICENSE, retrieved_at=utc_now(),
        title_path=[project_id, *(filter(None, (module, symbol or relative.name)))],
        technology=technology, content_type=content_type, locale=locale, text=text,
        source_path=relative.as_posix(), project_id=project_id, module=module, symbol=symbol,
        cloud_generation_allowed=cloud_generation_allowed,
    )


def build_material_chunks(project: Project, materials: list[Material]) -> list[Chunk]:
    """将已显式选定的材料规范化；空文本拒绝，防止空块进入索引。"""
    chunks = []
    for material in materials:
        if not material.text.strip():
            raise ValueError(f"项目材料正文为空: {material.path}")
        chunk = material_chunk(
            project_id=project.id, version=project.version, module=material.module,
            path=material.path, symbol=material.symbol, text=material.text,
            cloud_generation_allowed=project.cloud_generation_allowed,
            technology=material.technology, locale=material.locale,
            content_type=material.content_type,
        )
        chunk.validate()
        chunks.append(chunk)
    return chunks
