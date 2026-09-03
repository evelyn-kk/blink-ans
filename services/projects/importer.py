"""把项目说明材料规范化为可索引的 Chunk。

本模块刻意不读取工作区目录：调用方必须显式传入已获授权的文本和路径。
CLI 只读取说明文档；少量代码片段应嵌在经人工整理的说明材料中，避免项目库
退化成源码或配置的副本。
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from dataclasses import dataclass
from urllib.parse import quote

from packages.schemas.chunk import Chunk, PROJECT_LICENSE, estimate_tokens, utc_now
from services.projects.registry import Project
from services.sync.chunk import MAX_TOKENS, _merge_small, _split_body


@dataclass(frozen=True)
class Material:
    """调用方已获授权的一份项目说明材料；不接受绝对路径或隐式 glob。"""

    path: str
    text: str
    module: str | None = None
    symbol: str | None = None
    technology: str = "project"
    locale: str = "en"
    content_type: str = "mixed"


def _split_oversized_diagram(piece: str) -> list[str]:
    """把未标注语言的 ASCII 图按行拆为可独立引用的小图块。

    Markdown 的裸 ````` 围栏常用于流程图/数据流图而非代码。它们若超过上下文
    预算，可以在行边界拆开并重新围栏，文本不丢失。`````java`` 等真实代码块
    仍绝不自动拆断，由调用方显式选择可独立解释的片段。
    """
    lines = piece.splitlines()
    if (estimate_tokens(piece) <= MAX_TOKENS or len(lines) < 2
            or lines[0].strip() != "```" or lines[-1].strip() != "```"):
        return [piece]

    out: list[str] = []
    buf: list[str] = []
    # 留出围栏和换行的余量；图里的单行超过预算时仍应由人工整理，不能截断。
    budget = MAX_TOKENS - 4
    for line in lines[1:-1]:
        if estimate_tokens(line) > budget:
            raise ValueError("项目说明材料的单行图示超过上下文预算，请手动拆分")
        candidate = "\n".join([*buf, line])
        if buf and estimate_tokens(candidate) > budget:
            out.append("```\n" + "\n".join(buf) + "\n```")
            buf = [line]
        else:
            buf.append(line)
    if buf:
        out.append("```\n" + "\n".join(buf) + "\n```")
    return out


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
    """将已显式选定的材料规范化并按上下文预算切块。

    不从源码猜测函数/类名：猜错的 symbol 会成为错误检索过滤条件。调用方若已
    有可靠的符号清单可以显式提供；否则保留到文件粒度。
    """
    chunks = []
    for material in materials:
        if not material.text.strip():
            raise ValueError(f"项目材料正文为空: {material.path}")
        # 与官方材料相同的 400-token 上限，避免一份项目文件独占回答上下文。
        # `_merge_small` 可能把一个短尾巴并进已接近上限的前块。通用语料允许
        # 少量超额以保留章节完整性，项目问答的证据预算更紧，故在此重新展开。
        merged = _merge_small(_split_body(material.text.strip()))
        pieces = [
            subpiece
            for piece in merged
            for subpiece in (_split_body(piece) if estimate_tokens(piece) > MAX_TOKENS else [piece])
        ]
        for piece in pieces:
            for prepared in _split_oversized_diagram(piece):
                chunk = material_chunk(
                    project_id=project.id, version=project.version, module=material.module,
                    path=material.path, symbol=material.symbol, text=prepared,
                    cloud_generation_allowed=project.cloud_generation_allowed,
                    technology=material.technology, locale=material.locale,
                    content_type=material.content_type,
                )
                # 官方文档为保持可运行的完整代码围栏，会保留超长代码块；项目库的
                # 代码只是人工挑选的片段，不能让它绕过回答上下文的硬预算。拒绝并
                # 要求在说明材料中按独立示例拆开，比自动截断一段不可理解的代码安全。
                if chunk.token_estimate > MAX_TOKENS:
                    raise ValueError(
                        f"项目说明材料的单个片段超过 {MAX_TOKENS} token，请拆分后再导入: "
                        f"{material.path}（{chunk.token_estimate} token）"
                    )
                chunk.validate()
                chunks.append(chunk)
    return chunks


_DOCUMENT_SUFFIXES = {".md", ".markdown", ".adoc", ".rst", ".txt"}


def read_materials(project: Project, paths: list[str]) -> list[Material]:
    """读取逐个点名的项目说明文档，不递归扫描项目根目录。

    ``paths`` 只能是相对于注册根的文件；解析后再次检查真实路径仍在根内，
    所以 ``../`` 和指向根外的软链接都不能越界读取。源码和配置扩展名不在
    白名单中：必要代码须作为少量片段放进说明文档，而不是整文件入库。
    """
    if not paths:
        raise ValueError("至少需要一个 --file；项目导入不会自动扫描目录")
    root = project.root.resolve()
    materials: list[Material] = []
    seen: set[str] = set()
    for raw in paths:
        relative = PurePosixPath(raw)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"项目材料路径必须是相对路径且不得包含 '..': {raw}")
        normalized = relative.as_posix()
        if normalized in seen:
            raise ValueError(f"项目材料重复指定: {normalized}")
        seen.add(normalized)
        target = (root / Path(normalized)).resolve()
        if not target.is_relative_to(root):
            raise ValueError(f"项目材料不得通过链接离开注册根: {normalized}")
        if not target.is_file():
            raise ValueError(f"项目材料不存在或不是普通文件: {normalized}")
        suffix = target.suffix.lower()
        if suffix not in _DOCUMENT_SUFFIXES:
            raise ValueError(
                f"项目库只接纳说明文档/挑选片段，不接纳源码或配置文件: {normalized}"
            )
        text = target.read_text(encoding="utf-8")
        content_type = "mixed" if "```" in text else "prose"
        module = relative.parts[0] if len(relative.parts) > 1 else None
        materials.append(Material(
            path=normalized, text=text, module=module,
            content_type=content_type,
        ))
    return materials
