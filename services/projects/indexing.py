"""用户项目的增量索引构建。

项目材料绝不走同步器的网络抓取路径；调用方先由注册表确定根目录，再将逐个
点名的文件交给 ``read_materials``。这里仅处理已获授权的文本和本地向量化。
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from services.projects.importer import Material, build_material_chunks
from services.projects.registry import Project
from services.retrieval.embed import DEFAULT_MODEL
from services.retrieval.store import CURRENT, IndexBuilder, IndexStats

EMBED_BATCH_SIZE = 16


class _Embedder(Protocol):
    model_id: str

    def encode(self, texts: list[str]) -> list[list[float]]: ...


def rebuild_project_index(
    project: Project,
    materials: list[Material],
    embedder: _Embedder,
    *,
    source_index: Path = CURRENT,
    name: str = "current",
    activate: bool = True,
) -> IndexStats:
    """用新项目材料替换同一项目的旧块，并搬运其余索引内容。

    这是一种真正的增量构建：仅本项目重新嵌入，其他来源的既有向量按模型名
    校验后原样搬运。默认目标仍由调用方决定；CLI 默认发布为独立项目索引，
    不会在没有项目级回归集时静默替换全局 ``current``。
    """
    chunks = build_material_chunks(project, materials)
    if not chunks:
        raise ValueError("项目材料未产生可索引块")
    # 项目说明文档可能一次带来数百块。嵌入模型的批处理不会带来等比例吞吐收益，
    # 却会显著抬高统一内存峰值；小批顺序处理使首次导入在 M4 上可预测地完成。
    vectors: list[list[float]] = []
    for start in range(0, len(chunks), EMBED_BATCH_SIZE):
        vectors.extend(embedder.encode([c.text for c in chunks[start:start + EMBED_BATCH_SIZE]]))
    if len(vectors) != len(chunks):
        raise ValueError(f"嵌入数 {len(vectors)} 与项目块数 {len(chunks)} 不一致")

    builder = IndexBuilder(name)
    try:
        builder.carry_over(source_index, {f"project:{project.id}"}, embedder.model_id)
        builder.add(chunks, vectors)
        sources = builder.existing_versions(source_index)
        sources[f"project:{project.id}"] = project.version
        stats = builder.finalize(sources, embedder.model_id or DEFAULT_MODEL)
        if activate:
            builder.activate()
        return stats
    except Exception:
        builder.discard()
        raise
