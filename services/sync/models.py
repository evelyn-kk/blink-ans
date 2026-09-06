"""同步管线的报告数据结构。

拆成独立模块是为了让 `cards.py`（authored 来源）和 `pipeline.py`（拉取式来源）
都能引用 `SourceResult` 而不互相导入造成循环依赖——`pipeline.py` 按 `format`
分发到 `cards.py`，`cards.py` 又需要构造与拉取式来源同形状的 `SourceResult`。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SourceResult:
    source_id: str
    commit: str = ""
    files: int = 0
    chunks: int = 0
    rejected: int = 0
    reject_reasons: dict[str, int] = field(default_factory=dict)
    error: str | None = None


@dataclass
class SyncReport:
    sources: list[SourceResult] = field(default_factory=list)
    mode: str = "full"
    total_chunks: int = 0        # 本次同步实际写入的块（不含合并时搬运的）
    index_chunks: int = 0        # 暂存索引内的总块数
    carried_chunks: int = 0      # 合并更新时从当前索引搬运的块
    regression_passed: bool = False
    regression_failures: list[str] = field(default_factory=list)
    regression_skipped: list[str] = field(default_factory=list)
    activated: bool = False
    incomplete: bool = False     # 有来源未能同步，索引缺内容
    index_path: Path | None = None
    staging_path: Path | None = None
