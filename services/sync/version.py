"""索引内容转换实现的版本指纹。

索引里的正文不是原始语料：它已经过来源解析、切块，场景卡片和项目材料也
各有自己的文本转换。增量合并若只搬运旧块，就会把旧实现产生的派生结果带进
新索引。因此这里取这些实现文件的内容指纹，而不是依赖容易遗漏的手工版本号。
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[2]
_TRANSFORM_FILES = (
    "services/sync/parse.py",
    "services/sync/chunk.py",
    "services/sync/cards.py",
    "services/projects/importer.py",
)


def content_transform_version() -> str:
    """返回会影响入库文本形状的解析/切块实现版本。

    文件名也进入哈希，避免不同文件边界恰好拼出相同字节串。截短只用于存储和
    提示；它不是安全用途。任何实现（含注释）改动都会要求一次全量重建，宁可
    多重解析，也不能静默继续搬运可能陈旧的块。
    """
    digest = sha256()
    for relative in _TRANSFORM_FILES:
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update((_ROOT / relative).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()[:12]
