"""中文分词 —— 关键词检索的前提。

I0 实测：FTS5 的默认 unicode61 分词器把整串中文视为单个 token，
「慢查询」「生产环境」的检索命中数均为 0。因此中文正文必须预分词后写入。

jieba 的默认词典也不够用——它会把「慢查询」切成「慢 / 查询」，
靠通用词才勉强命中。专有技术词必须由 knowledge/tech_terms.txt 显式登记。

**索引侧与查询侧必须使用同一版本词典**，否则召回会静默劣化而不报错，
因此这里提供 dictionary_version() 供索引写入时记录、查询时校验。
"""

from __future__ import annotations

import hashlib
import re
import threading
from pathlib import Path

TERMS_PATH = Path(__file__).resolve().parents[2] / "knowledge" / "tech_terms.txt"

_lock = threading.Lock()
_ready = False
_version: str | None = None


def _load() -> None:
    global _ready, _version
    with _lock:
        if _ready:
            return
        import jieba

        raw = TERMS_PATH.read_bytes()
        _version = hashlib.sha256(raw).hexdigest()[:12]

        for line in raw.decode("utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.rsplit(" ", 1)
            if len(parts) == 2 and parts[1].isdigit():
                jieba.add_word(parts[0], freq=int(parts[1]))
            else:
                jieba.add_word(line)
        _ready = True


def dictionary_version() -> str:
    """词典内容哈希。词典变更等同于索引结构变更，必须整体重建索引。"""
    _load()
    assert _version
    return _version


_PUNCT = re.compile(r"[^\w一-鿿]+")


def tokenize(text: str) -> list[str]:
    """切词并去掉标点。英文与数字原样保留，中文按词典切分。"""
    _load()
    import jieba

    out: list[str] = []
    for tok in jieba.cut_for_search(text):
        tok = tok.strip()
        if not tok or _PUNCT.fullmatch(tok):
            continue
        out.append(tok.lower())
    return out


def to_fts_document(text: str) -> str:
    """索引侧：写入 FTS5 的空格分隔文档。"""
    return " ".join(tokenize(text))


def to_fts_query(text: str) -> str:
    """查询侧：转成 FTS5 的 OR 查询。

    用 OR 而非 AND：语音转写会写错技术名词（I0 实测 Kafka→CAFCA），
    要求全部词命中会让一个错字毁掉整次检索。相关性由 bm25 与 RRF 融合决定。
    """
    toks = [t.replace('"', "") for t in tokenize(text)]
    toks = [t for t in toks if t]
    return " OR ".join(f'"{t}"' for t in toks) if toks else '""'
