"""索引存储 —— SQLite + sqlite-vec + FTS5。

两个关键设计：

1. **双缓冲**：新索引建在临时文件里，回归检索通过后才原子替换当前索引
   （development.md 第 4 节第 5 条：索引失败不得覆盖当前可用索引）。

2. **词典版本绑定**：中文分词依赖 knowledge/tech_terms.txt，索引侧与查询侧
   词典不一致会让召回静默劣化。因此词典哈希写进 meta 表，打开索引时校验。
"""

from __future__ import annotations

import json
import os
import sqlite3
import struct
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from packages.schemas.chunk import Chunk  # noqa: E402

from .embed import DIM  # noqa: E402
from .tokenize import dictionary_version, to_fts_document  # noqa: E402

INDEX_DIR = Path(__file__).resolve().parents[2] / "data" / "index"
CURRENT = INDEX_DIR / "current.db"

_SCHEMA = f"""
CREATE TABLE chunks (
    id                INTEGER PRIMARY KEY,
    checksum          TEXT NOT NULL UNIQUE,
    source_url        TEXT NOT NULL,
    source_project    TEXT NOT NULL,
    version_or_commit TEXT NOT NULL,
    license           TEXT NOT NULL,
    retrieved_at      TEXT NOT NULL,
    title_path        TEXT NOT NULL,
    technology        TEXT NOT NULL,
    content_type      TEXT NOT NULL,
    locale            TEXT NOT NULL,
    token_estimate    INTEGER NOT NULL,
    anchor            TEXT,
    source_path       TEXT,
    text              TEXT NOT NULL
);
CREATE INDEX idx_chunks_tech ON chunks(technology);
CREATE INDEX idx_chunks_project ON chunks(source_project);

-- 中文必须预分词后写入；FTS5 默认分词器对中文完全失效（I0 实测命中 0）
CREATE VIRTUAL TABLE chunks_fts USING fts5(tokenized, content='');

CREATE VIRTUAL TABLE chunks_vec USING vec0(embedding float[{DIM}]);

CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


def _connect(path: Path, cross_thread: bool = False) -> sqlite3.Connection:
    # 网关在启动线程里打开索引，却从事件循环线程和多个工作线程读取。
    # sqlite3 默认禁止跨线程使用同一连接，因此这里放开检查，
    # 并由 ChunkStore.execute 的锁保证串行访问。
    db = sqlite3.connect(path, check_same_thread=not cross_thread)
    db.row_factory = sqlite3.Row
    db.enable_load_extension(True)
    import sqlite_vec

    sqlite_vec.load(db)
    db.enable_load_extension(False)
    return db


def _pack(vec: Sequence[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


class IndexError_(RuntimeError):
    pass


@dataclass
class IndexStats:
    chunks: int
    projects: dict[str, int]
    path: Path


class IndexBuilder:
    """把切块写入一个**新的**索引文件，不触碰当前索引。"""

    def __init__(self, name: str = "current") -> None:
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        self.target = INDEX_DIR / f"{name}.db"
        self.staging = INDEX_DIR / f"{name}.building.db"
        self.staging.unlink(missing_ok=True)
        self.db = _connect(self.staging)
        self.db.executescript(_SCHEMA)
        self._n = 0

    def add(self, chunks: list[Chunk], vectors: list[Sequence[float]]) -> int:
        """写入一批切块。调用方须保证每块已通过 validate()。"""
        if len(chunks) != len(vectors):
            raise IndexError_(f"切块数 {len(chunks)} 与向量数 {len(vectors)} 不一致")

        added = 0
        for c, v in zip(chunks, vectors):
            row = c.to_row()
            try:
                cur = self.db.execute(
                    """INSERT INTO chunks
                       (checksum, source_url, source_project, version_or_commit, license,
                        retrieved_at, title_path, technology, content_type, locale,
                        token_estimate, anchor, source_path, text)
                       VALUES (:checksum, :source_url, :source_project, :version_or_commit,
                               :license, :retrieved_at, :title_path, :technology,
                               :content_type, :locale, :token_estimate, :anchor,
                               :source_path, :text)""",
                    row,
                )
            except sqlite3.IntegrityError:
                continue  # checksum 重复：同一段正文在多处出现，只留一份
            rid = cur.lastrowid
            # 标题路径一并进全文索引：用户常用章节名而非正文原词提问
            doc = to_fts_document(" ".join(c.title_path) + "\n" + c.text)
            self.db.execute("INSERT INTO chunks_fts(rowid, tokenized) VALUES (?, ?)", (rid, doc))
            self.db.execute("INSERT INTO chunks_vec(rowid, embedding) VALUES (?, ?)", (rid, _pack(v)))
            added += 1
        self._n += added
        self.db.commit()
        return added

    def finalize(self, sources: dict[str, str], embedding_model: str) -> IndexStats:
        meta = {
            "dictionary_version": dictionary_version(),
            "embedding_model": embedding_model,
            "embedding_dim": str(DIM),
            "chunk_count": str(self._n),
            "sources": json.dumps(sources, ensure_ascii=False),
        }
        self.db.executemany("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", meta.items())
        self.db.commit()
        projects = {
            r["source_project"]: r["n"]
            for r in self.db.execute(
                "SELECT source_project, COUNT(*) n FROM chunks GROUP BY 1 ORDER BY 2 DESC"
            )
        }
        self.db.close()
        return IndexStats(chunks=self._n, projects=projects, path=self.staging)

    def activate(self) -> Path:
        """原子替换当前索引。仅在回归检索通过后调用。"""
        if not self.staging.exists():
            raise IndexError_("暂存索引不存在，无法激活")
        backup = self.target.with_suffix(".previous.db")
        if self.target.exists():
            os.replace(self.target, backup)
        os.replace(self.staging, self.target)
        return self.target

    def discard(self) -> None:
        self.staging.unlink(missing_ok=True)


class EmbeddingCache:
    """按正文校验和复用已有索引里的向量。

    嵌入是同步流程里最慢的一步（I0 实测 12.9 chunk/s，全量 13k 块约 51 分钟），
    但它只依赖正文。修正 URL 映射、标题路径这类元数据时正文没变，
    没有理由重算——这也是 development.md 第 4 节要求的增量同步的基础。
    """

    def __init__(self, path: Path | None = None, embedding_model: str | None = None) -> None:
        self.path = path or CURRENT
        self._db: sqlite3.Connection | None = None
        self.hits = 0
        self.misses = 0
        self.rejected_reason: str | None = None
        if not self.path.exists():
            return
        try:
            db = _connect(self.path)
        except Exception:
            return

        # 校验和只覆盖正文，换了嵌入模型后同一正文的向量语义完全不同。
        # 不比对模型名就会静默复用错的向量——报错都没有，只是检索悄悄变差。
        if embedding_model is not None:
            try:
                # sqlite3 会把损坏文件的错误推迟到首次查询，因此 _connect 成功
                # 不代表库可用。这里失败应回退为全量重算，而不是让整次同步崩掉。
                row = db.execute("SELECT value FROM meta WHERE key = 'embedding_model'").fetchone()
            except Exception as exc:
                self.rejected_reason = f"当前索引不可读（{type(exc).__name__}），向量将全部重算"
                db.close()
                return
            built = row["value"] if row else None
            if built != embedding_model:
                self.rejected_reason = (
                    f"索引由 {built} 建成，当前为 {embedding_model}，向量不可复用，将全部重算"
                )
                db.close()
                return
        self._db = db

    @property
    def available(self) -> bool:
        return self._db is not None

    def get(self, checksum: str) -> list[float] | None:
        if self._db is None:
            return None
        row = self._db.execute(
            """SELECT v.embedding AS e FROM chunks c
               JOIN chunks_vec v ON v.rowid = c.id
               WHERE c.checksum = ?""",
            (checksum,),
        ).fetchone()
        if row is None or row["e"] is None:
            self.misses += 1
            return None
        self.hits += 1
        blob = row["e"]
        return list(struct.unpack(f"{len(blob) // 4}f", blob))

    def close(self) -> None:
        if self._db is not None:
            self._db.close()
            self._db = None


class ChunkStore:
    """只读索引访问。"""

    def __init__(self, path: Path | None = None, *, check_dictionary: bool = True) -> None:
        self.path = path or CURRENT
        if not self.path.exists():
            raise IndexError_(f"索引不存在: {self.path}，请先运行 kb sync")
        self.db = _connect(self.path, cross_thread=True)
        self._lock = threading.Lock()
        self.meta = {r["key"]: r["value"] for r in self.db.execute("SELECT key, value FROM meta")}

        if check_dictionary:
            built = self.meta.get("dictionary_version")
            now = dictionary_version()
            if built != now:
                raise IndexError_(
                    f"分词词典已变更（索引 {built} vs 当前 {now}）。"
                    f"查询侧与索引侧词典不一致会让召回静默劣化，请重建索引"
                )

    def execute(self, sql: str, params: Sequence = ()) -> list[sqlite3.Row]:
        """串行化的只读查询入口。

        连接允许跨线程使用，因此所有访问必须经过这把锁——
        同一连接上的并发游标会互相干扰，症状是偶发的结果错乱而非报错。
        """
        with self._lock:
            return self.db.execute(sql, params).fetchall()

    def count(self) -> int:
        return self.execute("SELECT COUNT(*) c FROM chunks")[0]["c"]

    def stats(self) -> dict[str, int]:
        return {
            r["source_project"]: r["n"]
            for r in self.execute(
                "SELECT source_project, COUNT(*) n FROM chunks GROUP BY 1 ORDER BY 2 DESC"
            )
        }

    def close(self) -> None:
        with self._lock:
            self.db.close()
