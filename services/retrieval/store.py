"""索引存储 —— SQLite + sqlite-vec + FTS5。

两个关键设计：

1. **双缓冲**：新索引建在临时文件里，回归检索通过后才原子替换当前索引
   （development.md 第 4 节第 5 条：索引失败不得覆盖当前可用索引）。

2. **词典版本绑定**：中文分词依赖 knowledge/tech_terms.txt，索引侧与查询侧
   词典不一致会让召回静默劣化。因此词典哈希写进 meta 表，打开索引时校验。
"""

from __future__ import annotations

import hashlib
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
from services.sync.version import content_transform_version  # noqa: E402

from .embed import DIM  # noqa: E402
from .tokenize import dictionary_version, to_fts_document  # noqa: E402

INDEX_DIR = Path(__file__).resolve().parents[2] / "data" / "index"
CURRENT = INDEX_DIR / "current.db"

_SCHEMA = f"""
CREATE TABLE chunks (
    id                INTEGER PRIMARY KEY,
    checksum          TEXT NOT NULL,
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
    project_id        TEXT,
    module            TEXT,
    symbol            TEXT,
    cloud_generation_allowed INTEGER,
    text              TEXT NOT NULL,

    -- 唯一键必须带上来源身份。只按 checksum 去重会让"同一段说明出现在两个页面"
    -- 中的一个被静默丢弃，那个来源就再也引用不到（CR-003）。
    UNIQUE(source_url, checksum)
);
CREATE INDEX idx_chunks_tech ON chunks(technology);
CREATE INDEX idx_chunks_project ON chunks(source_project);
CREATE INDEX idx_chunks_project_id ON chunks(project_id);
CREATE INDEX idx_chunks_project_symbol ON chunks(project_id, symbol);
-- 向量复用按 checksum 单独查（EmbeddingCache.get），
-- 复合唯一键以 source_url 打头用不上，缺这条索引会退化成逐块全表扫描。
CREATE INDEX idx_chunks_checksum ON chunks(checksum);

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
    chunks: int                 # 索引内实际行数
    projects: dict[str, int]
    path: Path
    added: int = 0              # 本次同步新写入
    carried: int = 0            # 合并更新时从底座搬运


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
        self.carried = 0

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
                        token_estimate, anchor, source_path, project_id, module, symbol,
                        cloud_generation_allowed, text)
                       VALUES (:checksum, :source_url, :source_project, :version_or_commit,
                               :license, :retrieved_at, :title_path, :technology,
                               :content_type, :locale, :token_estimate, :anchor,
                               :source_path, :project_id, :module, :symbol,
                               :cloud_generation_allowed, :text)""",
                    row,
                )
            except sqlite3.IntegrityError:
                continue  # 同一来源页内的完全重复正文，只留一份
            rid = cur.lastrowid
            # 标题路径一并进全文索引：用户常用章节名而非正文原词提问
            doc = to_fts_document(" ".join(c.title_path) + "\n" + c.text)
            self.db.execute("INSERT INTO chunks_fts(rowid, tokenized) VALUES (?, ?)", (rid, doc))
            self.db.execute("INSERT INTO chunks_vec(rowid, embedding) VALUES (?, ?)", (rid, _pack(v)))
            added += 1
        self._n += added
        self.db.commit()
        return added

    def carry_over(
        self,
        source_index: Path,
        exclude_projects: set[str],
        embedding_model: str,
        exclude_source_path_prefixes: tuple[str, ...] = (),
    ) -> int:
        """从既有索引里搬运**不参与本次同步**的来源，用于合并更新（CR-004）。

        不用 DELETE 从副本里剔除，而是反过来把要保留的搬进空索引：
        `chunks_fts` 是 contentless 表，删除需要原始分词文本，
        而搬运时正文就在手边，按当前词典重算反而更正确——
        词典改了，被搬运来源的分词也随之更新，不会留下按旧词典切的残余。

        向量不能重算（太慢），因此嵌入模型必须一致，否则拒绝合并。

        `exclude_projects` 按 `source_project` 排除，对绝大多数来源足够——
        但 authored 来源（场景卡片）的块 `source_project` 是它引用的真实来源
        （如 debezium），不是卡片自己的来源 id，靠 `source_project` 认不出
        "这条块属于哪张卡片"。`exclude_source_path_prefixes` 按 `source_path`
        前缀补一道排除，卡片块的 `source_path` 是卡片文件自身的相对路径
        （见 services/sync/cards.py），据此才能在重新同步卡片来源时正确排除
        旧块，否则编辑或删除卡片小节后，旧版本会被当成"未参与本次同步"
        永久搬运下去。
        """
        if not source_index.exists():
            raise IndexError_(f"合并更新需要一个已有索引作为底座，但 {source_index} 不存在")
        src = _connect(source_index)
        try:
            built = src.execute(
                "SELECT value FROM meta WHERE key = 'embedding_model'"
            ).fetchone()
            built = built["value"] if built else None
            if built != embedding_model:
                raise IndexError_(
                    f"底座索引由 {built} 建成，当前嵌入模型为 {embedding_model}；"
                    f"向量语义不同不能混用，请改用全量重建"
                )

            # 块是 parse.py/chunk.py（以及 authored / 项目材料转换器）的派生物。
            # merge 若继续搬运不同版本的块，会让本轮 --only 未点名的来源永久停在
            # 旧解析结果；T-116 已实测发生过这种陈旧解析。缺少该元数据的旧索引
            # 同样不可信，必须失败关闭，而非把“还没记录版本”当成兼容。
            built_transform = src.execute(
                "SELECT value FROM meta WHERE key = 'content_transform_version'"
            ).fetchone()
            built_transform = built_transform["value"] if built_transform else None
            current_transform = content_transform_version()
            if built_transform != current_transform:
                raise IndexError_(
                    "底座索引的解析/切块版本与当前实现不一致"
                    f"（底座 {built_transform or '未记录'}，当前 {current_transform}）；"
                    "不能搬运可能陈旧的块，请执行不带 --only 的全量 kb sync"
                )

            moved = 0
            rows = src.execute(
                """SELECT c.*, v.embedding AS embedding FROM chunks c
                   JOIN chunks_vec v ON v.rowid = c.id
                   ORDER BY c.id"""
            )
            for r in rows:
                if r["source_project"] in exclude_projects:
                    continue
                path = r["source_path"]
                if path and any(
                    path == prefix or path.startswith(prefix.rstrip("/") + "/")
                    for prefix in exclude_source_path_prefixes
                ):
                    continue
                # current.db 可能由项目元数据扩列之前的版本构建。合并项目资料时
                # 不能因为旧官方块没有这些列就无法搬运；它们明确为 NULL，而不是
                # 猜测成某个项目。这次发布后，新索引自然带齐完整 schema。
                available = set(r.keys())
                cur = self.db.execute(
                    """INSERT INTO chunks
                       (checksum, source_url, source_project, version_or_commit, license,
                        retrieved_at, title_path, technology, content_type, locale,
                        token_estimate, anchor, source_path, project_id, module, symbol,
                        cloud_generation_allowed, text)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    tuple(r[k] if k in available else None for k in (
                        "checksum", "source_url", "source_project", "version_or_commit",
                        "license", "retrieved_at", "title_path", "technology",
                        "content_type", "locale", "token_estimate", "anchor",
                        "source_path", "project_id", "module", "symbol",
                        "cloud_generation_allowed", "text")),
                )
                rid = cur.lastrowid
                doc = to_fts_document(r["title_path"].replace(" › ", " ") + "\n" + r["text"])
                self.db.execute(
                    "INSERT INTO chunks_fts(rowid, tokenized) VALUES (?, ?)", (rid, doc)
                )
                self.db.execute(
                    "INSERT INTO chunks_vec(rowid, embedding) VALUES (?, ?)",
                    (rid, r["embedding"]),
                )
                moved += 1
        finally:
            src.close()
        self.db.commit()
        self.carried = moved
        return moved

    def existing_versions(self, source_index: Path) -> dict[str, str]:
        """读出底座索引记录的来源版本，合并时只覆盖本次同步的那几个。"""
        if not source_index.exists():
            return {}
        db = _connect(source_index)
        try:
            row = db.execute("SELECT value FROM meta WHERE key = 'sources'").fetchone()
        except Exception:
            return {}
        finally:
            db.close()
        return json.loads(row["value"]) if row else {}

    def finalize(self, sources: dict[str, str], embedding_model: str) -> IndexStats:
        total = self.db.execute("SELECT COUNT(*) c FROM chunks").fetchone()["c"]
        meta = {
            "dictionary_version": dictionary_version(),
            "content_transform_version": content_transform_version(),
            "embedding_model": embedding_model,
            "embedding_dim": str(DIM),
            "chunk_count": str(total),
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
        return IndexStats(chunks=total, added=self._n, carried=self.carried, projects=projects, path=self.staging)

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
               WHERE c.checksum = ? LIMIT 1""",
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


def index_fingerprint(store) -> str:
    """全量块的内容指纹：排序后的 `source_url\tchecksum` 之 SHA-256。

    **只记块数不足以标识索引**——同样是 17080 块可以是完全不同的内容
    （CR-159 复审给出的判据，CR-160 要求评测侧也用同一个）。
    `UNIQUE(source_url, checksum)` 保证这对值能唯一标识一块，排序后逐行入哈希，
    与插入顺序、rowid 分配都无关，因此索引重建后可独立复算比对。

    参数按鸭子类型取 `execute()` 与 `count()`，替身 store 也能直接用——
    这个函数是 `bench/bench_speculative_retrieval.py` 与
    `packages/evaltools/run_basic.py` **共用的同一份实现**，不是各抄一份：
    两份拷贝迟早会在某一次改动后算出不同的指纹，而指纹的全部价值就是跨工具可比。

    遍历数与 `count()` 不符时抛 `ValueError`，由调用方决定是失败关闭还是别的处置
    （CLI 侧一律失败关闭，见两个调用点）。
    """
    rows = store.execute(
        "SELECT source_url, checksum FROM chunks ORDER BY source_url, checksum"
    )
    # 在 Python 侧再排一次序：SQL 的 ORDER BY 已经保证了真实索引的顺序，但这样
    # 顺序无关就是**这个函数自己的性质**，而不是"调用方恰好传了个会排序的 store"。
    # 审查用的替身 store 通常不解析 SQL，只把行原样返回——没有这一步，"同内容
    # 反序仍同指纹"在替身上就不成立，而那正是审查方复核指纹时要验的性质。
    # 对真实索引结果不变：URL 与 checksum 都是 ASCII，SQLite 的 BINARY 排序与
    # Python 的字符串排序一致（本轮已对当前索引复算比对，见 R185 记录）。
    lines = sorted(
        f"{row['source_url']}\t{row['checksum']}\n" for row in rows
    )
    digest = hashlib.sha256()
    counted = 0
    for line in lines:
        digest.update(line.encode("utf-8"))
        counted += 1
    total = store.count()
    if counted != total:
        raise ValueError(f"索引指纹覆盖 {counted} 块，与 count() 的 {total} 不一致")
    return digest.hexdigest() if counted else ""
