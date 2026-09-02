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
TERM_MAP_PATH = Path(__file__).resolve().parents[2] / "knowledge" / "term_map.yaml"

_lock = threading.Lock()
_ready = False
_version: str | None = None
_term_map: dict[str, list[str]] = {}
# 每个映射键的组成词，用于"中间插了别的词也能命中"（见 _split_key）
_KEY_PARTS: dict[str, list[str]] = {}


def _load() -> None:
    global _ready, _version
    with _lock:
        if _ready:
            return
        import jieba

        raw = TERMS_PATH.read_bytes()

        import yaml

        spec = yaml.safe_load(TERM_MAP_PATH.read_text(encoding="utf-8")) or {}
        _term_map.update(spec.get("terms") or {})
        # 映射表也参与词典版本：它变了，查询侧展开就变了，需与索引一同复核
        _version = hashlib.sha256(raw + TERM_MAP_PATH.read_bytes()).hexdigest()[:12]

        for zh in _term_map:
            jieba.add_word(zh, freq=900)

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

    # 词典就绪后才能切词，故放在锁外、_ready 置位之后
    for zh in _term_map:
        parts = _split_key(zh)
        if parts:
            _KEY_PARTS[zh] = parts


def dictionary_version() -> str:
    """词典内容哈希。词典变更等同于索引结构变更，必须整体重建索引。"""
    _load()
    assert _version
    return _version


_PUNCT = re.compile(r"[^\w一-鿿]+")

# 这些词在对应语料里几乎每块都出现，作为检索词毫无区分度，
# 却会让 bm25 奖励"重复提到 kafka"的文档，把真正有信息量的词淹没。
# 实测: 查询「幂等生产者」时，ACL 文档因反复出现 producer 而排在首位。
# 正确用法是把它们当作元数据过滤条件（technology / project），而非检索词。
PROJECT_TERMS = {
    "kafka": "kafka",
    "kubernetes": "kubernetes", "k8s": "kubernetes",
    "postgresql": "postgresql", "postgres": "postgresql", "pg": "postgresql",
    "redis": "redis",
    "spring": "spring", "springboot": "spring",
}


def detect_technology(text: str) -> str | None:
    """从提问中识别技术域，用于检索过滤。

    返回第一个匹配到的技术域；匹配到多个时返回 None，
    因为跨技术域的问题（如"Spring Boot 连 Kafka"）不应被单域过滤掉。
    """
    low = text.lower()
    found = {tech for term, tech in PROJECT_TERMS.items() if term in low}
    return found.pop() if len(found) == 1 else None


# 驼峰标识符：livenessProbe、RedisCacheManager、httpGet。
# 分词器把它们当成单个 token（`livenessprobe`），于是查询里的
# `liveness OR probe` 永远匹配不上——Kubernetes 的 YAML 字段名和 Java 类名
# 几乎全是这个形式，对自然语言提问等于不可达。
# 点号形式（session.timeout.ms）会被标点规则切开，不受影响。
# 连续大写要整体成段，否则 PostgreSQL 会被切成 Postgre + S + Q + L
_CAMEL = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z0-9]*|[a-z0-9]+")


def _split_camel(tok: str) -> list[str]:
    """把驼峰标识符拆成组成词。整词一并保留——精确查 `livenessProbe` 时它仍要能命中。"""
    if not (tok[:1].isascii() and tok.isalnum()):
        return []
    low = tok.lower()
    # 产品名不拆：PostgreSQL 拆出的 `postgre` 出现在每个 PG 块里，
    # 零区分度却会被 bm25 奖励，正是 PROJECT_TERMS 要消除的那种噪声。
    if low in PROJECT_TERMS:
        return []
    parts = _CAMEL.findall(tok)
    # 至少两段、且确实含大写才算驼峰；全小写或全大写词拆出来的还是它自己
    if len(parts) < 2 or tok.islower() or tok.isupper():
        return []
    return [p.lower() for p in parts if len(p) > 1]


def tokenize(text: str) -> list[str]:
    """切词并去掉标点。英文与数字原样保留，中文按词典切分。

    驼峰标识符额外拆出组成词（`livenessProbe` → `livenessprobe`、`liveness`、`probe`），
    索引侧与查询侧共用本函数，因此两边的拆分口径天然一致。
    """
    _load()
    import jieba

    out: list[str] = []
    for tok in jieba.cut_for_search(text):
        tok = tok.strip()
        if not tok or _PUNCT.fullmatch(tok):
            continue
        out.append(tok.lower())
        out.extend(_split_camel(tok))
    return out


def to_fts_document(text: str) -> str:
    """索引侧：写入 FTS5 的空格分隔文档。"""
    return " ".join(tokenize(text))


# 展开出的英文里的功能词。它们在英文语料里几乎每块都出现，单独入 OR 查询
# 只会稀释信号——和 PROJECT_TERMS 是同一类问题，只是来源不同（CR-013）。
#
# 实测（T-025）：`失效 → not used / ignored / disabled / invalid` 让
# `not`、`used` 进入查询后，`PostgreSQL 索引失效` 的正确块掉到关键词路第 109 名。
#
# **只过滤单个 token，多词短语整体保留**：`"not used"` 作为短语仍有区分度，
# 拆出来的 `not`、`used` 没有。
EXPANSION_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "can", "do", "does",
    "for", "from", "in", "is", "it", "its", "not", "of", "on", "or", "that",
    "out", "the", "this", "to", "use", "used", "using", "when", "with",
}


def _split_key(key: str) -> list[str] | None:
    """把映射键切成一组**不重叠且完整覆盖它**的组成词；覆盖不全时返回 None。

    用途见 expand_terms：`索引失效` → `["索引", "失效"]`，于是
    `B-tree 索引什么时候会失效` 这种中间插了词的提问也能命中。

    **必须要求完整覆盖**。`慢查询` 经 jieba 只切出 `["查询"]`（单字子词被丢掉），
    若允许部分覆盖，任何含"查询"的提问都会被展开成慢查询的检索词。
    覆盖不全的键退回原来的子串匹配，宁可漏也不要错。
    """
    low = key.lower()
    parts = sorted(
        {t for t in tokenize(key) if t != low and t in low},
        key=len, reverse=True,
    )
    out, i = [], 0
    while i < len(low):
        for p in parts:
            if low.startswith(p, i):
                out.append(p)
                i += len(p)
                break
        else:
            return None
    # 只有一段等于键本身，子串匹配已经覆盖，无需额外规则
    return out if len(out) >= 2 else None


def matched_terms(text: str) -> list[str]:
    """本次提问命中了哪些映射键。展开逻辑与调试、测试共用这一个入口。

    两条匹配规则：
    1. 子串命中（原有行为）。
    2. **组成词全部命中**——`索引失效` 的 `索引` 与 `失效` 都在提问里就算命中，
       哪怕中间隔着"什么时候会"。CR-013 指出的第二例正是败在这里：
       子串匹配不上具体映射，却仍触发了通用的 `失效`，结果只剩纯噪声词。

    命中后做**最具体者胜**：`索引失效` 命中时丢掉 `失效`。
    通用键的英文展开必然更泛，与具体键叠加只会稀释信号。
    """
    _load()
    lowered = text.lower()
    qtoks = set(tokenize(text))
    hit = set()
    for zh in _term_map:
        if zh in text or zh.lower() in lowered:
            hit.add(zh)
            continue
        parts = _KEY_PARTS.get(zh)
        if parts and all(p in qtoks for p in parts):
            hit.add(zh)
    # 被更具体的命中键包含的，一律丢弃
    return sorted(z for z in hit if not any(z != o and z in o for o in hit))


def expand_terms(text: str) -> list[str]:
    """把查询中的中文技术概念展开为英文检索词。

    I1 实测：首批语料 100% 为英文，中文 token 永远匹配不上正文，
    关键词检索退化为按 "postgresql" 这类 ASCII 词排序的噪声。
    展开后关键词路才能真正参与排序。
    """
    out: list[str] = []
    for zh in matched_terms(text):
        out.extend(_term_map[zh])
    return out


def to_fts_query(text: str, expand: bool = True) -> str:
    """查询侧：转成 FTS5 的 OR 查询。

    用 OR 而非 AND：语音转写会写错技术名词（I0 实测 Kafka→CAFCA），
    要求全部词命中会让一个错字毁掉整次检索。相关性由 bm25 与 RRF 融合决定。

    OR 查询对噪声词没有抵抗力：多一个 `not` 就多一批高分无关块。
    因此展开出的功能词单独出现时要滤掉（EXPANSION_STOPWORDS），短语整体保留。
    """
    toks = [t.replace('"', "") for t in tokenize(text)]
    expanded: list[str] = []
    if expand:
        for phrase in expand_terms(text):
            # 英文映射词按空白切开后各自入查询，短语整体也保留
            expanded.extend(p.lower() for p in phrase.split())
            if " " in phrase:
                expanded.append(phrase.lower())
    seen, uniq = set(), []
    for t in toks + expanded:
        t = t.strip().replace('"', "")
        if not t or t in seen:
            continue
        # 项目名不参与打分：它们的区分度为零，只会稀释真正的信号
        if t in PROJECT_TERMS:
            continue
        # 功能词同理；短语（含空格）不受影响
        if " " not in t and t in EXPANSION_STOPWORDS:
            continue
        seen.add(t)
        uniq.append(t)
    return " OR ".join(f'"{t}"' for t in uniq) if uniq else '""'
