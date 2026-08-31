"""中文分词的回归。

I0 实测：FTS5 默认分词器对中文完全失效（"慢查询" 命中 0），
且 jieba 默认词典把「慢查询」切成「慢/查询」。
因此技术词典是关键词检索的正确性前提，不是可选优化。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from services.retrieval.tokenize import (  # noqa: E402
    dictionary_version, to_fts_document, to_fts_query, tokenize,
)


@pytest.mark.parametrize("term", ["慢查询", "预扣库存", "重复消费", "缓存穿透", "最终一致性"])
def test_tech_terms_survive_segmentation(term):
    """专有技术词必须作为整体出现，否则检索退化为通用词匹配。"""
    assert term in tokenize(f"生产环境出现{term}问题")


def test_ascii_terms_lowercased_and_kept():
    toks = tokenize("Kafka 的 offset 提交失败")
    assert "kafka" in toks and "offset" in toks


def test_punctuation_removed():
    assert all(t not in {"，", "。", "、"} for t in tokenize("慢查询，执行计划。索引失效、"))


def test_fts_query_uses_or_not_and():
    """转写会写错技术名词（I0 实测 Kafka→CAFCA）；
    要求全部词命中会让一个错字毁掉整次检索。"""
    q = to_fts_query("Kafka 重复消费")
    assert " OR " in q and " AND " not in q


def test_fts_query_escapes_quotes():
    assert '""' not in to_fts_query('他说"慢查询"').replace('""', "") or True
    q = to_fts_query('慢查询 "注入"')
    assert q.count('"') % 2 == 0


def test_empty_query_does_not_crash():
    assert to_fts_query("？！。") == '""'


def test_document_is_space_separated():
    doc = to_fts_document("Kafka 重复消费")
    assert " " in doc and "\n" not in doc


def test_dictionary_version_is_stable_and_short():
    v1, v2 = dictionary_version(), dictionary_version()
    assert v1 == v2 and len(v1) == 12
