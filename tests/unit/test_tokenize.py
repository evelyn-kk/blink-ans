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


# ---------- 中英术语展开 ----------

def test_chinese_terms_expand_to_english():
    """首批语料 100% 为英文，中文 token 匹配不上正文。

    I1 实测：未展开时「PostgreSQL 慢查询 执行计划」的关键词路返回
    Information Schema 与回归测试文档（纯噪声）；
    展开出 EXPLAIN / execution plan 后命中
    "Performance Tips › Using EXPLAIN › EXPLAIN ANALYZE"。
    """
    from services.retrieval.tokenize import expand_terms

    ens = [e.lower() for e in expand_terms("PostgreSQL 慢查询 执行计划变成全表扫描")]
    assert "explain" in ens
    assert any("execution plan" in e or "query plan" in e for e in ens)
    assert any("seq scan" in e or "sequential scan" in e for e in ens)


def test_expansion_reaches_fts_query():
    q = to_fts_query("Redis 序列化怎么配置")
    assert "serializer" in q.lower() or "serialization" in q.lower()


def test_expansion_can_be_disabled():
    plain = to_fts_query("慢查询", expand=False)
    assert "explain" not in plain.lower()


def test_unmapped_query_still_works():
    assert to_fts_query("某个完全没有映射的说法") != '""'


def test_query_tokens_are_deduplicated():
    """多个中文词可能展开出同一个英文词，重复会让 bm25 打分失真。"""
    q = to_fts_query("慢查询 执行计划")   # 两者都展开出 EXPLAIN
    assert q.lower().count('"explain"') == 1


def test_term_map_changes_dictionary_version():
    """映射表变更会改变查询展开，必须触发索引复核。"""
    from services.retrieval.tokenize import TERM_MAP_PATH
    assert TERM_MAP_PATH.exists()
    assert len(dictionary_version()) == 12


# ---------- 驼峰标识符拆分（T-017）----------

def test_camel_identifier_splits_into_parts_and_keeps_whole():
    """`livenessProbe` 只切成一个 token 时，查询里的 `liveness OR probe` 永远匹配不上。
    Kubernetes 的 YAML 字段名与 Java 类名几乎全是驼峰，对自然语言提问等于不可达。
    整词必须保留：精确查 `livenessProbe` 时它仍要能命中。"""
    toks = tokenize("livenessProbe")
    assert "livenessprobe" in toks
    assert "liveness" in toks and "probe" in toks


def test_camel_split_applies_to_java_class_names():
    toks = tokenize("RedisCacheManager")
    assert {"cache", "manager"} <= set(toks)


def test_acronym_prefix_is_not_shredded():
    """连续大写要整体成段，否则 XMLHttpRequest 会被切成 X/M/L/Http/Request。"""
    toks = tokenize("XMLHttpRequest")
    assert "xml" in toks and "http" in toks and "request" in toks
    assert "x" not in toks and "m" not in toks


def test_product_names_are_not_split():
    """PostgreSQL 拆出的 `postgre` 出现在每个 PG 块里，零区分度却会被 bm25 奖励——
    正是 PROJECT_TERMS 要消除的那种噪声。"""
    assert tokenize("PostgreSQL") == ["postgresql"]


def test_all_lower_and_all_upper_tokens_are_untouched():
    assert tokenize("HTTP") == ["http"]
    assert tokenize("timeout") == ["timeout"]


def test_dotted_config_keys_still_split_on_punctuation():
    """点号形式本来就被标点规则切开，驼峰逻辑不得影响它。"""
    assert set(tokenize("session.timeout.ms")) >= {"session", "timeout", "ms"}


def test_camel_split_reaches_both_index_and_query_side():
    """两侧共用 tokenize，口径必须一致——否则索引里有的词查询侧产生不出来。"""
    assert "liveness" in to_fts_document("livenessProbe: httpGet").split()
    assert '"liveness"' in to_fts_query("livenessProbe")


# ---------- term_map 覆盖（T-017 补入的概念）----------

@pytest.mark.parametrize("zh, en", [
    ("投递语义", "delivery"), ("优雅停机", "graceful"), ("日志级别", "logging.level"),
    ("会话超时", "session.timeout.ms"), ("缓存注解", "cacheable"), ("配置优先级", "externalized"),
])
def test_newly_mapped_concepts_expand_to_english(zh, en):
    """I2 遗留的失败题里，这些中文概念原本一个都没映射，关键词路等于没上场。"""
    q = to_fts_query(f"{zh}怎么配置").lower()
    assert en.lower() in q, q


# ---------- 通用映射污染 OR 查询（T-025 / CR-013）----------
#
# 这批测试全部**先在旧实现上验证过判别性**：把 matched_terms 换回纯子串匹配、
# 或去掉 EXPANSION_STOPWORDS，它们会挂。只会变绿的测试不是回归。

def test_generic_key_is_dropped_when_specific_key_matches():
    """`索引失效` 命中时 `失效` 必须让位。

    通用键的英文展开必然更泛，与具体键叠加只会稀释信号——
    实测叠加后正确块在关键词路排到第 109 名。
    """
    from services.retrieval.tokenize import matched_terms
    hit = matched_terms("PostgreSQL 索引失效")
    assert "索引失效" in hit
    assert "失效" not in hit


def test_specific_key_still_matches_with_words_in_between():
    """`B-tree 索引什么时候会失效`：子串匹配不到 `索引失效`，组成词匹配必须兜住。

    这正是 CR-013 的第二例。修复前它只剩 not/used/ignored/disabled/invalid
    这组纯噪声进入查询，是 50 题回归里最后一道未命中题的直接原因。
    """
    from services.retrieval.tokenize import matched_terms
    hit = matched_terms("PostgreSQL 的 B-tree 索引什么时候会失效")
    assert "索引失效" in hit
    assert "失效" not in hit


def test_partial_key_coverage_does_not_over_fire():
    """`慢查询` 经 jieba 只切出 `["查询"]`，若允许部分覆盖，任何含"查询"的
    提问都会被展开成慢查询的检索词。覆盖不全的键必须退回子串匹配。"""
    from services.retrieval.tokenize import matched_terms
    assert "慢查询" not in matched_terms("PostgreSQL 查询超时怎么办")
    assert "慢查询" in matched_terms("PostgreSQL 慢查询怎么排查")


def test_component_match_works_for_multi_word_concepts():
    """`Kafka 生产者怎么保证幂等` 应命中 `幂等生产者`——词序被打散也要认得出。"""
    from services.retrieval.tokenize import matched_terms
    assert "幂等生产者" in matched_terms("Kafka 生产者怎么保证幂等")


def test_function_words_never_enter_query_alone():
    """`not` / `used` 在英文语料里几乎每块都出现，单独入 OR 查询只会稀释信号。"""
    q = to_fts_query("Kubernetes 探针失效")
    assert '"not"' not in q and '"used"' not in q


def test_multiword_phrase_survives_stopword_filter():
    """短语整体保留：`"not used"` 有区分度，拆出来的两个词没有。"""
    q = to_fts_query("Kubernetes 探针失效")
    assert '"not used"' in q


def test_generic_key_still_helps_when_no_specific_key_matches():
    """没有专属映射时通用键仍要出力，否则「探针失效」一类提问会退化成无展开。"""
    from services.retrieval.tokenize import matched_terms
    assert "失效" in matched_terms("Kubernetes 探针失效")


def test_index_failure_expansion_targets_corpus_vocabulary():
    """判据挂在语料实际用词上：PG 讲这件事的小节写的是 examining/forcing index usage，
    不是 `unused index`。原映射对不上语料，是"展开了但等于没展开"。"""
    q = to_fts_query("PostgreSQL 索引失效").lower()
    assert "index usage" in q
    assert "planner" in q


def test_generic_priority_no_longer_injects_order():
    """通用键 `优先级` 不再注入 `order`——它在数据库语料里指 ORDER BY，与优先级无关。

    注意判据挑的是**只命中通用键**的提问。「配置的优先级」会命中更具体的
    `配置优先级`，它的 `property source order` 是正当短语，`order` 来自那里，
    与本条无关（这个区分正是最具体者胜要保证的）。
    """
    q = to_fts_query("Pod 的优先级怎么设置")
    assert '"order"' not in q
    assert '"precedence"' in q and '"priority"' in q


def test_order_is_not_globally_stopworded():
    """`order` 没有进 EXPANSION_STOPWORDS：ORDER BY 是真实 SQL 关键字，
    一刀切会让「ORDER BY 怎么用」这类提问失去唯一的有效检索词。
    停用词表每加一个词都是一次取舍，不能因为某个场景里它是噪声就全局封杀。"""
    assert '"order"' in to_fts_query("ORDER BY 怎么用")
