"""问答编排的判定逻辑测试。

这些断言不加载模型，只验证充分性判定、证据选取与引用统计——
它们是决定"何时该拒答"的核心逻辑，必须能快速回归。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from services.orchestrator.answering import (  # noqa: E402
    AnswerConfig, Sufficiency, assess, citation_coverage, select_evidence,
)
from services.retrieval.search import Hit  # noqa: E402

CFG = AnswerConfig()


def hit(dist=0.65, kw=1, tokens=200, i=1) -> Hit:
    return Hit(
        rowid=i, text=f"证据正文 {i}" * 10, title_path=f"A › B{i}",
        source_url=f"https://example.com/{i}.html", source_project="kafka",
        version_or_commit="abc", retrieved_at="2026-08-31T00:00:00+00:00",
        technology="kafka", content_type="prose", token_estimate=tokens,
        score=0.03, keyword_rank=kw, vector_rank=1, vector_distance=dist,
    )


# ---------- 充分性判定 ----------

def test_empty_hits_are_insufficient():
    a = assess([], CFG)
    assert a.level is Sufficiency.INSUFFICIENT


def test_close_evidence_is_sufficient():
    """实测：语料覆盖的问题 top1 距离在 0.60–0.70。"""
    assert assess([hit(dist=0.66)], CFG).level is Sufficiency.SUFFICIENT


def test_far_evidence_is_insufficient():
    """实测：语料不覆盖的问题 top1 距离在 0.75–0.83。

    判定不能靠空结果——FTS 用 OR 查询，几乎任何中文提问都会返回内容（I1 实测）。
    """
    assert assess([hit(dist=0.83, kw=0)], CFG).level is Sufficiency.INSUFFICIENT


def test_middle_distance_is_limited_regardless_of_keyword():
    """中间档不依赖关键词命中。

    技术名词被剔除出检索词后（tokenize.PROJECT_TERMS），中文提问打英文语料
    关键词命中常为 0。若中间档要求 kw>0，该档永不生效，边缘的有效问题会被一律误拒——
    50 题回归中「Spring Boot 怎么配置日志级别」（实测 0.7241）就因此被错判为超范围。
    """
    assert assess([hit(dist=0.74, kw=1)], CFG).level is Sufficiency.LIMITED
    assert assess([hit(dist=0.74, kw=None)], CFG).level is Sufficiency.LIMITED


def test_borderline_valid_question_is_not_refused():
    """实测值回归：0.7241 属有效问题，不得判为证据不足。"""
    assert assess([hit(dist=0.7241, kw=None)], CFG).level is not Sufficiency.INSUFFICIENT


def test_assessment_uses_closest_not_first():
    """命中列表按 RRF 排序，最相关的未必排在首位。"""
    a = assess([hit(dist=0.80, i=1), hit(dist=0.62, i=2)], CFG)
    assert a.level is Sufficiency.SUFFICIENT
    assert a.top_distance == pytest.approx(0.62)


def test_missing_vector_signal_degrades_to_limited():
    """只有关键词命中时无法评估语义相关性，保守处理而非直接作答。"""
    h = hit(); h.vector_distance = None
    assert assess([h], CFG).level is Sufficiency.LIMITED


# ---------- 证据选取 ----------

def test_selection_respects_token_budget():
    """预算约束来自 I0 实测：超出即首 token 时延不达标。"""
    cfg = AnswerConfig(evidence_budget=500, max_evidence=10)
    ev = select_evidence([hit(tokens=200, i=i) for i in range(1, 6)], cfg)
    assert sum(len(e.text) for e in ev) > 0
    assert len(ev) == 2


def test_selection_skips_oversized_and_keeps_filling():
    """放不下的跳过、继续找更小的——宁可多带一条短证据也不让预算空着。"""
    cfg = AnswerConfig(evidence_budget=300, max_evidence=10)
    ev = select_evidence([hit(tokens=900, i=1), hit(tokens=100, i=2), hit(tokens=100, i=3)], cfg)
    assert [e.index for e in ev] == [1, 2]
    assert all("证据正文 1" not in e.text for e in ev)


def test_selection_caps_count():
    cfg = AnswerConfig(evidence_budget=10_000, max_evidence=3)
    assert len(select_evidence([hit(tokens=50, i=i) for i in range(1, 9)], cfg)) == 3


def test_evidence_indices_are_sequential_from_one():
    """编号是引用的锚，必须从 1 连续——模型输出 [2] 时客户端据此对应来源卡片。"""
    cfg = AnswerConfig(evidence_budget=300, max_evidence=10)
    ev = select_evidence([hit(tokens=900, i=1), hit(tokens=100, i=2), hit(tokens=100, i=3)], cfg)
    assert [e.index for e in ev] == list(range(1, len(ev) + 1))


# ---------- 引用覆盖率 ----------

def test_citation_coverage_extracts_used_indices():
    assert citation_coverage("结论：见 [1] 与 [3]。", 4) == [1, 3]


def test_citation_coverage_ignores_out_of_range():
    """模型编造超出范围的编号时不得计入，否则覆盖率虚高。"""
    assert citation_coverage("参考 [9] 和 [2]", 3) == [2]


def test_citation_coverage_empty_when_no_evidence():
    assert citation_coverage("任何文本 [1]", 0) == []


def test_uncited_answer_reports_zero():
    """无引用的技术论断无法追溯，对本产品等同于不可用，必须能被观测到。"""
    assert citation_coverage("结论：应当调大线程池。", 3) == []


# ---------- CR-002: 未加载时的错误路径不依赖 MLX ----------

def test_stream_before_load_raises_runtime_error_without_mlx():
    """未加载模型时必须抛 RuntimeError，而不是导入 MLX 失败。

    代码评审 CR-002：错误路径若先导入 mlx_lm，在没有 Metal 的环境
    （如 CI）会抛 ImportError，纯逻辑测试就无法作为门禁运行。
    """
    from services.inference.engine import InferenceEngine

    eng = InferenceEngine()
    with pytest.raises(RuntimeError, match="尚未加载"):
        next(eng.stream("测试"))
