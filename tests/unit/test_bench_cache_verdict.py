"""远端基准的缓存判定与统计（CR-014 / CR-017）。

为什么值得单测：`bench_llm_remote.py` 要花钱才能跑，而它产出的报告要用来
决定生成后端路由（T-027）。判定逻辑本身必须能不花钱地验证——
否则「缓存到底有没有命中」这个结论只能靠人肉读日志。

**缓存不命中不报错**，只是变慢变贵。所以这里的核心断言是：
「没报字段」必须和「报了但为零」区分开，前者是测不了，后者才是没命中。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench"))
from bench_llm_remote import cache_verdict, summarize  # noqa: E402


def sample(read=None, write=None, ttft=1.0):
    return {
        "ttft_text_s": ttft, "first_event_s": 0.1, "total_s": 2.0,
        "text_chars": 10, "thinking_chars": 0,
        "prompt_tokens": 1000, "output_tokens": 20,
        "cache_write_tokens": write, "cache_read_tokens": read,
    }


# ---------- 两阶段断言：首轮写入、后续读取 ----------

def test_first_write_then_later_read_is_a_hit():
    v = cache_verdict([sample(write=900, read=0), sample(write=0, read=900)])
    assert v["status"] == "hit"
    assert v["first_write_tokens"] == 900
    assert v["later_read_tokens_max"] == 900


def test_zero_reads_across_later_runs_is_a_miss():
    """报了字段但后续读取全为 0 —— 这才是确认未命中，可以下结论。"""
    v = cache_verdict([sample(write=900, read=0), sample(write=0, read=0)])
    assert v["status"] == "miss"
    assert "最小可缓存长度" in v["reason"]


def test_absent_field_is_unverified_not_miss():
    """**本文件最重要的一条**：字段缺失 ≠ 未命中。

    把「这家没报」记成「没命中」，会让报告看起来验证过了，
    而 T-026 的完成条件恰恰要求显式验证。
    """
    v = cache_verdict([sample(read=None), sample(read=None)])
    assert v["status"] == "unverified"
    assert "未在 usage 中报告" in v["reason"]


def test_missing_first_write_is_unverified_even_if_later_read_is_positive():
    """后续读到缓存不等于本轮确实创建过缓存，不能跳过首阶段。"""
    v = cache_verdict([sample(write=None, read=0), sample(write=0, read=900)])
    assert v["status"] == "unverified"
    assert "首轮" in v["reason"]


def test_zero_first_write_is_a_miss_even_if_later_read_is_positive():
    """0 是供应商明确报告的“未创建”，与字段缺失的未验证不同。"""
    v = cache_verdict([sample(write=0, read=0), sample(write=0, read=900)])
    assert v["status"] == "miss"
    assert "未创建" in v["reason"]


def test_single_run_cannot_prove_a_hit():
    """一轮跑不出「后续读取」这件事，不能算验证通过。"""
    v = cache_verdict([sample(write=900, read=0)])
    assert v["status"] == "unverified"


def test_partial_hit_still_counts_but_reports_the_ratio():
    """三轮里只有一轮读到，也算命中，但比例要如实报出来供人判断。"""
    v = cache_verdict([sample(write=900, read=0), sample(read=0), sample(read=900)])
    assert v["status"] == "hit"
    assert v["later_runs_hit"] == "1/2"


# ---------- 统计不得把 None 当 0 ----------

def test_summarize_keeps_none_columns_as_none():
    """把 None 当 0 求中位数，会把「无法验证」算成「确认为零」。"""
    st = summarize([sample(read=None), sample(read=None)])
    assert st["median"]["cache_read_tokens"] is None
    assert st["p95"]["cache_read_tokens"] is None


def test_summarize_still_aggregates_numeric_columns():
    st = summarize([sample(ttft=1.0, read=None), sample(ttft=3.0, read=None)])
    assert st["median"]["ttft_text_s"] == 2.0


def test_one_missing_value_invalidates_the_whole_column():
    """混着报和不报时，整列作废——半真半假的中位数比没有更危险。"""
    st = summarize([sample(read=900), sample(read=None)])
    assert st["median"]["cache_read_tokens"] is None


def test_summarize_attaches_cache_verdict():
    st = summarize([sample(write=900, read=0), sample(read=900)])
    assert st["cache"]["status"] == "hit"


# ---------- P95 取实测值，不插值 ----------

@pytest.mark.parametrize("vals,expected", [
    ([1.0, 2.0, 3.0], 3.0),
    ([1.0, 1.0, 1.0, 1.0, 9.0], 9.0),
])
def test_p95_returns_an_observed_value(vals, expected):
    """插值会造出一个从未真实发生过的数字，而预算是按 P95 写死的。"""
    st = summarize([sample(ttft=v) for v in vals])
    assert st["p95"]["ttft_text_s"] == expected
