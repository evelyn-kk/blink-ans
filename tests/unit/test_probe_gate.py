"""排序探针门禁自身的回归（CR-015）。

为什么单独测：探针是**别的改动的判据**，它一旦悄悄失效，
后面所有"没退步"的结论都不作数。CR-015 就是这么一个洞——
门禁只在 `baseline is not None` 时判退步，而两道刚修好的目标题
`baseline: null`，于是它们从第 1 名跌到未进候选也照样 0 退出。

这里只测纯判定函数 `evaluate`，不加载索引与嵌入模型，因此毫秒级。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from packages.evaltools.probe_ranking import (  # noqa: E402
    PROBES, ProbeResult, evaluate,
)

TOP_K = 5


def mk(rank, *, baseline=None, known_open=False, top_k=TOP_K):
    return ProbeResult(
        question="q", gold="g", rank=rank,
        passed=rank is not None and rank <= top_k,
        baseline=baseline, known_open=known_open,
    )


def test_regression_against_baseline_is_reported():
    regressed, below = evaluate([mk(6, baseline=1)], TOP_K)
    assert len(regressed) == 1


def test_falling_out_of_candidates_is_a_regression():
    regressed, _ = evaluate([mk(None, baseline=1)], TOP_K)
    assert len(regressed) == 1


def test_known_open_gap_does_not_fail_when_stable():
    """既有缺口停在基线上不算退步，否则每次改动都被同一批红叉淹没。"""
    regressed, below = evaluate([mk(11, baseline=11, known_open=True)], TOP_K)
    assert not regressed and not below


def test_known_open_gap_still_fails_when_it_gets_worse():
    regressed, _ = evaluate([mk(20, baseline=11, known_open=True)], TOP_K)
    assert len(regressed) == 1


def test_missing_baseline_still_gated_by_top_k():
    """**CR-015 的核心**：没有基线的题不能因此免检。

    旧实现只看退步，`baseline is None` 时整条判据跳过，
    刚修好的题恰好失去门禁。
    """
    regressed, below = evaluate([mk(9, baseline=None)], TOP_K)
    assert not regressed          # 无基线，谈不上退步
    assert len(below) == 1        # 但没进 top_k，必须失败


def test_passing_probe_fails_nothing():
    regressed, below = evaluate([mk(1, baseline=1)], TOP_K)
    assert not regressed and not below


def test_no_probe_uses_null_baseline_as_a_backdoor():
    """`baseline: null` 只允许表示「从未测过」。

    若某条 known_open 的题用 null 当基线，它既不受退步判据约束，
    又被 top_k 判据豁免——两道闸同时打开。这里锁死这个组合。
    """
    spec = yaml.safe_load(PROBES.read_text(encoding="utf-8"))
    offenders = [
        p["q"] for p in spec["probes"]
        if p.get("known_open") and p.get("baseline") is None
    ]
    assert not offenders, f"这些题同时豁免了两道闸: {offenders}"


def test_every_probe_records_a_baseline():
    """每条探针都必须有基线。

    漏掉 `baseline` 的题只受 top_k 一道闸约束，
    从第 1 名跌到第 5 名不会被发现——门禁会随着题目增加而逐渐失真。
    实际发生过：修 CR-015 时才发现两条探针压根没有这个字段。
    """
    spec = yaml.safe_load(PROBES.read_text(encoding="utf-8"))
    missing = [p["q"] for p in spec["probes"] if "baseline" not in p]
    assert not missing, f"缺少 baseline: {missing}"


@pytest.mark.parametrize("field", ["q", "gold", "note"])
def test_every_probe_declares_gold_and_rationale(field):
    """金标准必须人工核对过，`note` 是核对留痕——没有理由的 gold 不可信。"""
    spec = yaml.safe_load(PROBES.read_text(encoding="utf-8"))
    for p in spec["probes"]:
        assert str(p.get(field, "")).strip(), f"{p.get('q')} 缺少 {field}"
