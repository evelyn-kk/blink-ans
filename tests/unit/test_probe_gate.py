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
    NOT_IN_CANDIDATES, PROBES, ProbeResult, evaluate,
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


# ---------- CR-085：gold 的定位精度 ----------

def _hit(title_path: str, text: str = "", project: str = "spring-framework"):
    """只带 _matches_gold 用得上的三个字段的最小替身。"""
    from types import SimpleNamespace
    return SimpleNamespace(title_path=title_path, text=text, source_project=project)


_SECTION = "Annotations › Using `@Transactional`"
_SELF_INVOCATION = "only external method calls coming in through the proxy are intercepted"


def test_prefix_gold_matches_unrelated_subsection():
    """前缀匹配会把子小节也算成命中——这是本轮实测到的假绿来源。

    真实案例：`… › Multiple Transaction Managers with @Transactional`
    与"自调用为什么不生效"毫无关系，却在第 2 名让探针判通过。
    """
    from packages.evaltools.probe_ranking import _matches_gold

    other = _hit(_SECTION + " › Multiple Transaction Managers with `@Transactional`",
                 "Most Spring applications need only a single transaction manager")
    assert _matches_gold(other, _SECTION, "spring-framework") is True


def test_exact_gold_rejects_subsections_but_not_same_section_siblings():
    """exact 挡住子小节；但同名小节的**兄弟块**它挡不住——所以还需要 contains。

    真实案例：exact 之下命中的是同一条 title_path 的另一块，讲的是
    `@EnableTransactionManagement` 的扫描范围，仍然不是自调用。
    """
    from packages.evaltools.probe_ranking import _matches_gold

    sub = _hit(_SECTION + " › Custom Composed Annotations", "compose your own annotation")
    assert _matches_gold(sub, _SECTION, "spring-framework", exact=True) is False

    sibling = _hit(_SECTION, "`@EnableTransactionManagement` ... look for `@Transactional` "
                             "only on beans in the same application context")
    assert _matches_gold(sibling, _SECTION, "spring-framework", exact=True) is True


def test_contains_pins_gold_to_the_manually_verified_chunk():
    """exact + contains 才真正指到人工核对过的那一块。"""
    from packages.evaltools.probe_ranking import _matches_gold

    sibling = _hit(_SECTION, "`@EnableTransactionManagement` scanning scope ...")
    real = _hit(_SECTION, f"NOTE: In proxy mode (which is the default), {_SELF_INVOCATION}. "
                          "This means that self-invocation ...")

    assert _matches_gold(sibling, _SECTION, "spring-framework",
                         exact=True, contains=_SELF_INVOCATION) is False
    assert _matches_gold(real, _SECTION, "spring-framework",
                         exact=True, contains=_SELF_INVOCATION) is True


def test_gold_contains_must_accompany_a_title_path_gold():
    """纪律锁死：contains 只做同小节内的定位，不得单独拿关键词圈 gold。

    单独用关键词圈 gold 会把上百块全算对（`ranking_probe.yaml` 开头的禁令）。
    """
    spec = yaml.safe_load(PROBES.read_text(encoding="utf-8"))
    offenders = [p["q"] for p in spec["probes"]
                 if p.get("gold_contains") and not str(p.get("gold", "")).strip()]
    assert not offenders, f"gold_contains 必须与 title_path gold 合用: {offenders}"


# ---------- CR-085：地板基线 not_in_candidates ----------

def test_floor_baseline_does_not_report_regression_when_still_absent():
    """已经在地板上，没有更差的状态可退——这不是豁免，是没有可判的退步。"""
    regressed, below = evaluate(
        [mk(None, baseline=NOT_IN_CANDIDATES, known_open=True)], TOP_K)
    assert regressed == []
    assert below == []


def test_floor_baseline_without_known_open_still_fails():
    """地板基线不豁免 top_k 那道闸：不写 known_open 照样让命令失败。

    判别性：这正是它与 `baseline: null` 的区别——null 是两道闸同时打开
    （CR-015），地板基线只关掉"退步"这一道，因为那一道本来就无从判起。
    """
    regressed, below = evaluate(
        [mk(None, baseline=NOT_IN_CANDIDATES, known_open=False)], TOP_K)
    assert regressed == []
    assert len(below) == 1


def test_floor_baseline_probe_must_declare_known_open():
    """YAML 里用地板基线就必须配 known_open，否则等于悄悄留一条永远红的探针。"""
    spec = yaml.safe_load(PROBES.read_text(encoding="utf-8"))
    offenders = [p["q"] for p in spec["probes"]
                 if p.get("baseline") == NOT_IN_CANDIDATES and not p.get("known_open")]
    assert not offenders, f"地板基线必须配 known_open: {offenders}"


def test_numeric_baseline_still_catches_falling_out_of_candidates():
    """加了地板取值之后，数字基线的退步判据必须原样有效（防止改坏 CR-015 的修复）。"""
    regressed, _ = evaluate([mk(None, baseline=6, known_open=True)], TOP_K)
    assert len(regressed) == 1


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
