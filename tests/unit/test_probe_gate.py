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
    regressed, below, needs = evaluate([mk(6, baseline=1)], TOP_K)
    assert len(regressed) == 1


def test_falling_out_of_candidates_is_a_regression():
    regressed, _, _ = evaluate([mk(None, baseline=1)], TOP_K)
    assert len(regressed) == 1


def test_known_open_gap_does_not_fail_when_stable():
    """既有缺口停在基线上不算退步，否则每次改动都被同一批红叉淹没。"""
    regressed, below, needs = evaluate([mk(11, baseline=11, known_open=True)], TOP_K)
    assert not regressed and not below


def test_known_open_gap_still_fails_when_it_gets_worse():
    regressed, _, _ = evaluate([mk(20, baseline=11, known_open=True)], TOP_K)
    assert len(regressed) == 1


def test_missing_baseline_still_gated_by_top_k():
    """**CR-015 的核心**：没有基线的题不能因此免检。

    旧实现只看退步，`baseline is None` 时整条判据跳过，
    刚修好的题恰好失去门禁。
    """
    regressed, below, needs = evaluate([mk(9, baseline=None)], TOP_K)
    assert not regressed          # 无基线，谈不上退步
    assert len(below) == 1        # 但没进 top_k，必须失败


def test_passing_probe_fails_nothing():
    regressed, below, needs = evaluate([mk(1, baseline=1)], TOP_K)
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
    """已经在地板上，没有更差的状态可退——这不是豁免，是没有可判的退步。

    这是地板基线**唯一**允许静默通过的情形（仍然缺席）。
    """
    regressed, below, needs = evaluate(
        [mk(None, baseline=NOT_IN_CANDIDATES, known_open=True)], TOP_K)
    assert regressed == []
    assert below == []
    assert needs == []


def _r76_verdict(results, top_k):
    """R76 那版 `evaluate` 的判据，逐字复刻（CR-087 的对照基准）。

    与 `_r75_verdict` 同一个用途、同一条理由：把实现换回旧版跑这些用例，
    失败原因会是别的（这次连形状都没变，更需要真对照）。
    这些复刻件是**冻结的历史快照**，故意不随实现演进——它们的职责只是
    回答"旧判据当时对这个输入怎么判"。
    """
    def _breached_floor(r):
        return r.baseline == NOT_IN_CANDIDATES and r.rank is not None

    def _improved_numeric(r):
        return isinstance(r.baseline, int) and r.rank is not None and r.rank < r.baseline

    regressed = [r for r in results
                 if isinstance(r.baseline, int) and (r.rank is None or r.rank > r.baseline)]
    below = [r for r in results if not r.passed and not r.known_open]
    needs = [r for r in results if _breached_floor(r) or _improved_numeric(r)]
    return 1 if (regressed or below or needs) else 0


def _r75_verdict(results, top_k):
    """R75 那版 `evaluate` 的判据，逐字复刻在这里。

    为什么要复刻：把 `probe_ranking.py` 换回旧版跑下面几条用例，失败原因是
    `evaluate` 只返回两元组（解包 ValueError），那属于 §5.2 说的**弱判别性**
    ——因为形状不同而失败，不是因为判断不同。复刻一份不依赖新代码的旧判据，
    才能把"旧判据放它过、新判据拦下来"这个真正的差别写成断言。
    """
    regressed = [r for r in results
                 if isinstance(r.baseline, int) and (r.rank is None or r.rank > r.baseline)]
    below = [r for r in results if not r.passed and not r.known_open]
    return 1 if (regressed or below) else 0


# CR-086：地板值被突破必须非零退出，否则它就是个什么都不判的后门。
# 这四个名次正是审查方独立构造出来、旧实现全部静默通过的那组。
@pytest.mark.parametrize("rank", [20, 6, 5, 1])
def test_floor_baseline_entering_candidates_must_fail(rank):
    """从"未进候选"改善到任意名次，都必须报出来并强制固化数字基线。

    这不是退步，但门禁描述的现实已经变了，不固化就等于永久失去信号。
    """
    probe = mk(rank, baseline=NOT_IN_CANDIDATES, known_open=True)

    # 真正的判别性对照，不依赖新代码的形状：同一个输入，旧判据判过、新判据判失败。
    assert _r75_verdict([probe], TOP_K) == 0, f"R75 那版对第 {rank} 名确实是静默通过的"

    regressed, below, needs = evaluate([probe], TOP_K)
    assert regressed == [], "这不是退步"
    assert len(needs) == 1, f"第 {rank} 名必须触发基线固化"
    assert needs[0].rank == rank


def test_floor_baseline_upgraded_to_number_then_catches_falling_back():
    """固化成数字基线之后，再掉回未进候选必须按退步报出来。

    这是 CR-086 要求的闭环后半段：None -> 20 触发固化，
    固化成 `baseline: 20` 之后 20 -> None 必须再次失败。
    """
    regressed, below, needs = evaluate(
        [mk(None, baseline=20, known_open=True)], TOP_K)
    assert len(regressed) == 1, "数字基线之下掉回未进候选就是退步"
    assert needs == []


def test_numeric_baseline_improvement_must_also_be_fixed():
    """数字基线变好也必须固化——否则从新水平滑回旧基线是**静默**的。

    与地板值那一半是同一个洞（超出 CR-086 字面范围，一并补上）：
    `baseline=5, rank=1` 在旧判据下同样退出 0。
    """
    probe = mk(1, baseline=5, known_open=False)
    assert _r75_verdict([probe], TOP_K) == 0, "旧判据对 5 -> 1 是静默通过的"
    regressed, below, needs = evaluate([probe], TOP_K)
    assert regressed == [] and below == []
    assert len(needs) == 1 and needs[0].rank == 1


# CR-087：`baseline: null` 的语义是「从未测过」，而这一跑就测到了。
# 不当场固化，就会留下 1 -> 5 全程无信号的后门（两名都在 top_k 内）。
@pytest.mark.parametrize("rank", [1, 5, 6, 20])
def test_null_baseline_with_a_measured_rank_must_be_fixed(rank):
    probe = mk(rank, baseline=None, known_open=False)

    # 真行为对照：R76 那版对 1 / 5 两格是静默通过的（6 / 20 则由 below 判失败）。
    expected_old = 0 if rank <= TOP_K else 1
    assert _r76_verdict([probe], TOP_K) == expected_old

    regressed, below, needs = evaluate([probe], TOP_K)
    assert len(needs) == 1, f"null 基线测到第 {rank} 名必须要求固化"
    assert needs[0].rank == rank


def test_null_baseline_backdoor_closed_end_to_end():
    """CR-087 点名的闭环：null 首次测到第 1 名要求固化，固化成 1 之后跌到 5 报退步。"""
    # 第一步：null + 第 1 名 —— R76 静默，现在要求固化。
    first = mk(1, baseline=None, known_open=False)
    assert _r76_verdict([first], TOP_K) == 0, "这正是 CR-087 复现出来的那格"
    _, _, needs = evaluate([first], TOP_K)
    assert len(needs) == 1 and needs[0].rank == 1

    # 第二步：按要求把 baseline 固化成 1，此后跌到第 5 名 —— 必须报退步。
    later = mk(5, baseline=1, known_open=False)
    regressed, _, needs_after = evaluate([later], TOP_K)
    assert len(regressed) == 1, "固化之后 1 -> 5 必须有信号"
    assert needs_after == []


def test_null_baseline_still_absent_is_left_to_the_yaml_layer():
    """null + 仍未进候选：非 known_open 由 below 判失败；

    known_open 的那一格 `evaluate` 确实不判，靠 YAML 层禁止这个组合本身
    （见 test_no_probe_uses_null_baseline_as_a_backdoor）。这是 R76 那句
    "防线在 YAML 层"**唯一**成立的一格——CR-087 指出它当时被过度推广到了 7 格。
    """
    _, below, needs = evaluate([mk(None, baseline=None, known_open=False)], TOP_K)
    assert len(below) == 1 and needs == []
    assert any(evaluate([mk(None, baseline=None, known_open=True)], TOP_K)) is False


def test_baseline_equal_to_measured_rank_is_silent():
    """名次与基线相符是唯一该安静的常态，别把它也判成失败。"""
    assert any(evaluate([mk(5, baseline=5)], TOP_K)) is False
    assert any(evaluate([mk(1, baseline=1)], TOP_K)) is False


# §5.4「列一遍：哪些输入会让它什么都不判」。把整个输入空间钉死，
# 将来任何改动只要多开出一个静默格子，这条就会失败。
def test_which_inputs_make_the_gate_judge_nothing():
    """穷举 (baseline 种类 × rank × known_open)，锁住允许静默的那几格。"""
    silent = set()
    for baseline, tag in [(None, "null"), (NOT_IN_CANDIDATES, "floor"), (5, "num5")]:
        for rank in (None, 1, 5, 6, 20):
            for known_open in (False, True):
                probe = mk(rank, baseline=baseline, known_open=known_open)
                if not any(evaluate([probe], TOP_K)):
                    silent.add((tag, rank, known_open))

    expected = {
        # baseline: null + known_open + 仍未进候选。**只有这一格**靠 YAML 层
        # 兜底（test_no_probe_uses_null_baseline_as_a_backdoor 禁止
        # null + known_open 这个组合本身）。
        #
        # CR-087 的教训就写在这里：R76 我把 null 的 **7 格**全列成允许静默，
        # 理由写的是"防线在 YAML 层"——**那句话我没核实**。实际上
        # test_no_probe_uses_null_baseline_as_a_backdoor 只禁止
        # null + known_open，test_every_probe_records_a_baseline 只检查
        # `"baseline" not in p`（显式写 `baseline: null` 时键是在的），
        # 两条都放行 `null + known_open=False`。于是用 null 首次测到第 1 名、
        # 此后跌到第 5 名**全程无信号**。现在 null 只要测出名次就必须固化。
        ("null", None, True),
        # 地板值：只剩「仍然缺席」这一格，也就是它本来的语义（CR-086 之后）。
        ("floor", None, True),
        # 数字基线：只有「进了 top_k 且不比基线好」才安静。
        ("num5", 5, False), ("num5", 5, True),
    }
    assert silent == expected, (
        f"静默格子变了。多出来的: {silent - expected}；少掉的: {expected - silent}"
    )


def test_floor_baseline_probe_exit_code_is_nonzero_on_breach():
    """把三类判据合起来看一眼退出码：任一非空都必须失败。"""
    breached_probe = mk(3, baseline=NOT_IN_CANDIDATES, known_open=True)
    stable_probe = mk(None, baseline=NOT_IN_CANDIDATES, known_open=True)

    assert any(evaluate([breached_probe], TOP_K)) is True, "地板被突破 -> 非零"
    assert any(evaluate([stable_probe], TOP_K)) is False, "仍然缺席 -> 零"

    # 旧判据对这两种输入都给 0——这正是 CR-086 说的"什么都不判"。
    assert _r75_verdict([breached_probe], TOP_K) == 0
    assert _r75_verdict([stable_probe], TOP_K) == 0


def test_floor_baseline_without_known_open_still_fails():
    """地板基线不豁免 top_k 那道闸：不写 known_open 照样让命令失败。

    判别性：这正是它与 `baseline: null` 的区别——null 是两道闸同时打开
    （CR-015），地板基线只关掉"退步"这一道，因为那一道本来就无从判起。
    """
    regressed, below, needs = evaluate(
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
    regressed, _, _ = evaluate([mk(None, baseline=6, known_open=True)], TOP_K)
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
