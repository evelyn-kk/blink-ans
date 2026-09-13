"""RRF 融合分的算术，按纯函数钉死（CR-096）。

为什么值得一条独立回归：T-112 那轮的根因说明把
`1/(60+1) == 2/(60+61)` 写成了等式——分母错、等号也错（两路都第 60 名是
`2/120`）。结论方向恰好不受影响，所以它在两轮审查里都没被数值证据挡住，
只能靠人眼读注释发现。**注释会错，纯函数不会**：这里把"一路第 1 vs 两路
都第 60"这组关系变成可执行断言，日后再有人改 RRF 或重写那段注释，
数值对不上会立刻失败。

这些用例不依赖索引、不加载模型，毫秒级。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from services.retrieval.search import RRF_K, rrf_fuse  # noqa: E402


def _only_score(fused: dict) -> float:
    assert len(fused) == 1, fused
    return next(iter(fused.values()))[0]


def _score_of(fused: dict, rowid: int) -> float:
    return fused[rowid][0]


def test_single_path_rank_one_is_one_over_k_plus_one():
    """一路第 1、另一路缺席 = `w/(k+1)`。k=60 时 1/61 ≈ 0.016393。"""
    fused = rrf_fuse([(42, 0.0)], [], k=60)
    assert _score_of(fused, 42) == pytest.approx(1 / 61, rel=1e-12)
    assert _score_of(fused, 42) == pytest.approx(0.016393442622950821, rel=1e-12)


def test_both_paths_rank_sixty_is_two_over_one_hundred_twenty():
    """两路都第 60 = `2w/(k+60)`。k=60 时 2/120 ≈ 0.016667——**不是 2/121**，
    那个分母对应的是第 61 名。这正是 CR-096 指出的错误。
    """
    ranked = [(i, 0.0) for i in range(1, 61)]          # 目标块 rowid=60，排第 60
    fused = rrf_fuse(ranked, ranked, k=60)
    assert _score_of(fused, 60) == pytest.approx(2 / 120, rel=1e-12)
    assert _score_of(fused, 60) != pytest.approx(2 / 121, rel=1e-12)


def test_two_mediocre_paths_beat_one_perfect_path_at_k_sixty():
    """T-112 的根因本身：k=60 下"两路都第 60"压过"一路第 1"。

    这条是结论方向，与上面两条数值分开写——CR-096 的错误恰恰是**结论对、
    算术错**，两者必须分别可证伪。
    """
    single = _score_of(rrf_fuse([(1, 0.0)], [], k=60), 1)
    ranked = [(i, 0.0) for i in range(1, 61)]
    both = _score_of(rrf_fuse(ranked, ranked, k=60), 60)
    assert single < both
    assert (both - single) == pytest.approx(2 / 120 - 1 / 61, rel=1e-9)


def test_the_relation_flips_at_small_k():
    """k 变小则反过来——这是当初想调 k 的全部动机，也钉一条，
    免得日后有人读注释读成"k 怎么调都一样"（验证集显示它在真实题上惰性，
    但算术层面的机制是真的）。
    """
    ranked = [(i, 0.0) for i in range(1, 61)]
    for k in (5, 10):
        single = _score_of(rrf_fuse([(1, 0.0)], [], k=k), 1)
        both = _score_of(rrf_fuse(ranked, ranked, k=k), 60)
        assert single > both, f"k={k} 时一路第 1 应当胜出"


def test_production_constant_is_sixty():
    """生产常量当前是 60（R86 那次 10 已按 CR-097 回退）。"""
    assert RRF_K == 60
