"""RRF 那段活跃注释里的算式，做文本级断言（CR-101）。

为什么单独一条：`test_rrf_arithmetic.py` 钉的是 `rrf_fuse()` 的**返回值**，
它**不读注释**。R88 我却写下"注释会错，纯函数不会"，把纯函数回归说成了
注释的保护伞——那是一句没有实现支撑的承诺，CR-101 直接点了出来。

要么删掉那句承诺，要么让它成真。这里选后者：把"活跃注释必须写对算式、
错式只能出现在明确标注的留痕块里"变成可执行断言。

判定方式刻意简单（只查字符串），因为它要防的正是散文里的手误，
不需要解析语义；写复杂了反而会自己长出 bug。
"""

from __future__ import annotations

import re
from pathlib import Path

SEARCH_PY = Path(__file__).resolve().parents[2] / "services" / "retrieval" / "search.py"

# 历史错式。CR-096 指出过两处错：分母（第 60 名应是 60+60=120，不是 121）
# 与等号（应为小于）。
_WRONG_FORMS = ("2/(60+61)", "2/121")
_TRACE_BEGIN = "RRF-TRACE-BEGIN"
_TRACE_END = "RRF-TRACE-END"


def _source() -> str:
    return SEARCH_PY.read_text(encoding="utf-8")


def _split_trace(text: str) -> tuple[str, str]:
    """拆成（留痕块之外的正文, 留痕块）。留痕块允许引用错式，正文不允许。"""
    begin, end = text.index(_TRACE_BEGIN), text.index(_TRACE_END)
    return text[:begin] + text[end:], text[begin:end]


# 完整等式，按"折叠空白后整体匹配"来判（CR-103）。
#
# 第一版是分别查 `"2/(60+60)" in text` 与 `"2/120" in text`，审查方当场复现了
# 它的洞：把等式改写成"2/(60+60) **与** 2/120"——等号没了、两个片段都还在，
# 测试照样全绿。**分别断言片段，证明不了它们之间的关系**，而这次事故的
# 全部内容恰恰就是关系写错了（`==` 该是 `<`，分母该是 60+60）。
_ACTIVE_EQUATIONS = (
    (r"1/\(60\+1\)\s*=\s*1/61\s*=\s*0\.016393", "一路第 1：1/(60+1) = 1/61 = 0.016393…"),
    (r"2/\(60\+60\)\s*=\s*2/120\s*=\s*0\.016666", "两路都第 60：2/(60+60) = 2/120 = 0.016666…"),
)


def _collapsed(text: str) -> str:
    """折叠空白，好让断言不依赖注释里的对齐方式。"""
    return re.sub(r"[ \t]+", " ", text)


def test_active_comment_states_the_complete_equations():
    outside = _collapsed(_split_trace(_source())[0])
    for pattern, human in _ACTIVE_EQUATIONS:
        assert re.search(pattern, outside), f"活跃注释里没有完整写出：{human}"


def test_the_higher_score_is_marked_on_the_two_path_line():
    """哪一边更高必须标在**同一行**上——分开写就又变成"两个片段"了。"""
    for line in _collapsed(_split_trace(_source())[0]).splitlines():
        if re.search(r"2/\(60\+60\)", line):
            assert "← 更高" in line, f"这一行应标出它更高: {line.strip()}"
            break
    else:
        raise AssertionError("活跃注释里找不到两路都第 60 的那一行")


def test_wrong_forms_appear_only_inside_the_marked_trace_block():
    """错式可以留档（它是这次事故的证据），但只能待在留痕块里。

    判别性：把任一错式搬到留痕块外面，这条立刻失败——本轮写它时就是先把
    `2/121` 复制到活跃段落里跑了一次红，再删掉的。
    """
    outside, trace = _split_trace(_source())
    leaked = [w for w in _WRONG_FORMS if w in outside]
    assert not leaked, f"错式泄漏到活跃注释里了: {leaked}"
    assert any(w in trace for w in _WRONG_FORMS), (
        "留痕块里应当保留原始错式，否则日后无从对照这次事故"
    )


def test_active_comment_does_not_equate_the_two_scores():
    """CR-096 的另一半：两者是**小于**关系，不是相等。

    只查活跃注释里有没有把两个算式用 `==` 连起来——这正是原句的形状
    （`1/(60+1) == 2/(60+61)`）。
    """
    outside, _ = _split_trace(_source())
    assert not re.search(r"1/\(60\+1\)\s*==", outside), "活跃注释不得把两者写成相等"
    assert "← 更高" in outside or "<" in outside, "活跃注释应当标出哪一边更高"


def test_the_trace_markers_are_a_matched_pair():
    text = _source()
    assert text.count(_TRACE_BEGIN) == 1 and text.count(_TRACE_END) == 1
    assert text.index(_TRACE_BEGIN) < text.index(_TRACE_END)
