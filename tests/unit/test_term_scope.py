"""内容例外举证工具的体检（CR-104）。

这个工具的作用是把"范围有限"从一句话变成三件可核对的产出。它自己必须满足
两条，否则举证就成了走过场：
  1. **碰撞负例真的被检查**——负例触发时必须报出来（非零退出）；
  2. **缺负例不算通过**——CR-104 的原话是"9 道无关提问不能证明整体范围有限"，
     所以"一条负例都不给"必须判为举证不完整，而不是默认通过。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "packages" / "evaltools"))

import term_scope as TS  # noqa: E402


def test_changed_keys_reports_added_keys_with_their_expansions():
    """以 R85 那次真实改动为样本（固定 SHA，CR-093）。"""
    changed = TS.changed_keys("2cbe470")
    assert set(changed) == {"覆盖索引", "回表", "可见性映射", "索引只扫描", "仅索引扫描"}
    old, new = changed["回表"]
    assert old is None, "这五个键在那次改动里都是新增"
    assert "visibility map" in new


def test_all_eval_questions_covers_every_registered_set():
    labels = {label for label, _ in TS.all_eval_questions()}
    assert labels == {"basic", "scenario", "probe", "validation"}, (
        "四份评测题集都要参与'预期命中'统计，漏一份就等于少看一片触发面"
    )


def test_missing_negatives_is_not_a_pass(capsys, monkeypatch):
    """不给碰撞负例 → 退出码 1（举证不完整）。"""
    monkeypatch.setattr(sys, "argv", ["term_scope.py", "--since", "2cbe470"])
    assert TS.main() == 1
    assert "举证不完整" in capsys.readouterr().out


def test_a_triggered_negative_fails(capsys, monkeypatch):
    """负例被触发 → 退出码 1。这里故意拿一条**会**触发的提问当"负例"，
    它就是判别性本身：工具若不检查负例，这条会静默通过。
    """
    monkeypatch.setattr(sys, "argv", [
        "term_scope.py", "--since", "2cbe470",
        "--negatives", "覆盖索引为什么还要回表",
    ])
    assert TS.main() == 1
    assert "范围不是你以为的那个" in capsys.readouterr().out


def test_clean_negatives_pass(capsys, monkeypatch):
    """三件产出齐全且负例都不触发 → 退出码 0。"""
    monkeypatch.setattr(sys, "argv", [
        "term_scope.py", "--since", "2cbe470",
        "--negatives", "Kafka 消息堆积了怎么办", "JVM 堆内存怎么调",
        "索引扫描和顺序扫描怎么选", "事务回滚之后表里的数据还在吗",
    ])
    assert TS.main() == 0
    out = capsys.readouterr().out
    assert "三件产出齐了" in out
    assert "本工具不下这个结论" in out, "工具不该替审查方判断命中面是否可接受"
