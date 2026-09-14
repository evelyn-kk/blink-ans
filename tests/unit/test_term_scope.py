"""内容例外举证工具的体检（CR-104，R92 按 CR-106/107/108 重写）。

这个工具的作用是把"范围有限"从一句话变成三件可核对的产出。R91 的第一版有
三条可绕过路径，都被 codex 复现了，因此这里对每一条都留一个判别性用例：

- **CR-106**：只看新词典的命中 → **删除/遮蔽**一条映射时会被误报成零影响。
  现在比较旧/新 `expand_terms()` 差分，删除同样可见。
- **CR-107**：`--expect ''` 前缀匹配可通配一切并退出 0。现在只接受精确题面
  + 完整 added/removed，逐词比对。
- **CR-108**：`--since HEAD~1` 会漂移（CR-093 已定过"固定 SHA"的规矩）。
  现在拒绝非 SHA，并把解析后的完整 commit 写进产物。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "packages" / "evaltools"))

import term_scope as TS  # noqa: E402

R85_SHA = "2cbe470"          # R85 改 term_map 之前的那个提交（固定 SHA，CR-093）
R85_Q = "已经建了包含查询所有列的覆盖索引，为什么执行时还是要回表访问堆，没有变快"
EXPECT_FILE = ROOT / "bench/audits/term-map-r85-expect.yaml"


# ---------- CR-108：--since 必须是固定 SHA ----------

@pytest.mark.parametrize("ref", ["HEAD", "HEAD~1", "main", "v1.0", "HEAD^"])
def test_drifting_refs_are_rejected(ref):
    with pytest.raises(TS.ScopeError, match="固定 SHA"):
        TS.resolve_sha(ref)


def test_a_real_sha_resolves_to_the_full_commit():
    full = TS.resolve_sha(R85_SHA)
    assert len(full) == 40 and full.startswith(R85_SHA)


def test_json_product_records_the_resolved_commit(tmp_path, monkeypatch, capsys):
    out = tmp_path / "scope.json"
    monkeypatch.setattr(sys, "argv", [
        "term_scope.py", "--since", R85_SHA,
        "--expect-file", str(EXPECT_FILE),
        "--negatives", "JVM 堆内存怎么调",
        "--json", str(out),
    ])
    assert TS.main() == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert len(data["since_resolved"]) == 40, "产物里必须留完整 commit，短 SHA 日后可能歧义"


# ---------- CR-106：删除/遮蔽必须可见 ----------

def test_deleting_a_mapping_is_reported_as_impact():
    """判别性：只看新词典命中时，删除一条映射后该题"不再命中"，
    旧实现因此报零影响；差分实现必须把它记成 removed。
    """
    old = {"覆盖索引": ["covering index", "index-only scan"]}
    diff = TS.expansion_diff(old, {}, [R85_Q])
    assert R85_Q in diff, "删除映射必须算作受影响"
    assert diff[R85_Q]["removed"] == ["covering index", "index-only scan"]
    assert diff[R85_Q]["added"] == []


def test_shadowing_change_is_visible_even_when_key_count_is_unchanged():
    """更隐蔽的一种：键数没变、但"最具体者胜"的遮蔽关系变了。

    旧词典里具体键 `覆盖索引` 遮住通用键 `索引`；把具体键删掉、通用键保留，
    键差分只看到"删了一个键"，而真正的效果是**通用键的展开开始生效**。
    差分实现把两侧都报出来。
    """
    old = {"索引": ["index"], "覆盖索引": ["covering index"]}
    new = {"索引": ["index"]}
    diff = TS.expansion_diff(old, new, [R85_Q])
    assert diff[R85_Q]["removed"] == ["covering index"]
    assert diff[R85_Q]["added"] == ["index"], "具体键被删后，通用键的展开应当浮现"


def test_no_change_means_no_affected_questions():
    terms = {"覆盖索引": ["covering index"]}
    assert TS.expansion_diff(terms, terms, [R85_Q]) == {}


# ---------- CR-107：预期必须精确 ----------

def test_empty_or_blank_question_in_expectations_is_rejected(tmp_path):
    for bad in ("", "   "):
        p = tmp_path / "e.yaml"
        p.write_text(yaml.safe_dump({"expect": [{"q": bad}]}, allow_unicode=True), encoding="utf-8")
        with pytest.raises(TS.ScopeError, match="精确题面"):
            TS.load_expectations(p)


def test_duplicate_questions_in_expectations_are_rejected(tmp_path):
    p = tmp_path / "e.yaml"
    p.write_text(yaml.safe_dump({"expect": [{"q": R85_Q}, {"q": R85_Q}]}, allow_unicode=True),
                 encoding="utf-8")
    with pytest.raises(TS.ScopeError, match="题面重复"):
        TS.load_expectations(p)


def test_expectation_must_match_the_exact_registered_question():
    """前缀不再被接受：CR-107 复现的 `--expect ''` 通配路径由此关闭。"""
    actual = {R85_Q: {"added": ["covering index"], "removed": []}}
    problems = TS.compare([{"q": "已经建了", "added": ["covering index"], "removed": []}],
                          actual, {R85_Q})
    assert any("不在任何已登记评测集中" in p for p in problems)


def test_expectation_compares_word_by_word():
    actual = {R85_Q: {"added": ["covering index", "index-only scan"], "removed": []}}
    problems = TS.compare([{"q": R85_Q, "added": ["covering index"], "removed": []}],
                          actual, {R85_Q})
    assert any("added 对不上" in p for p in problems)


def test_unexpected_affected_question_is_reported():
    actual = {R85_Q: {"added": ["covering index"], "removed": []}}
    assert any("没写进预期" in p for p in TS.compare([], actual, {R85_Q}))


# ---------- 三件产出齐全才通过 ----------

def test_missing_expect_file_is_not_a_pass(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [
        "term_scope.py", "--since", R85_SHA, "--negatives", "JVM 堆内存怎么调"])
    assert TS.main() == 1
    assert "没有事先声明预期" in capsys.readouterr().out


def test_missing_negatives_is_not_a_pass(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [
        "term_scope.py", "--since", R85_SHA, "--expect-file", str(EXPECT_FILE)])
    assert TS.main() == 1
    assert "举证不完整" in capsys.readouterr().out


def test_a_negative_whose_expansion_changes_fails(monkeypatch, capsys):
    """拿一条**会**受影响的提问当负例：工具若不检查负例就会静默通过。"""
    monkeypatch.setattr(sys, "argv", [
        "term_scope.py", "--since", R85_SHA, "--expect-file", str(EXPECT_FILE),
        "--negatives", R85_Q])
    assert TS.main() == 1
    assert "范围不是你以为的那个" in capsys.readouterr().out


def test_complete_evidence_passes(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [
        "term_scope.py", "--since", R85_SHA, "--expect-file", str(EXPECT_FILE),
        "--negatives", "Kafka 消息堆积了怎么办", "JVM 堆内存怎么调",
        "索引扫描和顺序扫描怎么选", "事务回滚之后表里的数据还在吗"])
    assert TS.main() == 0
    out = capsys.readouterr().out
    assert "三件产出齐了" in out and "本工具不下这个结论" in out


def test_all_eval_questions_covers_every_registered_set():
    labels = {label for label, _ in TS.all_eval_questions()}
    assert labels == {"basic", "scenario", "probe", "validation"}
