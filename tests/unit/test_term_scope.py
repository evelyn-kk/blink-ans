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

TS._terms_at_orig = TS._terms_at   # 供上面那条"完整举证"用例构造新词典

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
    """产物里必须留完整 commit——短 SHA 日后可能歧义。

    退出码在这条里是 **1**（R85 那批键有 3 个在观测集里没有见证，CR-110），
    产物仍然要写出来：举证不通过时更需要留下当时的实测数据。
    """
    out = tmp_path / "scope.json"
    monkeypatch.setattr(sys, "argv", [
        "term_scope.py", "--since", R85_SHA,
        "--expect-file", str(EXPECT_FILE),
        "--negatives", "JVM 堆内存怎么调",
        "--json", str(out),
    ])
    assert TS.main() == 1
    data = json.loads(out.read_text(encoding="utf-8"))
    assert len(data["since_resolved"]) == 40
    assert set(data["unwitnessed_keys"]) == {"仅索引扫描", "可见性映射", "索引只扫描"}


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


# ---------- CR-110：逐键正向见证 ----------

def test_witness_is_attributed_by_matched_key_not_by_shared_word():
    """判别性：几个键常共享同一个展开词，**按词归因会发假证明**。

    第一版就是按词匹配的：`index-only scan` 同时出现在 `覆盖索引`/`回表`/
    `索引只扫描`/`仅索引扫描` 的展开里，于是一道只命中了前两个键的题，
    被算成四个键的见证。现在按**命中的键**归因。
    """
    old: dict[str, list[str]] = {}
    new = {
        "覆盖索引": ["covering index", "index-only scan"],
        "索引只扫描": ["index-only scan"],          # 与上一个键共享展开词
    }
    w = TS.key_witnesses(old, new, [R85_Q])
    assert w["覆盖索引"] == [R85_Q], "题面里有『覆盖索引』，这个键应当有见证"
    assert w["索引只扫描"] == [], (
        "题面里没有『索引只扫描』，它不该因为共享 index-only scan 就拿到见证"
    )


def test_deleted_key_also_gets_a_witness():
    """见证要覆盖减法：删掉一个原本会命中的键，同样算"有落脚点"。"""
    old = {"覆盖索引": ["covering index"]}
    w = TS.key_witnesses(old, {}, [R85_Q])
    assert w["覆盖索引"] == [R85_Q]


def test_a_key_without_witness_blocks_the_pass(monkeypatch, capsys):
    """R85 那次改动的真实情况：5 个键里 3 个在观测集里没有落脚点 → 退出码 1。

    这条同时是 CR-110 那句裁定的执行：观测集是**固定的 120 道题**，
    没有见证说明的是"不知道它影响什么"，不是"它没有影响"。
    """
    monkeypatch.setattr(sys, "argv", [
        "term_scope.py", "--since", R85_SHA, "--expect-file", str(EXPECT_FILE),
        "--negatives", "JVM 堆内存怎么调"])
    assert TS.main() == 1
    out = capsys.readouterr().out
    assert "没有正向见证" in out
    for key in ("仅索引扫描", "可见性映射", "索引只扫描"):
        assert key in out


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


def test_complete_evidence_passes(monkeypatch, capsys, tmp_path):
    """三件产出齐全、且每个改动键都有见证 → 退出码 0。

    用 R85 那批键里**有见证的那两个**构造场景：`--since` 仍取真实 SHA，但把
    `_terms_at` 的"新词典"换成只含这两个键，其余不变——这样既走完整流程，
    又不依赖"某次历史改动恰好每个键都有见证"。
    """
    real_new = TS._terms_at(None)
    trimmed = {k: v for k, v in real_new.items()
               if k not in ("仅索引扫描", "可见性映射", "索引只扫描")}
    monkeypatch.setattr(TS, "_terms_at",
                        lambda sha: TS.__dict__["_terms_at_orig"](sha) if sha else trimmed)
    monkeypatch.setattr(sys, "argv", [
        "term_scope.py", "--since", R85_SHA, "--expect-file", str(EXPECT_FILE),
        "--negatives", "Kafka 消息堆积了怎么办", "JVM 堆内存怎么调",
        "索引扫描和顺序扫描怎么选", "事务回滚之后表里的数据还在吗"])
    assert TS.main() == 0
    out = capsys.readouterr().out
    assert "三件产出齐了" in out and "本工具不下这个结论" in out


# ---------- `_expansions_under()` 换词典之后必须原样还原 ----------

def test_swapping_dictionaries_restores_the_tokenizer_state():
    """差分实现要把两版词典轮流装进 `tokenize` 模块，跑完必须还原。

    不还原的后果很隐蔽：同一个进程里后续任何一次 `expand_terms()` 都会用着
    临时词典——而 `term_scope` 常常和别的评测脚本跑在同一个会话里。
    这条是自陈薄弱处里登记后当场补掉的（§5.1）。
    """
    from services.retrieval import tokenize as tk

    # 先热一次，确保分词器已加载——否则快照到的是"还没加载"那个瞬间的状态，
    # 断言就变成了和一个陈旧快照比（写这条时先踩了这个坑）。
    before_expand = sorted(tk.expand_terms(R85_Q))
    before_path = tk.TERM_MAP_PATH
    before_ready = tk._ready
    before_map = dict(tk._term_map)
    before_version = tk._version
    before_dict_version = tk.dictionary_version()
    assert before_ready is True, "热身之后分词器应当已加载"

    TS.expansion_diff({"覆盖索引": ["covering index"]}, {}, [R85_Q])

    assert tk.TERM_MAP_PATH == before_path, "词典路径没还原"
    assert tk._ready == before_ready
    assert dict(tk._term_map) == before_map, "词典内容没还原"
    assert sorted(tk.expand_terms(R85_Q)) == before_expand, (
        "跑完差分之后，真实词典下的展开结果必须与跑之前一致"
    )
    # CR-109：上一版就漏在这里——`_version` 没还原，`dictionary_version()`
    # 会停在临时 YAML 的版本上，而它正是"索引词典版本必须与查询侧一致"
    # 那道护栏的依据（`store.ChunkStore`）。跑完一次差分就能让后面任何一次
    # 开索引报假的不一致。
    assert tk._version == before_version, "_version 没还原"
    assert tk.dictionary_version() == before_dict_version, (
        "dictionary_version() 必须回到真实词典的版本，否则索引一致性护栏会误报"
    )


def test_cleanup_does_not_delete_words_jieba_already_knew():
    """清理临时词时不能误删 jieba 本来就认识的词。

    **这条是实测踩出来的**：第一版 `finally` 里对每个临时键无条件
    `del_word`，于是临时词典里放一个 `索引`（jieba 原本认识）跑完之后，
    `tests/unit/test_tokenize.py` 里一条切词用例直接挂了——工具把公共分词器
    改坏了。现在按词记住进来之前的词频，跑完原样放回。
    """
    from services.retrieval import tokenize as tk

    # 复刻当时真正挂掉的那条断言（tests/unit/test_tokenize.py 的 CR-013 用例）：
    # 它依赖"组成词全部出现"这条规则，而组成词是**切出来**的——jieba 少认识
    # 一个词，这条就崩。只比 tokenize() 的输出不够灵敏，要比到 matched_terms。
    probe = "PostgreSQL 的 B-tree 索引什么时候会失效"
    before = sorted(tk.matched_terms(probe))
    assert "索引失效" in before, "前置条件：真实词典下这条本来就该命中"

    TS.expansion_diff({"索引": ["index"]}, {}, [R85_Q])

    assert sorted(tk.matched_terms(probe)) == before, (
        "jieba 原本认识的词被清理逻辑删掉了，组成词匹配随之失效"
    )


def test_temporary_dictionary_does_not_leak_custom_words_into_jieba(capsys):
    """临时词典引入的自定义词必须从 jieba 里删掉（CR-109 的同源问题）。

    `_load()` 会把每个键 `jieba.add_word`，那是**全局副作用**：不清理的话，
    同进程后续的分词会认得一个本不存在的词，`matched_terms()` 的"组成词全部
    出现"规则因此可能改判。
    """
    from services.retrieval import tokenize as tk

    probe = "临时造的词到底会不会被切出来"
    before = tk.tokenize(probe)
    TS.expansion_diff({"临时造的词": ["temp"]}, {}, [R85_Q])
    assert tk.tokenize(probe) == before, "临时词典的自定义词泄漏进了 jieba"


def test_all_eval_questions_covers_every_registered_set():
    labels = {label for label, _ in TS.all_eval_questions()}
    assert labels == {"basic", "scenario", "probe", "validation"}
