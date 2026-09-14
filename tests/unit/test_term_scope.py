"""内容例外举证工具的体检（CR-104，R92 按 CR-106/107/108 重写）。

这个工具的作用是把"范围有限"从一句话变成三件可核对的产出。R91 的第一版有
三条可绕过路径，都被 codex 复现了，因此这里对每一条都留一个判别性用例：

- **CR-106**：只看新词典的命中 → **删除/遮蔽**一条映射时会被误报成零影响。
  现在比较旧/新 `expand_terms()` 差分，删除同样可见。
- **CR-107**：`--expect ''` 前缀匹配可通配一切并退出 0。现在只接受精确题面
  + 完整 added/removed，逐词比对。
- **CR-108**：`--since HEAD~1` 会漂移（CR-093 已定过"固定 SHA"的规矩）。
  现在拒绝非 SHA，并把解析后的完整 commit 写进产物。
- **CR-111**：R93 的差分在同一个进程里换词典再还原，**两版词典其实互相看得见**
  ——`_load()` 往 jieba 的全局前缀树里 `add_word`，还原清单管不到。现在两版
  各自在短生命周期子进程里算，判别性用例见文件末尾。
- **CR-112**：R94 声称"四次调用只落成两次子进程"，实测是 4 次——缓存不会把
  两批题合成一批。现在主动并批，并用**起了几个 worker** 这个计数钉住。
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
    assert all(len(qs) == len(set(qs)) for qs in data["key_witnesses"].values()), (
        "见证列表里出现了重复题面——观测面被数了两遍"
    )


def test_the_observation_set_is_deduplicated(monkeypatch, capsys):
    """登记题面里有重复（同一道题同时登记在两个评测集里），观测面必须按去重算。

    R93 存档的产物里 `回表`/`覆盖索引` 各写着"2 道见证"，其实是同一道题
    被列了两遍——不是假证明（CR-110 那种），但会把观测面说大一倍。
    """
    labelled = TS.all_eval_questions()
    distinct = len({q for _, q in labelled})
    assert distinct < len(labelled), "前置条件：登记题面里确实有重复"

    monkeypatch.setattr(sys, "argv", [
        "term_scope.py", "--since", R85_SHA, "--expect-file", str(EXPECT_FILE),
        "--negatives", "JVM 堆内存怎么调"])
    TS.main()
    assert f"登记 {len(labelled)} 条题面、去重后 {distinct} 道" in capsys.readouterr().out


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


# ---------- CR-111：两版词典必须互不可见 ----------

def test_old_side_does_not_see_words_that_only_the_new_dictionary_adds():
    """判别性：**新词典独有的键泄漏到旧侧**时，差分会漏报。

    R93 的实现在本进程里换词典再还原。还原清单管得住 `tokenize` 的模块状态，
    管不住 jieba：`_load()` 对每个键 `add_word(freq=900)`，改的是第三方库的
    全局前缀树，而清单只记得住"我们知道要记"的那些词。于是主进程一旦加载过
    当前词典（`dictionary_version()`、任何一次 `expand_terms()` 都会），
    当前独有的键就一直在 jieba 里，**以旧词典运行时也还在**。

    下面这组是能把后果显出来的最小构造：`同步机制` 是当前词典里的真实键，
    只出现在新侧；它在 jieba 里会让提问多切出一个 `同步机`，而那正是
    `同步机故障` 这个键的组成词之一（`_split_key` 切成 `同步机` + `故障`）。

    - 隔离之后：旧侧没有 `同步机` → `同步机故障` 不命中；新侧命中 →
      `added` 里两个键的展开都在。
    - R93 的进程内实现：旧侧因为 jieba 里残留的 `同步机制` **也**命中了
      `同步机故障`，两侧抵消 → `added` 只剩 `replication`，`sync failure`
      被静默吞掉。

    键名是为这条用例构造的（要让贪心切分正好落在跨词边界的三字词上），
    但机制不是：实测加载当前词典后以 R85 旧词典运行，120 道登记题里有
    19 道的 `tokenize()` 输出与干净进程不同。
    """
    from services.retrieval import tokenize as tk

    # 先把真实词典装进本进程——这正是污染的来源。不热身的话，跑在别的用例
    # 前面时 jieba 里恰好没有 `同步机制`，旧实现也能蒙混过关。
    tk.dictionary_version()

    q = "Kafka 分区副本的同步机制故障怎么办"
    old = {"同步机故障": ["sync failure"]}
    new = {"同步机故障": ["sync failure"], "同步机制": ["replication"]}

    diff = TS.expansion_diff(old, new, [q])
    assert diff[q]["added"] == ["replication", "sync failure"], (
        "旧侧看见了只属于新词典的 `同步机制`，`sync failure` 因此被抵消掉"
    )
    assert TS.key_witnesses(old, new, [q]) == {"同步机制": [q]}


# ---------- 主进程的分词器状态：现在是"压根没碰过" ----------

def test_the_diff_does_not_touch_the_process_tokenizer_at_all():
    """CR-109 那批还原断言的继任者：不再是"还原得对不对"，而是"有没有碰"。

    换成子进程之后，主进程不再改 `tokenize` 的模块状态，也不再改 jieba，
    因此可以直接断言**全等**——比逐项还原强，也不用再维护还原清单
    （CR-109 就是漏在清单少了一项 `_version` 上）。
    """
    import jieba

    from services.retrieval import tokenize as tk

    # 先热一次：否则快照到的是"还没加载"那个瞬间，断言就成了和陈旧快照比。
    before_expand = sorted(tk.expand_terms(R85_Q))
    before_dict_version = tk.dictionary_version()
    before_module = (tk.TERM_MAP_PATH, tk._ready, tk._version,
                     dict(tk._term_map), dict(tk._KEY_PARTS))
    before_freq = dict(jieba.dt.FREQ)

    TS.expansion_diff({"覆盖索引": ["covering index"]}, {}, [R85_Q])

    assert (tk.TERM_MAP_PATH, tk._ready, tk._version,
            dict(tk._term_map), dict(tk._KEY_PARTS)) == before_module
    assert jieba.dt.FREQ == before_freq, "jieba 的全局词频被改了"
    # CR-109 的两个外部可观测量仍然逐条断言：它们才是当时真正出事的地方
    # （`dictionary_version()` 是"索引词典版本与查询侧一致"那道护栏的依据）。
    assert tk.dictionary_version() == before_dict_version
    assert sorted(tk.expand_terms(R85_Q)) == before_expand


def test_public_tokenizer_still_works_after_a_diff():
    """CR-109 第二个问题的回归：那次是清理逻辑把 jieba 本来认识的词删掉了，
    `matched_terms()` 的"组成词全部出现"规则随之失效
    （`tests/unit/test_tokenize.py` 的 CR-013 用例当场挂掉）。

    隔离之后这条按理不可能再犯，但断言留着：它比字段自比灵敏，
    盯的是"公共分词器还能不能用"这个外部可观测量。
    """
    from services.retrieval import tokenize as tk

    probe = "PostgreSQL 的 B-tree 索引什么时候会失效"
    leak_probe = "临时造的词到底会不会被切出来"
    before_matched = sorted(tk.matched_terms(probe))
    before_tokens = tk.tokenize(leak_probe)
    assert "索引失效" in before_matched, "前置条件：真实词典下这条本来就该命中"

    TS.expansion_diff({"索引": ["index"]}, {}, [R85_Q])          # 会误删 jieba 原有词的那种
    TS.expansion_diff({"临时造的词": ["temp"]}, {}, [R85_Q])      # 会把自定义词泄漏进去的那种

    assert sorted(tk.matched_terms(probe)) == before_matched
    assert tk.tokenize(leak_probe) == before_tokens, "临时词典的自定义词泄漏进了 jieba"


def test_a_full_run_starts_one_worker_per_dictionary_version(monkeypatch):
    """CR-112：缓存**不会**自己把两批题合成一次子进程，得主动并批。

    R94 我在文档里写了"旧/新 × 观测集/负例四次调用只落成两次子进程"，
    但没有任何东西钉住这句话——审查方一数，实际是 **4 次**：负例是另一批题，
    缓存只能各自命中各自那批，两版词典于是各起了两个 worker。

    子进程的固定开销（起 Python + 载 jieba 词典）约 0.5 秒，是这里唯一值钱的
    东西，所以判据按**版本数**写死：一版词典一个 worker，与调用它几次无关。
    """
    import term_scope

    spawns: list[list[str]] = []
    real_run = term_scope.subprocess.run

    def counting(cmd, *a, **kw):
        if len(cmd) > 1 and cmd[1] == str(term_scope.WORKER):
            spawns.append(cmd)
        return real_run(cmd, *a, **kw)

    monkeypatch.setattr(term_scope.subprocess, "run", counting)
    TS._CACHE.clear()          # 别让别的用例先把结果算好了，那样一次都不会起
    monkeypatch.setattr(sys, "argv", [
        "term_scope.py", "--since", R85_SHA, "--expect-file", str(EXPECT_FILE),
        "--negatives", "JVM 堆内存怎么调", "Kafka 消息堆积了怎么办"])
    TS.main()

    assert len(spawns) == 2, (
        f"旧/新两版词典应当各起一个 worker，实际起了 {len(spawns)} 次"
    )


def test_worker_failure_is_an_evidence_error_not_a_silent_zero():
    """子进程算不出来时必须炸，不能当成"零影响"——那是 CR-106 同一类静默通过。"""
    import term_scope

    orig = term_scope.WORKER
    try:
        term_scope.WORKER = ROOT / "packages" / "evaltools" / "no-such-worker.py"
        TS._CACHE.clear()
        with pytest.raises(TS.ScopeError, match="隔离子进程"):
            TS.expansion_diff({"覆盖索引": ["covering index"]}, {}, [R85_Q])
    finally:
        term_scope.WORKER = orig
        TS._CACHE.clear()


def test_all_eval_questions_covers_every_registered_set():
    labels = {label for label, _ in TS.all_eval_questions()}
    assert labels == {"basic", "scenario", "probe", "validation"}
