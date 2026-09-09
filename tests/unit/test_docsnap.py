"""`infra/scripts/docsnap` 的回归（CR-068）。

为什么单独测：2026-09-08 一次批量替换脚本误用开放式切片，把
`progress.md` 从中部截断到文件末尾，约 1400 行历史记录**不可恢复**——
这九份协作文档全部在 `.gitignore` 里，没有 git 副本。CR-068 明确要求
把"下次小心点"这种操作约束升级成真正的恢复能力，并且这套恢复能力本身
必须是**可演练、可验证**的，而不是写在文档里的一句承诺。

因此这里直接跑脚本自带的 `drill` 子命令：它在临时目录里真实走一遍
"写前快照 → 把文件截断成事故那种样子 → check 必须报错 → restore 必须
逐字节还原"，不碰仓库里的真实文档。这条测试的意义是防止有人改坏
docsnap 之后，恢复能力悄悄失效却没人发现——门禁失效比没有门禁更危险
（`AGENTS.md` §5.4 的同一类教训）。
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOCSNAP = ROOT / "infra" / "scripts" / "docsnap"


def test_docsnap_exists_and_is_executable():
    assert DOCSNAP.exists(), "CR-068 要求的写前快照脚本不存在"
    assert DOCSNAP.stat().st_mode & 0o111, "docsnap 必须可执行"


def test_docsnap_drill_recovers_a_truncated_document():
    """演练：2000 行的文档被截断成 561 行（就是事故的形状）之后，
    `check` 必须判为异常缩水（退出码 1），`restore` 必须逐字节还原。
    """
    r = subprocess.run([str(DOCSNAP), "drill"], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, f"恢复演练失败：\n{r.stdout}\n{r.stderr}"
    assert "check 退出码 1" in r.stdout
    assert "2000 行原样找回" in r.stdout


def test_docsnap_check_flags_a_truncated_file_and_restore_brings_it_back(tmp_path):
    """把演练拆成显式的三步再测一遍：save → 截断 → check 报错 → restore 还原。

    与 `drill` 的区别是这里由测试自己控制每一步，断言的是具体的退出码和
    文件内容，而不是脚本自报的成功信息——避免"脚本说自己通过了"这种
    自证（`AGENTS.md` §4.2"不得引用实现轮的结论当证据"的同一精神）。
    """
    repo, store = tmp_path / "repo", tmp_path / "store"
    repo.mkdir()
    doc = repo / "progress.md"
    original = "\n".join(f"第 {i} 行" for i in range(1, 1001)) + "\n"
    doc.write_text(original, encoding="utf-8")
    env = {
        "DOCSNAP_ROOT": str(repo), "DOCSNAP_STORE": str(store),
        "DOCSNAP_DOCS": "progress.md", "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
    }

    def run(*args):
        return subprocess.run([str(DOCSNAP), *args], capture_output=True, text=True, env=env, timeout=60)

    assert run("save", "before-edit").returncode == 0
    assert run("check").returncode == 0, "刚快照完，check 应该判为无变化"

    doc.write_text("\n".join(f"第 {i} 行" for i in range(1, 201)) + "\n", encoding="utf-8")
    truncated = run("check")
    assert truncated.returncode == 1, "1000 -> 200 行必须被判成异常缩水"
    assert "疑似大段截断" in truncated.stdout + truncated.stderr

    snap = sorted(p.name for p in store.iterdir())[-1]
    assert run("restore", snap, "progress.md").returncode == 0
    assert doc.read_text(encoding="utf-8") == original, "恢复后必须与原文逐字节相同"


def test_docsnap_check_tolerates_normal_growth(tmp_path):
    """判别性的另一半：正常追加内容不能被误报成缩水，否则这条门禁会被
    当成噪声关掉（只会变红的门禁和不会变红的一样没用）。
    """
    repo, store = tmp_path / "repo", tmp_path / "store"
    repo.mkdir()
    doc = repo / "progress.md"
    doc.write_text("\n".join(f"第 {i} 行" for i in range(1, 1001)) + "\n", encoding="utf-8")
    env = {
        "DOCSNAP_ROOT": str(repo), "DOCSNAP_STORE": str(store),
        "DOCSNAP_DOCS": "progress.md", "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
    }
    subprocess.run([str(DOCSNAP), "save"], capture_output=True, env=env, timeout=60)
    with doc.open("a", encoding="utf-8") as f:
        f.write("\n".join(f"新增第 {i} 行" for i in range(1, 51)) + "\n")
    r = subprocess.run([str(DOCSNAP), "check"], capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 0, f"追加 50 行不应报错：\n{r.stdout}\n{r.stderr}"


# ---- CR-069：任何删除都必须非零，有意删减走显式 accept ----
#
# codex R62 复审复现：`check` 原来只在"少 ≥200 行或 ≥20%"时返回 1，
# 于是 1000 行文档中部丢 100 行（10%）走普通分支、返回 0——而那 100 行
# 完全可能正是一段基准或风险记录。这与 CR-068 要建立的收尾门禁自相
# 矛盾：门禁在，但恰好对"不够大的丢失"什么都不判（`AGENTS.md` §5.4
# 那条"门禁不能有豁免后门"的第三个实例）。
#
# 现在的判定：任何行数/字节数减少，或任何"单块净删 ≥10 行"，一律非零；
# 有意删减用 `docsnap accept "理由"` 记录理由并重建基线。


def _mkrepo(tmp_path, lines=1000):
    repo, store = tmp_path / "repo", tmp_path / "store"
    repo.mkdir()
    doc = repo / "progress.md"
    doc.write_text("\n".join(f"第 {i} 行" for i in range(1, lines + 1)) + "\n", encoding="utf-8")
    env = {
        "DOCSNAP_ROOT": str(repo), "DOCSNAP_STORE": str(store),
        "DOCSNAP_DOCS": "progress.md", "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
    }

    def run(*args):
        return subprocess.run([str(DOCSNAP), *args], capture_output=True, text=True, env=env, timeout=60)

    run("save", "baseline")
    return doc, store, run


def test_pre_cr069_thresholds_would_have_passed_a_mid_file_deletion():
    """判别性基线：把旧判据（少 <200 行且 <20% 就算正常）内联复现一遍，
    确认 1000 -> 900 行这个具体案例在旧规则下确实返回"正常"——这正是
    审查方复现的漏判，不是我复述它的结论。
    """
    snapshot_lines, current_lines = 1000, 900
    drop = snapshot_lines - current_lines
    ratio = drop * 100 // snapshot_lines
    old_rule_flags = drop >= 200 or ratio >= 20
    assert not old_rule_flags, "旧阈值本应（错误地）把中部丢 100 行判成正常变更"


def test_cr069_mid_file_deletion_is_flagged(tmp_path):
    """新规则：同一个案例必须非零，并且报出"单块最多净删 100 行"。"""
    doc, _store, run = _mkrepo(tmp_path)
    kept = [l for i, l in enumerate(doc.read_text(encoding="utf-8").splitlines(), 1)
            if not (400 <= i <= 499)]
    doc.write_text("\n".join(kept) + "\n", encoding="utf-8")
    r = run("check")
    assert r.returncode == 1, f"1000 -> 900 行必须判非零：\n{r.stdout}\n{r.stderr}"
    assert "真正丢失 100 行" in r.stdout + r.stderr


def test_cr069_deletion_masked_by_appends_is_flagged(tmp_path):
    """更隐蔽的一种：中间删 100 行、末尾追加 300 行，净增长 200 行。
    只比行数的判据看不出来，必须靠 diff 里的整块删除判出来。
    """
    doc, _store, run = _mkrepo(tmp_path)
    kept = [l for i, l in enumerate(doc.read_text(encoding="utf-8").splitlines(), 1)
            if not (400 <= i <= 499)]
    kept += [f"本轮新增第 {i} 行" for i in range(1, 301)]
    doc.write_text("\n".join(kept) + "\n", encoding="utf-8")
    r = run("check")
    assert r.returncode == 1, f"净增长 200 行也必须报出中间那 100 行的删除：\n{r.stdout}\n{r.stderr}"
    assert "真正丢失" in r.stdout + r.stderr


def test_cr069_small_in_place_edit_that_grows_is_still_clean(tmp_path):
    """判据不能严到把普通编辑也拦下来：原地改几行、内容变长，仍应为 0。
    （只会变红的门禁会被人关掉，`AGENTS.md` §5.2 的对偶情形。）
    """
    doc, _store, run = _mkrepo(tmp_path)
    lines = doc.read_text(encoding="utf-8").splitlines()
    for i in (10, 20, 30):
        lines[i] = lines[i] + "：补充一点说明"
    doc.write_text("\n".join(lines) + "\n", encoding="utf-8")
    r = run("check")
    assert r.returncode == 0, f"原地改写且内容变长不应报错：\n{r.stdout}\n{r.stderr}"


def test_cr069_in_place_shrink_is_flagged(tmp_path):
    """反过来：行数没变但内容变短（一行被改短）同样判非零。
    这比 CR-069 明确要求的"行数减少"更严——**有意为之**：内容变短同样是
    内容丢失，只是丢在行内。代价是"把长句改短"这类正常编辑也要走一次
    `accept`，已在 wait-for-review 自陈里登记，供审查方判断是否过严。
    """
    doc, _store, run = _mkrepo(tmp_path)
    lines = doc.read_text(encoding="utf-8").splitlines()
    lines[10] = "短"
    doc.write_text("\n".join(lines) + "\n", encoding="utf-8")
    r = run("check")
    assert r.returncode == 1
    assert "内容变短" in r.stdout + r.stderr


def test_cr069_accept_requires_a_reason(tmp_path):
    """`accept` 是"有记录的基线重建"，不是"把判据关掉"：不给理由就拒绝。"""
    doc, _store, run = _mkrepo(tmp_path)
    doc.write_text("只剩一行\n", encoding="utf-8")
    r = run("accept", "progress.md")
    assert r.returncode == 2
    assert "都必填" in r.stdout + r.stderr
    assert run("check").returncode == 1, "拒绝之后门禁必须仍然是红的"


def test_cr069_accept_records_reason_and_rebaselines(tmp_path):
    """有意删减：accept 之后 check 归零，且理由与新基线都落了盘。"""
    doc, store, run = _mkrepo(tmp_path)
    kept = [l for i, l in enumerate(doc.read_text(encoding="utf-8").splitlines(), 1)
            if not (400 <= i <= 499)]
    doc.write_text("\n".join(kept) + "\n", encoding="utf-8")
    assert run("check").returncode == 1
    r = run("accept", "progress.md", "删掉了 400-499 行那段过时记录，已人工确认")
    assert r.returncode == 0, r.stderr
    assert run("check").returncode == 0, "accept 之后同样的状态不应再报错"
    accepted = list(store.glob("*/ACCEPTED.txt"))
    assert accepted, "接受理由必须落盘存档"
    assert "已人工确认" in accepted[0].read_text(encoding="utf-8")
    # 新基线是"删减后"的内容：能被 restore 回来的必须是当前这份
    latest = sorted(p.name for p in store.iterdir())[-1]
    assert (store / latest / "progress.md").read_text(encoding="utf-8") == doc.read_text(encoding="utf-8")


# ---- CR-070 / CR-071：accept 必须逐文件，hunk 净删不设阈值 ----
#
# codex R63 复审的两条：
# - **CR-070（P1）**：`accept "理由"` 把九份文档的当前内容一起做成新基线，
#   接受 A 的有意删减时 B 的意外截断会被静默洗白，事后也无从追溯。
# - **CR-071（P2）**：`BLOCK_DELETE=10` 是另一条豁免——十处各净删 9 行、
#   再追加足量内容，三条判据全部绕过。
# 两条的共同点与 CR-069 一样：门禁在，却对某一类输入什么都不判。


def _mkrepo2(tmp_path, names=("progress.md", "architecture.md"), lines=1000):
    """两份文档的工作区，用来验证 accept 的作用域。"""
    repo, store = tmp_path / "repo", tmp_path / "store"
    repo.mkdir()
    for n in names:
        (repo / n).write_text(
            "\n".join(f"{n} 第 {i} 行" for i in range(1, lines + 1)) + "\n", encoding="utf-8")
    env = {
        "DOCSNAP_ROOT": str(repo), "DOCSNAP_STORE": str(store),
        "DOCSNAP_DOCS": " ".join(names), "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
    }

    def run(*args):
        return subprocess.run([str(DOCSNAP), *args], capture_output=True, text=True, env=env, timeout=60)

    run("save", "baseline")
    return repo, store, run


def test_cr070_accepting_one_file_does_not_launder_another(tmp_path):
    """接受 A 的有意删减之后，B 的未接受截断必须仍让 check 非零。"""
    repo, _store, run = _mkrepo2(tmp_path)
    a, b = repo / "progress.md", repo / "architecture.md"
    a.write_text("\n".join(
        l for i, l in enumerate(a.read_text(encoding="utf-8").splitlines(), 1)
        if not (300 <= i <= 399)) + "\n", encoding="utf-8")          # A：有意删 100 行
    b.write_text("\n".join(b.read_text(encoding="utf-8").splitlines()[:100]) + "\n",
                 encoding="utf-8")                                     # B：意外截断
    assert run("check").returncode == 1

    r = run("accept", "progress.md", "删掉了 300-399 那段过时记录，已人工确认")
    assert r.returncode == 0, r.stderr
    after = run("check")
    assert after.returncode == 1, "接受 A 之后，B 的截断必须仍然报红"
    assert "architecture.md" in after.stdout + after.stderr
    assert "progress.md" not in [
        line.split()[1] for line in (after.stdout + after.stderr).splitlines()
        if line.strip().startswith("✗") and len(line.split()) > 1
    ], "被接受的 A 不应再报红"


def test_cr070_unaccepted_file_keeps_its_old_baseline(tmp_path):
    """而且 B 在新基线里必须仍是**截断前**的版本——否则事故内容就没了，
    报红也救不回来。
    """
    repo, store, run = _mkrepo2(tmp_path)
    a, b = repo / "progress.md", repo / "architecture.md"
    before_b = b.read_text(encoding="utf-8")
    a.write_text("\n".join(
        l for i, l in enumerate(a.read_text(encoding="utf-8").splitlines(), 1)
        if not (300 <= i <= 399)) + "\n", encoding="utf-8")
    b.write_text("\n".join(b.read_text(encoding="utf-8").splitlines()[:100]) + "\n",
                 encoding="utf-8")
    run("accept", "progress.md", "有意删减")
    latest = sorted(p.name for p in store.iterdir())[-1]
    assert (store / latest / "architecture.md").read_text(encoding="utf-8") == before_b
    assert (store / latest / "progress.md").read_text(encoding="utf-8") == a.read_text(encoding="utf-8")


def test_cr070_unaccepted_baseline_survives_repeated_accepts(tmp_path):
    """R64 我自陈过一个"连续 accept 会把未接受文件的旧版本轮换掉"的风险，
    codex R64 复审指出它在当前实现下不成立：每次 accept 都会把未点名文件
    从上一份基线**继承**到新基线，所以旧版本一直跟着往前走。这里把这个
    性质钉成回归——它是 CR-070 修法能否长期成立的前提，不能只停留在
    "复审说不成立"。
    """
    repo, store, run = _mkrepo2(tmp_path)
    a, b = repo / "progress.md", repo / "architecture.md"
    before_b = b.read_text(encoding="utf-8")
    b.write_text("被截断了\n", encoding="utf-8")          # B 出事，一直不接受
    for i in range(3):                                     # A 连续三轮有意删减
        lines = a.read_text(encoding="utf-8").splitlines()
        a.write_text("\n".join(lines[:-50]) + "\n", encoding="utf-8")
        assert run("accept", "progress.md", f"第 {i + 1} 轮有意删减").returncode == 0
    latest = sorted(p.name for p in store.iterdir())[-1]
    assert (store / latest / "architecture.md").read_text(encoding="utf-8") == before_b, \
        "未被接受的 B 必须一路继承到最新基线，否则它的事故前版本会被轮换掉"
    assert run("check").returncode == 1, "B 的截断也必须一直报红"


def test_cr070_accept_records_which_file_and_what_was_deleted(tmp_path):
    """存档必须能追溯"接受的是哪一份、删掉的是什么"，否则橡皮图章盖完
    什么痕迹都不留。
    """
    repo, store, run = _mkrepo2(tmp_path)
    a = repo / "progress.md"
    a.write_text("\n".join(
        l for i, l in enumerate(a.read_text(encoding="utf-8").splitlines(), 1)
        if not (300 <= i <= 399)) + "\n", encoding="utf-8")
    run("accept", "progress.md", "删掉了 300-399 那段过时记录")
    latest = sorted(p.name for p in store.iterdir())[-1]
    text = (store / latest / "ACCEPTED.txt").read_text(encoding="utf-8")
    assert "接受的文档: progress.md" in text
    assert "删掉了 300-399 那段过时记录" in text
    assert "-progress.md 第 300 行" in text, "存档里要能看到被接受掉的具体行"


def test_cr070_accept_rejects_a_file_outside_the_baseline(tmp_path):
    _repo, _store, run = _mkrepo2(tmp_path)
    r = run("accept", "不存在.md", "理由")
    assert r.returncode == 2
    assert run("check").returncode == 0


def test_pre_cr071_block_threshold_would_have_passed_scattered_deletions():
    """判别性基线：内联复现 `BLOCK_DELETE=10` 的判定——十处各净删 9 行时
    每个 hunk 都 < 10，旧规则判"正常"。
    """
    per_hunk_net_deletes = [9] * 10
    old_rule_flags = any(d >= 10 for d in per_hunk_net_deletes)
    assert not old_rule_flags, "旧阈值本应（错误地）放过十处各 9 行的删除"


def test_cr071_scattered_small_deletions_masked_by_appends_are_flagged(tmp_path):
    """新规则：十处各删 9 行 + 末尾追加 300 行（净 +210 行、字节也增长），
    必须非零。
    """
    doc, _store, run = _mkrepo(tmp_path, lines=2000)
    dropped = set()
    for k in range(10):
        lo = 200 + k * 100
        dropped.update(range(lo, lo + 9))
    kept = [l for i, l in enumerate(doc.read_text(encoding="utf-8").splitlines(), 1)
            if i not in dropped]
    kept += [f"本轮新增第 {i} 行" for i in range(1, 301)]
    doc.write_text("\n".join(kept) + "\n", encoding="utf-8")
    assert len(kept) > 2000, "构造前提：净行数必须是增长的，否则测的不是这个形状"
    r = run("check")
    assert r.returncode == 1, f"分散的小删除必须被判非零：\n{r.stdout}\n{r.stderr}"
    assert "真正丢失" in r.stdout + r.stderr


def test_cr071_equal_size_rewrite_is_still_clean(tmp_path):
    """反向判别性：同一处删 1 行加 1 行（净删 0）且内容变长，仍应是 0——
    去掉阈值不能把普通的原地改写变成红灯。
    """
    doc, _store, run = _mkrepo(tmp_path)
    lines = doc.read_text(encoding="utf-8").splitlines()
    lines[500] = lines[500] + "：这里补一句说明"
    doc.write_text("\n".join(lines) + "\n", encoding="utf-8")
    r = run("check")
    assert r.returncode == 0, f"原地等量改写（变长）不应报错：\n{r.stdout}\n{r.stderr}"


# ---- CR-072：accept 的审计证据不能只是预览 ----


def test_cr072_full_diff_is_archived_and_totals_recorded(tmp_path):
    """删掉 151 行（远超预览的 40 行）之后：
    - `ACCEPTED.txt` 必须写明删除总数，并说明自己只是预览；
    - `ACCEPTED.diff` 必须存下完整 diff，第 41 行以后的删除也能取回。
    这是 accept 唯一的审计证据——上一基线被保留策略轮换掉之后，它就是
    唯一还能回答"当时到底删了什么"的东西。
    """
    doc, store, run = _mkrepo(tmp_path, lines=500)
    kept = [l for i, l in enumerate(doc.read_text(encoding="utf-8").splitlines(), 1)
            if not (100 <= i <= 250)]
    doc.write_text("\n".join(kept) + "\n", encoding="utf-8")
    assert run("accept", "progress.md", "删掉 100-250 那段过时记录").returncode == 0

    latest = sorted(p.name for p in store.iterdir())[-1]
    txt = (store / latest / "ACCEPTED.txt").read_text(encoding="utf-8")
    assert "删除行数: 151" in txt, f"摘要里必须有删除总数：\n{txt[:400]}"
    assert "预览" in txt and "ACCEPTED.diff" in txt, "必须写明这里只是预览、完整 diff 在哪"
    assert "其余 111 行删除见 ACCEPTED.diff" in txt

    diff_text = (store / latest / "ACCEPTED.diff").read_text(encoding="utf-8")
    deleted = [l[1:] for l in diff_text.splitlines()
               if l.startswith("-") and not l.startswith("---")]
    assert len(deleted) == 151, "完整 diff 里必须有全部 151 行删除"
    # 第 41 行以后（预览截断处之外）确实能从归档里取到
    assert "第 200 行" in deleted[-1] or any("第 200 行" in d for d in deleted)
    assert any("第 250 行" in d for d in deleted), "最后一行删除也必须在归档里"


def test_cr072_preview_note_is_absent_when_nothing_is_truncated(tmp_path):
    """反向：删的行数不到 40 时不应出现"其余 N 行"这句——摘要不能说假话。"""
    doc, store, run = _mkrepo(tmp_path, lines=100)
    kept = [l for i, l in enumerate(doc.read_text(encoding="utf-8").splitlines(), 1)
            if not (10 <= i <= 14)]
    doc.write_text("\n".join(kept) + "\n", encoding="utf-8")
    run("accept", "progress.md", "删掉 5 行")
    latest = sorted(p.name for p in store.iterdir())[-1]
    txt = (store / latest / "ACCEPTED.txt").read_text(encoding="utf-8")
    assert "删除行数: 5" in txt
    assert "其余" not in txt


# ---- CR-073：_prune 是唯一会 rm -rf 的分支，必须有边界回归 ----
#
# R62/R63/R64 连着三轮把"保留策略没有测试覆盖"登记为自陈却没有当场验证，
# 违反 `AGENTS.md` §5.1。这里按当前策略（至少留最近 KEEP_MIN=20 份；
# 超出的部分里只删超过 KEEP_DAYS=30 天的）把边界钉死。
#
# 造快照不需要跑 save：`_prune` 只看目录名里的 UTC 时间戳，不读内容。

_KEEP_MIN = 20
_KEEP_DAYS = 30


def _fake_snapshots(store, ages_days):
    """按给定的"天数前"造快照目录，返回目录名（按时间从老到新）。"""
    store.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    names = []
    for i, age in enumerate(sorted(ages_days, reverse=True)):
        ts = (now - timedelta(days=age, seconds=i)).strftime("%Y%m%dT%H%M%SZ")
        d = store / f"{ts}__01__fake{i:02d}"
        d.mkdir()
        (d / "MANIFEST.tsv").write_text("progress.md\t1\t3\tdeadbeef\n", encoding="utf-8")
        (d / "progress.md").write_text("hi\n", encoding="utf-8")
        names.append(d.name)
    return names


def _prune_env(tmp_path):
    repo, store = tmp_path / "repo", tmp_path / "store"
    repo.mkdir()
    (repo / "progress.md").write_text("hi\n", encoding="utf-8")
    env = {
        "DOCSNAP_ROOT": str(repo), "DOCSNAP_STORE": str(store),
        "DOCSNAP_DOCS": "progress.md", "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
    }

    def run(*args):
        return subprocess.run([str(DOCSNAP), *args], capture_output=True, text=True, env=env, timeout=60)

    return repo, store, run


def test_cr073_prune_keeps_everything_within_keep_min_even_if_ancient(tmp_path):
    """保底数量优先：只有 15 份（都 400 天前）时，一份都不许删。"""
    _repo, store, run = _prune_env(tmp_path)
    names = _fake_snapshots(store, [400] * 15)
    assert run("save", "new").returncode == 0
    left = {p.name for p in store.iterdir()}
    assert set(names) <= left, "未超过保底数量时不得删除任何快照"
    assert len(left) == 16


def test_cr073_prune_deletes_only_the_excess_that_is_also_old(tmp_path):
    """超额且超过 30 天的才删；超额但年轻的、以及保底 20 份，都要留下。"""
    _repo, store, run = _prune_env(tmp_path)
    old = _fake_snapshots(store, [60] * 5)          # 5 份 60 天前（会被删）
    young = _fake_snapshots(store, [5] * 20)        # 20 份 5 天前（保底）
    assert run("save", "new").returncode == 0       # 共 26 份，超额 6 份
    left = {p.name for p in store.iterdir()}
    assert not (set(old) & left), f"超额且超过 30 天的 5 份应被删除，实际还剩 {set(old) & left}"
    assert set(young) <= left, "保底 20 份必须全部留下（哪怕它们也在超额计算里）"
    assert len(left) == 21


def test_cr073_prune_boundary_exactly_30_days_is_kept_31_is_deleted(tmp_path):
    """边界：策略写的是"超过 30 天"，因此 30 天整必须留、31 天必须删。"""
    _repo, store, run = _prune_env(tmp_path)
    kept_edge = _fake_snapshots(store, [30])        # 30 天整 -> 留
    deleted_edge = _fake_snapshots(store, [31])     # 31 天 -> 删
    young = _fake_snapshots(store, [1] * 20)
    assert run("save", "new").returncode == 0       # 22 份，超额 2 份，正好是上面两份
    left = {p.name for p in store.iterdir()}
    assert kept_edge[0] in left, "刚好 30 天不算'超过 30 天'，必须留下"
    assert deleted_edge[0] not in left, "31 天且超额，必须删除"
    assert set(young) <= left


def test_cr073_after_pruning_latest_and_restore_still_work(tmp_path):
    """删完之后基线仍然可用：`check` 认最新那份，`restore` 能拿回内容。"""
    repo, store, run = _prune_env(tmp_path)
    _fake_snapshots(store, [60] * 5 + [5] * 20)
    doc = repo / "progress.md"
    doc.write_text("\n".join(f"第 {i} 行" for i in range(1, 201)) + "\n", encoding="utf-8")
    assert run("save", "real").returncode == 0
    assert len({p.name for p in store.iterdir()}) == 21
    assert run("check").returncode == 0, "刚存完的基线应当是干净的"

    doc.write_text("被截断了\n", encoding="utf-8")
    assert run("check").returncode == 1
    latest = sorted(p.name for p in store.iterdir())[-1]
    assert run("restore", latest, "progress.md").returncode == 0
    assert doc.read_text(encoding="utf-8").splitlines()[-1] == "第 200 行"


# ---- CR-074：正文里以 -- 开头的行不是 diff 的文件头 ----


def _old_rule_deleted_count(diff_text):
    """R65 那版按字面前缀认文件头的算法，内联复现用于判别性对照。"""
    return sum(1 for l in diff_text.splitlines()
               if l.startswith("-") and not l.startswith("---"))


def test_cr074_deleted_separator_lines_are_counted(tmp_path):
    """删掉的是 Markdown 分隔线（`---`）这类以 `--` 开头的正文行时，
    旧算法会把它们当成 unified diff 的文件头跳过，删除数少算甚至归零。
    """
    doc, store, run = _mkrepo(tmp_path, lines=20)
    doc.write_text("--- 分隔线一\n--- 分隔线二\n--- 分隔线三\n正文一\n正文二\n", encoding="utf-8")
    run("save", "baseline")
    doc.write_text("正文一\n正文二\n", encoding="utf-8")

    r = run("check")
    assert r.returncode == 1
    assert "删 3 行" in r.stdout + r.stderr, f"三行分隔线都必须算进删除数：\n{r.stdout}\n{r.stderr}"

    assert run("accept", "progress.md", "删掉三条分隔线").returncode == 0
    latest = sorted(p.name for p in store.iterdir())[-1]
    txt = (store / latest / "ACCEPTED.txt").read_text(encoding="utf-8")
    assert "删除行数: 3" in txt, f"摘要里的删除总数必须是 3：\n{txt}"
    diff_text = (store / latest / "ACCEPTED.diff").read_text(encoding="utf-8")
    assert _old_rule_deleted_count(diff_text) == 0, (
        "判别性前提：旧算法在这份 diff 上确实数出 0 行删除（三行都被当成文件头）")
    assert "-  分隔线一" in txt or "分隔线一" in txt, "预览里也要能看到被删的分隔线"


def test_cr074_separator_deletion_masked_by_appends_is_flagged(tmp_path):
    """最坏的组合：删掉的全是 `---` 行、又用追加把行数和字节数都做成增长。
    旧算法下三条判据会同时失效（净行数增、字节增、每 hunk 算出的净删为 0），
    这正是 CR-074 从"计数不准"升级成"能绕过门禁"的地方。
    """
    doc, _store, run = _mkrepo(tmp_path, lines=5)
    doc.write_text("".join(f"--- 分隔线 {i}\n" for i in range(1, 11)) + "正文\n", encoding="utf-8")
    run("save", "baseline")
    doc.write_text("正文\n" + "".join(f"新增第 {i} 行\n" for i in range(1, 31)), encoding="utf-8")

    r = run("check")
    assert r.returncode == 1, f"被追加掩盖的分隔线删除必须仍然报红：\n{r.stdout}\n{r.stderr}"
    assert "删 10 行" in r.stdout + r.stderr


# ---- CR-075：形似时间戳的目录不等于可用基线 ----


def _bogus_dir(store, name, manifest=None, files=None):
    d = store / name
    d.mkdir(parents=True)
    if manifest is not None:
        (d / "MANIFEST.tsv").write_text(manifest, encoding="utf-8")
    for fn, content in (files or {}).items():
        (d / fn).write_text(content, encoding="utf-8")
    return d


def test_cr075_invalid_timestamp_dir_is_not_used_as_baseline(tmp_path):
    """`20269999T999999Z__01__bogus` 这种名字能过正则、过不了 `date`。
    旧实现把它选成最新基线，`check` 打印一句找不到 MANIFEST 之后**退出 0**
    ——门禁被整个绕过。现在必须：仍拿真正的基线比对，并把坏目录报出来。
    """
    doc, store, run = _mkrepo(tmp_path, lines=100)
    _bogus_dir(store, "20269999T999999Z__01__bogus")
    doc.write_text("只剩一行\n", encoding="utf-8")

    r = run("check")
    assert r.returncode == 1, "真实截断必须被判非零，而不是被坏目录带成 0"
    out = r.stdout + r.stderr
    assert "疑似大段截断" in out, "必须仍然与真正的基线比对"
    assert "20269999T999999Z__01__bogus" in out, "坏目录必须被报出来"


def test_cr075_only_broken_snapshots_fails_closed(tmp_path):
    """一份有效基线都没有时必须失败关闭（退出 2），不能当作"没问题"。"""
    repo, store, run = _prune_env(tmp_path)
    _bogus_dir(store, "20269999T999999Z__01__bogus")
    r = run("check")
    assert r.returncode == 2
    assert "失败关闭" in r.stdout + r.stderr


def test_cr075_snapshot_missing_a_listed_file_is_invalid(tmp_path):
    """清单里点名的文件不在目录里 = 这份快照不可用（半份 save 的形状）。"""
    doc, store, run = _mkrepo(tmp_path, lines=50)
    _bogus_dir(store, "29990101T000000Z__01__halfwritten",
               manifest="progress.md\t50\t100\tdeadbeef\n")   # 只有清单，没有文件
    doc.write_text("只剩一行\n", encoding="utf-8")
    r = run("check")
    assert r.returncode == 1
    assert "halfwritten" in r.stdout + r.stderr
    assert "疑似大段截断" in r.stdout + r.stderr, "仍应与真正可用的基线比对"


def test_cr075_restore_refuses_an_invalid_snapshot(tmp_path):
    doc, store, run = _mkrepo(tmp_path, lines=50)
    _bogus_dir(store, "29990101T000000Z__01__halfwritten",
               manifest="progress.md\t50\t100\tdeadbeef\n")
    before = doc.read_text(encoding="utf-8")
    r = run("restore", "29990101T000000Z__01__halfwritten", "progress.md")
    assert r.returncode == 2
    assert "不完整" in r.stdout + r.stderr
    assert doc.read_text(encoding="utf-8") == before, "拒绝恢复时不得改动工作区"


def test_cr075_save_leaves_no_half_written_directory(tmp_path):
    """save 先写临时目录再原子改名：正常存完之后不应留下任何半成品，
    而且遗留的 `.partial-` 目录不会被当成快照（名字不匹配时间戳格式）。
    """
    _doc, store, run = _mkrepo(tmp_path, lines=10)
    (store / ".partial-20260101T000000Z__01__interrupted").mkdir()
    assert run("save", "again").returncode == 0
    snapshots = [p.name for p in store.iterdir() if not p.name.startswith(".partial-")]
    for name in snapshots:
        assert (store / name / "MANIFEST.tsv").exists(), f"{name} 是半成品"
    assert run("check").returncode == 0, "遗留的 .partial- 目录不该把门禁带红"


# ---- CR-076：演练的输出本身也是证据，不能带噪声 ----


def test_cr076_drill_writes_nothing_to_stderr():
    """`drill` 里一句提示原来用双引号裹着反引号 `---`，bash 把它当命令
    替换执行了，stderr 上冒出 `---: command not found`，而演练照样报成功。
    审查方是从 stderr 里看出来的——**演练报"通过"却在报错，本身就是
    一种不可信**。这条断言把它钉死：整条演练的 stderr 必须为空。
    """
    r = subprocess.run([str(DOCSNAP), "drill"], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stderr == "", f"演练不该往 stderr 写任何东西，实际：\n{r.stderr}"


# ---- CR-077：受监控的文档缺失时，save 不能悄悄少存一份 ----


def test_cr077_save_refuses_when_a_monitored_doc_is_missing(tmp_path):
    """最坏的失效形状：文件没了 -> save 静默跳过 -> 快照"有效"但不覆盖它
    -> check 返回 0 -> 那份文档永久脱离检测，而门禁一直是绿的。
    """
    repo, store, run = _mkrepo2(tmp_path)
    (repo / "architecture.md").unlink()
    r = run("save", "少了一份")
    assert r.returncode == 1, "受监控文档缺失时必须拒绝存快照"
    assert "architecture.md" in r.stdout + r.stderr
    names = [p.name for p in store.iterdir() if not p.name.startswith(".partial-")]
    assert len(names) == 1, "被拒绝的那次不该留下任何快照目录"


def test_cr077_allow_missing_records_the_gap(tmp_path):
    """确实是有意去掉的文档：显式 `--allow-missing`，缺失清单必须落盘。"""
    repo, store, run = _mkrepo2(tmp_path)
    (repo / "architecture.md").unlink()
    r = run("save", "--allow-missing", "architecture.md 已按计划移除")
    assert r.returncode == 0, r.stderr
    latest = sorted(p.name for p in store.iterdir())[-1]
    missing = (store / latest / "MISSING.txt").read_text(encoding="utf-8")
    assert "- architecture.md" in missing
    assert "architecture.md 已按计划移除" in missing, "CR-079：理由必须一起落盘"


def test_cr077_check_flags_a_monitored_doc_not_covered_by_the_baseline(tmp_path):
    """基线里没有这份文档 != 它不该被盯着。用只覆盖一份的基线去 check
    两份文档，未被覆盖的那份必须报出来并判非零——否则一次"缺文件的快照"
    会让它永久脱离检测。
    """
    repo, _store, _run = _mkrepo2(tmp_path)
    env_one = {
        "DOCSNAP_ROOT": str(repo), "DOCSNAP_STORE": str(tmp_path / "store"),
        "DOCSNAP_DOCS": "progress.md", "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
    }
    subprocess.run([str(DOCSNAP), "save", "only-one"], capture_output=True, env=env_one, timeout=60)
    env_two = dict(env_one, DOCSNAP_DOCS="progress.md architecture.md")
    r = subprocess.run([str(DOCSNAP), "check"], capture_output=True, text=True, env=env_two, timeout=60)
    assert r.returncode == 1
    assert "architecture.md" in r.stdout + r.stderr
    assert "没有被基线覆盖" in r.stdout + r.stderr


def test_cr077_check_flags_a_monitored_doc_that_vanished_entirely(tmp_path):
    """文件既不在工作区、也不在基线里（先删再存了一份 --allow-missing 的
    快照），仍然必须报——这正是"脱离检测"的那条路径。
    """
    repo, _store, run = _mkrepo2(tmp_path)
    (repo / "architecture.md").unlink()
    assert run("save", "--allow-missing", "有意移除").returncode == 0
    r = run("check")
    assert r.returncode == 1
    assert "不该凭空消失" in r.stdout + r.stderr


# ---- CR-078：accept 也必须先写临时目录再原子改名 ----


def test_cr078_accept_leaves_no_half_written_snapshot(tmp_path):
    """accept 是唯一会把删除写进基线的操作，半途而废的后果比 save 更糟。
    这里验证它和 save 一样不会留下半成品，且遗留的 `.partial-` 不影响它。
    """
    repo, store, run = _mkrepo2(tmp_path)
    (store / ".partial-20260101T000000Z__01__interrupted").mkdir()
    a = repo / "progress.md"
    a.write_text("\n".join(a.read_text(encoding="utf-8").splitlines()[:-100]) + "\n",
                 encoding="utf-8")
    assert run("accept", "progress.md", "有意删减").returncode == 0
    snapshots = [p.name for p in store.iterdir() if not p.name.startswith(".partial-")]
    for name in snapshots:
        assert (store / name / "MANIFEST.tsv").exists(), f"{name} 是半成品"
    assert not [p for p in store.iterdir()
                if p.name.startswith(".partial-") and p.name != ".partial-20260101T000000Z__01__interrupted"], \
        "accept 结束后不该留下自己的临时目录"
    assert run("check").returncode == 0


# ---- CR-079：--allow-missing 的理由必填 ----


def test_cr079_allow_missing_without_a_reason_is_rejected(tmp_path):
    """`--allow-missing` 是"我知道少了几份文档、照样建基线"的唯一出口。
    空理由也能建成的话，事后只知道少了哪几份、不知道是有意移除还是某次
    误删被顺手放行——与 `accept` 同一口径：理由必填。
    """
    repo, store, run = _mkrepo2(tmp_path)
    (repo / "architecture.md").unlink()
    before = {p.name for p in store.iterdir()}
    r = run("save", "--allow-missing")
    assert r.returncode == 2, "空理由必须被拒绝"
    assert "理由必填" in r.stdout + r.stderr
    assert {p.name for p in store.iterdir()} == before, "被拒绝的那次不该留下快照"


def test_cr079_plain_save_still_needs_no_reason(tmp_path):
    """反向：普通的写前快照仍然可以不带说明——理由必填只针对
    `--allow-missing` 这个明知有缺失还要建基线的动作，不是给日常操作
    加负担。
    """
    _repo, _store, run = _mkrepo2(tmp_path)
    assert run("save").returncode == 0


def test_cr079_missing_txt_records_reason_time_and_files(tmp_path):
    """存档要能独立回答"少了哪几份、为什么、什么时候"三个问题。"""
    repo, store, run = _mkrepo2(tmp_path)
    (repo / "architecture.md").unlink()
    assert run("save", "--allow-missing", "架构文档已并入 scope.md").returncode == 0
    latest = sorted(p.name for p in store.iterdir())[-1]
    text = (store / latest / "MISSING.txt").read_text(encoding="utf-8")
    assert "理由: 架构文档已并入 scope.md" in text
    assert "- architecture.md" in text
    assert "缺失时间: " in text


def test_real_collaboration_docs_are_all_covered(tmp_path):
    """`.gitignore` 里那几份没有 git 副本的协作文档，必须全部在 docsnap
    的默认清单里——漏掉哪一份，哪一份就还是裸奔。
    """
    # 只取"整份文档"那种忽略项：带路径分隔符或通配符的（如
    # bench/reports/*.md）是产物目录，不属于协作文档。
    ignored = [
        line.strip() for line in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip().endswith(".md")
        and not line.strip().startswith(("#", "!"))
        and "/" not in line.strip() and "*" not in line.strip()
    ]
    assert len(ignored) >= 8, f"协作文档清单看起来不对: {ignored}"
    covered = DOCSNAP.read_text(encoding="utf-8").split("DEFAULT_DOCS=", 1)[1].split("\n", 1)[0]
    missing = [d for d in ignored if d not in covered]
    assert not missing, f"这些没有 git 副本的文档没被 docsnap 覆盖: {missing}"
