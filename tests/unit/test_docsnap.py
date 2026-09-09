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

import shutil
import subprocess
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
    assert "单块最多净删 100 行" in r.stdout + r.stderr


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
    assert "整块删除" in r.stdout + r.stderr


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
    r = run("accept")
    assert r.returncode == 2
    assert "理由必填" in r.stdout + r.stderr
    assert run("check").returncode == 1, "拒绝之后门禁必须仍然是红的"


def test_cr069_accept_records_reason_and_rebaselines(tmp_path):
    """有意删减：accept 之后 check 归零，且理由与新基线都落了盘。"""
    doc, store, run = _mkrepo(tmp_path)
    kept = [l for i, l in enumerate(doc.read_text(encoding="utf-8").splitlines(), 1)
            if not (400 <= i <= 499)]
    doc.write_text("\n".join(kept) + "\n", encoding="utf-8")
    assert run("check").returncode == 1
    r = run("accept", "删掉了 400-499 行那段过时记录，已人工确认")
    assert r.returncode == 0, r.stderr
    assert run("check").returncode == 0, "accept 之后同样的状态不应再报错"
    accepted = list(store.glob("*/ACCEPTED.txt"))
    assert accepted, "接受理由必须落盘存档"
    assert "已人工确认" in accepted[0].read_text(encoding="utf-8")
    # 新基线是"删减后"的内容：能被 restore 回来的必须是当前这份
    latest = sorted(p.name for p in store.iterdir())[-1]
    assert (store / latest / "progress.md").read_text(encoding="utf-8") == doc.read_text(encoding="utf-8")


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
