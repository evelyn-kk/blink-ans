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
    assert "疑似异常缩水" in truncated.stdout + truncated.stderr

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
