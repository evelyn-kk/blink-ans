"""来源拉取与许可证实测校验。

两个设计要点：

1. **稀疏拉取**：登记的仓库有 200–750 MB，但真正需要的文档目录往往只占几个百分点。
   用 `--filter=blob:none --sparse --depth 1` 只取需要的路径，避免下载全部历史与二进制。

2. **许可证不信声明**：注册表里的 license 字段只是期望值，同步时从仓库内的许可文件
   读取实际内容做特征匹配。声明与实测不符即失败，不得入库——
   这条防的是上游换许可而注册表未同步的情况（Redis 官方文档就换过）。
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .registry import Source

DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "corpus"

# 许可证特征串。匹配的是许可文本里最不容易改动的措辞，
# 而非标题——很多项目的许可文件没有标准标题。
_LICENSE_SIGNATURES: dict[str, tuple[str, ...]] = {
    "Apache-2.0": ("Apache License", "Version 2.0"),
    "MIT": ("MIT License", "Permission is hereby granted, free of charge"),
    "BSD-3-Clause": ("Redistribution and use in source and binary forms",
                     "Neither the name"),
    "BSD-2-Clause": ("Redistribution and use in source and binary forms",),
    "PostgreSQL": ("PostgreSQL Database Management System",
                   "Permission to use, copy, modify, and distribute this software"),
    "CC-BY-4.0": ("Creative Commons Attribution 4.0 International",),
    "CC-BY-SA-4.0": ("Attribution-ShareAlike 4.0 International",),
    "CC-BY-NC-SA-4.0": ("Attribution-NonCommercial-ShareAlike 4.0 International",),
}

# 出现这些字样说明许可含额外限制，即便 SPDX 匹配也要拒绝入库
_RESTRICTIVE_MARKERS = ("NonCommercial", "Non-Commercial", "no Derivative")


class FetchError(RuntimeError):
    pass


class LicenseError(RuntimeError):
    """许可证实测结果与注册表声明不符，或含额外使用限制。"""


@dataclass
class Fetched:
    source: Source
    root: Path
    commit: str
    license_verified: str


def _git(*args: str, cwd: Path | None = None) -> str:
    r = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise FetchError(f"git {' '.join(args)} 失败:\n{r.stderr.strip()[:500]}")
    return r.stdout.strip()


def clone_or_update(src: Source, root: Path | None = None) -> Path:
    """稀疏拉取来源仓库到本地缓存目录，返回工作树根。"""
    root = (root or DATA_ROOT) / src.slug
    sparse = [*src.paths, src.license_file]

    if (root / ".git").exists():
        try:
            _git("fetch", "--depth", "1", "origin", src.ref, cwd=root)
            _git("checkout", "-f", "FETCH_HEAD", cwd=root)
            return root
        except FetchError:
            # 缓存损坏或 ref 变更时重新克隆，而不是留下半坏的工作树
            shutil.rmtree(root, ignore_errors=True)

    root.parent.mkdir(parents=True, exist_ok=True)
    _git(
        "clone", "--filter=blob:none", "--sparse", "--depth", "1",
        "--branch", src.ref, src.repo, str(root),
    )
    _git("sparse-checkout", "set", "--no-cone", *sparse, cwd=root)
    return root


def head_commit(root: Path) -> str:
    return _git("rev-parse", "HEAD", cwd=root)[:12]


def verify_license(src: Source, root: Path) -> str:
    """读取仓库内的许可文件，校验其与注册表声明一致。

    返回实测到的 SPDX 标识；不一致或含额外限制时抛 LicenseError。
    """
    path = root / src.license_file
    if not path.exists():
        raise LicenseError(
            f"{src.id}: 许可文件 {src.license_file} 不存在于仓库中，无法校验许可"
        )

    text = path.read_text(encoding="utf-8", errors="replace")

    matched = [
        spdx for spdx, sigs in _LICENSE_SIGNATURES.items()
        if all(sig.lower() in text.lower() for sig in sigs)
    ]
    # NC 变体同时会匹配 CC-BY-SA 的特征串，取最具体的那个
    if "CC-BY-NC-SA-4.0" in matched:
        matched = ["CC-BY-NC-SA-4.0"]

    if not matched:
        raise LicenseError(
            f"{src.id}: 无法从 {src.license_file} 识别出已知许可证，"
            f"拒绝入库（前 120 字符: {text[:120]!r}）"
        )

    if src.license not in matched:
        raise LicenseError(
            f"{src.id}: 注册表声明 {src.license}，但仓库实测为 {'/'.join(matched)}。"
            f"上游可能变更了许可，须人工确认后再更新注册表"
        )

    if src.ingest:
        for marker in _RESTRICTIVE_MARKERS:
            if marker.lower() in text.lower():
                raise LicenseError(
                    f"{src.id}: 许可文本含 {marker!r}，属使用受限资料，"
                    f"不得入核心语料；应在注册表中置 ingest: false 并走实时链接检索"
                )

    return src.license


def collect_files(src: Source, root: Path) -> list[Path]:
    """收集该来源需要解析的文件。"""
    exts = {
        "markdown": (".md",),
        "asciidoc": (".adoc",),
        "html": (".html", ".htm"),
        "docbook": (".sgml", ".xml"),
    }[src.format]

    files: list[Path] = []
    for rel in src.paths:
        base = root / rel
        if not base.exists():
            raise FetchError(f"{src.id}: 登记路径 {rel} 不存在于仓库中（上游可能已重构目录）")
        files.extend(p for p in base.rglob("*") if p.suffix.lower() in exts and p.is_file())

    if not files:
        # 静默的 0 文件比报错更糟：索引会悄悄少掉一整个来源而回归可能仍然通过。
        # Kafka 就发生过这种情况——文档从 HTML 迁到 Markdown，而注册表仍写 html。
        raise FetchError(
            f"{src.id}: 在 {', '.join(src.paths)} 下没有找到任何 {src.format} 文件"
            f"（后缀 {'/'.join(exts)}）。上游可能变更了文档格式或目录结构，请核对注册表"
        )
    return sorted(files)


def fetch(src: Source, root: Path | None = None) -> Fetched:
    """完整拉取流程：克隆 → 校验许可 → 返回工作树信息。"""
    work = clone_or_update(src, root)
    spdx = verify_license(src, work)
    return Fetched(source=src, root=work, commit=head_commit(work), license_verified=spdx)
