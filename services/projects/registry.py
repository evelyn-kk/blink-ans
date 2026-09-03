"""用户项目注册表：把导入根、版本和出网许可变成显式可审计配置。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ProjectRegistryError(ValueError):
    pass


@dataclass(frozen=True)
class Project:
    id: str
    version: str
    root: Path
    cloud_generation_allowed: bool


def load_projects(path: Path) -> list[Project]:
    """读取显式登记的项目；根目录必须存在，避免静默导入错误目录。"""
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    entries = raw.get("projects", [])
    if not isinstance(entries, list):
        raise ProjectRegistryError("projects 必须是列表")
    out: list[Project] = []
    for item in entries:
        missing = [k for k in ("id", "version", "root", "cloud_generation_allowed") if k not in item]
        if missing:
            raise ProjectRegistryError(f"项目 {item.get('id', '?')!r} 缺少: {', '.join(missing)}")
        ident = str(item["id"]).strip()
        root = Path(str(item["root"])).expanduser()
        # 清单应能随项目一起移动；相对 root 的基准是清单自身而非启动命令的 cwd。
        # resolve 也让后续 read_materials 的边界检查针对真实根目录进行。
        if not root.is_absolute():
            root = path.parent / root
        root = root.resolve()
        if not ident or "/" in ident or ".." in ident:
            raise ProjectRegistryError(f"非法项目 ID: {ident!r}")
        if not root.is_dir():
            raise ProjectRegistryError(f"项目 {ident!r} 的 root 不存在或不是目录: {root}")
        if not isinstance(item["cloud_generation_allowed"], bool):
            raise ProjectRegistryError(f"项目 {ident!r} 的 cloud_generation_allowed 必须是布尔值")
        out.append(Project(ident, str(item["version"]), root, item["cloud_generation_allowed"]))
    ids = [p.id for p in out]
    if len(ids) != len(set(ids)):
        raise ProjectRegistryError("项目 ID 不得重复")
    return out
