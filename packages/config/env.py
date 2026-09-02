"""仓库根 `.env` 的最小加载器。

只支持本项目所需的 ``KEY=value`` 形式；不引入 python-dotenv，避免为少量
凭据增加运行时依赖。值永远不记录或回显。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import MutableMapping


@dataclass(frozen=True)
class EnvFileState:
    """加载结果，仅保留键名和文件状态，绝不保存或暴露值。"""

    path: Path
    exists: bool
    declared: frozenset[str]
    empty: frozenset[str]
    loaded: frozenset[str]


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def load_dotenv(path: Path, environ: MutableMapping[str, str] | None = None) -> EnvFileState:
    """仅填充尚未设置的环境变量；文件不存在时静默返回状态。"""
    env = os.environ if environ is None else environ
    if not path.is_file():
        return EnvFileState(path, False, frozenset(), frozenset(), frozenset())

    declared: set[str] = set()
    empty: set[str] = set()
    loaded: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key.removeprefix("export ").strip()
        if not key:
            continue
        declared.add(key)
        value = _unquote(value)
        if not value:
            empty.add(key)
        # “已设置”（包括显式空字符串）同样优先于文件，避免临时 export 被覆盖。
        if key not in env:
            env[key] = value
            loaded.add(key)
    return EnvFileState(path, True, frozenset(declared), frozenset(empty), frozenset(loaded))


def missing_credential_message(key: str, state: EnvFileState,
                               environ: MutableMapping[str, str] | None = None) -> str:
    """说明凭据为什么不可用，只返回键名与来源，永不返回值。"""
    env = os.environ if environ is None else environ
    if key in env and not env[key]:
        if key in state.empty and key in state.loaded:
            return f"{state.path.name} 中的 {key} 值为空"
        return f"环境变量 {key} 已设置但值为空"
    if not state.exists:
        return f"未找到 {state.path.name}，且环境变量 {key} 未设置"
    if key not in state.declared:
        return f"{state.path.name} 未声明 {key}，且环境变量未设置"
    return f"环境变量 {key} 未设置"
