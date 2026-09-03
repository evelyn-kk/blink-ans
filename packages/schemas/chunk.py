"""知识切块的元数据契约。

字段清单来自 architecture.md 5.2。这里的校验是**硬门**：
任何缺失必填字段的切块一律拒绝入库，不允许"先入库后补"——
回答必须能回链到 source_url 并显示版本或抓取日期，元数据缺失会让引用失去可追溯性。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

# 只接受明确许可、允许本地再利用的语料。许可更严格的资料走 I6 实时链接检索，
# 不进核心语料库（见 architecture.md 4.2）。
ALLOWED_LICENSES = {
    "Apache-2.0",
    "MIT",
    "BSD-3-Clause",
    "BSD-2-Clause",
    "PostgreSQL",
    "CC-BY-4.0",
    "CC-BY-SA-4.0",
}

CONTENT_TYPES = {"prose", "code", "config", "table", "mixed"}


class MetadataError(ValueError):
    """元数据不满足契约。附带字段名以便同步管线定位问题来源。"""

    def __init__(self, field_name: str, reason: str) -> None:
        self.field_name = field_name
        super().__init__(f"{field_name}: {reason}")


@dataclass
class Chunk:
    """一个可检索、可引用的知识片段。"""

    # ---- 必填：缺失即拒绝入库 ----
    source_url: str          # 可点击的原页地址，含锚点
    source_project: str      # 来源项目，如 spring-boot
    version_or_commit: str   # tag 或 commit，用于版本过滤
    license: str             # SPDX 标识
    retrieved_at: str        # ISO8601 UTC 抓取时间
    title_path: list[str]    # 标题层级，如 ["Web", "Servlet", "Embedded Container"]
    technology: str          # 技术域，用于路由过滤
    content_type: str        # prose / code / config / table / mixed
    locale: str              # 正文语言，如 en / zh
    text: str                # 正文

    # ---- 自动派生 ----
    checksum: str = ""       # 正文 sha256，用于增量同步与去重
    token_estimate: int = 0  # 粗略 token 数，供检索期控制上下文预算

    # ---- 可选 ----
    anchor: str | None = None
    source_path: str | None = None
    # 项目材料专用。通用官方语料保持 None，避免把来源项目误作用户项目。
    # 这些字段须进 SQLite 列：仅放 extra 会导致查询无法过滤、merge 会静默丢失。
    project_id: str | None = None
    module: str | None = None
    symbol: str | None = None
    cloud_generation_allowed: bool | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.checksum:
            self.checksum = hashlib.sha256(self.text.encode("utf-8")).hexdigest()
        if not self.token_estimate:
            self.token_estimate = estimate_tokens(self.text)

    # ---- 校验 ----

    def validate(self) -> None:
        """不满足契约时抛 MetadataError。同步管线在写入前逐块调用。"""
        for name in ("source_url", "source_project", "version_or_commit",
                     "license", "retrieved_at", "technology", "content_type",
                     "locale", "text"):
            if not str(getattr(self, name) or "").strip():
                raise MetadataError(name, "必填字段为空")

        if not self.source_url.startswith(("http://", "https://")):
            raise MetadataError("source_url", f"必须是可访问的 URL，实际为 {self.source_url!r}")

        if self.license not in ALLOWED_LICENSES:
            raise MetadataError(
                "license",
                f"{self.license!r} 不在允许清单内；许可受限的资料只做实时链接检索，不入核心语料",
            )

        if self.content_type not in CONTENT_TYPES:
            raise MetadataError("content_type", f"未知类型 {self.content_type!r}")

        if not self.title_path:
            raise MetadataError("title_path", "标题层级不得为空，引用需要显示定位路径")

        try:
            datetime.fromisoformat(self.retrieved_at)
        except ValueError as exc:
            raise MetadataError("retrieved_at", f"不是合法 ISO8601 时间: {exc}") from exc

        # checksum 必须与正文一致，防止切块后正文被改写而校验和未更新
        actual = hashlib.sha256(self.text.encode("utf-8")).hexdigest()
        if self.checksum != actual:
            raise MetadataError("checksum", "与正文不一致，疑似正文在计算校验和后被修改")

    # ---- 序列化 ----

    def to_row(self) -> dict[str, Any]:
        d = asdict(self)
        d["title_path"] = " › ".join(self.title_path)
        if d["cloud_generation_allowed"] is not None:
            d["cloud_generation_allowed"] = int(d["cloud_generation_allowed"])
        d.pop("extra")
        return d

    @property
    def citation(self) -> str:
        """回答中展示的引用字符串，含版本与抓取日期。"""
        date = self.retrieved_at[:10]
        return f"{self.source_project} {self.version_or_commit} · {' › '.join(self.title_path)} · 抓取于 {date}"


_CJK = re.compile(r"[一-鿿]")


def estimate_tokens(text: str) -> int:
    """粗略估算 token 数，不加载分词器。

    中文约 1 字 1 token，英文约 4 字符 1 token。
    仅用于检索期的上下文预算控制——I0 实测 prefill 352 tok/s，
    上下文长度直接决定首 token 时延，因此每块都要带这个数。
    """
    cjk = len(_CJK.findall(text))
    rest = len(text) - cjk
    return cjk + rest // 4


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
