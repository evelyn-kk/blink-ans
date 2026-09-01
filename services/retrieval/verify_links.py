"""引用链接可达性抽样验证。

I1 的完成定义是"检索结果能回链原页锚点"。仓库目录结构与官方站点 URL 结构
并不总是对应（Kafka 的仓库已迁到 Hugo，官网却仍是单页锚点），
靠推断很容易产出一批 404 而无人发觉，因此必须实测。

只做抽样：每个来源取 N 条，HEAD 请求看状态码。全量验证会对官方站点造成不必要的压力。
"""

from __future__ import annotations

import random
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field

from .store import ChunkStore

USER_AGENT = "blink-ans-link-check/0.1 (local knowledge base integrity check)"


@dataclass
class LinkReport:
    by_project: dict[str, dict[str, int]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))
    failures: list[tuple[str, str, int | str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures


def _status(url: str, timeout: float) -> int | str:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status
    except urllib.error.HTTPError as e:
        # 部分站点对 HEAD 返回 405，改用 GET 只读首字节
        if e.code == 405:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    return r.status
            except Exception as exc:
                return f"{type(exc).__name__}"
        return e.code
    except Exception as exc:
        return f"{type(exc).__name__}"


def verify(
    store: ChunkStore, sample_per_project: int = 5, timeout: float = 10.0, seed: int = 0
) -> LinkReport:
    report = LinkReport()
    rng = random.Random(seed)

    projects = [r["source_project"] for r in store.execute(
        "SELECT DISTINCT source_project FROM chunks"
    )]

    for proj in projects:
        urls = sorted({
            r["source_url"] for r in store.execute(
                "SELECT DISTINCT source_url FROM chunks WHERE source_project = ?", (proj,)
            )
        })
        for url in rng.sample(urls, min(sample_per_project, len(urls))):
            code = _status(url, timeout)
            bucket = "ok" if code == 200 else str(code)
            report.by_project[proj][bucket] += 1
            if code != 200:
                report.failures.append((proj, url, code))

    return report
