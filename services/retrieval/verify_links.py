"""引用链接可达性抽样验证。

I1 的完成定义是"检索结果能回链原页锚点"。仓库目录结构与官方站点 URL 结构
并不总是对应（Kafka 的仓库已迁到 Hugo，官网却仍是单页锚点），
靠推断很容易产出一批 404 而无人发觉，因此必须实测。

**页面 200 不等于引用正确**：锚点错了链接照样返回 200，只是落到页面顶部。
Spring 与 Kubernetes 的文档大量使用显式锚点（`[[documentation.first-steps]]`、
`{#increase-load}`），靠标题文字 slug 化推不出来，而这个错误正是靠只查状态码
的旧实现漏过去的。因此 `--check-anchors` 会取回页面正文，核对 id 是否真的存在。

只做抽样：每个来源取 N 条。全量验证会对官方站点造成不必要的压力。
"""

from __future__ import annotations

import random
import re
import urllib.error
import urllib.parse
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


def _fetch(url: str, timeout: float) -> tuple[int | str, str]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read(2_000_000).decode("utf-8", errors="replace")
            return r.status, body
    except urllib.error.HTTPError as e:
        return e.code, ""
    except Exception as exc:
        return f"{type(exc).__name__}", ""


def _has_anchor(body: str, anchor: str) -> bool:
    """页面里是否真的存在这个 id。

    只认 id/name 属性，不认正文里碰巧出现的同名字符串——后者会让检查形同虚设。
    属性值可以不带引号：Kubernetes 站点的 HTML 是压缩过的，写作 `<h2 id=api>`，
    只认带引号的形式会把整站误判为"锚点不存在"。
    片段标识符区分大小写，因此这里也区分——PostgreSQL 发布的 HTML 把 SGML 里的
    小写 id 全部转成大写，不区分就发现不了。
    """
    a = re.escape(urllib.parse.unquote(anchor))
    return re.search(
        rf"""(?:id|name)\s*=\s*(?:"{a}"|'{a}'|{a}(?=[\s/>]))""", body
    ) is not None


def verify(
    store: ChunkStore,
    sample_per_project: int = 5,
    timeout: float = 10.0,
    seed: int = 0,
    check_anchors: bool = False,
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
        if check_anchors:
            # 只抽带锚点的链接：不带锚点的页面本来就没有可核对的定位
            urls = [u for u in urls if "#" in u and u.rsplit("#", 1)[1]]
        for url in rng.sample(urls, min(sample_per_project, len(urls))):
            if not check_anchors:
                code = _status(url, timeout)
                bucket = "ok" if code == 200 else str(code)
                report.by_project[proj][bucket] += 1
                if code != 200:
                    report.failures.append((proj, url, code))
                continue

            page, anchor = url.rsplit("#", 1)
            code, body = _fetch(page, timeout)
            if code != 200:
                report.by_project[proj][str(code)] += 1
                report.failures.append((proj, url, code))
            elif _has_anchor(body, anchor):
                report.by_project[proj]["ok"] += 1
            else:
                # 页面在、锚点不在：链接可点，但落到页面顶部而非被引用的小节
                report.by_project[proj]["锚点不存在"] += 1
                report.failures.append((proj, url, "锚点不存在"))

    return report
