"""代码围栏遮罩的回归（CR-009）。

`_mask_fenced()` 只为一件事存在：代码块里的 `# download` 是 shell 注释，不是标题。
但旧实现每命中一个围栏就复位状态，于是把**收尾**围栏又当成下一段的开围栏，
从第一个代码块一路遮到文件末尾——

- 代码块之后的标题全部消失，那些小节不再进索引；
- 它们的正文被并进上一节，引用因此张冠李戴，
  直接违反 I1 确立的「块正文不跨小节」不变量。

实测：一篇「装-配-验」三节的 Markdown，修复前只解析出第一节。
下面每条都锁住围栏配对的一种形态。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from services.sync.parse import _mask_fenced, parse_asciidoc, parse_markdown  # noqa: E402


def _titles(sections) -> list[str]:
    return [s.title_path[-1] for s in sections]


BODY = "这是一段足够长的正文，避免被空正文与导航页规则过滤掉。"


def test_heading_after_code_block_survives():
    """CR-009 的最小复现：代码块之后的标题不得消失。"""
    src = (
        f"## Install\n{BODY}\n\n"
        "```bash\n# download the tarball\ncurl -O https://example.com/x.tgz\n```\n\n"
        f"## Configure\n{BODY}\n"
    )
    assert _titles(parse_markdown(src, "Doc")) == ["Install", "Configure"]


def test_every_heading_after_the_first_code_block_survives():
    """旧实现是从第一个围栏一路遮到文件末尾，丢的不止一节。"""
    src = (
        f"## A\n{BODY}\n\n```bash\n# x\n```\n\n"
        f"## B\n{BODY}\n\n~~~yaml\nkey: value\n~~~\n\n"
        f"## C\n{BODY}\n"
    )
    assert _titles(parse_markdown(src, "Doc")) == ["A", "B", "C"]


def test_body_does_not_swallow_the_next_section():
    """标题丢失的连带后果：下一节正文被并进上一节，引用会指错小节。"""
    src = (
        f"## Install\n安装说明{BODY}\n\n```bash\n# x\n```\n\n"
        "## Configure\n配置说明只应出现在 Configure 一节。\n"
    )
    secs = parse_markdown(src, "Doc")
    install = next(s for s in secs if s.title_path[-1] == "Install")
    assert "配置说明" not in install.body


def test_shell_comments_inside_code_are_still_masked():
    """遮罩本身不能失效：代码块里的 `#` 注释不得变成标题与伪造锚点。"""
    src = f"## Install\n{BODY}\n\n```bash\n# 下载 tarball\n## 解压\n```\n"
    assert _titles(parse_markdown(src, "Doc")) == ["Install"]


def test_unclosed_fence_masks_to_end_of_file():
    """未闭合的围栏按 CommonMark 延伸到文档结尾，页面也是这样渲染的。

    这里宁可少收几节，也不能把代码里的 `#` 注释当标题——
    伪造的锚点会写进 source_url，引用指向页面上不存在的位置。
    """
    src = f"## Intro\n{BODY}\n\n```bash\n# not a heading\n## also not a heading\n"
    assert _titles(parse_markdown(src, "Doc")) == ["Intro"]


def test_longer_outer_fence_wraps_shorter_inner_fence():
    """````  包裹的示例里含 ``` ，收尾必须认「不短于开围栏」这条规则。"""
    src = (
        f"## A\n{BODY}\n\n"
        "````markdown\n```\n# inner\n```\n````\n\n"
        f"## B\n{BODY}\n"
    )
    assert _titles(parse_markdown(src, "Doc")) == ["A", "B"]


def test_backtick_fence_is_not_closed_by_tilde():
    """字符不同不构成配对；把 ~~~ 当收尾会让其后的正文错位。"""
    masked = _mask_fenced("```\ncode\n~~~\nstill code\n```\nprose\n")
    assert "prose" in masked
    assert "still code" not in masked


def test_mask_preserves_offsets_and_newlines():
    """遮罩只换字符不换长度，否则匹配到的偏移量回原文取正文就会错位。"""
    src = "## A\n\n```\ncode\n```\n\n## B\n正文\n"
    masked = _mask_fenced(src)
    assert len(masked) == len(src)
    assert masked.count("\n") == src.count("\n")


# ---------- AsciiDoc（`----` 归一为 ``` 后走同一条路径）----------

def test_adoc_heading_after_listing_block_survives():
    src = (
        "= Page\n\n[[a.install]]\n== Install\n\n"
        f"{BODY}\n\n[source,bash]\n----\n# download\ncurl -O x\n----\n\n"
        f"[[a.configure]]\n== Configure\n\n{BODY}\n"
    )
    secs = parse_asciidoc(src, "Page")
    assert _titles(secs) == ["Install", "Configure"]
    # 锚点也必须跟着回来，否则这些小节即使入库也引用不到
    assert [s.anchor for s in secs] == ["a.install", "a.configure"]
