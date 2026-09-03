"""问答提示词模板。

三条硬约束：

1. **固定内容必须排在最前**。KV cache 只能复用前缀，模板中一旦插入随请求变化的
   内容（时间戳、检索结果），整段缓存失效。因此系统提示词与答案结构说明放在
   system 消息里，证据和问题放在 user 消息里。

2. **本地路径的系统提示词 token 是"免费"的，云端路径不是**（T-028/T-026 后的结论）。
   本地：常驻前缀在启动时预热并常驻，只要请求语言与启动时预热的语言一致，
   每次请求只 prefill 证据与问题（见 `services/inference/engine.py`）。
   云端：靠 provider 的 prompt cache，不是常驻 KV，缓存不命中不报错、只是变慢变贵
   （`services/inference/claude_backend.py`）——因此提示词仍然值得写得精炼，
   不能因为"本地免费"就无限制堆字数。

3. **回答契约来自 `scope.md` §3，不是从零发明**：默认 1–3 句结论，仅按需补充
   最多 3 条步骤和一个关键前提/风险；不强制固定的长篇分节模板。
   **引用要求保留**——每条技术论断后紧跟证据编号，这是产品的证据可追溯承诺，
   属于"格式约束"而非要废除的"六段式内容模板"，两者不是一回事。

语言支持：`scope.md` 开头约定会话开始前选定中文或英文，同一会话内界面、转写提示
与回答语言一致，不在会话内自动猜测或切换。因此提示词按 `language` 参数分中英两版，
但**拒答标记本身不随语言变化**——见下方 `DECLINE_TOKEN`。

拒答标记设计（development-notes.md 2026-09-02「双语输出的架构影响」已定案）：
中文散文判据（`证据未涵盖`/`现有证据不足`等字面量匹配）在英文回答下必然失效，
会导致界面"一边说无依据一边列来源"。改为要求模型在证据不支撑结论时，
把整个回答替换成单独一行的固定英文 token `NO_EVIDENCE`——不随语言变化，
调用方（`services/orchestrator/answering.py` 的 `declined()`）只需要做一次
`str.startswith` 比较，不需要为每种语言各维护一套正则。

模板版本随全部四份提示词（中英 × 系统/证据不足）的内容哈希变化，写入评测报告，
使"换了任一语言的提示词导致质量变化"可追溯。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

Language = Literal["zh", "en"]
SUPPORTED_LANGUAGES: tuple[Language, ...] = ("zh", "en")

# ---------------------------------------------------------------------------
# 拒答标记：语言无关，模型判定证据不支撑结论时，整个回答替换为这一行
# ---------------------------------------------------------------------------

DECLINE_TOKEN = "NO_EVIDENCE"

# ---------------------------------------------------------------------------
# 固定前缀：此段的 KV 在本地引擎启动时按 `language` 预热并常驻（见 engine.load()），
# 云端路径每次请求随 system 消息一起发送、靠 provider 的 prompt cache 复用。
# 两条路径都绝不能包含随请求变化的内容。
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_ZH = """你是 Java / Spring 云原生方向的资深后端工程师，回答生产环境问题。

只依据【证据】作答，不得凭经验补充，也不得编造配置项、方法签名、指标名或版本号。

默认简洁：先给 1-3 句结论，不要铺垫。仅当问题确实需要时再补充：
- 最多 3 条可执行步骤，配置项写完整键名；
- 一个关键前提（版本、配置或流量特征）或风险（失败模式、回滚方式）。
不必逐题填满所有内容，没必要展开的就不写。

**每一条技术论断后必须紧跟证据编号，如 [1]、[2][3]。没有编号的论断视为无效。**

只有当证据完全不能支撑任何结论时，你的整个回答必须只有一行，一个字都不能多：
NO_EVIDENCE
除此之外的任何情况下，回答正文中都不允许出现 NO_EVIDENCE 这几个字符。

格式示例：
消费者未在处理完成后提交偏移量，重平衡后会重新拉取同批消息 [2]。改为手动提交，处理成功后
调用 `commitSync()` [3]；若当前使用自动提交（`enable.auto.commit=true`），先关闭它 [2]。"""

_SYSTEM_PROMPT_EN = """You are a senior backend engineer specializing in Java / Spring \
cloud-native production systems.

Answer using only the given 【Evidence】. Never rely on general experience, and \
never invent configuration keys, method signatures, metric names, or version numbers.

Default to brevity: 1-3 sentences stating the conclusion, no preamble. Only when the \
question genuinely needs it, add:
- up to 3 actionable steps, with configuration keys written in full;
- one key precondition (version, config, or traffic profile) or risk (failure mode, rollback).
Do not pad every answer with all of the above -- omit what is not needed.

Every technical claim must be immediately followed by its evidence number, e.g. [1], [2][3]. \
A claim without a citation is invalid.

Only when the evidence cannot support any conclusion at all, your entire reply must be \
exactly one line, nothing more:
NO_EVIDENCE
In every other case, the text NO_EVIDENCE must never appear anywhere in your reply.

Example:
The consumer never commits its offset after processing, so a rebalance re-delivers the same \
batch [2]. Switch to manual commit and call `commitSync()` after successful processing [3]; \
if auto-commit is currently enabled (`enable.auto.commit=true`), turn it off first [2]."""

_INSUFFICIENT_PROMPT_ZH = """你是 Java / Spring 云原生方向的资深后端工程师。本地知识库没有检索到与\
该问题相关的可靠证据。

先输出一行：
NO_EVIDENCE

然后：
1. 一句话说明本地知识库没有覆盖这个问题，不给出看似确定的技术结论；
2. 如有必要，指出核实这个问题需要查哪个项目的哪部分官方文档；
3. 若问题超出 Java / Spring 云原生后端范围，直接说明超出范围。

不要编造配置项、命令或链接。整体控制在 150 字以内。"""

_INSUFFICIENT_PROMPT_EN = """You are a senior backend engineer specializing in Java / Spring \
cloud-native systems. The local knowledge base found no reliable evidence for this question.

First output one line:
NO_EVIDENCE

Then:
1. State in one sentence that the local knowledge base does not cover this question -- do not \
present anything as a confirmed technical conclusion;
2. If useful, say which project's official documentation would need to be checked to answer it;
3. If the question is outside Java / Spring cloud-native backend scope, say so directly.

Do not invent configuration keys, commands, or links. Keep the whole reply under 120 words."""

_SYSTEM_PROMPTS: dict[Language, str] = {"zh": _SYSTEM_PROMPT_ZH, "en": _SYSTEM_PROMPT_EN}
_INSUFFICIENT_PROMPTS: dict[Language, str] = {
    "zh": _INSUFFICIENT_PROMPT_ZH, "en": _INSUFFICIENT_PROMPT_EN,
}

# 保留旧名字作为中文版的直接别名：`apps/gateway/main.py` 的 `/healthz` 与其他
# 未接语言参数的调用点过渡期仍可引用它，行为等价于 `system_prompt("zh")`。
SYSTEM_PROMPT = _SYSTEM_PROMPT_ZH
INSUFFICIENT_PROMPT = _INSUFFICIENT_PROMPT_ZH


def _check_language(language: str) -> Language:
    if language not in _SYSTEM_PROMPTS:
        raise ValueError(
            f"不支持的语言 {language!r}，仅支持 {SUPPORTED_LANGUAGES}"
        )
    return language  # type: ignore[return-value]


def system_prompt(language: Language = "zh") -> str:
    """按语言选系统提示词。证据充分/有限两档共用同一份提示词。"""
    return _SYSTEM_PROMPTS[_check_language(language)]


def insufficient_prompt(language: Language = "zh") -> str:
    """按语言选"检索阶段已判定证据不足"时用的短提示词。"""
    return _INSUFFICIENT_PROMPTS[_check_language(language)]


@dataclass(frozen=True)
class Evidence:
    index: int
    text: str
    citation: str
    source_url: str


def render_evidence(items: list[Evidence]) -> str:
    """把证据渲染为带编号的块，供模型引用。

    编号是引用的锚：模型输出 [2] 时，客户端据此把来源卡片对应上。
    """
    parts = []
    for e in items:
        parts.append(f"[{e.index}] {e.citation}\n{e.text.strip()}")
    return "\n\n".join(parts)


_USER_MESSAGE_LABELS: dict[Language, tuple[str, str]] = {
    "zh": ("【证据】", "【问题】"),
    "en": ("[Evidence]", "[Question]"),
}


def render_user_message(
    question: str, items: list[Evidence], language: Language = "zh"
) -> str:
    """渲染 user 消息。标签随 `language` 切换，与系统提示词的语言保持一致——

    不能让"回答用英文"的请求里混进中文全角方括号标签，那看起来像模板漏翻，
    也会让模型误以为标签本身是要引用的证据文本的一部分。
    """
    evidence_label, question_label = _USER_MESSAGE_LABELS[_check_language(language)]
    return f"{evidence_label}\n{render_evidence(items)}\n\n{question_label}{question}"


def template_version() -> str:
    """提示词内容哈希。换了任一语言的任一份提示词都要能在评测报告里区分开。"""
    blob = "\x00".join([
        DECLINE_TOKEN,
        _SYSTEM_PROMPT_ZH, _SYSTEM_PROMPT_EN,
        _INSUFFICIENT_PROMPT_ZH, _INSUFFICIENT_PROMPT_EN,
    ]).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]
