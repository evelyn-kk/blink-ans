"""问答提示词模板。

两条硬约束来自 I0 实测：

1. **固定内容必须排在最前**。KV cache 只能复用前缀，模板中一旦插入随请求变化的
   内容（时间戳、检索结果），整段缓存失效。因此系统提示词与答案结构说明放在
   system 消息里，证据和问题放在 user 消息里。

2. **系统提示词的 token 是"免费"的**。它的 KV 在启动时预热并常驻，
   每次请求只 prefill 证据与问题。因此这里可以写得足够明确——
   加一个格式示例来提高引用标注的依从性，不会增加首 token 时延。
   真正受 prefill 预算约束的是证据部分（见 orchestrator/answering.py）。

模板版本随内容哈希变化，写入评测报告，使"换了提示词导致质量变化"可追溯。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# 固定前缀：此段的 KV 在服务启动时预热并常驻，绝不能包含随请求变化的内容
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """你是 Java / Spring 云原生方向的资深后端工程师，回答生产环境问题。

只依据【证据】作答。证据未提及的内容，一律说"证据未涵盖"，不得凭经验补充。
不得编造配置项名称、方法签名、指标名或版本号。

**每一条技术论断后必须紧跟证据编号，如 [1]、[2][3]。没有编号的论断视为无效。**
若某节内容全部来自你的经验而非证据，该节写"证据未涵盖"。

按以下结构作答，每节一行标题，无内容的节写"证据未涵盖"：

结论：一句话判断，不铺垫。
适用前提：在什么版本、配置、流量特征下成立。
实施步骤：可执行的操作，配置项写完整键名。
失败模式：什么情况下会失效，失效时的现象。
监控与验证：改动后看哪些指标，判断生效的依据。
来源：本次回答用到的全部证据编号，如 [1][3]。

涉及生产变更时必须说明回滚方式。不同大版本的配置不得混用。

格式示例：

结论：消费者未在处理完成后提交偏移量，重平衡后会重新拉取同批消息 [2]。
适用前提：使用自动提交且 `enable.auto.commit=true` 时 [2][3]。
实施步骤：改为手动提交，处理成功后调用 `commitSync()` [3]。
失败模式：证据未涵盖。
监控与验证：观察消费者组的 lag 是否随提交恢复 [1]。
来源：[1][2][3]"""

INSUFFICIENT_PROMPT = """你是 Java / Spring 云原生方向的资深后端工程师。

本地知识库没有检索到与该问题相关的可靠证据。请：
1. 明确告知"本地知识库没有覆盖这个问题"，不要给出看似确定的技术结论。
2. 指出要回答它需要核实哪些具体资料（例如哪个项目的哪部分官方文档）。
3. 若问题超出 Java / Spring 云原生后端范围，直接说明超出范围。

不要编造配置项、命令或链接。回答控制在 150 字以内。"""


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


def render_user_message(question: str, items: list[Evidence]) -> str:
    return f"【证据】\n{render_evidence(items)}\n\n【问题】{question}"


def template_version() -> str:
    """提示词内容哈希。换了提示词必须能在评测报告里区分开。"""
    blob = (SYSTEM_PROMPT + INSUFFICIENT_PROMPT).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]
