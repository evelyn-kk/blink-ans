"""T-011 场景评测雏形：关键点覆盖率 + 来源命中，而不只是"有没有引用"。

`run_basic.py`（I2）只验证结构性属性：有没有来源、拒答判据对不对，
明确不评判答案的技术正确性——那是 I3 的场景评测要做的事。这里补上
"答案是否真的说中了关键点"这一层：`knowledge/eval/scenario_questions.yaml`
里每题的 `expect_keypoints` 是人工核对过、场景卡片正文确实包含的具体
事实点（正则，不区分大小写），`expect_sources` 是答案引用中必须全部
出现的来源项目。

判据用确定性正则而非 LLM 判分——延续 `ranking_probe.yaml`"金标准要人工
核对"与 `AGENTS.md` §5.3"判据不得承诺未测量的因果"的方法论，不引入本
项目尚未验证过的 LLM-judge 环节。但这也意味着 `expect_keypoints` 必须
贴着模型的真实措辞校准，不能靠猜——`development-notes.md` 2026-08-31
"答案不可逐字复现"讲的是同一个坑：新增关键点后必须实跑一遍，看它是否
真的在多次生成里稳定命中，而不是断言一个从没在真实输出里出现过的短语。

可选的 `forbid_patterns`（每题一组正则）用来标记"已知的可疑/越界措辞"
（比如卡片明确不支持的排他性断言）——命中只会被记入 `forbidden_hit`
并在输出里显著标出"待人工复核"，**不会**自动判定这道题失败（CR-054~
060 的完整教训见 `_score()` 上方注释：自动判断"这次命中是否被否定"
在正则层面被反复证明做不对，与其继续加窗口/连词打补丁，不如老实交给
人看）。

用法:
    python packages/evaltools/run_scenarios.py [--limit N] [--offline] [--language zh|en]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import yaml  # noqa: E402

from packages.config.env import load_dotenv  # noqa: E402
from packages.prompts.answer import SUPPORTED_LANGUAGES, system_prompt  # noqa: E402
from services.inference.backend import LocalBackend  # noqa: E402
from services.inference.claude_backend import ClaudeBackend  # noqa: E402
from services.inference.engine import DEFAULT_MODEL, InferenceEngine  # noqa: E402
from services.inference.router import Router  # noqa: E402
from services.orchestrator.answering import (  # noqa: E402
    AnswerConfig, AnswerRequest, Orchestrator, declined,
)
from services.retrieval.embed import Embedder  # noqa: E402
from services.retrieval.store import ChunkStore  # noqa: E402

QUESTIONS = ROOT / "knowledge" / "eval" / "scenario_questions.yaml"
REPORTS = ROOT / "bench" / "reports"


@dataclass
class ScenarioCase:
    question: str
    expect_keypoints: list[str]
    expect_sources: list[str]
    forbid_patterns: list[str] = field(default_factory=list)
    answer_text: str = ""
    sufficiency: str = ""
    served_by: str = ""
    ttft_s: float = 0.0
    declined: bool = False
    sources: int = 0
    cited_projects: list[str] = field(default_factory=list)
    keypoints_hit: list[str] = field(default_factory=list)
    keypoints_missed: list[str] = field(default_factory=list)
    forbidden_hit: list[str] = field(default_factory=list)
    sources_missed: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    @property
    def keypoint_coverage(self) -> float:
        total = len(self.expect_keypoints)
        return len(self.keypoints_hit) / total if total else 1.0

    @property
    def ok(self) -> bool:
        """`forbid_patterns` 命中**不计入**这里——见 `forbidden_hit` 字段
        与 `_score()` 顶部关于 CR-060 的说明：它是"标记待人工复核"，
        不再是自动判定失败的判据。"""
        return not self.failures


def _project_of(citation: str) -> str:
    """citation 格式为 `<project> <version> · <标题路径> · 抓取于 <日期>`。

    与 run_basic.py 的同名函数刻意保持独立实现（两份评测脚本互不依赖），
    但取的都是权威字段第一段，不是从 URL 反猜——见 code-review.md CR-008。
    """
    return citation.split(" ", 1)[0] if citation else ""


# CR-054~059：`forbid_patterns` 原本想在正则层面自动判断"这次命中是不是
# 被否定了"（"不是只能用 Lua，也可以采用其他机制"字面包含"只能…Lua"这个
# 触发词组，但明确推翻了排他性主张，不该算命中）。这条自动否定检测启发式
# 先后被找到六个反例（CR-055 跨命题窗口泄漏 → CR-056 分句边界不认转折
# 连词 → CR-057"零容忍紧邻"矫枉过正 → CR-059 分句边界不认加合连词 →
# CR-060 单命中场景下加合连词接无关否定仍会泄漏），每次的修法都在
# "窗口多大/边界怎么划/词表要不要加词"这几个维度来回打补丁，因为它想用
# "这段前文有没有否定词"回答一个本质上是句法结构的问题（"这个否定词在
# 语法上修饰的是不是这个短语"），而中文/英文的连接词理论上是一个开放
# 集合，枚举永远追不完，句法结构也不是关键词表能穷尽的。
#
# CR-060（R55 复审、六个反例里的最后一个）之后决定停止在这条路上继续
# 打补丁：`forbid_patterns` 命中**不再做任何自动否定判定**，也**不再
# 自动计入 `failures`/影响 `ok`**——命中就如实记入 `forbidden_hit` 并在
# CLI 输出、JSON 报告里显著标出"待人工复核"，由人判断这次命中到底是
# 真实的越界断言还是已被正确否定的表述。这不是放弃 CR-053 想要的东西
# （"已知的错误论断不能被正向关键点掩盖而悄悄放过"）——`forbidden_hit`
# 依然会被打印出来、写进报告，不会像 CR-053 修复前那样完全没有痕迹；
# 放弃的只是"自动分辨这次命中是否被否定"这一步，把它交还给人工。
def _score(
    answer: str, expect_keypoints: list[str], forbid_patterns: list[str],
) -> tuple[list[str], list[str], list[str], list[str]]:
    """纯判定：不涉及生成或网络，可在没有 Metal/模型的环境里单测。

    返回 `(keypoints_hit, keypoints_missed, forbidden_hit, failures)`。
    `forbidden_hit` 不进入 `failures`——命中 `forbid_patterns` 只是标记
    "这段文字里出现了已知的可疑措辞，需要人工看一眼"，不再自动判定这题
    失败（见上方 CR-054~060 的完整说明）。
    """
    hit: list[str] = []
    missed: list[str] = []
    forbidden: list[str] = []
    failures: list[str] = []
    for pattern in expect_keypoints:
        if re.search(pattern, answer):
            hit.append(pattern)
        else:
            missed.append(pattern)
            failures.append(f"未命中关键点: {pattern!r}")
    for pattern in forbid_patterns:
        if re.search(pattern, answer):
            forbidden.append(pattern)
    return hit, missed, forbidden, failures


def run_case(orch: Orchestrator, spec: dict, language: str) -> ScenarioCase:
    c = ScenarioCase(
        question=spec["q"],
        expect_keypoints=list(spec.get("expect_keypoints", [])),
        expect_sources=list(spec.get("expect_sources", [])),
        forbid_patterns=list(spec.get("forbid_patterns", [])),
    )
    answer = ""
    for ev in orch.answer(AnswerRequest(question=c.question, max_tokens=400, language=language)):
        t = ev["type"]
        if t == "retrieval":
            c.sufficiency = ev["sufficiency"]
        elif t == "answer_delta":
            answer += ev["text"]
        elif t == "sources":
            c.sources = len(ev["items"])
            c.cited_projects = sorted({_project_of(i["citation"]) for i in ev["items"]})
        elif t == "done":
            c.ttft_s = ev["ttft_s"]
            c.served_by = ev.get("served_by", "")
        elif t == "error":
            c.failures.append(f"错误 {ev['stage']}: {ev['message']}")

    c.answer_text = answer
    c.declined = declined(answer)

    if c.declined:
        c.failures.append("模型判定证据不足而拒答，本题按设计应有覆盖场景卡片可回答")
        return c

    c.keypoints_hit, c.keypoints_missed, c.forbidden_hit, kp_failures = _score(
        answer, c.expect_keypoints, c.forbid_patterns,
    )
    c.failures.extend(kp_failures)

    for project in c.expect_sources:
        if project not in c.cited_projects:
            c.sources_missed.append(project)
            c.failures.append(
                f"引用中缺少期望来源 {project!r}（实际 {', '.join(c.cited_projects) or '无'}）"
            )

    return c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--offline", action="store_true",
                     help="强制走本地兜底，不尝试云端 Claude（省钱/可复现）")
    ap.add_argument("--language", choices=SUPPORTED_LANGUAGES, default="zh")
    args = ap.parse_args()
    load_dotenv(ROOT / ".env")

    specs = yaml.safe_load(QUESTIONS.read_text(encoding="utf-8"))["questions"]
    if args.limit:
        specs = specs[: args.limit]

    prompt = system_prompt(args.language)
    engine = InferenceEngine(DEFAULT_MODEL)
    engine.load(prompt)
    if not engine.status.loaded:
        print(f"模型加载失败: {engine.status.error}", file=sys.stderr)
        return 2
    embedder = Embedder(); embedder.load()
    store = ChunkStore()
    router = Router(LocalBackend(engine), ClaudeBackend(prompt), offline=args.offline)
    orch = Orchestrator(store, embedder, router, config=AnswerConfig(default_language=args.language))

    print(f"运行 {len(specs)} 题场景评测（语言 {args.language}）\n")
    cases: list[ScenarioCase] = []
    t0 = time.perf_counter()
    for i, spec in enumerate(specs, 1):
        c = run_case(orch, spec, args.language)
        cases.append(c)
        mark = "✓" if c.ok else "✗"
        print(f"  {mark} [{i:>2}/{len(specs)}] {c.question[:40]:<42} "
              f"关键点 {len(c.keypoints_hit)}/{len(c.expect_keypoints)} "
              f"来源 {', '.join(c.cited_projects) or '无'}")
        for f in c.failures:
            print(f"        └─ {f}")
        for p in c.forbidden_hit:
            print(f"        ⚠ 命中可疑模式（未计入判定，需人工复核）: {p!r}")

    passed = sum(1 for c in cases if c.ok)
    total_keypoints = sum(len(c.expect_keypoints) for c in cases)
    hit_keypoints = sum(len(c.keypoints_hit) for c in cases)
    coverage = hit_keypoints / total_keypoints if total_keypoints else 1.0
    flagged = [c for c in cases if c.forbidden_hit]

    print(f"\n{'='*60}")
    print(f"通过 {passed}/{len(cases)}")
    print(f"关键点覆盖率: {hit_keypoints}/{total_keypoints}（{coverage*100:.0f}%）")
    if flagged:
        print(f"⚠ {len(flagged)} 题命中可疑模式，需人工复核（未计入上面的通过/失败判定）：")
        for c in flagged:
            print(f"    - {c.question[:50]}")
    print(f"耗时 {time.perf_counter()-t0:.0f}s")

    REPORTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = REPORTS / f"eval-scenarios-{stamp}.json"
    path.write_text(json.dumps({
        "language": args.language,
        "offline_mode": args.offline,
        "passed": passed,
        "total": len(cases),
        "keypoint_coverage": coverage,
        "cases": [vars(c) for c in cases],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"报告 {path.name}")

    store.close()
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
