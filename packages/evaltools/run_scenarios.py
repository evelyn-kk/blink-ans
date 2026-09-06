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
    answer_text: str = ""
    sufficiency: str = ""
    served_by: str = ""
    ttft_s: float = 0.0
    declined: bool = False
    sources: int = 0
    cited_projects: list[str] = field(default_factory=list)
    keypoints_hit: list[str] = field(default_factory=list)
    keypoints_missed: list[str] = field(default_factory=list)
    sources_missed: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    @property
    def keypoint_coverage(self) -> float:
        total = len(self.expect_keypoints)
        return len(self.keypoints_hit) / total if total else 1.0

    @property
    def ok(self) -> bool:
        return not self.failures


def _project_of(citation: str) -> str:
    """citation 格式为 `<project> <version> · <标题路径> · 抓取于 <日期>`。

    与 run_basic.py 的同名函数刻意保持独立实现（两份评测脚本互不依赖），
    但取的都是权威字段第一段，不是从 URL 反猜——见 code-review.md CR-008。
    """
    return citation.split(" ", 1)[0] if citation else ""


def run_case(orch: Orchestrator, spec: dict, language: str) -> ScenarioCase:
    c = ScenarioCase(
        question=spec["q"],
        expect_keypoints=list(spec.get("expect_keypoints", [])),
        expect_sources=list(spec.get("expect_sources", [])),
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

    for pattern in c.expect_keypoints:
        if re.search(pattern, answer):
            c.keypoints_hit.append(pattern)
        else:
            c.keypoints_missed.append(pattern)
            c.failures.append(f"未命中关键点: {pattern!r}")

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

    passed = sum(1 for c in cases if c.ok)
    total_keypoints = sum(len(c.expect_keypoints) for c in cases)
    hit_keypoints = sum(len(c.keypoints_hit) for c in cases)
    coverage = hit_keypoints / total_keypoints if total_keypoints else 1.0

    print(f"\n{'='*60}")
    print(f"通过 {passed}/{len(cases)}")
    print(f"关键点覆盖率: {hit_keypoints}/{total_keypoints}（{coverage*100:.0f}%）")
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
