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
（比如卡片明确不支持的排他性断言）——命中记入 `forbidden_hit`，但
**不再**在正则层面自动判断这次命中是否真的构成问题（CR-054~060 的
完整教训见 `_score()` 上方注释：自动判断"这次命中是否被否定"在正则
层面被反复证明做不对，与其继续加窗口/连词打补丁，不如老实交给人看）。

每道题因此有三种状态（CR-061，CR-062/063 修正）：`passed`（无
failures，且没有未裁决的 forbidden_hit）、`failed`（有
failures——缺关键点/缺来源/拒答/生成出错；**或者** forbidden_hit 命中
一条已被人工确认为真实问题的模式）、`review_required`（无 failures，
有 forbidden_hit，但找不到匹配当前这次具体回答的人工复核结论）。
`review_required` **不计入通过数，也不能让整次评测的退出码为 0**。
人工复核结论持久化存放在 `knowledge/eval/scenario_review.yaml`，按
`(question, pattern, answer_hash)` 三元组匹配——**必须连着这次具体
回答的内容哈希一起匹配**（CR-063），不能只按题目和正则匹配：本地
模型每次生成的措辞会变，同一个 pattern 这次命中的可能是一句正确的
否定表述，下次命中的可能是一句真实的排他性断言，一条历史复核结论
不能不看内容就放行未来所有命中。`verdict: confirmed_ok` 且哈希匹配
才转为 `passed`；`verdict: confirmed_issue` 且哈希匹配转为
**`failed`**（这是一个已经完成的人工判断，不是"还没人看过"，不该
和 `review_required` 混为一谈——CR-062）；哈希不匹配（包括记录压根
不存在）一律 `review_required`，需要针对这次新出现的具体内容重新
复核。

用法:
    python packages/evaltools/run_scenarios.py [--limit N] [--offline] [--language zh|en]
"""

from __future__ import annotations

import argparse
import hashlib
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
REVIEWS = ROOT / "knowledge" / "eval" / "scenario_review.yaml"
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
    status: str = "passed"

    @property
    def keypoint_coverage(self) -> float:
        total = len(self.expect_keypoints)
        return len(self.keypoints_hit) / total if total else 1.0


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
# 打补丁：`forbid_patterns` 命中**不再做任何自动否定判定**——命中就如实
# 记入 `forbidden_hit`，由人判断这次命中到底是真实的越界断言还是已被
# 正确否定的表述。
#
# CR-061：光记入 `forbidden_hit` 并打印出来还不够——如果这依然不影响
# 这道题的通过判定，等于只是换了个说法重新引入 CR-053 想堵住的洞（已知
# 可疑断言被悄悄计入统计）。因此命中 `forbidden_hit` 但没有对应人工
# 复核确认的题，判定为独立的第三种状态 `review_required`：不算 `passed`，
# 也不能让整次评测的退出码为 0——具体的三态判定逻辑见 `_case_status()`，
# 人工复核结论的持久化格式见 `knowledge/eval/scenario_review.yaml`。
def _score(
    answer: str, expect_keypoints: list[str], forbid_patterns: list[str],
) -> tuple[list[str], list[str], list[str], list[str]]:
    """纯判定：不涉及生成或网络，可在没有 Metal/模型的环境里单测。

    返回 `(keypoints_hit, keypoints_missed, forbidden_hit, failures)`。
    `forbidden_hit` 不进入 `failures`——命中 `forbid_patterns` 只是标记
    "这段文字里出现了已知的可疑措辞，需要人工看一眼"，本身不判定这题
    是否失败；最终的三态判定（含要不要采信人工复核结论）在 `_case_
    status()` 里做，见上方 CR-054~061 的完整说明。
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


def _answer_hash(answer: str) -> str:
    """CR-063：复核结论必须绑定"人工实际看过的那次具体回答"，不能只按
    题目和 pattern 匹配——本地模型每次生成的措辞会变，同一个 pattern
    这次命中的可能是一句正确的否定表述，下次命中的可能是一句真实的
    排他性断言。取 `answer_text` 的 SHA-256 前 16 位十六进制字符作为
    这次回答的指纹，写进复核记录、也用来做查找时的匹配键。
    """
    return hashlib.sha256(answer.encode("utf-8")).hexdigest()[:16]


def _lookup_review_verdict(
    question: str, pattern: str, answer_hash: str, reviews: list[dict],
) -> str | None:
    """在 `knowledge/eval/scenario_review.yaml` 的记录里找
    `(question, pattern, answer_hash)` 这个三元组对应的人工复核结论。
    三者必须**同时**匹配——哪怕 question/pattern 都对得上，只要这次的
    `answer_hash` 和记录里的不一样（说明模型这次生成了不同的内容），
    就当作没有复核过，返回 `None`（CR-063）。
    """
    for r in reviews:
        if (
            r.get("question") == question
            and r.get("pattern") == pattern
            and r.get("answer_hash") == answer_hash
        ):
            return r.get("verdict")
    return None


def _case_status(c: ScenarioCase, reviews: list[dict]) -> str:
    """三态判定（CR-061，CR-062/063 修正）：
    `passed` / `failed` / `review_required`。

    有 `failures`（缺关键点、缺来源、拒答、生成出错）一律 `failed`，
    与 `forbidden_hit` 无关。没有 `failures` 但有 `forbidden_hit` 时，
    对每一条命中的 pattern 查找绑定了这次 `answer_hash` 的复核记录：
    - 只要有一条被人工确认为 `confirmed_issue`，整题判 **`failed`**——
      这是一个已经完成的人工判断（"我看过，这确实是问题"），不是"还
      没人看过"，不该和 `review_required` 混为一谈（CR-062）。
    - 没有 `confirmed_issue`，但至少有一条找不到匹配记录（问题+pattern
      对得上、内容对不上，或者压根没记录），判 `review_required`。
    - 只有**每一条**命中都能找到匹配当前内容的 `confirmed_ok` 记录，
      才判 `passed`。
    """
    if c.failures:
        return "failed"
    if not c.forbidden_hit:
        return "passed"
    answer_hash = _answer_hash(c.answer_text)
    verdicts = [
        _lookup_review_verdict(c.question, pattern, answer_hash, reviews)
        for pattern in c.forbidden_hit
    ]
    if any(v == "confirmed_issue" for v in verdicts):
        return "failed"
    if any(v != "confirmed_ok" for v in verdicts):
        return "review_required"
    return "passed"


def run_case(orch: Orchestrator, spec: dict, language: str, reviews: list[dict]) -> ScenarioCase:
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
        c.status = _case_status(c, reviews)
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

    c.status = _case_status(c, reviews)
    return c


def _summarize(cases: list[ScenarioCase]) -> dict:
    """不涉及生成或网络，可在没有 Metal/模型的环境里单测（CR-064）——
    R57 复审指出，`main()` 里内联的汇总/退出码计算完全没有独立测试
    覆盖，`--limit 1` 这类端到端手工验证也无法稳定命中
    `review_required` 分支来证明它确实会让退出码非零。这里把汇总和
    退出码判定拆成一个纯函数，只依赖每个 `ScenarioCase.status`，不需要
    真的跑一次评测就能测。
    """
    passed = sum(1 for c in cases if c.status == "passed")
    failed = sum(1 for c in cases if c.status == "failed")
    review_required = sum(1 for c in cases if c.status == "review_required")
    return {
        "passed": passed,
        "failed": failed,
        "review_required": review_required,
        "exit_code": 0 if passed == len(cases) else 1,
    }


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
    reviews = yaml.safe_load(REVIEWS.read_text(encoding="utf-8"))["reviews"] if REVIEWS.exists() else []

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
    marks = {"passed": "✓", "failed": "✗", "review_required": "△"}
    for i, spec in enumerate(specs, 1):
        c = run_case(orch, spec, args.language, reviews)
        cases.append(c)
        print(f"  {marks[c.status]} [{i:>2}/{len(specs)}] {c.question[:40]:<42} "
              f"关键点 {len(c.keypoints_hit)}/{len(c.expect_keypoints)} "
              f"来源 {', '.join(c.cited_projects) or '无'}")
        for f in c.failures:
            print(f"        └─ {f}")
        answer_hash = _answer_hash(c.answer_text)
        for p in c.forbidden_hit:
            verdict = _lookup_review_verdict(c.question, p, answer_hash, reviews) or "无匹配复核记录"
            print(f"        ⚠ 命中可疑模式（{verdict}）: {p!r}")

    summary = _summarize(cases)
    passed, failed, review_required = summary["passed"], summary["failed"], summary["review_required"]
    total_keypoints = sum(len(c.expect_keypoints) for c in cases)
    hit_keypoints = sum(len(c.keypoints_hit) for c in cases)
    coverage = hit_keypoints / total_keypoints if total_keypoints else 1.0

    print(f"\n{'='*60}")
    print(f"通过 {passed}/{len(cases)}（失败 {failed}，待复核 {review_required}）")
    print(f"关键点覆盖率: {hit_keypoints}/{total_keypoints}（{coverage*100:.0f}%）")
    if review_required:
        print(f"△ {review_required} 题命中可疑模式且无匹配的 confirmed_ok 复核记录，"
              f"不计入通过、整次评测不算成功——需要在 "
              f"{REVIEWS.relative_to(ROOT)} 补一条复核结论（记得带上这次的 "
              f"answer_hash，否则下次生成内容一变又会失效）：")
        for c in cases:
            if c.status == "review_required":
                print(f"    - {c.question[:50]}")
    print(f"耗时 {time.perf_counter()-t0:.0f}s")

    REPORTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = REPORTS / f"eval-scenarios-{stamp}.json"
    path.write_text(json.dumps({
        "language": args.language,
        "offline_mode": args.offline,
        "passed": passed,
        "failed": failed,
        "review_required": review_required,
        "total": len(cases),
        "keypoint_coverage": coverage,
        "cases": [vars(c) for c in cases],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"报告 {path.name}")

    store.close()
    return summary["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
