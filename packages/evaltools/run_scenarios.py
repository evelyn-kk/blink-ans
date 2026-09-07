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
        return not self.failures


def _project_of(citation: str) -> str:
    """citation 格式为 `<project> <version> · <标题路径> · 抓取于 <日期>`。

    与 run_basic.py 的同名函数刻意保持独立实现（两份评测脚本互不依赖），
    但取的都是权威字段第一段，不是从 URL 反猜——见 code-review.md CR-008。
    """
    return citation.split(" ", 1)[0] if citation else ""


# CR-054：一段答案里出现"只能用 Lua"这个字面串，不代表它在断言排他性——
# "不是只能用 Lua，也可以采用其他机制"是明确推翻这个论断的正确表述，字面
# 却包含同一个触发词组。只做字符串/正则命中判断不了极性，这里需要一个
# 能验证的启发式来判断"这次命中是不是被否定了"。这条启发式已经改过三次，
# 每次的教训都记在下面，因为这三次改动互相牵制（放宽一处会重新踩上一处
# 的坑），后续再改必须先看完这一串历史，不能只看最后一版：
#
# CR-055：发现"紧邻前文一小段窗口"如果只是直接量字符数（原为 15），不管
# 中间隔了什么，就会把**不相关的另一个命题**的否定词也当成这次命中的
# 否定标记——"不能；没有条件检查，但不只是先读再写，仍然只能用 Lua
# 脚本。"里"不只是"否定的是"先读再写"，跟后面"只能用 Lua 脚本"这句独立
# 的真实排他性断言毫无关系。当时的修法：按标点分句边界截断窗口。
#
# CR-056：指出 CR-055 的分句边界只认标点，"但是/不过"这类转折连词不产生
# 标点边界——"并非只能用 Lua 但是仍然只能用 Lua 脚本"里，没有标点分隔
# 两个命题，前一命题的"并非"照样泄漏给了后一个独立的真实排他性断言。
# 当时判断"窗口多大/按什么边界截断"这个方向本身在打不完的补丁，换成
# "否定标记必须**紧邻**触发词本身"（`str.endswith`，几乎零容忍）。
#
# CR-057：指出 CR-056 的"紧邻"矫枉过正——"并不一定只能用 Lua 脚本"
# （否定词和触发词之间隔着"一定"）、英文 "definitely not the only way to
# use Lua"（"not"和"only way"之间隔着"the"）这类**自然的修饰语**，同样
# 会被"零容忍紧邻"拒之门外，把明确的否定语境也判成了真实违规。
#
# CR-057 之后综合三轮教训：真正需要的不是"零窗口"也不是"无限窗口"，而是
# 把 CR-055 的"按分句边界截断"和 CR-056/057 指出的两个缺口一起补上：
# 分句边界不能只认标点，也要认得出"但是/不过"这类转折连词（补 CR-056
# 的洞）；边界内的窗口不能收到零容忍，要给"一定/the"这类自然修饰语留出
# 空间，但仍然要有一个不算大的上限（补 CR-057 的洞，同时不重新打开
# CR-055 的洞）。否定标记本身的词表也要覆盖"不一定/并不一定"这类比
# "不是/并非"更松散的否定构式。
#
# CR-059：指出光靠"给 `_CLAUSE_BOUNDARY` 加连词"这条路会没完没了——
# "并非只能用 Lua **且**必须使用 Lua 脚本"里，"且"是加合连词而不是转折
# 连词，没在词表里，12 字符窗口重新让"并非"泄漏给了"必须使用 Lua 脚本"
# 这句独立的真实排他性断言。中文里能起加合/转折作用的连词理论上是
# 一个开放集合（且/而/并且/而且/同时/但是/不过/然而/……），靠枚举永远
# 追不完——这正是这条启发式反复被打补丁的根本原因：它想用"这段前文里
# 有没有否定词"去回答一个本质上是句法结构的问题（"这个否定词在语法上
# 修饰的是不是这个短语"），而句法结构不是靠关键词表能穷尽的。
#
# 因此这次**没有**再往 `_CLAUSE_BOUNDARY` 里加"且/而/并且"这些词——按
# R54 指出的"收益递减"结论，继续枚举连词只是换一种反例、迟早再挨一刀。
# 换成一个不依赖连词词表也能生效的结构性约束：同一个 forbid pattern 在
# 答案里命中多次时，**后一次命中的否定检索范围不能越过前一次命中的
# 结尾**——如果否定词和当前命中之间隔着"另一个刚刚出现过的、同类的
# 真实断言"，那本身就是两个独立命题的强信号，不管中间用的是哪个连词、
# 甚至没有连词，都不需要先认出那是个连词才能生效。这条约束天然覆盖了
# CR-059 的复现句（验证见下方 `_is_negated()`/`_score()` 与测试）。
# residual 风险如实登记：这条结构性约束只覆盖"同一个 pattern 命中过
# 不止一次"的场景；如果未来出现"只命中一次、前面隔着一个还没被列进
# 词表的连词接无关否定"这种单命中场景，现有词表（标点 + 但是/不过/
# 然而/却/仍然/but/however/though/yet）依然可能漏判——这是本轮明确
# 选择不去堵的口子，理由同上：不想为了堵一个尚未被真实复现的假设场景，
# 继续往词表里加没有实际反例支撑的连词。
_NEGATION_MARKERS = ("不是", "并非", "不一定", "isn't", "not")
_NEGATION_WINDOW = 12  # 留够"并不一定"（4 字）、"definitely not the "（约 8 字符）的空间
_CLAUSE_BOUNDARY = re.compile(
    r"[，。；！？、,.;!?\n]|但是|不过|然而|却|仍然|but|however|though|yet"
)


def _is_negated(
    answer: str, match_start: int, *, lower_bound: int = 0, window: int = _NEGATION_WINDOW,
) -> bool:
    """`lower_bound`：否定标记搜索范围的硬下界——CR-059 之后，调用方在
    同一个 pattern 的多次命中之间传入上一次命中的结尾位置，防止否定词
    跨过一个独立的、更早的真实命中泄漏过来（见上方 CR-059 说明）。
    """
    clause_start = lower_bound
    for m in _CLAUSE_BOUNDARY.finditer(answer, lower_bound, match_start):
        clause_start = m.end()
    prefix = answer[max(clause_start, match_start - window):match_start]
    return any(marker in prefix for marker in _NEGATION_MARKERS)


def _score(
    answer: str, expect_keypoints: list[str], forbid_patterns: list[str],
) -> tuple[list[str], list[str], list[str], list[str]]:
    """纯判定：不涉及生成或网络，可在没有 Metal/模型的环境里单测。

    返回 `(keypoints_hit, keypoints_missed, forbidden_hit, failures)`。

    CR-053：只看"该出现的短语出现了没有"这一种判据，漏掉了另一种失败
    模式——答案里混进一句**已知错误的越界断言**，却因为凑巧命中了两条
    比较宽泛的正向 `expect_keypoints`（例如"不能"/"无法…判断"）而被记成
    "关键点全部命中"，实际上这段解释本身是错的（R48/R49 已指出
    `RedisAtomicLong` 断言"仅支持原子增减""必须用 Lua"这类说法超出了
    引用来源的支撑范围，见 CR-052/CR-053）。`forbid_patterns` 是可选的
    负向判据，用来登记"已知这句话是错的，不能因为正向关键点凑巧命中就
    判通过"——命中任一条就判失败，不管正向关键点是否已经全部命中。

    CR-054：命中判断按 `_is_negated()` 做极性过滤——一个 forbid pattern
    在答案里出现多次时，只要有一次命中不是被否定的，就记为命中；如果
    每一次命中前面都紧跟着否定标记，则不算命中这条 forbid pattern。

    CR-055/CR-056/CR-057：`_is_negated()` 按分句边界（含标点与"但是/
    不过"这类转折连词）截断否定标记的搜索范围，边界内留一个不大的窗口
    容纳"一定/the"这类自然修饰语——见 `_is_negated()` 定义处的完整说明，
    含四轮教训的具体反例。

    CR-059：同一个 pattern 的多次命中之间，后一次命中的否定检索不能
    越过前一次命中的结尾——防止"并非只能用 Lua **且**必须使用 Lua
    脚本"这类中间隔着一个还没被列进连词词表的加合词时，前一个命中的
    否定标记泄漏给后一个独立的真实命中。
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
        matches = list(re.finditer(pattern, answer))
        lower_bound = 0
        has_real_violation = False
        for m in matches:
            if not _is_negated(answer, m.start(), lower_bound=lower_bound):
                has_real_violation = True
            lower_bound = m.end()
        if has_real_violation:
            forbidden.append(pattern)
            failures.append(f"命中禁止模式（已知的越界/错误论断）: {pattern!r}")
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
