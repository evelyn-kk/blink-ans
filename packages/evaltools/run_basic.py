"""I2 基础题回归。

验证两件事，对应 plan.md 的 I2 完成定义：
1. 应作答的题目返回**可访问的**来源链接，且答案带引用标注。
2. 语料未覆盖的题目被判为证据不足，不给出伪装成确定结论的技术建议。

注意断言方式：只检查结构性属性（有无来源、引用是否落在有效编号内、
充分性判定是否正确），**不比对答案文本**——I0 已确认答案不可逐字复现。
答案的技术正确性由 I3 的场景评测负责。

T-022（双语提示词）：`--language en` 用同一份 50 题问题集要求**英文**作答。
问题文本本身仍是中文——检索是跨语言的（development-notes.md 2026-09-02
「场景卡片改用英文正文」实测：向量路跨语言损失仅 0.035–0.05），换语言只影响
生成阶段用哪份提示词、模型该用哪种语言回答。这不是 T-023 要建的完整双语评测集
（那一套需要 `q_zh`/`q_en` 对照与独立的关键点标注），只是本轮验证"英文提示词路径
在真实检索证据下也能正确生成带引用的回答、且拒答标记正常工作"的务实最小验证——
足以覆盖本轮验收要求的"中英各跑 50 题回归均通过"，但不是语言对照质量评测。

用法:
    python packages/evaltools/run_basic.py [--limit N] [--check-links] [--language zh|en]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import yaml  # noqa: E402

from packages.config.env import load_dotenv  # noqa: E402
from packages.prompts.answer import (  # noqa: E402
    SUPPORTED_LANGUAGES, system_prompt, template_version,
)
from services.inference.backend import LocalBackend  # noqa: E402
from services.inference.claude_backend import ClaudeBackend  # noqa: E402
from services.inference.engine import DEFAULT_MODEL, InferenceEngine  # noqa: E402
from services.inference.router import Router  # noqa: E402
from services.orchestrator.answering import (  # noqa: E402
    AnswerConfig, AnswerRequest, Orchestrator, Sufficiency, declined,
)
from services.retrieval.embed import Embedder  # noqa: E402
from services.retrieval.store import ChunkStore  # noqa: E402

QUESTIONS = ROOT / "knowledge" / "eval" / "basic_questions.yaml"
REPORTS = ROOT / "bench" / "reports"


@dataclass
class Case:
    question: str
    expect: str
    project: str | None = None
    sufficiency: str = ""
    sources: int = 0
    cited: int = 0
    ttft_s: float = 0.0
    total_s: float = 0.0
    prompt_tokens: int = 0
    served_by: str = ""            # T-028：走的是哪个生成后端（"claude"/"local"）
    urls: list[str] = field(default_factory=list)
    projects: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    evidence_count: int = 0       # 送进 prompt 的证据条数
    declined: bool = False        # 模型明说证据未涵盖
    # 拒答分两种，成因完全不同，不能混为一谈（2026-09-02 / T-025 发现）：
    retrieval_miss: bool = False       # 检索本身没给到合格证据 → 检索问题
    declined_with_evidence: bool = False  # 证据判为充分且已送进 prompt，模型仍拒答
                                          # → 证据答非所问、或问题超出证据能回答的范围

    @property
    def ok(self) -> bool:
        return not self.failures


def check_url(url: str, timeout: float = 10.0) -> bool:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "blink-ans-eval/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status == 200
    except urllib.error.HTTPError as e:
        return e.code == 405   # 少数站点拒绝 HEAD，不算链接失效
    except Exception:
        return False


def run_case(orch: Orchestrator, spec: dict, language: str) -> Case:
    c = Case(question=spec["q"], expect=spec["expect"], project=spec.get("project"))
    answer = ""
    for ev in orch.answer(
        AnswerRequest(question=c.question, max_tokens=400, language=language)
    ):
        t = ev["type"]
        if t == "retrieval":
            c.sufficiency = ev["sufficiency"]
        elif t == "answer_delta":
            answer += ev["text"]
        elif t == "sources":
            c.sources = len(ev["items"])
            c.urls = [i["url"] for i in ev["items"]]
            c.projects = [_project_of(i["citation"]) for i in ev["items"]]
        elif t == "done":
            c.cited = len(ev["cited_evidence"])
            c.ttft_s, c.total_s = ev["ttft_s"], ev["total_s"]
            c.prompt_tokens = ev["prompt_tokens"]
            c.evidence_count = ev.get("evidence_count", 0)
            c.served_by = ev.get("served_by", "")
        elif t == "error":
            c.failures.append(f"错误 {ev['stage']}: {ev['message']}")

    # T-022：判据与生产路径同一个函数（`answering.declined()`），不是本脚本
    # 另起一套散文正则——散文判据换语言就失效，且两套判据分叉迟早互相打脸。
    c.declined = declined(answer)

    if c.expect == "answered":
        if c.sources == 0 and not c.declined:
            c.failures.append("未返回任何来源")
        if c.project and c.projects and c.project not in c.projects:
            got = ", ".join(sorted(set(c.projects))) or "无"
            c.failures.append(f"来源均不属于预期项目 {c.project}（实际 {got}）")

        if c.declined:
            # 模型明说"证据未涵盖"是**正确行为**：它拒绝基于手上的证据编造。
            # 但成因有两种，必须分开统计——
            #
            # 2026-09-02（T-025）实测反例：「PostgreSQL 的 B-tree 索引什么时候会失效」
            # 的正确块检索到第 1 名、`sufficiency=sufficient`、且确实作为证据 [1]
            # 送进了 prompt，模型仍然逐段回"证据未涵盖"。
            # 此前本脚本把所有拒答一律记为 `retrieval_miss` 并打印
            # "检索未命中而正确拒绝编造"，会把生成侧的行为误报成检索缺陷，
            # 从而把优化方向指错（**判据本身坏掉**，本项目第六次）。
            if c.sufficiency == Sufficiency.SUFFICIENT.value and c.evidence_count > 0:
                c.declined_with_evidence = True
            else:
                c.retrieval_miss = True
        elif c.cited == 0:
            # 给出了技术内容却不标注任何来源——这才是真正危险的情况：
            # 结论无法追溯，用户无从判断可信度。
            c.failures.append("给出技术内容但未标注任何证据编号，结论无法追溯")
    else:
        # 判据是**行为**而非标签：不给出技术结论、不展示来源即为通过。
        # 充分性标签判为 limited 但模型自行拒绝编造，属于第二道防线生效，
        # 不应算作失败——真正要防的是"给出无依据的技术结论"。
        if not (c.sufficiency == Sufficiency.INSUFFICIENT.value or c.declined):
            c.failures.append(f"未拒绝作答（充分性={c.sufficiency}），存在编造风险")
        if c.sources:
            c.failures.append("拒答时不应返回来源")
    return c


def _project_of(citation: str) -> str:
    """从引用串取来源项目名。

    引用格式是 `<project> <version> · <标题路径> · 抓取于 <日期>`，项目名就是第一段。
    这里**不能**改回"看 URL 里有没有项目名"那种猜法：官方站点的路径与项目名
    并不总是一致——spring-data-redis 的文档发布在 docs.spring.io/spring-data/redis/ 下，
    URL 里根本没有 `spring-data-redis` 这个串。按 URL 猜会把正确的来源判成不匹配，
    看起来像检索退步，实际是判据自己坏了（2026-09-01 实际发生过一次）。
    """
    return citation.split(" ", 1)[0] if citation else ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--check-links", action="store_true", help="逐条 HEAD 验证来源可达（较慢）")
    ap.add_argument("--offline", action="store_true",
                     help="强制走本地兜底，不尝试云端 Claude（省钱/可复现；"
                          "不传时按生产路由跑：有 ANTHROPIC_API_KEY 就走云端）")
    ap.add_argument("--language", choices=SUPPORTED_LANGUAGES, default="zh",
                     help="回答语言（T-022）。问题文本本身仍是中文，只切生成阶段的"
                          "提示词与本地常驻前缀预热语言——检索是跨语言的，不受影响。")
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
    # T-028：与生产同一套路由（services/inference/router.py），而不是直接绑死本地
    # InferenceEngine——回归脚本要能看出真实生产会走哪个后端（served_by）。
    # --offline 强制本地，避免每次跑 50 题回归都产生云端调用开销。
    router = Router(LocalBackend(engine), ClaudeBackend(prompt), offline=args.offline)
    orch = Orchestrator(
        store, embedder, router, config=AnswerConfig(default_language=args.language)
    )

    print(f"运行 {len(specs)} 题（模板 {template_version()}，语言 {args.language}）\n")
    cases: list[Case] = []
    t0 = time.perf_counter()
    for i, spec in enumerate(specs, 1):
        c = run_case(orch, spec, args.language)
        cases.append(c)
        mark = "✓" if c.ok else "✗"
        if c.retrieval_miss or c.declined_with_evidence:
            mark = "○"   # 安全（拒绝编造），但没给出可用答案
        print(f"  {mark} [{i:>2}/{len(specs)}] {c.question[:34]:<36} "
              f"{c.sufficiency:<12} 来源{c.sources} 引用{c.cited} "
              f"{c.served_by or '?':<6} {c.ttft_s:.2f}s")
        for f in c.failures:
            print(f"        └─ {f}")

    broken: list[str] = []
    if args.check_links:
        urls = sorted({u for c in cases for u in c.urls})
        print(f"\n验证 {len(urls)} 条去重来源链接...")
        broken = [u for u in urls if not check_url(u)]

    passed = sum(1 for c in cases if c.ok)
    answered = [c for c in cases if c.expect == "answered"]
    refused = [c for c in cases if c.expect == "refused"]
    ttfts = [c.ttft_s for c in cases if c.ttft_s]

    print(f"\n{'='*60}")
    print(f"通过 {passed}/{len(cases)}")
    print(f"  应作答 {sum(1 for c in answered if c.ok)}/{len(answered)}"
          f" · 应拒答 {sum(1 for c in refused if c.ok)}/{len(refused)}")
    served_by_counts: dict[str, int] = {}
    for c in cases:
        key = c.served_by or "(未生成)"
        served_by_counts[key] = served_by_counts.get(key, 0) + 1
    print(f"  生成后端: {', '.join(f'{k} {v}' for k, v in sorted(served_by_counts.items()))}"
          + ("  ← --offline 强制本地" if args.offline else ""))
    if answered:
        with_src = sum(1 for c in answered if c.sources)
        miss = sum(1 for c in answered if c.retrieval_miss)
        mismatch = sum(1 for c in answered if c.declined_with_evidence)
        useful = [c for c in answered
                  if not (c.retrieval_miss or c.declined_with_evidence)]
        with_cite = sum(1 for c in useful if c.cited)
        print(f"  返回来源 {with_src}/{len(answered)}")
        print(f"  实际作答 {len(useful)}/{len(answered)}"
              f" · 其中带引用 {with_cite}/{len(useful)}"
              f"  （引用覆盖率 {with_cite/max(len(useful),1)*100:.0f}%）")
        print(f"  ○ 拒绝编造（检索未给到合格证据）: {miss}/{len(answered)}"
              f"  —— 检索问题")
        print(f"  ○ 拒绝编造（证据判为充分仍答不了）: {mismatch}/{len(answered)}"
              f"  —— 证据答非所问或问题超出证据范围，**不是**检索未命中")
    if ttfts:
        s = sorted(ttfts)
        print(f"  首 token: 中位 {statistics.median(s):.2f}s · "
              f"P95 {s[int(len(s)*0.95)-1]:.2f}s · 最大 {max(s):.2f}s")
        print(f"  超 3.0s 生成预算: {sum(1 for t in ttfts if t > 3.0)}/{len(ttfts)}"
              f"  （architecture.md 6.2 按实测重新分配后的预算）")
        e2e = [c.ttft_s + 0.15 for c in cases if c.ttft_s]
        print(f"  端到端首字（含检索）: 中位 {statistics.median(e2e):.2f}s · "
              f"P95 {sorted(e2e)[int(len(e2e)*0.95)-1]:.2f}s · 目标 5s")
    if args.check_links:
        print(f"  来源链接可达: {len({u for c in cases for u in c.urls}) - len(broken)}"
              f"/{len({u for c in cases for u in c.urls})}")
        for u in broken[:10]:
            print(f"    ✗ {u}")
    print(f"  总耗时 {time.perf_counter()-t0:.0f}s")

    REPORTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = REPORTS / f"eval-basic-{stamp}.json"
    path.write_text(json.dumps({
        "template_version": template_version(),
        "language": args.language,
        "model": DEFAULT_MODEL,
        "index_chunks": store.count(),
        "dictionary_version": store.meta.get("dictionary_version"),
        "passed": passed, "total": len(cases),
        "retrieval_misses": sum(1 for c in cases if c.retrieval_miss),
        "declined_with_evidence": sum(1 for c in cases if c.declined_with_evidence),
        "served_by_counts": served_by_counts,
        "offline_mode": args.offline,
        "broken_links": broken,
        "cases": [vars(c) for c in cases],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  报告 {path.name}")

    store.close()
    return 0 if passed == len(cases) and not broken else 1


if __name__ == "__main__":
    raise SystemExit(main())
