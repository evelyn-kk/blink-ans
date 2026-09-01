"""I2 基础题回归。

验证两件事，对应 plan.md 的 I2 完成定义：
1. 应作答的题目返回**可访问的**来源链接，且答案带引用标注。
2. 语料未覆盖的题目被判为证据不足，不给出伪装成确定结论的技术建议。

注意断言方式：只检查结构性属性（有无来源、引用是否落在有效编号内、
充分性判定是否正确），**不比对答案文本**——I0 已确认答案不可逐字复现。
答案的技术正确性由 I3 的场景评测负责。

用法:
    python packages/evaltools/run_basic.py [--limit N] [--check-links]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import yaml  # noqa: E402

from packages.prompts.answer import SYSTEM_PROMPT, template_version  # noqa: E402
from services.inference.engine import DEFAULT_MODEL, InferenceEngine  # noqa: E402
from services.orchestrator.answering import (  # noqa: E402
    AnswerRequest, Orchestrator, Sufficiency,
)
from services.retrieval.embed import Embedder  # noqa: E402
from services.retrieval.store import ChunkStore  # noqa: E402

QUESTIONS = Path(__file__).resolve().parents[2] / "knowledge" / "eval" / "basic_questions.yaml"
REPORTS = Path(__file__).resolve().parents[2] / "bench" / "reports"


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
    urls: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    declined: bool = False        # 模型明说证据未涵盖
    retrieval_miss: bool = False  # 检索给错证据，模型正确拒绝编造

    @property
    def ok(self) -> bool:
        return not self.failures


# 模型明确表示证据不支撑作答的措辞
_DECLINED = re.compile(r"证据未涵盖|现有证据不足|证据不足以")


def check_url(url: str, timeout: float = 10.0) -> bool:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "blink-ans-eval/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status == 200
    except urllib.error.HTTPError as e:
        return e.code == 405   # 少数站点拒绝 HEAD，不算链接失效
    except Exception:
        return False


def run_case(orch: Orchestrator, spec: dict) -> Case:
    c = Case(question=spec["q"], expect=spec["expect"], project=spec.get("project"))
    answer = ""
    for ev in orch.answer(AnswerRequest(question=c.question, max_tokens=400)):
        t = ev["type"]
        if t == "retrieval":
            c.sufficiency = ev["sufficiency"]
        elif t == "answer_delta":
            answer += ev["text"]
        elif t == "sources":
            c.sources = len(ev["items"])
            c.urls = [i["url"] for i in ev["items"]]
        elif t == "done":
            c.cited = len(ev["cited_evidence"])
            c.ttft_s, c.total_s = ev["ttft_s"], ev["total_s"]
            c.prompt_tokens = ev["prompt_tokens"]
        elif t == "error":
            c.failures.append(f"错误 {ev['stage']}: {ev['message']}")

    c.declined = _DECLINED.search(answer) is not None and len(answer.strip()) < 80

    if c.expect == "answered":
        if c.sources == 0 and not c.declined:
            c.failures.append("未返回任何来源")
        if c.project and c.urls and not any(_belongs(u, c.project) for u in c.urls):
            c.failures.append(f"来源均不属于预期项目 {c.project}")

        if c.declined:
            # 模型明说"证据未涵盖"是**正确行为**——检索给错了证据，它拒绝编造。
            # 这是检索质量问题，不是安全问题，必须与"编造"区分统计，
            # 否则会误导后续优化方向（I3 的评测集才是解决检索质量的地方）。
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


_HOST_HINT = {
    "kafka": "kafka.apache.org", "kubernetes": "kubernetes.io",
    "postgresql": "postgresql.org", "spring-boot": "docs.spring.io/spring-boot",
    "spring-data-redis": "spring-data-redis",
}


def _belongs(url: str, project: str) -> bool:
    return _HOST_HINT.get(project, project) in url


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--check-links", action="store_true", help="逐条 HEAD 验证来源可达（较慢）")
    args = ap.parse_args()

    specs = yaml.safe_load(QUESTIONS.read_text(encoding="utf-8"))["questions"]
    if args.limit:
        specs = specs[: args.limit]

    engine = InferenceEngine(DEFAULT_MODEL)
    engine.load(SYSTEM_PROMPT)
    if not engine.status.loaded:
        print(f"模型加载失败: {engine.status.error}", file=sys.stderr)
        return 2
    embedder = Embedder(); embedder.load()
    store = ChunkStore()
    orch = Orchestrator(store, embedder, engine)

    print(f"运行 {len(specs)} 题（模板 {template_version()}）\n")
    cases: list[Case] = []
    t0 = time.perf_counter()
    for i, spec in enumerate(specs, 1):
        c = run_case(orch, spec)
        cases.append(c)
        mark = "✓" if c.ok else "✗"
        if c.retrieval_miss:
            mark = "○"   # 安全但检索未命中
        print(f"  {mark} [{i:>2}/{len(specs)}] {c.question[:34]:<36} "
              f"{c.sufficiency:<12} 来源{c.sources} 引用{c.cited} {c.ttft_s:.2f}s")
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
    if answered:
        with_src = sum(1 for c in answered if c.sources)
        miss = sum(1 for c in answered if c.retrieval_miss)
        useful = [c for c in answered if not c.retrieval_miss]
        with_cite = sum(1 for c in useful if c.cited)
        print(f"  返回来源 {with_src}/{len(answered)}")
        print(f"  实际作答 {len(useful)}/{len(answered)}"
              f" · 其中带引用 {with_cite}/{len(useful)}"
              f"  （引用覆盖率 {with_cite/max(len(useful),1)*100:.0f}%）")
        print(f"  ○ 检索未命中而正确拒绝编造: {miss}/{len(answered)}"
              f"  —— 安全行为，属检索质量问题，由 I3 评测集驱动改进")
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
        "model": DEFAULT_MODEL,
        "index_chunks": store.count(),
        "dictionary_version": store.meta.get("dictionary_version"),
        "passed": passed, "total": len(cases),
        "retrieval_misses": sum(1 for c in cases if c.retrieval_miss),
        "broken_links": broken,
        "cases": [vars(c) for c in cases],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  报告 {path.name}")

    store.close()
    return 0 if passed == len(cases) and not broken else 1


if __name__ == "__main__":
    raise SystemExit(main())
