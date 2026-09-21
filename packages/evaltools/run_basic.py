"""I2 基础题回归。

验证两件事，对应 plan.md 的 I2 完成定义：
1. 应作答的题目返回**可访问的**来源链接，且答案带引用标注。
2. 语料未覆盖的题目被判为证据不足，不给出伪装成确定结论的技术建议。

T-023 将每道题显式写成 `q_zh` / `q_en`，而不是在运行时翻译中文题。两种题面共用
同一条 `expect_keypoints`（中英文都可命中的事实正则）和 `expect_sources`，所以报告能
按语言比较事实、来源与拒答结果。关键点是确定性文字判据，不是 LLM 判分；在真正跑
本地/云端模型校准前，它只如实报告命中，不能被写成“英文更快或更准”的结论。

用法:
    python packages/evaltools/run_basic.py [--limit N] [--check-links] \\
        [--language zh] [--language en]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
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
from services.retrieval.store import ChunkStore, index_fingerprint  # noqa: E402

QUESTIONS = ROOT / "knowledge" / "eval" / "basic_questions.yaml"
REPORTS = ROOT / "bench" / "reports"
_FULL_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")


@dataclass
class Case:
    id: str
    language: str
    question: str
    expect: str
    expect_keypoints: list[str] = field(default_factory=list)
    expect_sources: list[str] = field(default_factory=list)
    answer_text: str = ""           # 事实判据命中/漏失必须可人工回读，不只留计数
    sufficiency: str = ""
    sources: int = 0
    cited: int = 0
    ttft_s: float = 0.0
    total_s: float = 0.0
    prompt_tokens: int = 0
    served_by: str = ""            # T-028：走的是哪个生成后端（"claude"/"local"）
    # T-029：cache 命中与成本埋点，字段名与 done 事件保持一致。云端没命中缓存、
    # 或本地场景（没有这个概念）时为 None，不伪造成 0——与 done 事件同一口径。
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    prefix_reused: bool = False
    cost_usd: float = 0.0          # --offline 模式下应恒为 0，见 main() 里的隐含正确性检查
    urls: list[str] = field(default_factory=list)
    projects: list[str] = field(default_factory=list)
    # 评测报告必须保留实际进入 prompt 的来源身份；只留 URL 时，同一页被切成多块会
    # 无法在事后审计到底选了哪一块（CR-145）。顺序是 sources 事件的原始顺序，
    # `index` 是生成 prompt 使用的引用编号，二者都不能从 URL 或 DB 重算代替。
    selected_evidence: list[dict[str, int | str]] = field(default_factory=list)
    keypoints_hit: list[str] = field(default_factory=list)
    keypoints_missed: list[str] = field(default_factory=list)
    sources_missed: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    evidence_count: int = 0       # 送进 prompt 的证据条数
    # 登记正则在**证据文本**里的出现情况（`keypoint_evidence()` 填），与
    # `keypoints_hit`（看答案）是两处不同的文本，合起来才分得清
    # 「答案写了但证据里没有」和「证据里有但答案没写」。不参与通过判定。
    keypoint_evidence: list[dict] = field(default_factory=list)
    evidence_rows_unverified: int = 0   # 按 chunk_id 回查后身份对不上的证据行
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


def runtime_git_commit() -> str:
    """在评测启动时钉住实现身份，不能等报告写盘时再读取漂移的 HEAD。"""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD^{commit}"],
            cwd=ROOT, text=True, capture_output=True, check=False,
        )
    except OSError as exc:
        raise RuntimeError(f"无法在评测启动时读取 Git commit: {exc}") from exc
    commit = result.stdout.strip()
    if result.returncode or not _FULL_GIT_COMMIT.fullmatch(commit):
        detail = result.stderr.strip() or repr(commit)
        raise RuntimeError(f"无法验证评测启动时的完整 Git commit: {detail}")
    return commit


def index_identity(store) -> dict:
    """CR-160：**查询前**钉住索引的内容身份，缺身份就不跑、不写报告。

    此前报告只存 `index_chunks`，而块数不是内容身份——同为 17080 块的两份索引
    内容可以完全不同，于是 44/50 对 41/50 和六对 `top_distance` 都只能自洽，
    无法在索引重建后独立复验（CR-159 已在 bench 侧踩过同一个坑）。

    指纹算法复用 `services/retrieval/store.index_fingerprint`，与
    `bench/bench_speculative_retrieval.py` 是**同一份实现**而不是各抄一份。
    另记 `embedding_model`：距离结论依赖它，换模型后同一份索引的距离也会变。
    """
    identity = {
        "index_chunks": store.count(),
        "index_fingerprint": index_fingerprint(store),
        "dictionary_version": store.meta.get("dictionary_version"),
        "embedding_model": store.meta.get("embedding_model"),
    }
    missing = [k for k, v in identity.items() if not v]
    if missing:
        raise RuntimeError(f"索引缺少身份字段，拒绝产出不可复验的报告: {missing}")
    return identity


def _question_for(spec: dict, language: str) -> str:
    """取人工登记的该语言题面；绝不在评测时翻译或回退到另一种语言。"""
    question = spec.get(f"q_{language}")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"{spec.get('id', '<unknown>')}: 缺 q_{language}，不能拿另一语言题面代替")
    return question


def _normalize_for_match(text: str) -> str:
    """把换行折成一个空格，供登记正则匹配。

    Markdown 流的换行只表示排版，不能让 ``.`` 的有限距离窗口把同一事实误判
    为断开。仅将一个或多个换行及其相邻水平空白折为一个普通空格；不折叠正文
    内的词、不删除字符。原始文本（``answer_text``、索引里的块正文）都不受影响。

    答案侧与证据侧**必须共用这一个函数**：两边各写一套归一化，"答案命中而证据
    没命中"就会混进纯粹由换行折叠方式不同造成的差异。
    """
    return re.sub(r"[^\S\r\n]*[\r\n]+[^\S\r\n]*", " ", text)


def _score_keypoints(answer: str, patterns: list[str]) -> tuple[list[str], list[str]]:
    """事实断言：每个登记正则都必须在非拒答回答中出现。"""
    normalized = _normalize_for_match(answer)
    hit = [pattern for pattern in patterns if re.search(pattern, normalized)]
    missed = [pattern for pattern in patterns if pattern not in hit]
    return hit, missed


def keypoint_evidence(store, case: Case) -> tuple[list[dict], int]:
    """同一条登记正则，在**实际进入 prompt 的证据文本**里找不找得到。

    与 `_score_keypoints()` 看的是两处不同的文本：那个看答案，这个看证据。
    分开之后，"答案写了、但它引的块里根本没有这句话"才有名字可叫——
    R187/R188 的逐块审计发现这类**通过**不止一例：`redis-pool` 的
    `commons-pool2`、`postgres-connections` 的 `max_connections`
    都不在各自引用的任何一块证据里。

    **它不判对错、不进 `failures`、不改变任何一道题的通过与否。**
    证据完全可以用别的措辞表达同一事实，这条正则照样找不到；
    所以字段名只说到"这条正则在证据文本里出现过没有"，没有承诺"有没有依据"。

    第二个返回值是**没能核验身份的证据行数**：按 `chunk_id` 回查索引后，
    `source_url` 与正文 SHA-256 必须与 `sources` 事件登记的一致，
    否则读到的正文就不是当时送进 prompt 的那一份，这条正则的结论也就不成立——
    这种行只计数，不参与匹配（调用方据此失败关闭）。
    """
    texts: dict[int, str] = {}
    unverified = 0
    for item in case.selected_evidence:
        rows = store.execute(
            "SELECT source_url, text FROM chunks WHERE id = ?", (item["chunk_id"],)
        )
        if not rows or rows[0]["source_url"] != item["url"]:
            unverified += 1
            continue
        text = rows[0]["text"]
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != item["text_sha256"]:
            unverified += 1
            continue
        texts[item["chunk_id"]] = _normalize_for_match(text)

    rows_out = []
    for pattern in case.expect_keypoints:
        compiled = re.compile(pattern)
        rows_out.append({
            "pattern": pattern,
            "in_answer": pattern in case.keypoints_hit,
            "in_evidence_chunk_ids": sorted(
                cid for cid, text in texts.items() if compiled.search(text)
            ),
        })
    return rows_out, unverified


def keypoints_without_corpus_match(store, patterns: list[str]) -> list[str]:
    """哪些登记正则在**整个索引**里一块都匹配不到。

    这类判据从证据出发永远通不过：谁"通过"了它，靠的是模型自己写出那个词串
    （R187 在 `spring-graceful-shutdown`、`spring-auto-configuration` 上实测为 0 块）。
    它是**这份索引**上的性质，索引换了要重算，所以跟着每次运行一起记。
    """
    corpus = [_normalize_for_match(r["text"]) for r in store.execute("SELECT text FROM chunks")]
    return [p for p in dict.fromkeys(patterns)
            if not any(re.compile(p).search(t) for t in corpus)]


def run_case(orch: Orchestrator, spec: dict, language: str) -> Case:
    c = Case(
        id=spec["id"], language=language, question=_question_for(spec, language),
        expect=spec["expect"],
        expect_keypoints=list(spec.get("expect_keypoints", [])),
        expect_sources=list(spec.get("expect_sources", [])),
    )
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
            items = ev["items"]
            c.sources = len(items)
            c.urls = [i["url"] for i in items]
            c.projects = [_project_of(i["citation"]) for i in items]
            try:
                c.selected_evidence = [
                    {
                        "index": i["index"],
                        "chunk_id": i["chunk_id"],
                        "citation": i["citation"],
                        "url": i["url"],
                        "text_sha256": i["text_sha256"],
                    }
                    for i in items
                ]
            except KeyError as exc:
                # 不能让未来报告再次“有来源却没有证据身份”；保留事件数量以使旧的
                # 来源判据仍可诊断，但明确把评测 case 判为不完整，而非猜 rowid。
                c.failures.append(f"来源事件缺少审计字段 {exc.args[0]!r}")
        elif t == "done":
            c.cited = len(ev["cited_evidence"])
            c.ttft_s, c.total_s = ev["ttft_s"], ev["total_s"]
            c.prompt_tokens = ev["prompt_tokens"]
            c.evidence_count = ev.get("evidence_count", 0)
            c.served_by = ev.get("served_by", "")
            c.cache_read_tokens = ev.get("cache_read_tokens")
            c.cache_write_tokens = ev.get("cache_write_tokens")
            c.prefix_reused = bool(ev.get("prefix_reused"))
            c.cost_usd = ev.get("cost_usd", 0.0) or 0.0
        elif t == "error":
            c.failures.append(f"错误 {ev['stage']}: {ev['message']}")

    # T-022：判据与生产路径同一个函数（`answering.declined()`），不是本脚本
    # 另起一套散文正则——散文判据换语言就失效，且两套判据分叉迟早互相打脸。
    c.answer_text = answer
    c.declined = declined(answer)

    if c.expect == "answered":
        if c.sources == 0 and not c.declined:
            c.failures.append("未返回任何来源")

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
            c.keypoints_hit, c.keypoints_missed = _score_keypoints(
                answer, c.expect_keypoints,
            )
            for pattern in c.keypoints_missed:
                c.failures.append(f"未命中关键事实: {pattern!r}")
            got_projects = sorted(set(c.projects))
            for project in c.expect_sources:
                if project not in got_projects:
                    c.sources_missed.append(project)
                    got = ", ".join(got_projects) or "无"
                    c.failures.append(f"引用中缺少期望来源 {project!r}（实际 {got}）")
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


def validate_specs(specs: list[dict]) -> None:
    """T-023 题库契约在加载时失败关闭，避免默默退回中文题或空断言。"""
    seen: set[str] = set()
    for spec in specs:
        case_id = spec.get("id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("每道基础题必须有唯一的非空 id")
        if case_id in seen:
            raise ValueError(f"基础题 id 重复: {case_id}")
        seen.add(case_id)
        for language in SUPPORTED_LANGUAGES:
            _question_for(spec, language)
        if spec.get("expect") not in {"answered", "refused"}:
            raise ValueError(f"{case_id}: expect 必须是 answered 或 refused")
        for field in ("expect_keypoints", "expect_sources"):
            if not isinstance(spec.get(field), list):
                raise ValueError(f"{case_id}: {field} 必须是列表")
        if spec["expect"] == "answered" and not spec["expect_keypoints"]:
            raise ValueError(f"{case_id}: 应作答题必须登记至少一个事实断言")
        if spec["expect"] == "answered" and not spec["expect_sources"]:
            raise ValueError(f"{case_id}: 应作答题必须登记至少一个来源断言")


def keypoint_evidence_counts(cases: list[Case]) -> dict[str, int]:
    """把每条登记正则按「答案里有没有 × 证据里有没有」分到四格。

    `in_answer_only` 就是 R187/R188 反复撞到的那一格：判据命中，但它命中的那句话
    在这道题自己引用的证据里找不到。四格加起来等于参与统计的正则条数——
    只统计**真的算过证据**的题（拒答题与没跑到证据的题不在内）。

    **有证据行核不上身份的题整道排除**，另记在
    `cases_excluded_for_unverified_evidence`：那种题读到的正文不是当时送进
    prompt 的那一份，把它算进 `in_answer_only` 会凭空造出一条"无据通过"。
    """
    counts = {"in_answer_and_evidence": 0, "in_answer_only": 0,
              "in_evidence_only": 0, "in_neither": 0,
              "cases_counted": 0, "cases_excluded_for_unverified_evidence": 0}
    for case in cases:
        if case.evidence_rows_unverified:
            counts["cases_excluded_for_unverified_evidence"] += 1
            continue
        if case.keypoint_evidence:
            counts["cases_counted"] += 1
        for row in case.keypoint_evidence:
            in_answer, in_evidence = row["in_answer"], bool(row["in_evidence_chunk_ids"])
            if in_answer and in_evidence:
                counts["in_answer_and_evidence"] += 1
            elif in_answer:
                counts["in_answer_only"] += 1
            elif in_evidence:
                counts["in_evidence_only"] += 1
            else:
                counts["in_neither"] += 1
    return counts


def summarize_by_language(cases: list[Case]) -> dict[str, dict]:
    """按题目的实际 language 汇总，不能把中英文混成一条“总通过率”。"""
    grouped: dict[str, list[Case]] = {}
    for case in cases:
        grouped.setdefault(case.language, []).append(case)
    return {
        language: {
            "passed": sum(1 for c in group if c.ok),
            "total": len(group),
            "retrieval_misses": sum(1 for c in group if c.retrieval_miss),
            "declined_with_evidence": sum(1 for c in group if c.declined_with_evidence),
            "keypoints_hit": sum(len(c.keypoints_hit) for c in group),
            "keypoints_total": sum(len(c.expect_keypoints) for c in group),
            "keypoint_evidence_counts": keypoint_evidence_counts(group),
            "sources_missed": sum(len(c.sources_missed) for c in group),
            "cases": [vars(c) for c in group],
        }
        for language, group in sorted(grouped.items())
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--check-links", action="store_true", help="逐条 HEAD 验证来源可达（较慢）")
    ap.add_argument("--offline", action="store_true",
                     help="强制走本地兜底，不尝试云端 Claude（省钱/可复现；"
                          "不传时按生产路由跑：有 ANTHROPIC_API_KEY 就走云端）")
    ap.add_argument("--language", choices=SUPPORTED_LANGUAGES, default="zh",
                     help="评测题面与回答语言；分别运行 zh/en 会生成各自 language 分组。")
    args = ap.parse_args()
    run_id = os.environ.get("BLINK_EVAL_RUN_ID")
    started_at = os.environ.get("BLINK_EVAL_STARTED_AT") or datetime.now(timezone.utc).isoformat()
    startup_pid = os.getpid()
    load_dotenv(ROOT / ".env")

    try:
        implementation_commit = runtime_git_commit()
    except RuntimeError as exc:
        print(f"评测身份验证失败: {exc}", file=sys.stderr)
        return 2

    specs = yaml.safe_load(QUESTIONS.read_text(encoding="utf-8"))["questions"]
    try:
        validate_specs(specs)
    except ValueError as exc:
        print(f"题库格式错误: {exc}", file=sys.stderr)
        return 2
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
    # CR-160：身份在**任何一次检索之前**确定。放到写报告时再算就绑不住实际跑的
    # 那份索引——中途换了 current.db 也看不出来。
    try:
        identity = index_identity(store)
    except (RuntimeError, ValueError) as exc:
        print(f"索引身份验证失败: {exc}", file=sys.stderr)
        store.close()
        return 2
    print(f"索引 {identity['index_chunks']} 块 · 指纹 {identity['index_fingerprint'][:12]}…"
          f" · 词典 {identity['dictionary_version']} · 嵌入 {identity['embedding_model']}")
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
              f"{c.served_by or '?':<6} {c.ttft_s:.2f}s "
              f"cache读{c.cache_read_tokens if c.cache_read_tokens is not None else '-'} "
              f"写{c.cache_write_tokens if c.cache_write_tokens is not None else '-'} "
              f"${c.cost_usd:.6f}")
        for f in c.failures:
            print(f"        └─ {f}")

    # 证据侧的登记正则统计。放在全部题目跑完之后：它只读索引、不进生成路径，
    # 混进循环会把 SQL 读的耗时算进逐题时延。**不改变任何一道题的通过与否**。
    unverified_rows = 0
    for c in cases:
        c.keypoint_evidence, unverified = keypoint_evidence(store, c)
        c.evidence_rows_unverified = unverified
        unverified_rows += unverified
    unsatisfiable = keypoints_without_corpus_match(
        store, [p for c in cases for p in c.expect_keypoints]
    )

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

    # T-029：cache 命中与成本汇总。--offline 模式下所有请求都走本地，
    # 本地 cost_usd 恒为 0.0——total_cost_usd 不为 0 说明哪里算错了，
    # 用它当一次隐含的正确性检查（见下方 offline_cost_ok）。
    total_cost_usd = round(sum(c.cost_usd for c in cases), 6)
    cache_hits = sum(1 for c in cases if c.prefix_reused)
    offline_cost_ok = not args.offline or total_cost_usd == 0.0
    print(f"  cache 命中（prefix_reused）: {cache_hits}/{len(cases)}")
    print(f"  云端成本合计: ${total_cost_usd:.6f}"
          + ("  ← --offline 应恒为 0" if args.offline else ""))
    if not offline_cost_ok:
        print(f"        └─ 异常：--offline 模式下成本合计应为 0，实际 ${total_cost_usd:.6f}，"
              f"cost_usd 计算或透传有误")
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
        cells = keypoint_evidence_counts(cases)
        print(f"  登记正则 × 证据文本（不参与通过判定）: "
              f"答案与证据都有 {cells['in_answer_and_evidence']}、"
              f"**只在答案里 {cells['in_answer_only']}**、"
              f"只在证据里 {cells['in_evidence_only']}、"
              f"两边都没有 {cells['in_neither']}")
        if unsatisfiable:
            print(f"  ⚠ 在这份索引里一块都匹配不到的登记正则: {len(unsatisfiable)} 条"
                  f"  —— 它们从证据出发永远通不过")
            for pattern in unsatisfiable:
                print(f"    · {pattern}")
        if unverified_rows:
            print(f"  ⚠ 身份核不上的证据行: {unverified_rows} 条"
                  f"  —— 相关题目整道退出上面的统计，不影响通过判定")
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
    by_language = summarize_by_language(cases)
    path.write_text(json.dumps({
        # 3（R189）：每道题新增 `keypoint_evidence` 与 `evidence_rows_unverified`，
        # 报告新增 `keypoint_evidence_counts` 与 `keypoints_without_corpus_match`。
        # 都是新增字段，旧消费者按名取值不受影响。
        "schema_version": 3,
        "implementation_commit": implementation_commit,
        "run_id": run_id,
        "startup_pid": startup_pid,
        "started_at_utc": started_at,
        # This child can attest only that its JSON was written.  The outer
        # runner records the authoritative post-exit terminal state.
        "completion_state": "report_written",
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "template_version": template_version(),
        "language": args.language,
        "by_language": by_language,
        "model": DEFAULT_MODEL,
        # CR-160：写的是**查询前**钉住的那份身份，不在这里重读 store——
        # 重读等于把"报告写盘那一刻的索引"当成"跑题时的索引"。
        **identity,
        "passed": passed, "total": len(cases),
        "retrieval_misses": sum(1 for c in cases if c.retrieval_miss),
        "declined_with_evidence": sum(1 for c in cases if c.declined_with_evidence),
        "served_by_counts": served_by_counts,
        "offline_mode": args.offline,
        "cache_hits": cache_hits,
        "total_cost_usd": total_cost_usd,
        "offline_cost_ok": offline_cost_ok,
        "broken_links": broken,
        "keypoint_evidence_counts": keypoint_evidence_counts(cases),
        "keypoints_without_corpus_match": unsatisfiable,
        "evidence_rows_unverified": unverified_rows,
        "cases": [vars(c) for c in cases],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  报告 {path.name}")

    store.close()
    # 证据侧统计**不参与退出码**：它是测量不是门禁。核不上身份的行只会让对应的题
    # 退出统计（见 `keypoint_evidence_counts`），数量记在报告里由人看。
    # 让一个信息性统计决定 CI 红绿，等于给它加一道没人要求过的门。
    return 0 if passed == len(cases) and not broken and offline_cost_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
