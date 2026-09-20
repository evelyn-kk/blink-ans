"""同身份两臂的失败题归因：这道题为什么只在一种语言上失败（T-023 R187）。

R184/R185 把中英两臂拉到同一实现、同一索引跑完 50×2：中文 44/50、英文 41/50，
失败集合只重叠 2 条，**英文独有 7 条一直没有做归因**（中文侧有 R150/R154 的对应审计）。
这个脚本把归因需要的信号变成可独立复跑的产物。

**它只产出机械信号，不产出"根因"**（§5.3）。每条信号的名字就是它测到的那件事：

| 字段 | 它测的是 |
| --- | --- |
| `in_answer` | 登记的 keypoint 正则在该臂答案文本上是否匹配 |
| `in_selected_evidence` | 同一条正则在该臂**实际进入 prompt 的块**正文上匹配了哪几块 |
| `in_question` | 同一条正则在**题面自己**上是否匹配——匹配意味着复述题面即可通过这道判据 |
| `min_proximity_window_chars` | 把 `.{0,40}` 放宽到多少字符，该臂答案才会命中；`null` 表示两侧词项在答案里**任何距离上都没有同时出现** |
| `shared_evidence_chunk_ids` | 两臂实际选中的块的交集 |
| `failure_kinds` | 失败按报告字段（而不是失败文案）分类：keypoint / source / refusal / citation |
| `term_expansion` | `expand_terms()` 在**每一道题面**上展开出多少个词——两臂分别统计 |

"证据里没有这条事实"与"模型没写这条事实"是两件不同的事，这些字段把它们分开；
**哪一件构成失败的原因，由 `.md` 审计里的人工判断给出，不由本脚本断言**。

身份前提（CR-159/CR-160）：两份报告的 `index_fingerprint`、`embedding_model`、
`dictionary_version`、`template_version`、`model`、`index_chunks` 必须逐项相同，
否则拒绝比较——跨身份的失败集合差异说明不了语言。
证据完整性：每条 `selected_evidence` 都按 `chunk_id` 回查索引，要求
`source_url` 与报告一致、正文 SHA-256 与报告登记的 `text_sha256` 一致，
任一条对不上就中止、不读正文（R154 的口径）。

用法：.venv/bin/python bench/bench_language_failure_attribution.py \
        --zh-report bench/reports/<zh>.json --en-report bench/reports/<en>.json \
        --json bench/audits/t023-r187-english-only-failure-attribution.json \
        [--expect-index-fingerprint <sha256>]

不需要 Metal，也不调用 Embedder / 检索 / Router / LLM：纯读两份报告与
`data/index/current.db`。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from services.retrieval.store import ChunkStore, index_fingerprint  # noqa: E402
from services.retrieval.tokenize import detect_technology, expand_terms  # noqa: E402

# 两臂必须逐项相同的身份字段。少一项都不足以说"同一实验条件"：
# 同为 17080 块可以是完全不同的内容（CR-159），同一索引换个模板也会换答案。
IDENTITY_FIELDS = (
    "index_fingerprint", "index_chunks", "embedding_model",
    "dictionary_version", "template_version", "model",
)
_PROXIMITY = re.compile(r"\{0,(\d+)\}")
# 放宽上限：取到这个值仍不匹配就记 null（"任何距离上都没同时出现"）。
# 报告里最长的答案不到 600 字符，600 已覆盖"同一条答案内的任意两处"。
_MAX_WINDOW = 600


def shared_identity(zh_report: dict, en_report: dict) -> dict:
    """两份报告的共同身份；任一字段不同就拒绝比较。

    这不是防御性检查而是判据本身：身份不同时，"英文独有失败 7 条"里有多少
    来自语言、有多少来自索引或模板换了，无法分开。
    """
    for arm, report, expected in (("zh", zh_report, "zh"), ("en", en_report, "en")):
        actual = report.get("language")
        if actual != expected:
            raise SystemExit(f"{arm} 报告的 language 是 {actual!r}，不是 {expected!r}")
    identity = {}
    for field in IDENTITY_FIELDS:
        zh_value, en_value = zh_report.get(field), en_report.get(field)
        if zh_value is None or en_value is None:
            raise SystemExit(f"报告缺少身份字段 {field}：zh={zh_value!r} en={en_value!r}")
        if zh_value != en_value:
            raise SystemExit(
                f"两臂身份不同（{field}：zh={zh_value!r} en={en_value!r}），"
                f"跨身份的失败集合差异说明不了语言"
            )
        identity[field] = zh_value
    return identity


def failure_partition(zh_cases: dict, en_cases: dict) -> dict[str, list[str]]:
    """按"哪一臂失败"把题分三份。只看 `failures` 是否为空，不看失败类型。"""
    if set(zh_cases) != set(en_cases):
        raise SystemExit(
            f"两臂题目集合不同：zh 独有 {sorted(set(zh_cases) - set(en_cases))}，"
            f"en 独有 {sorted(set(en_cases) - set(zh_cases))}"
        )
    zh_failed = {i for i, c in zh_cases.items() if c["failures"]}
    en_failed = {i for i, c in en_cases.items() if c["failures"]}
    return {
        "both": sorted(zh_failed & en_failed),
        "zh_only": sorted(zh_failed - en_failed),
        "en_only": sorted(en_failed - zh_failed),
    }


def registered_window(keypoint: str) -> int | None:
    """邻近判据登记的窗口字符数；不是邻近判据则为 None。"""
    found = _PROXIMITY.findall(keypoint)
    if not found:
        return None
    if len(set(found)) != 1:
        raise SystemExit(f"keypoint 含多个不同的邻近窗口，无法单值放宽：{keypoint!r}")
    return int(found[0])


def min_proximity_window(keypoint: str, text: str) -> int | None:
    """把窗口放宽到多少字符，这条邻近判据才会在 `text` 上命中。

    返回 `None` 有两种含义，由 `registered_window()` 区分：不是邻近判据，
    或者**两侧词项在 `text` 里任何距离上都没有同时出现**。后者意味着放宽窗口
    也救不回来，和"只是隔得远了点"是不同的失败。
    """
    if registered_window(keypoint) is None or not text:
        return None
    for n in range(0, _MAX_WINDOW + 1):
        if re.search(_PROXIMITY.sub("{0,%d}" % n, keypoint), text, re.S):
            return n
    return None


def failure_kinds(case: dict) -> list[str]:
    """失败按**报告字段**分类，不解析失败文案。

    文案是中文模板，解析它等于把判据绑在一句话的措辞上；这里改看产生那句话的字段。
    一道题可以同时属于多类（例如既漏来源又没标编号）。
    """
    kinds = []
    if case["keypoints_missed"]:
        kinds.append("keypoint")
    if case["sources_missed"]:
        kinds.append("source")
    if case["expect"] == "refused" and not case["declined"]:
        kinds.append("refusal")
    if case["expect"] == "answered" and not case["declined"] and case["cited"] == 0:
        kinds.append("citation")
    return kinds


def verify_evidence(store, case: dict) -> dict[int, str]:
    """按 `chunk_id` 取回实际进入 prompt 的块正文，逐条核对身份。

    URL 或正文哈希与报告登记的不符即中止：那说明索引与报告不是同一份内容，
    此后读到的任何正文都不是当时那条证据（R154 的口径）。
    """
    texts: dict[int, str] = {}
    for item in case["selected_evidence"]:
        chunk_id = item["chunk_id"]
        rows = store.execute(
            "SELECT source_url, text FROM chunks WHERE id = ?", (chunk_id,)
        )
        if not rows:
            raise SystemExit(f"{case['id']}：索引里没有 chunk {chunk_id}")
        url, text = rows[0]["source_url"], rows[0]["text"]
        if url != item["url"]:
            raise SystemExit(
                f"{case['id']} chunk {chunk_id} 的 source_url 与报告不符：{url!r} vs {item['url']!r}"
            )
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest != item["text_sha256"]:
            raise SystemExit(
                f"{case['id']} chunk {chunk_id} 的正文哈希与报告不符："
                f"{digest} vs {item['text_sha256']}"
            )
        texts[chunk_id] = text
    return texts


def corpus_matches(store, pattern: str, cache: dict[str, list[int]]) -> list[int]:
    """整个索引里有哪些块能匹配这条 keypoint。

    它把"判据要的事实**语料里就没有**"与"语料里有、只是没进这次的证据"分开——
    只看进入 prompt 的那 5 块，两者长得一模一样。
    """
    if pattern not in cache:
        compiled = re.compile(pattern)
        cache[pattern] = [
            row["id"] for row in store.execute("SELECT id, text FROM chunks")
            if compiled.search(row["text"])
        ]
    return cache[pattern]


def keypoint_signals(keypoint: str, zh_case: dict, en_case: dict,
                     texts: dict[str, dict[int, str]],
                     corpus: list[int] | None = None) -> dict:
    """一条 keypoint 在两臂上的机械信号。字段名即它测到的那件事，不含因果。"""
    pattern = re.compile(keypoint)
    window = registered_window(keypoint)
    signals = {
        "pattern": keypoint,
        "registered_window_chars": window,
        "in_question": {
            arm: bool(pattern.search(case["question"]))
            for arm, case in (("zh", zh_case), ("en", en_case))
        },
        "in_answer": {
            arm: bool(pattern.search(case["answer_text"] or ""))
            for arm, case in (("zh", zh_case), ("en", en_case))
        },
        "in_selected_evidence": {
            arm: sorted(cid for cid, text in texts[arm].items() if pattern.search(text))
            for arm in ("zh", "en")
        },
    }
    if window is not None:
        signals["min_proximity_window_chars"] = {
            arm: min_proximity_window(keypoint, case["answer_text"] or "")
            for arm, case in (("zh", zh_case), ("en", en_case))
        }
    if corpus is not None:
        signals["in_corpus_chunk_count"] = len(corpus)
        signals["in_corpus_sample_chunk_ids"] = corpus[:8]
    return signals


def case_signals(zh_case: dict, en_case: dict, texts: dict[str, dict[int, str]],
                 partition: str, corpus: dict[str, list[int]] | None = None) -> dict:
    zh_ids = sorted(texts["zh"]); en_ids = sorted(texts["en"])
    return {
        "id": zh_case["id"],
        "partition": partition,
        "expect": zh_case["expect"],
        "question": {"zh": zh_case["question"], "en": en_case["question"]},
        "failures": {"zh": zh_case["failures"], "en": en_case["failures"]},
        "failure_kinds": {"zh": failure_kinds(zh_case), "en": failure_kinds(en_case)},
        "detect_technology": {"zh": detect_technology(zh_case["question"]),
                              "en": detect_technology(en_case["question"])},
        "expanded_terms": {"zh": expand_terms(zh_case["question"]),
                           "en": expand_terms(en_case["question"])},
        "answer_chars": {"zh": len(zh_case["answer_text"] or ""),
                         "en": len(en_case["answer_text"] or "")},
        "sufficiency": {"zh": zh_case["sufficiency"], "en": en_case["sufficiency"]},
        "served_by": {"zh": zh_case["served_by"], "en": en_case["served_by"]},
        "cited": {"zh": zh_case["cited"], "en": en_case["cited"]},
        "evidence_count": {"zh": zh_case["evidence_count"], "en": en_case["evidence_count"]},
        "selected_chunk_ids": {"zh": zh_ids, "en": en_ids},
        "shared_evidence_chunk_ids": sorted(set(zh_ids) & set(en_ids)),
        "expect_sources": zh_case["expect_sources"],
        "cited_projects": {"zh": sorted(set(zh_case["projects"])),
                           "en": sorted(set(en_case["projects"]))},
        "keypoints": [
            keypoint_signals(kp, zh_case, en_case, texts,
                             None if corpus is None else corpus[kp])
            for kp in zh_case["expect_keypoints"]
        ],
    }


def arm_totals(cases: dict) -> dict:
    """整臂（50 道，不只失败题）的机械统计。

    `keypoint` 一类的总数是这份审计最关键的对照：两臂各漏几条 keypoint，
    与"哪几道漏"是两个问题，只看失败总数会把它们混成一句"英文更差"。
    `expanded_terms_nonempty` 同理——它统计的是 `expand_terms()` 在题面上
    展开出词的题数，不是这些展开词对名次有多少影响。
    """
    answered = [c for c in cases.values() if c["expect"] == "answered"]
    kinds: dict[str, int] = {}
    for case in cases.values():
        for kind in failure_kinds(case):
            kinds[kind] = kinds.get(kind, 0) + 1
    lengths = sorted(len(c["answer_text"] or "") for c in answered)
    return {
        "cases": len(cases),
        "failed_cases": sum(1 for c in cases.values() if c["failures"]),
        "failure_kind_counts": kinds,
        "keypoints_registered": sum(len(c["expect_keypoints"]) for c in answered),
        "keypoints_missed": sum(len(c["keypoints_missed"]) for c in answered),
        "answered_with_zero_citations": sum(1 for c in answered if c["cited"] == 0),
        "median_answer_chars": lengths[len(lengths) // 2] if lengths else None,
        "expanded_terms_nonempty": sum(
            1 for c in cases.values() if expand_terms(c["question"])
        ),
    }


def wider_window_would_match(case: dict) -> list[dict]:
    """哪些"未命中"只要把窗口放宽就会命中——即窗口本身是那一条的约束。

    它不说明窗口该不该放宽，只说明这条失败与"事实缺失"不是一回事。
    """
    out = []
    for kp in case["keypoints"]:
        window = kp["registered_window_chars"]
        if window is None:
            continue
        for arm in ("zh", "en"):
            need = kp["min_proximity_window_chars"][arm]
            if not kp["in_answer"][arm] and need is not None and need > window:
                out.append({"id": case["id"], "arm": arm, "pattern": kp["pattern"],
                            "registered_window_chars": window,
                            "min_proximity_window_chars": need})
    return out


def question_echo_passes(case: dict) -> list[dict]:
    """哪些 keypoint 在题面自己上就匹配——复述题面即可通过那道判据。"""
    out = []
    for kp in case["keypoints"]:
        for arm in ("zh", "en"):
            if kp["in_question"][arm]:
                out.append({"id": case["id"], "arm": arm, "pattern": kp["pattern"]})
    return out


def probe_retrieval(store, cases: list[dict], corpus_by_pattern: dict[str, list[int]],
                    mix_limit: int = 10, rank_limit: int = 50) -> dict:
    """现场跑两类只读检索探针（**唯一需要 Metal 的一段**，默认不执行）。

    - `source_mix`：两臂证据零交集的题，各自 top-N 的来源构成，外加一个**操纵实验**——
      英文题面拼上中文侧 `expand_terms()` 的展开词。它回答的是
      "英文侧缺的就是这些展开词吗"，不是在提议改任何默认值。
    - `keypoint_rank`：判据要的事实语料里有、却没进这次证据的题，那些
      **能匹配判据的块在两臂检索里排到第几**。`null` 表示前 `rank_limit` 名里没有，
      即这次连取都没取到，而不是"取到了没选上"。

    两类探针都不改任何共享默认值：`limit` 之外的参数全部取 `AnswerConfig()` 的现值。
    """
    from services.orchestrator.answering import AnswerConfig
    from services.retrieval.embed import Embedder
    from services.retrieval.search import hybrid_search

    embedder = Embedder(); embedder.load()
    cfg = AnswerConfig()

    def search(query: str, limit: int):
        return hybrid_search(store, query, embedder.encode_one(query), limit=limit,
                             technology=detect_technology(query), candidates=cfg.candidates)

    source_mix = []
    for case in cases:
        if case["expect"] != "answered" or case["shared_evidence_chunk_ids"]:
            continue
        queries = {
            "zh_question": case["question"]["zh"],
            "en_question": case["question"]["en"],
            "en_question_plus_zh_expansion_terms":
                " ".join([case["question"]["en"], *case["expanded_terms"]["zh"]]),
        }
        source_mix.append({
            "id": case["id"], "limit": mix_limit, "queries": queries,
            "expect_sources": case["expect_sources"],
            "top_source_projects": {
                name: [h.source_project for h in search(q, mix_limit)]
                for name, q in queries.items()
            },
        })

    keypoint_rank = []
    for case in cases:
        for kp in case["keypoints"]:
            wanted = set(corpus_by_pattern.get(kp["pattern"], ()))
            if not wanted:
                continue
            if kp["in_selected_evidence"]["zh"] and kp["in_selected_evidence"]["en"]:
                continue
            row = {"id": case["id"], "pattern": kp["pattern"],
                   "in_corpus_chunk_count": kp["in_corpus_chunk_count"],
                   "rank_limit": rank_limit, "best_rank": {}, "best_chunk_id": {}}
            for arm in ("zh", "en"):
                hits = search(case["question"][arm], rank_limit)
                found = next(((r, h.rowid) for r, h in enumerate(hits, 1)
                              if h.rowid in wanted), (None, None))
                row["best_rank"][arm], row["best_chunk_id"][arm] = found
            keypoint_rank.append(row)

    return {"source_mix": source_mix, "keypoint_rank": keypoint_rank,
            "note": "best_rank 对全部匹配块算，不受产物里那 8 个样本块的限制；"
                    "null = 前 rank_limit 名里一块都没有，即这次连取都没取到。"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zh-report", required=True)
    ap.add_argument("--en-report", required=True)
    ap.add_argument("--json", required=True, help="审计产物路径")
    ap.add_argument("--expect-index-fingerprint",
                    help="预期索引指纹；与报告或当前索引不符即在读正文前失败关闭")
    ap.add_argument("--probe-retrieval", action="store_true",
                    help="对两臂证据零交集的题现场量一次来源构成（需要 Metal）")
    args = ap.parse_args()

    zh_report = json.loads(Path(args.zh_report).read_text(encoding="utf-8"))
    en_report = json.loads(Path(args.en_report).read_text(encoding="utf-8"))
    identity = shared_identity(zh_report, en_report)

    expected = args.expect_index_fingerprint
    if expected and expected != identity["index_fingerprint"]:
        raise SystemExit(
            f"报告的索引指纹不符：预期 {expected}，实际 {identity['index_fingerprint']}"
        )

    store = ChunkStore()
    live = {"index_chunks": store.count(),
            "index_fingerprint": index_fingerprint(store),
            "dictionary_version": store.meta.get("dictionary_version"),
            "embedding_model": store.meta.get("embedding_model")}
    for field in ("index_fingerprint", "index_chunks",
                  "dictionary_version", "embedding_model"):
        if live[field] != identity[field]:
            store.close()
            raise SystemExit(
                f"当前索引与报告不是同一份（{field}：索引 {live[field]!r} vs "
                f"报告 {identity[field]!r}），拒绝用它回查当时的证据"
            )

    zh_cases = {c["id"]: c for c in zh_report["cases"]}
    en_cases = {c["id"]: c for c in en_report["cases"]}
    partition = failure_partition(zh_cases, en_cases)

    cases, verified, cache = [], 0, {}
    for name in ("en_only", "zh_only", "both"):
        for case_id in partition[name]:
            texts = {"zh": verify_evidence(store, zh_cases[case_id]),
                     "en": verify_evidence(store, en_cases[case_id])}
            verified += len(texts["zh"]) + len(texts["en"])
            corpus = {kp: corpus_matches(store, kp, cache)
                      for kp in zh_cases[case_id]["expect_keypoints"]}
            cases.append(
                case_signals(zh_cases[case_id], en_cases[case_id], texts, name, corpus)
            )

    probe = probe_retrieval(store, cases, cache) if args.probe_retrieval else None
    store.close()

    out = {
        "round": "R187",
        "zh_report": str(Path(args.zh_report)),
        "en_report": str(Path(args.en_report)),
        "zh_implementation_commit": zh_report["implementation_commit"],
        "en_implementation_commit": en_report["implementation_commit"],
        **identity,
        "passed": {"zh": zh_report["passed"], "en": en_report["passed"]},
        "total": {"zh": zh_report["total"], "en": en_report["total"]},
        "note": "只读归因，未跑检索/生成，未改题库、阈值与门禁。"
                "字段只记机械信号，失败原因的判断在同名 .md 审计里。",
        "evidence_rows_verified": verified,
        "arm_totals": {"zh": arm_totals(zh_cases), "en": arm_totals(en_cases)},
        "failure_partition": partition,
        "retrieval_probe": probe,
        "wider_window_would_match": [w for c in cases for w in wider_window_would_match(c)],
        "question_echo_passes": [q for c in cases for q in question_echo_passes(c)],
        "cases": cases,
    }
    path = Path(args.json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"指纹 {identity['index_fingerprint'][:12]}… · {identity['index_chunks']} 块 · "
          f"核对 {verified} 条证据")
    print(f"失败：英文独有 {len(partition['en_only'])}、中文独有 {len(partition['zh_only'])}、"
          f"两臂共有 {len(partition['both'])}")
    for arm in ("zh", "en"):
        t = out["arm_totals"][arm]
        print(f"  {arm}：失败 {t['failed_cases']}，按字段分类 {t['failure_kind_counts']}，"
              f"keypoint 漏 {t['keypoints_missed']}/{t['keypoints_registered']}，"
              f"题面展开非空 {t['expanded_terms_nonempty']}/{t['cases']}")
    print(f"放宽窗口即可命中 {len(out['wider_window_would_match'])} 处；"
          f"题面自身即可通过 {len(out['question_echo_passes'])} 处")
    print(f"写入 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
