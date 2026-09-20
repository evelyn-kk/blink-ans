"""拒答题的充分性档位随语言怎么变（T-023 R184 发现，R185 按 CR-160 补身份）。

R184 实测：6 道登记拒答题里，英文题面的 `top_distance` 有 5 道比中文更近，
且那 5 道的档位**每一道都更宽**（limited→sufficient、insufficient→limited）。
语料全是英文官方文档，英文提问天然离语料更近——外域安全边际在英文侧更薄。

这个脚本把那张表变成可独立复跑的产物，**不改产品代码、不改任何阈值**：
它只读索引，逐对算 `top_distance`、档位与 `must_refuse_limited_out_of_scope()`
是否触发。

CR-160：查询前失败关闭地钉住索引内容指纹（与 `run_basic.py`、
`bench_speculative_retrieval.py` 共用 `services.retrieval.store.index_fingerprint`），
否则"同为 17080 块"的不同索引会被当成同一实验条件。

用法：.venv/bin/python bench/bench_refusal_band.py \
        --json bench/audits/t023-r185-refusal-band-by-language.json \
        [--expect-index-fingerprint <sha256>]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from services.orchestrator.answering import (  # noqa: E402
    AnswerConfig, AnswerRequest, assess, must_refuse_limited_out_of_scope,
)
from services.retrieval.embed import Embedder  # noqa: E402
from services.retrieval.search import hybrid_search  # noqa: E402
from services.retrieval.store import ChunkStore, index_fingerprint  # noqa: E402
from services.retrieval.tokenize import detect_technology  # noqa: E402

QUESTIONS = ROOT / "knowledge" / "eval" / "basic_questions.yaml"
_FULL_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")
# 从严到宽；"更宽"意味着更容易放行，是这份审计关心的方向。
_BANDS = ("insufficient", "limited", "sufficient")


def runtime_identity(store) -> dict:
    """查询前钉住身份，缺一项就不跑（CR-159 的口径，CR-160 要求本脚本也守）。"""
    result = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD^{commit}"],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    commit = result.stdout.strip()
    if result.returncode or not _FULL_GIT_COMMIT.fullmatch(commit):
        raise SystemExit(f"无法验证完整 Git commit：{result.stderr.strip() or commit!r}")
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, text=True,
                           capture_output=True, check=False).stdout.strip()
    identity = {
        "implementation_commit": commit,
        "working_tree_dirty": bool(dirty),
        "index_chunks": store.count(),
        "index_fingerprint": index_fingerprint(store),
        "dictionary_version": store.meta.get("dictionary_version"),
        "embedding_model": store.meta.get("embedding_model"),
    }
    missing = [k for k in ("index_chunks", "index_fingerprint",
                           "dictionary_version", "embedding_model") if not identity[k]]
    if missing:
        raise SystemExit(f"索引缺少身份字段，拒绝产出不可复验的审计：{missing}")
    return identity


def measure(store, embedder, cfg: AnswerConfig, text: str, language: str) -> dict:
    tech = detect_technology(text)
    verdict = assess(
        hybrid_search(store, text, embedder.encode_one(text),
                      limit=cfg.max_evidence * 2, technology=tech,
                      candidates=cfg.candidates),
        cfg,
    )
    return {
        "question": text,
        "top_distance": round(verdict.top_distance, 4) if verdict.top_distance else None,
        "band": verdict.level.value,
        "keyword_hits": verdict.keyword_hits,
        "detect_technology": tech,
        "policy_guard_fires": must_refuse_limited_out_of_scope(
            AnswerRequest(text, language=language), verdict
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="审计产物路径")
    ap.add_argument("--expect-index-fingerprint",
                    help="预期索引指纹；不符则在查询前失败关闭")
    args = ap.parse_args()

    store = ChunkStore()
    identity = runtime_identity(store)
    expected = args.expect_index_fingerprint
    if expected and expected != identity["index_fingerprint"]:
        store.close()
        raise SystemExit(
            f"索引指纹不符：预期 {expected}，实际 {identity['index_fingerprint']}；"
            f"拒绝在非预期索引上产出审计"
        )
    if expected:
        identity["expected_index_fingerprint"] = expected

    embedder = Embedder(); embedder.load()
    cfg = AnswerConfig()
    specs = yaml.safe_load(QUESTIONS.read_text(encoding="utf-8"))["questions"]

    pairs = []
    for spec in specs:
        if spec.get("expect") != "refused":
            continue
        row = {"id": spec["id"]}
        for language in ("zh", "en"):
            row[language] = measure(store, embedder, cfg, spec[f"q_{language}"], language)
        row["en_closer_than_zh"] = (
            row["en"]["top_distance"] is not None
            and row["zh"]["top_distance"] is not None
            and row["en"]["top_distance"] < row["zh"]["top_distance"]
        )
        row["en_band_wider_than_zh"] = (
            _BANDS.index(row["en"]["band"]) > _BANDS.index(row["zh"]["band"])
        )
        pairs.append(row)

    out = {
        "round": "R185",
        **identity,
        "thresholds": {"sufficient_distance": cfg.sufficient_distance,
                       "limited_distance": cfg.limited_distance},
        "note": "只读测量，未改任何阈值。guard 需要 LIMITED 带才触发，"
                "故落在 sufficient 带的题它结构上够不到。",
        "en_closer_count": sum(p["en_closer_than_zh"] for p in pairs),
        "en_band_wider_count": sum(p["en_band_wider_than_zh"] for p in pairs),
        "pairs": pairs,
    }
    path = Path(args.json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    store.close()
    print(f"索引 {identity['index_chunks']} 块 · 指纹 {identity['index_fingerprint'][:12]}…")
    print(f"{len(pairs)} 对拒答题：英文更近 {out['en_closer_count']}，"
          f"英文档位更宽 {out['en_band_wider_count']}")
    print(f"写入 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
