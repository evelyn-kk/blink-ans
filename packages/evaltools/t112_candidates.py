"""T-112 融合候选的可复现审计运行器。

这不是又一个“调到探针变绿”的脚本。它一次运行固定的 baseline 与四个候选，
把每份 probe/held-out 结果、完整候选参数、实现提交、索引校验和与 meta 写进 JSON，
再由同一比较程序列出相对 baseline 的所有名次退步。产物不能手填。

用法：
    .venv/bin/python packages/evaltools/t112_candidates.py \
        --output-dir bench/audits --prefix t112-r106

所有候选共享同一个已打开的只读索引和 Embedder；运行前后均计算 current.db 指纹，
中途索引若被替换就失败，而不会把不同数据源的名次混成一次比较。
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from packages.evaltools import probe_ranking as ranking  # noqa: E402
from packages.evaltools import probe_validation as validation  # noqa: E402
from services.retrieval.embed import Embedder  # noqa: E402
from services.retrieval.search import FusionExperiment  # noqa: E402
from services.retrieval.store import ChunkStore  # noqa: E402

SCHEMA = "t112-candidate-audit/v1"
PROBES = ROOT / "knowledge" / "eval" / "ranking_probe.yaml"
VALIDATION = ROOT / "knowledge" / "eval" / "ranking_validation.yaml"
IMPLEMENTATION_FILES = (
    ROOT / "services" / "retrieval" / "search.py",
    ROOT / "packages" / "evaltools" / "probe_ranking.py",
    ROOT / "packages" / "evaltools" / "probe_validation.py",
    Path(__file__),
)


@dataclasses.dataclass(frozen=True)
class Candidate:
    """一个候选必须完整声明；不能在调用方临时 monkeypatch 隐藏规则。"""

    name: str
    candidates: int = 30
    experiment: FusionExperiment | None = None
    rationale: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "candidates": self.candidates,
            "experiment": dataclasses.asdict(self.experiment) if self.experiment else None,
            "rationale": self.rationale,
        }


CANDIDATES = (
    Candidate("baseline", rationale="生产 RRF_K=60、两路各取 30；无实验规则"),
    Candidate(
        "single-path-credit",
        experiment=FusionExperiment(
            "single-path-credit", imputed_keyword_rank=31, imputed_vector_rank=31,
        ),
        rationale="任一单路候选为缺席的另一路补假定第 31 名 RRF 信号",
    ),
    Candidate(
        "vector-only-credit",
        experiment=FusionExperiment("vector-only-credit", imputed_keyword_rank=31),
        rationale="仅向量命中而关键词候选外时补假定关键词第 31 名",
    ),
    Candidate(
        "keyword-tail",
        experiment=FusionExperiment("keyword-tail", keyword_candidate_depth=150, keyword_score_depth=150),
        rationale="仅关键词候选与计分深度从 30 扩至 150；向量仍为 30，不补虚构名次",
    ),
    Candidate(
        "vector-keyword-rescue",
        experiment=FusionExperiment(
            "vector-keyword-rescue", keyword_candidate_depth=150, keyword_score_depth=30,
            vector_candidate_depth=150, vector_score_depth=30,
            keyword_rescue_depth=150, vector_rescue_max_rank=15, rescue_credit_rank="vector_zero",
        ),
        rationale="两路各取 150 以保留过滤后的真实名次，但只计各自前 30；关键词真实排在 31..150 且向量前 15 时，再按该向量名次补一条关键词权重 RRF 分（R105 临时实现的实际规则）",
    ),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip()


def _implementation() -> dict[str, Any]:
    return {
        "git_commit": _git_commit(),
        "files": {
            str(path.relative_to(ROOT)): _sha256(path)
            for path in IMPLEMENTATION_FILES
        },
    }


def _index_identity(store: ChunkStore) -> dict[str, Any]:
    return {
        "path": str(store.path.relative_to(ROOT)),
        "sha256": _sha256(store.path),
        "chunk_count": store.count(),
        "meta": dict(sorted(store.meta.items())),
    }


def _rank(value: int | None) -> float:
    """候选外比任意实际名次差；不读取 known_open 或 YAML baseline。"""
    return float("inf") if value is None else value


def compare_ranks(
    baseline: Iterable[dict[str, Any]], candidate: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """统一比较程序：所有题（含 known_open）相对实测 baseline 的名次变化。

    这里故意不使用 `ranking_probe.yaml` 的历史 baseline，也不看 known_open；二者
    是生产门禁概念，而本函数回答的是“此候选相对于这次同数据基线变差了吗”。
    因而 None→整数是改善、整数→None 是退步，不能靠排除既有缺口改变计数。
    """
    before = {row["q"]: row["rank"] for row in baseline}
    after = {row["q"]: row["rank"] for row in candidate}
    if before.keys() != after.keys():
        raise ValueError("候选与基线的题集不一致，拒绝比较")
    regressions = [
        {"q": q, "baseline_rank": before[q], "candidate_rank": after[q]}
        for q in before
        if _rank(after[q]) > _rank(before[q])
    ]
    improvements = [
        {"q": q, "baseline_rank": before[q], "candidate_rank": after[q]}
        for q in before
        if _rank(after[q]) < _rank(before[q])
    ]
    unchanged = [q for q in before if _rank(after[q]) == _rank(before[q])]
    return {
        "regression_count": len(regressions), "regressions": regressions,
        "improvement_count": len(improvements), "improvements": improvements,
        "unchanged_count": len(unchanged), "unchanged": unchanged,
    }


def _ranking_rows(results: list[ranking.ProbeResult]) -> list[dict[str, Any]]:
    return [
        {"q": r.question, "gold": r.gold, "rank": r.rank, "passed": r.passed,
         "top": r.top, "fts_query": r.fts_query}
        for r in results
    ]


def _validation_rows(spec: dict, store: ChunkStore, embedder: Embedder, candidate: Candidate) -> list[dict[str, Any]]:
    return [
        {"q": case["q"],
         "rank": validation.run(
             case, store, embedder, spec["limit"], candidates=candidate.candidates,
             experiment=candidate.experiment,
         ),
         "rank_at_k60": case.get("rank_at_k60")}
        for case in spec["cases"]
    ]


def _audit(
    *, kind: str, common: dict[str, Any], candidate: Candidate,
    results: list[dict[str, Any]], comparison: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA, "kind": kind, "provenance": common,
        "candidate": candidate.as_dict(), "results": results, "comparison_to_baseline": comparison,
    }


def run_all(command: str) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    """在同一输入身份下运行 baseline 与全部候选，返回尚未写盘的 JSON。"""
    ranking_spec = yaml.safe_load(PROBES.read_text(encoding="utf-8"))
    validation_spec = yaml.safe_load(VALIDATION.read_text(encoding="utf-8"))
    store = ChunkStore()
    try:
        missing = validation.unresolvable_golds(validation_spec, store)
        if missing:
            raise RuntimeError(f"held-out gold 在索引中不可解析: {missing}")
        before_identity = _index_identity(store)
        embedder = Embedder()
        embedder.load()
        if embedder.error:
            raise RuntimeError(f"嵌入模型未就绪: {embedder.error}")
        common = {
            "command": command,
            "implementation": _implementation(),
            "index": before_identity,
            "embedding_model": embedder.model_id,
            "datasets": {
                str(PROBES.relative_to(ROOT)): _sha256(PROBES),
                str(VALIDATION.relative_to(ROOT)): _sha256(VALIDATION),
            },
        }
        raw: dict[str, tuple[Candidate, list[dict[str, Any]], list[dict[str, Any]]]] = {}
        for candidate in CANDIDATES:
            probe_rows = _ranking_rows(ranking.run(
                ranking_spec, store, embedder, limit=validation_spec["limit"],
                candidates=candidate.candidates, experiment=candidate.experiment,
            ))
            validation_rows = _validation_rows(validation_spec, store, embedder, candidate)
            raw[candidate.name] = (candidate, probe_rows, validation_rows)
        if _index_identity(store) != before_identity:
            raise RuntimeError("实验期间 current.db 的内容或 meta 已改变，拒绝混合产物")
    finally:
        store.close()

    base_probe = raw["baseline"][1]
    base_validation = raw["baseline"][2]
    return {
        name: (
            _audit(kind="probe", common=common, candidate=candidate, results=probe_rows,
                   comparison=None if name == "baseline" else compare_ranks(base_probe, probe_rows)),
            _audit(kind="validation", common=common, candidate=candidate, results=validation_rows,
                   comparison=None if name == "baseline" else compare_ranks(base_validation, validation_rows)),
        )
        for name, (candidate, probe_rows, validation_rows) in raw.items()
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--prefix", required=True, help="文件名前缀，例如 t112-r106")
    args = ap.parse_args()
    command = (
        ".venv/bin/python packages/evaltools/t112_candidates.py "
        f"--output-dir {args.output_dir} --prefix {args.prefix}"
    )
    audits = run_all(command)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, (probe, validation_audit) in audits.items():
        for kind, audit in (("probe", probe), ("validation", validation_audit)):
            path = args.output_dir / f"{args.prefix}-{name}-{kind}.json"
            path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            comparison = audit["comparison_to_baseline"]
            suffix = "baseline" if comparison is None else f"probe/validation regressions={comparison['regression_count']}"
            print(f"已写入 {path} ({suffix})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
