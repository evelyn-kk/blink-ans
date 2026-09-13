"""检索验证集（held-out）本身的体检（CR-097）。

这份集合的价值全在"独立于调参探针"上，因此有两件事必须被机械钉住：
题面不得与排序探针重叠、gold 必须真的能在索引里解析出块。后者是
`probe_validation.py` 里唯一会失败的检查——`rank=None` 同时意味着
"检索没找到"和"gold 写错了"，不分开就等于这把尺子可能什么都没量。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "packages" / "evaltools"))

from services.retrieval.store import CURRENT, ChunkStore  # noqa: E402

import probe_validation as PV  # noqa: E402

VALIDATION = yaml.safe_load((ROOT / "knowledge/eval/ranking_validation.yaml").read_text(encoding="utf-8"))
PROBES = yaml.safe_load((ROOT / "knowledge/eval/ranking_probe.yaml").read_text(encoding="utf-8"))
BASIC = yaml.safe_load((ROOT / "knowledge/eval/basic_questions.yaml").read_text(encoding="utf-8"))


def test_validation_questions_do_not_overlap_the_tuning_probes():
    """独立性的第一条：题面不得与用来选参数的 13 条探针重叠。"""
    probe_qs = {p["q"] for p in PROBES["probes"]}
    overlap = [c["q"] for c in VALIDATION["cases"] if c["q"] in probe_qs]
    assert not overlap, f"验证集与调参探针重叠，独立性失效: {overlap}"


def test_validation_questions_come_from_the_pre_existing_question_set():
    """独立性的第二条：题面逐字取自 I2 时期就存在的 50 题集，不是本轮为了
    验证某个参数而现编的——现编题面等于换一批靶子，一样是自证。
    """
    basic_qs = {q["q"] for q in BASIC["questions"]}
    invented = [c["q"] for c in VALIDATION["cases"] if c["q"] not in basic_qs]
    assert not invented, f"这些题不在 basic_questions.yaml 里: {invented}"


def test_every_case_records_why_the_gold_answers_the_question():
    """gold 必须附人工核对的依据（与排序探针同一条规矩）。"""
    for c in VALIDATION["cases"]:
        assert c.get("why", "").strip(), f'{c["q"][:30]} 缺 why'


@pytest.mark.skipif(not CURRENT.exists(), reason="需要本机已建好的 current.db")
def test_every_gold_resolves_to_a_real_chunk():
    store = ChunkStore(CURRENT, check_dictionary=False)
    try:
        assert PV.unresolvable_golds(VALIDATION, store) == []
    finally:
        store.close()


@pytest.mark.skipif(not CURRENT.exists(), reason="需要本机已建好的 current.db")
def test_the_guard_catches_a_gold_that_does_not_exist():
    """判别性：把 gold 改成一个不存在的小节，检查必须报出来。"""
    store = ChunkStore(CURRENT, check_dictionary=False)
    try:
        broken = {"cases": [{"q": "编的题", "gold": "No Such Section › Nope", "project": "kafka"}]}
        assert PV.unresolvable_golds(broken, store)
    finally:
        store.close()
