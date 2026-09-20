"""CR-161：拒答档位审计的两个计数与 weather 的非跨档情形。

R184 把"英文更近"和"英文档位更宽"合成一句写成"更近的 5 对每一道都更宽"，
实测是 5 与 4——`refuse-beijing-weather` 英文更近却没有跨档。CR-160 复审已确认
机械计数为 5/6 更近、4/6 跨档；这组回归把它钉住，避免那句话再回来。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench"))

import bench_refusal_band as band  # noqa: E402

AUDIT = ROOT / "bench" / "audits" / "t023-r185-refusal-band-by-language.json"


def _side(distance, band_name):
    return {"top_distance": distance, "band": band_name}


# ---------- compare_pair：两问必须分开判 ----------

def test_weather_shape_is_closer_but_does_not_cross_a_band():
    """CR-161 的反例本身：英文更近，档位不变。

    这是 R184 那句话唯一的反例，也是它被撤回的原因；
    若将来有人把两问又合成一个判据，这条会失败。
    """
    got = band.compare_pair(_side(0.8012, "insufficient"), _side(0.7830, "insufficient"))

    assert got["en_closer_than_zh"] is True
    assert got["en_band_wider_than_zh"] is False


def test_scifi_shape_is_closer_and_crosses_into_a_wider_band():
    got = band.compare_pair(_side(0.7571, "limited"), _side(0.7171, "sufficient"))

    assert got["en_closer_than_zh"] is True
    assert got["en_band_wider_than_zh"] is True


def test_rust_shape_is_farther_and_does_not_cross():
    """唯一一道英文反而更远的题。"""
    got = band.compare_pair(_side(0.7219, "limited"), _side(0.7469, "limited"))

    assert got["en_closer_than_zh"] is False
    assert got["en_band_wider_than_zh"] is False


def test_narrower_english_band_is_not_counted_as_wider():
    """方向性：英文档位更**严**时不得记成"更宽"。"""
    got = band.compare_pair(_side(0.70, "sufficient"), _side(0.79, "insufficient"))

    assert got["en_band_wider_than_zh"] is False


def test_missing_distance_is_not_guessed_as_closer():
    assert band.compare_pair(_side(None, "limited"), _side(0.71, "limited"))[
        "en_closer_than_zh"] is False
    assert band.compare_pair(_side(0.71, "limited"), _side(None, "limited"))[
        "en_closer_than_zh"] is False


def test_bands_run_strict_to_permissive():
    """`_BANDS` 的顺序就是"更宽"的定义，写反了上面所有判定都会翻转。"""
    assert band._BANDS == ("insufficient", "limited", "sufficient")


# ---------- 已记录产物：计数必须是 5 与 4 ----------

def test_recorded_audit_counts_are_five_closer_and_four_wider():
    """钉住 R185 产物里的两个机械计数，防止撤回的说法复活。"""
    report = json.loads(AUDIT.read_text(encoding="utf-8"))

    assert report["en_closer_count"] == 5
    assert report["en_band_wider_count"] == 4
    assert len(report["pairs"]) == 6


def test_recorded_audit_counts_match_recomputing_them_from_the_pairs():
    """计数不能与逐对数据脱节——用 compare_pair 从原始距离/档位重算一遍。"""
    report = json.loads(AUDIT.read_text(encoding="utf-8"))

    recomputed = [band.compare_pair(p["zh"], p["en"]) for p in report["pairs"]]

    assert sum(r["en_closer_than_zh"] for r in recomputed) == report["en_closer_count"]
    assert sum(r["en_band_wider_than_zh"] for r in recomputed) == report["en_band_wider_count"]
    for pair, again in zip(report["pairs"], recomputed):
        assert pair["en_closer_than_zh"] == again["en_closer_than_zh"], pair["id"]
        assert pair["en_band_wider_than_zh"] == again["en_band_wider_than_zh"], pair["id"]


def test_recorded_weather_pair_is_the_closer_but_not_wider_case():
    """点名核对那一对，而不是只看总数——总数相同也可能是别的题在抵消。"""
    report = json.loads(AUDIT.read_text(encoding="utf-8"))
    weather = next(p for p in report["pairs"] if p["id"] == "refuse-beijing-weather")

    assert weather["en_closer_than_zh"] is True
    assert weather["en_band_wider_than_zh"] is False
    assert weather["zh"]["band"] == weather["en"]["band"] == "insufficient"


def test_recorded_audit_is_bound_to_the_verified_index_fingerprint():
    """CR-160 的要求在产物里仍然成立（与本轮回归同源，不重复造一份身份口径）。"""
    report = json.loads(AUDIT.read_text(encoding="utf-8"))

    assert report["index_fingerprint"] == (
        "fe11b6b41ded3d60f422768727c67a9aefe4dac8ebc83dac625efc6b1d176305"
    )
    assert report["index_chunks"] == 17080
    assert report["embedding_model"]
    assert report["dictionary_version"]


# ---------- 文件头不得再写撤回的说法 ----------

@pytest.mark.parametrize("retracted", ["每一道都更宽", "每一对都更宽"])
def test_module_docstring_does_not_repeat_the_retracted_claim(retracted):
    """CR-161：文档与产物必须一致。

    这条判据只看这一句措辞，**不验证任何数值**——数值由上面的计数回归负责。
    """
    assert retracted not in (band.__doc__ or "")
