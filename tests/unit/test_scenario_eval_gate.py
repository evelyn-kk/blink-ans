"""场景评测判据自身的回归（CR-053~061）。

为什么单独测：判据是场景评测结果的门禁，判据本身漏判会让"通过/不通过"
这个数字失去意义——就像 `test_probe_gate.py` 对排序探针门禁（CR-015）
做的事一样。

这里测的是 `forbid_patterns` 这个负向判据本身的完整教训史：

- **CR-053**：`run_scenarios.py` 原先只看正向的 `expect_keypoints`，一条
  答案哪怕混进一句已知错误的越界断言（真实保存输出，见下方
  `_CR053_ANSWER`：模型断言 `RedisAtomicLong` "仅支持原子增减…需使用
  Lua 脚本"，这超出了 CR-052 已核实的来源支撑范围），只要凑巧命中了
  两条比较宽泛的正向关键点（"不能"/"无法…判断"），照样被记成"关键点
  2/2 全部命中"、判为通过——新增 `forbid_patterns`，命中即记入
  `forbidden_hit`。
- **CR-054~060**：`forbid_patterns` 上线后，又花了六轮反复调整"命中之后
  要不要自动判断这次命中是否被否定"这条逻辑，每次修一个反例就暴露一个
  新反例（跨命题窗口泄漏 → 分句边界不认转折连词 → 零容忍紧邻矫枉过正 →
  分句边界不认加合连词 → 单命中场景加合连词接无关否定仍泄漏）。根因是
  这条启发式想用"这段前文有没有否定词"回答一个本质上是句法结构的问题，
  而连接词是开放集合，枚举永远追不完。CR-060 决定不再尝试自动判断否定
  极性：命中只记入 `forbidden_hit`，交给人工判断。
- **CR-061**：指出 CR-060 的修法本身还不完整——`forbidden_hit` 虽然会
  打印警告，却完全不影响这道题的通过判定，真实报告里"命中可疑模式 +
  正向关键点全部命中"的题照样被算进"通过"，等于换了个说法重新引入
  CR-053 想堵住的洞。修法：三态判定（`passed`/`failed`/
  `review_required`），命中 `forbidden_hit` 但没有人工复核确认
  （`knowledge/eval/scenario_review.yaml` 里 `verdict: confirmed_ok`）
  的题目一律 `review_required`——不计入通过数，也不能让整次评测的
  退出码为 0。

这里只测纯判定函数 `_score`/`_case_status`/`_lookup_review_verdict`，
不加载模型/索引，因此毫秒级，可进快速门禁。
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from packages.evaltools.run_scenarios import (  # noqa: E402
    REVIEWS, ScenarioCase, _case_status, _lookup_review_verdict, _score,
)

# 逐字取自 bench/reports/eval-scenarios-20260907T055103Z.json 第 18 题的
# answer_text（CR-053 的具体案例），未做任何删改。
_CR053_ANSWER = (
    "不能。RedisAtomicLong 仅支持原子增减，无法实现读-改-写完整逻辑，"
    "无法判断库存是否充足 [1]。需使用 Lua 脚本实现读取+判断+扣减的原子操作 [1]。  \n"
    "配置 `RedisTemplate` 时应使用 `DefaultRedisScript` 缓存 Lua 脚本 SHA1，"
    "避免每次调用重复传输 [2]。  \n"
    "风险：若未使用 Lua 脚本，多个请求并发时可能触发超卖"
    "（如库存为 1，两个请求同时读取为 1 后扣减） [1]。"
)

# `knowledge/eval/scenario_questions.yaml` 里 RedisAtomicLong 那道题当前
# 实际使用的判据。
_QUESTION = (
    "spring-data-redis 的 RedisAtomicLong 这类原子计数器，如果只调用它的"
    "无条件递增/递减操作来做库存扣减，能不能防止超卖"
)
_POSITIVE_KEYPOINTS = [
    r"(?i)(不能|不行|不足以|does.{0,10}not|cannot)",
    r"(?i)(单个?命令|single command|(没有|无法|不(能|会)).{0,8}(条件|判断|检查|阈值)|no.{0,10}(condition|check))",
]
_FORBID_PATTERNS = [
    r"(?i)(仅支持.{0,6}(原子)?(增减|递增|递减|increment)|only support.{0,10}(atomic )?increment(/decrement)?)",
    r"(?i)((必须|只能|需要?|唯一(的)?(方法|方式|办法)|only way|must|need).{0,10}(Lua|脚本|script))",
]

# CR-054~060 期间反复出现的反例句：不管否定语境到底站不站得住，"只能/
# 必须/需要"这类触发词组只要字面出现，就该被记进 forbidden_hit 供人工
# 复核——不再要求 `_score` 自己判断这些例句里的否定语境是否成立。
_HISTORICAL_REPROS = {
    "CR-055 跨命题窗口泄漏": "不能；没有条件检查，但不只是先读再写，仍然只能用 Lua 脚本。",
    "CR-056 转折连词泄漏": "不能；没有条件检查，并非只能用 Lua 但是仍然只能用 Lua 脚本。",
    "CR-057 松散否定（中文）": "不能；没有条件检查，并不一定只能用 Lua 脚本，也可以采用其他机制。",
    "CR-057 松散否定（英文）": (
        "cannot; no condition check; it is definitely not the only way to use Lua; "
        "other mechanisms work."
    ),
    "CR-059 加合连词泄漏": "不能；没有条件检查，并非只能用 Lua 且必须使用 Lua 脚本。",
    "CR-060 单命中+加合连词": "不能；没有条件检查，并非其他机制可靠 且必须使用 Lua 脚本。",
}

# 一句明确、干净地否定排他性主张、且不含任何独立真实违规的表述——即使
# 不自动判断极性，这句话依然会被记入 forbidden_hit（因为字面包含
# "只能…Lua"），这是本轮接受的代价，用测试如实固定下来，不假装它会被
# 自动放过。
_CLEANLY_NEGATED_ANSWER = "不能直接防止超卖，但这不是只能用 Lua 脚本才能解决，也可以采用其他机制。"


def _make_case(*, failures=(), forbidden_hit=()) -> ScenarioCase:
    return ScenarioCase(
        question=_QUESTION,
        expect_keypoints=_POSITIVE_KEYPOINTS,
        expect_sources=[],
        forbid_patterns=_FORBID_PATTERNS,
        failures=list(failures),
        forbidden_hit=list(forbidden_hit),
    )


def test_forbid_pattern_hit_is_recorded_in_score_output():
    """`_score` 本身只负责记录，不判定状态——命中 forbid_patterns 记入
    forbidden_hit，但不出现在 failures 里，即使正向关键点全部命中。
    """
    hit, missed, forbidden, failures = _score(_CR053_ANSWER, _POSITIVE_KEYPOINTS, _FORBID_PATTERNS)
    assert hit == _POSITIVE_KEYPOINTS
    assert not missed
    assert forbidden, "真实存在的越界断言仍应被记入 forbidden_hit，供人工复核"
    assert not failures, "forbid_patterns 命中不应出现在 failures 里"


def test_missing_keypoints_still_fail_the_case_independent_of_forbid_patterns():
    """确认这条判据完全不受 forbid_patterns 语义变化的影响。"""
    answer = "这段回答完全跑题，什么都没提到。"
    hit, missed, forbidden, failures = _score(answer, _POSITIVE_KEYPOINTS, _FORBID_PATTERNS)
    assert not hit
    assert missed == _POSITIVE_KEYPOINTS
    assert not forbidden
    assert failures and all("未命中关键点" in f for f in failures)


def test_all_historical_negation_repros_are_flagged_in_forbidden_hit():
    """CR-054~060 六轮反例，不管否定语境是否成立，字面上都包含真实的
    forbid_patterns 触发词组——不会再有任何一句被自动逻辑错误放行、
    完全没有痕迹（是否计入通过是 `_case_status` 的事，这里只测 `_score`
    本身的记录是否完整）。
    """
    for label, answer in _HISTORICAL_REPROS.items():
        _, _, forbidden, failures = _score(answer, [], _FORBID_PATTERNS)
        assert forbidden, f"{label}: 应至少命中一条 forbid_patterns 供人工复核"
        assert not failures, f"{label}: forbid_patterns 命中不应出现在 failures 里"


def test_old_ok_semantics_would_have_wrongly_counted_review_required_as_passed():
    """判别性基线：CR-060 刚上线时 `ScenarioCase.ok` 就是 `not failures`，
    完全不管 `forbidden_hit`——用 `_CR053_ANSWER` 的真实 `_score()` 结果
    构造一个 case，复现"命中禁止模式 + 正向关键点全部命中"时，旧逻辑会
    把它算成"通过"。这正是 CR-061 指出的洞：只标记不改变判定，等于没有
    约束。
    """
    _, _, forbidden, failures = _score(_CR053_ANSWER, _POSITIVE_KEYPOINTS, _FORBID_PATTERNS)
    old_ok = not failures  # CR-060 时期 ScenarioCase.ok 的定义
    assert forbidden, "复现前提：这条真实答案必须命中 forbid_patterns"
    assert old_ok, "旧逻辑应该（错误地）把这道题算作通过——这正是 CR-061 的洞"


def test_case_status_is_failed_when_failures_present_regardless_of_reviews():
    """有 failures 的题一律 failed，不管有没有人工复核确认——两者是
    独立的判据，复核确认不能拿来抵消关键点缺失这类硬失败。
    """
    c = _make_case(failures=["未命中关键点: xxx"], forbidden_hit=["yyy"])
    reviews = [{"question": _QUESTION, "pattern": "yyy", "verdict": "confirmed_ok"}]
    assert _case_status(c, reviews) == "failed"


def test_case_status_is_passed_when_no_forbidden_hit():
    """没有命中 forbid_patterns 的题，只要没有 failures 就直接 passed，
    完全不需要查复核记录。
    """
    c = _make_case()
    assert _case_status(c, []) == "passed"


def test_case_status_is_review_required_without_a_matching_review_record():
    """CR-061 的核心行为：命中 forbid_patterns、没有 failures，但找不到
    对应的复核记录——必须是 review_required，不能算 passed。
    """
    c = _make_case(forbidden_hit=[_FORBID_PATTERNS[1]])
    assert _case_status(c, []) == "review_required"


def test_case_status_stays_review_required_with_a_confirmed_issue_verdict():
    """人工看过、确认这确实是个真实问题（`confirmed_issue`）时，依然是
    review_required——"标记已复核"不等于"标记为已解决"，两者是不同的
    状态，不能用 verdict 记录本身冒充"这道题没问题了"。
    """
    c = _make_case(forbidden_hit=[_FORBID_PATTERNS[1]])
    reviews = [{"question": _QUESTION, "pattern": _FORBID_PATTERNS[1], "verdict": "confirmed_issue"}]
    assert _case_status(c, reviews) == "review_required"


def test_case_status_becomes_passed_with_a_confirmed_ok_verdict():
    """只有明确的 `confirmed_ok` 复核记录，才能把命中转为 passed。"""
    c = _make_case(forbidden_hit=[_FORBID_PATTERNS[1]])
    reviews = [{"question": _QUESTION, "pattern": _FORBID_PATTERNS[1], "verdict": "confirmed_ok"}]
    assert _case_status(c, reviews) == "passed"


def test_case_status_requires_every_forbidden_hit_to_be_confirmed():
    """一道题命中了两条 forbid_patterns 时，必须每一条都有 confirmed_ok
    记录才能 passed——只确认其中一条不够，不能靠部分复核冒充全部复核。
    """
    c = _make_case(forbidden_hit=[_FORBID_PATTERNS[0], _FORBID_PATTERNS[1]])
    reviews = [{"question": _QUESTION, "pattern": _FORBID_PATTERNS[0], "verdict": "confirmed_ok"}]
    assert _case_status(c, reviews) == "review_required"


def test_lookup_review_verdict_matches_on_exact_question_and_pattern():
    """复核记录按 (question, pattern) 精确匹配——问法或正则稍有不同都不
    应该命中同一条记录（避免卡片/问法改写后误用过期的复核结论）。
    """
    reviews = [{"question": _QUESTION, "pattern": _FORBID_PATTERNS[1], "verdict": "confirmed_ok"}]
    assert _lookup_review_verdict(_QUESTION, _FORBID_PATTERNS[1], reviews) == "confirmed_ok"
    assert _lookup_review_verdict(_QUESTION, "不同的正则", reviews) is None
    assert _lookup_review_verdict("不同的问题", _FORBID_PATTERNS[1], reviews) is None


def test_cleanly_negated_answer_is_still_flagged_not_silently_dropped():
    """如实固定接受的代价：一句语境上正确的否定表述，字面依然包含触发
    词组，依然会被记入 forbidden_hit（进而在没有复核记录时判
    review_required）——这不是 bug，是"不再自动判断极性"这个设计决定的
    直接后果。
    """
    _, _, forbidden, failures = _score(_CLEANLY_NEGATED_ANSWER, [], _FORBID_PATTERNS)
    assert forbidden, "即使语境上是正确的否定表述，字面命中依然应记入 forbidden_hit"
    assert not failures


def test_real_scenario_review_yaml_entries_declare_required_fields():
    """`knowledge/eval/scenario_review.yaml` 里的每条记录都必须有
    question/pattern/verdict/note/reviewed_at，且 verdict 只能是两个
    受支持的值——防止漏填字段导致复核记录悄悄失效（`_lookup_review_
    verdict` 找不到就等于没复核过）。
    """
    data = yaml.safe_load(REVIEWS.read_text(encoding="utf-8"))
    reviews = data.get("reviews", [])
    assert reviews, "至少应有一条真实复核记录（当前已知 RedisAtomicLong 那题的命中）"
    for r in reviews:
        for field in ("question", "pattern", "verdict", "note", "reviewed_at"):
            assert str(r.get(field, "")).strip(), f"记录缺少 {field}: {r}"
        assert r["verdict"] in ("confirmed_ok", "confirmed_issue"), (
            f"verdict 只能是 confirmed_ok/confirmed_issue，实际: {r['verdict']!r}"
        )
