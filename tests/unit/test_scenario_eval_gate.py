"""场景评测判据自身的回归（CR-053~065）。

为什么单独测：判据是场景评测结果的门禁，判据本身漏判会让"通过/不通过"
这个数字失去意义——就像 `test_probe_gate.py` 对排序探针门禁（CR-015）
做的事一样。

这里测的是 `forbid_patterns` 这个负向判据本身的完整教训史：

- **CR-053**：`run_scenarios.py` 原先只看正向的 `expect_keypoints`，一条
  答案哪怕混进一句已知错误的越界断言，只要凑巧命中了两条比较宽泛的正向
  关键点，照样被记成"关键点全部命中"、判为通过——新增 `forbid_patterns`，
  命中即记入 `forbidden_hit`。
- **CR-054~060**：`forbid_patterns` 上线后，又花了六轮反复调整"命中之后
  要不要自动判断这次命中是否被否定"这条逻辑，每次修一个反例就暴露一个
  新反例。根因是这条启发式想用"这段前文有没有否定词"回答一个本质上是
  句法结构的问题，而连接词是开放集合，枚举永远追不完。CR-060 决定不再
  尝试自动判断否定极性：命中只记入 `forbidden_hit`，交给人工判断。
- **CR-061**：指出 CR-060 的修法本身还不完整——`forbidden_hit` 完全不
  影响这道题的通过判定，等于换了个说法重新引入 CR-053 想堵住的洞。修法：
  三态判定（`passed`/`failed`/`review_required`），命中但没有人工复核
  确认的题目一律 `review_required`，不计入通过数、不能让整次评测的
  退出码为 0；复核结论持久化存放在
  `knowledge/eval/scenario_review.yaml`。
- **CR-062**：指出 `confirmed_issue`（人工已经看过、确认这确实是问题）
  被 CR-061 错误地也归为 `review_required`，和"根本没人看过"混在一起，
  下游没法区分"待办"和"已确认的失败"。修法：`confirmed_issue` 映射为
  `failed`，只有找不到任何匹配记录才是 `review_required`。
- **CR-063**：指出 CR-061/062 的复核记录只按 `(question, pattern)` 匹配，
  不看具体回答内容——同一题同一 pattern，这次命中的可能是正确的否定
  表述，下次命中的可能是真实的排他性断言，一条历史 `confirmed_ok` 会
  连未来完全不同的违规内容一起放行。修法：复核记录额外绑定
  `answer_hash`（这次具体回答的内容指纹），三者必须同时匹配才算数，
  模型换一次说法旧记录就自动失效。
- **CR-064**：指出 `main()` 里的汇总/退出码计算完全没有独立测试，
  `--limit 1` 这类端到端手工验证也没能真正命中 `review_required` 分支
  来证明它确实会让退出码非零。修法：把汇总/退出码判定拆成纯函数
  `_summarize()`，可以脱离模型/索引单测。
- **CR-065**：指出复核档案没有校验 `(question, pattern, answer_hash)`
  三元组的唯一性——如果同一个三元组意外出现两条记录、一条
  `confirmed_ok` 一条 `confirmed_issue`，`_lookup_review_verdict()`
  原来按 YAML 里出现的顺序取第一条，`confirmed_ok` 排在前面就会把
  一道真实有问题的题错误判为通过。修法：新增 `_validate_reviews()`
  在加载时拒绝重复三元组；`_lookup_review_verdict()` 本身也加一层
  独立的失败关闭兜底——遇到冲突的多条记录返回 `None`，不按顺序挑一条。

这里只测纯判定函数（`_score`/`_case_status`/`_lookup_review_verdict`/
`_validate_reviews`/`_summarize`/`_answer_hash`），不加载模型/索引，
因此毫秒级，可进快速门禁。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from packages.evaltools.run_scenarios import (  # noqa: E402
    QUESTIONS, REVIEWS, ScenarioCase, _answer_hash, _case_status,
    _lookup_review_verdict, _score, _summarize, _validate_reviews,
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

# CR-063 复现：同一题同一 pattern，两份内容截然不同的回答——一份是正确
# 的否定表述，一份是真实的排他性断言。
_CR063_NEGATED = "不能直接防止超卖，但这不是只能用 Lua 脚本才能解决，也可以采用其他机制。"
_CR063_VIOLATION = "不能直接防止超卖。必须使用 Lua 脚本才能补上条件检查。"


def _make_case(*, answer_text=_CR053_ANSWER, failures=(), forbidden_hit=()) -> ScenarioCase:
    return ScenarioCase(
        question=_QUESTION,
        expect_keypoints=_POSITIVE_KEYPOINTS,
        expect_sources=[],
        forbid_patterns=_FORBID_PATTERNS,
        answer_text=answer_text,
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
    复现"命中禁止模式 + 正向关键点全部命中"时，旧逻辑会把它算成"通过"。
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
    reviews = [{
        "question": _QUESTION, "pattern": "yyy",
        "answer_hash": _answer_hash(c.answer_text), "verdict": "confirmed_ok",
    }]
    assert _case_status(c, reviews) == "failed"


def test_case_status_is_passed_when_no_forbidden_hit():
    """没有命中 forbid_patterns 的题，只要没有 failures 就直接 passed，
    完全不需要查复核记录。
    """
    c = _make_case()
    assert _case_status(c, []) == "passed"


def test_case_status_is_review_required_without_a_matching_review_record():
    """核心行为：命中 forbid_patterns、没有 failures，但找不到对应的
    复核记录——必须是 review_required，不能算 passed。
    """
    c = _make_case(forbidden_hit=[_FORBID_PATTERNS[1]])
    assert _case_status(c, []) == "review_required"


def test_case_status_becomes_failed_with_a_confirmed_issue_verdict():
    """CR-062 修复：人工看过、确认这确实是个真实问题（`confirmed_issue`）
    时，判 failed——这是一个已完成的人工判断，不是"还没人看"，不该和
    review_required（真的没人看过）混为一谈。
    """
    c = _make_case(forbidden_hit=[_FORBID_PATTERNS[1]])
    reviews = [{
        "question": _QUESTION, "pattern": _FORBID_PATTERNS[1],
        "answer_hash": _answer_hash(c.answer_text), "verdict": "confirmed_issue",
    }]
    assert _case_status(c, reviews) == "failed"


def test_pre_cr062_confirmed_issue_was_wrongly_left_as_review_required():
    """判别性基线：CR-061 刚上线时的三态判定只区分"有没有 confirmed_ok"，
    把 confirmed_issue 和"完全没有记录"同等对待，都判 review_required——
    这正是 CR-062 指出的洞：已经完成的人工判断被晾在"待办"里。
    """
    c = _make_case(forbidden_hit=[_FORBID_PATTERNS[1]])
    reviews = [{
        "question": _QUESTION, "pattern": _FORBID_PATTERNS[1],
        "answer_hash": _answer_hash(c.answer_text), "verdict": "confirmed_issue",
    }]

    def _pre_cr062_status(c, reviews):
        if c.failures:
            return "failed"
        if not c.forbidden_hit:
            return "passed"
        for pattern in c.forbidden_hit:
            verdict = next(
                (r.get("verdict") for r in reviews
                 if r.get("question") == c.question and r.get("pattern") == pattern),
                None,
            )
            if verdict != "confirmed_ok":
                return "review_required"
        return "passed"

    assert _pre_cr062_status(c, reviews) == "review_required", (
        "旧逻辑应该（错误地）把已确认的问题也晾在 review_required 里——这正是 CR-062 的洞"
    )
    assert _case_status(c, reviews) == "failed"


def test_case_status_becomes_passed_with_a_matching_confirmed_ok_verdict():
    """只有明确的 `confirmed_ok` 复核记录、且 answer_hash 匹配这次具体
    回答，才能把命中转为 passed。
    """
    c = _make_case(forbidden_hit=[_FORBID_PATTERNS[1]])
    reviews = [{
        "question": _QUESTION, "pattern": _FORBID_PATTERNS[1],
        "answer_hash": _answer_hash(c.answer_text), "verdict": "confirmed_ok",
    }]
    assert _case_status(c, reviews) == "passed"


def test_case_status_requires_every_forbidden_hit_to_be_confirmed():
    """一道题命中了两条 forbid_patterns 时，必须每一条都有匹配的
    confirmed_ok 记录才能 passed——只确认其中一条不够。
    """
    c = _make_case(forbidden_hit=[_FORBID_PATTERNS[0], _FORBID_PATTERNS[1]])
    reviews = [{
        "question": _QUESTION, "pattern": _FORBID_PATTERNS[0],
        "answer_hash": _answer_hash(c.answer_text), "verdict": "confirmed_ok",
    }]
    assert _case_status(c, reviews) == "review_required"


def test_confirmed_ok_bound_to_one_answer_does_not_leak_to_a_different_violation():
    """CR-063 复现：同一题同一 pattern，`_CR063_NEGATED`（正确的否定
    表述）和 `_CR063_VIOLATION`（真实的排他性断言）是两份完全不同的
    回答。只给 `_CR063_NEGATED` 的具体内容记一条 `confirmed_ok`，不应该
    连带放行 `_CR063_VIOLATION`——否则一次对否定语境的确认会放行未来
    完全不同的真实违规，重新打开 CR-053 的洞。
    """
    negated_case = _make_case(answer_text=_CR063_NEGATED, forbidden_hit=[_FORBID_PATTERNS[1]])
    violation_case = _make_case(answer_text=_CR063_VIOLATION, forbidden_hit=[_FORBID_PATTERNS[1]])
    assert _answer_hash(_CR063_NEGATED) != _answer_hash(_CR063_VIOLATION), (
        "复现前提：两份回答必须产生不同的 answer_hash"
    )
    reviews = [{
        "question": _QUESTION, "pattern": _FORBID_PATTERNS[1],
        "answer_hash": _answer_hash(_CR063_NEGATED), "verdict": "confirmed_ok",
    }]
    assert _case_status(negated_case, reviews) == "passed"
    assert _case_status(violation_case, reviews) == "review_required", (
        "只确认过否定表述的那次内容，不应该连带放行内容完全不同的真实违规"
    )


def test_pre_cr063_question_pattern_only_matching_would_have_leaked():
    """判别性基线：如果复核记录只按 (question, pattern) 匹配、不看
    answer_hash（CR-061/062 时期的实现），同一条 confirmed_ok 记录会
    把 `_CR063_VIOLATION` 也放行——这正是 CR-063 指出的洞。
    """
    def _pre_cr063_lookup(question, pattern, reviews):
        for r in reviews:
            if r.get("question") == question and r.get("pattern") == pattern:
                return r.get("verdict")
        return None

    reviews = [{"question": _QUESTION, "pattern": _FORBID_PATTERNS[1], "verdict": "confirmed_ok"}]
    assert _pre_cr063_lookup(_QUESTION, _FORBID_PATTERNS[1], reviews) == "confirmed_ok", (
        "旧的按 (question, pattern) 匹配的查找，应该（错误地）对任何内容都放行"
    )


def test_lookup_review_verdict_requires_all_three_keys_to_match():
    """复核记录按 (question, pattern, answer_hash) 三元组精确匹配——
    任何一项不同都不应该命中同一条记录。
    """
    h = _answer_hash(_CR053_ANSWER)
    reviews = [{"question": _QUESTION, "pattern": _FORBID_PATTERNS[1], "answer_hash": h, "verdict": "confirmed_ok"}]
    assert _lookup_review_verdict(_QUESTION, _FORBID_PATTERNS[1], h, reviews) == "confirmed_ok"
    assert _lookup_review_verdict(_QUESTION, "不同的正则", h, reviews) is None
    assert _lookup_review_verdict("不同的问题", _FORBID_PATTERNS[1], h, reviews) is None
    assert _lookup_review_verdict(_QUESTION, _FORBID_PATTERNS[1], "0000000000000000", reviews) is None


def test_cleanly_negated_answer_is_still_flagged_not_silently_dropped():
    """如实固定接受的代价：一句语境上正确的否定表述，字面依然包含触发
    词组，依然会被记入 forbidden_hit（进而在没有匹配复核记录时判
    review_required）——这不是 bug，是"不再自动判断极性"这个设计决定的
    直接后果。
    """
    _, _, forbidden, failures = _score(_CR063_NEGATED, [], _FORBID_PATTERNS)
    assert forbidden, "即使语境上是正确的否定表述，字面命中依然应记入 forbidden_hit"
    assert not failures


def test_real_scenario_review_yaml_entries_declare_required_fields():
    """`knowledge/eval/scenario_review.yaml` 里的每条记录都必须有
    question/pattern/answer_hash/verdict/note/reviewed_at，且 verdict
    只能是两个受支持的值——防止漏填字段导致复核记录悄悄失效
    （`_lookup_review_verdict` 找不到就等于没复核过）。
    """
    data = yaml.safe_load(REVIEWS.read_text(encoding="utf-8"))
    reviews = data.get("reviews", [])
    assert reviews, "至少应有一条真实复核记录（当前已知 RedisAtomicLong 那题的命中）"
    for r in reviews:
        for field in ("question", "pattern", "answer_hash", "verdict", "note", "reviewed_at"):
            assert str(r.get(field, "")).strip(), f"记录缺少 {field}: {r}"
        assert r["verdict"] in ("confirmed_ok", "confirmed_issue"), (
            f"verdict 只能是 confirmed_ok/confirmed_issue，实际: {r['verdict']!r}"
        )
        assert len(r["answer_hash"]) == 16, f"answer_hash 应为 16 位十六进制指纹: {r['answer_hash']!r}"
    _validate_reviews(reviews)  # CR-065：真实档案不应该有重复的三元组


# ---- CR-065：复核档案的三元组唯一性校验 ----

def test_validate_reviews_accepts_a_clean_list():
    reviews = [
        {"question": _QUESTION, "pattern": _FORBID_PATTERNS[1], "answer_hash": "aaaa", "verdict": "confirmed_ok"},
        {"question": _QUESTION, "pattern": _FORBID_PATTERNS[0], "answer_hash": "aaaa", "verdict": "confirmed_issue"},
    ]
    _validate_reviews(reviews)  # 不应该抛异常


def test_validate_reviews_rejects_a_conflicting_duplicate_triple():
    """CR-065 复现：同一个 (question, pattern, answer_hash) 三元组出现
    两条记录、verdict 互相矛盾——必须在加载时就拒绝。
    """
    reviews = [
        {"question": _QUESTION, "pattern": _FORBID_PATTERNS[1], "answer_hash": "aaaa", "verdict": "confirmed_ok"},
        {"question": _QUESTION, "pattern": _FORBID_PATTERNS[1], "answer_hash": "aaaa", "verdict": "confirmed_issue"},
    ]
    with pytest.raises(ValueError, match="重复的"):
        _validate_reviews(reviews)


def test_validate_reviews_rejects_an_exact_duplicate_triple_too():
    """即使两条记录的 verdict 完全一致，重复的三元组本身也说明数据有
    问题（复制粘贴遗留），同样应该拒绝，不只是"verdict 冲突"才拒绝。
    """
    reviews = [
        {"question": _QUESTION, "pattern": _FORBID_PATTERNS[1], "answer_hash": "aaaa", "verdict": "confirmed_ok"},
        {"question": _QUESTION, "pattern": _FORBID_PATTERNS[1], "answer_hash": "aaaa", "verdict": "confirmed_ok"},
    ]
    with pytest.raises(ValueError, match="重复的"):
        _validate_reviews(reviews)


def test_pre_cr065_lookup_silently_picked_the_first_matching_record():
    """判别性基线：CR-061~064 时期的 `_lookup_review_verdict()` 只是
    遍历列表、返回第一条匹配的记录——如果两条记录的三元组相同但 verdict
    冲突，且 `confirmed_ok` 恰好排在前面，旧实现会把这道真实有问题的题
    错误地判定为通过。这里内联复现那个旧实现，证明这个洞是真实存在的。
    """
    reviews = [
        {"question": _QUESTION, "pattern": _FORBID_PATTERNS[1], "answer_hash": "aaaa", "verdict": "confirmed_ok"},
        {"question": _QUESTION, "pattern": _FORBID_PATTERNS[1], "answer_hash": "aaaa", "verdict": "confirmed_issue"},
    ]

    def _pre_cr065_lookup(question, pattern, answer_hash, reviews):
        for r in reviews:
            if (r.get("question") == question and r.get("pattern") == pattern
                    and r.get("answer_hash") == answer_hash):
                return r.get("verdict")
        return None

    assert _pre_cr065_lookup(_QUESTION, _FORBID_PATTERNS[1], "aaaa", reviews) == "confirmed_ok", (
        "旧实现应该（错误地）采信排在前面的 confirmed_ok，即使后面还有一条互相矛盾的 confirmed_issue"
    )


def test_lookup_review_verdict_fails_closed_on_conflicting_duplicate_records():
    """CR-065 修复：`_lookup_review_verdict()` 本身也做了独立的失败
    关闭兜底——遇到同一三元组的多条冲突记录，返回 `None`（等价于"没有
    复核过"），不像旧实现那样按列表顺序采信第一条。这是防御性的第二
    道防线，即使某处绕过了 `_validate_reviews()` 也不会被曾经的洞
    绕过去。
    """
    reviews = [
        {"question": _QUESTION, "pattern": _FORBID_PATTERNS[1], "answer_hash": "aaaa", "verdict": "confirmed_ok"},
        {"question": _QUESTION, "pattern": _FORBID_PATTERNS[1], "answer_hash": "aaaa", "verdict": "confirmed_issue"},
    ]
    assert _lookup_review_verdict(_QUESTION, _FORBID_PATTERNS[1], "aaaa", reviews) is None


# ---- CR-064：main() 汇总/退出码逻辑的独立测试，不需要模型/索引 ----

def _status_only_case(status: str) -> ScenarioCase:
    c = _make_case()
    c.status = status
    return c


def test_summarize_all_passed_gives_exit_code_zero():
    cases = [_status_only_case("passed"), _status_only_case("passed")]
    summary = _summarize(cases)
    assert summary == {"passed": 2, "failed": 0, "review_required": 0, "exit_code": 0}


def test_summarize_with_a_failed_case_gives_nonzero_exit_code():
    cases = [_status_only_case("passed"), _status_only_case("failed")]
    summary = _summarize(cases)
    assert summary["exit_code"] == 1


def test_summarize_with_only_review_required_gives_nonzero_exit_code():
    """CR-064 的核心行为：即使没有任何 failed，只要有 review_required，
    整次评测也不能算成功退出——这是 `--limit 1` 端到端手工验证没能稳定
    覆盖到的分支，这里用纯函数直接测。
    """
    cases = [_status_only_case("passed"), _status_only_case("review_required")]
    summary = _summarize(cases)
    assert summary["passed"] == 1
    assert summary["review_required"] == 1
    assert summary["exit_code"] == 1, "只有 review_required、没有 failed 时，退出码也必须非零"


def test_pre_cr061_naive_summary_would_have_returned_zero_for_review_required():
    """判别性基线：如果沿用 CR-060 之前"只看 not failures"的朴素汇总（把
    review_required 当成"没有 failures 所以算通过"），同样的用例会得出
    退出码 0——这正是 CR-061/064 要堵住的洞。
    """
    cases = [_status_only_case("passed"), _status_only_case("review_required")]

    def _naive_passed(c):
        return c.status != "failed"  # 旧思路：不是明确 failed 就算过

    naive_passed_count = sum(1 for c in cases if _naive_passed(c))
    naive_exit_code = 0 if naive_passed_count == len(cases) else 1
    assert naive_exit_code == 0, "朴素汇总应该（错误地）把这种情况判定为整体成功"
    assert _summarize(cases)["exit_code"] == 1


# ---- CR-066：关键点只查术语、不约束结论方向 ----
#
# codex R60 审查指出：第四张卡片（PostgreSQL 慢 SQL）新增的 8 道题，
# `expect_keypoints` 检查的是"术语有没有出现"，不是"结论对不对"，因此
# 一个**事实词全中、结论恰好相反**的答案会被判 passed。审查方给了两个
# 复现：一个是保存报告里的真实输出（Q20 建议用 total_exec_time 看单次
# 极端延迟），一个是纯 `_score()` 构造的反例（Q22 说"对不上就说明计划
# 有问题"）。
#
# 修法两条通道，下面的测试分别覆盖：
#   1. 正向 keypoint 改成"绑定式"——术语必须和它正确的含义出现在同一句
#      里。挡的是"把术语背对了"的答案。
#   2. 结论方向靠 `forbid_patterns` 进三态通道（命中 → review_required
#      → 人工复核）。正向正则原理上挡不住"事实全对、结论反过来"，因为
#      对手把两条事实也写进去就能全中——这正是 Q22 反例的构造方式。
#
# 每条测试都成对写："旧规则会放行"（判别性基线，内联复现改之前的正则）
# 与"新规则不会放行"（读 `knowledge/eval/scenario_questions.yaml` 里
# **当前真实生效**的判据，不复制一份，避免判据改了测试还在测旧副本）。

_QUESTIONS_YAML = yaml.safe_load(QUESTIONS.read_text(encoding="utf-8"))["questions"]


def _spec(prefix: str) -> dict:
    """按题干前缀取当前真实生效的题目判据。"""
    matches = [q for q in _QUESTIONS_YAML if q["q"].startswith(prefix)]
    assert len(matches) == 1, f"题干前缀 {prefix!r} 应唯一匹配一题，实际 {len(matches)} 条"
    return matches[0]


def _status_of(spec: dict, answer: str) -> str:
    """把 `_score` + `_case_status` 串起来，得到这次回答的三态结果。
    复核档案传空列表：这些反例都是新出现的内容，本来就不该有复核记录。
    """
    hit, missed, forbidden, failures = _score(
        answer, spec["expect_keypoints"], spec.get("forbid_patterns", [])
    )
    case = ScenarioCase(
        question=spec["q"],
        expect_keypoints=spec["expect_keypoints"],
        expect_sources=[],
        forbid_patterns=spec.get("forbid_patterns", []),
        answer_text=answer,
        keypoints_hit=hit,
        keypoints_missed=missed,
        forbidden_hit=forbidden,
        failures=failures,
    )
    return _case_status(case, [])


# 逐字取自 bench/reports/eval-scenarios-20260908T012349Z.json 第 20 题的
# answer_text（CR-066 的真实误放行案例），未做任何删改。
_CR066_Q20_REAL_ANSWER = (
    "按 total_exec_time 排序会突出执行总耗时最长的语句，反映整体性能瓶颈；"
    "按 mean_exec_time 排序反映单次执行的平均耗时，更适用于识别频繁执行但每次耗时较高的语句 [1][2]。  \n"
    "若需优化高负载场景，应优先看 mean_exec_time，因其反映常态执行效率；"
    "若关注单次极端延迟，可看 total_exec_time [1][2]。  \n"
    "注意：total_exec_time 可能被长尾请求拉高，而 mean_exec_time 更稳定 [1][2]。"
)

# 逐字取自 code-review.md CR-066 的复现描述。
_CR066_Q22_COUNTEREXAMPLE = (
    "**是**，cost 虽然是任意单位，估算行数也要和实际比较，"
    "但只要两者对不上就说明计划有问题。"
)

# 改判据之前，这 8 道题用的正向关键点（判别性基线，内联复现）。
_PRE_CR066_KEYPOINTS = {
    "Q19": [
        r"(?i)(不是|并非|不一定|not the|规范化|归一化|normaliz|代表性?(的)?(文本|语句|查询)|representative)",
        r"(?i)(\$1|常量|字面量|参数符号|placeholder|literal)",
    ],
    "Q20": [
        r"(?i)(total_exec_time|累计|总(的)?(执行)?(时间|耗时)|总耗时)",
        r"(?i)(mean_exec_time|平均|单次|每次|per.?call)",
    ],
    "Q21": [
        r"(?i)(所有|全部|每(一)?(条|个)|不(只|仅)是?.{0,12}(慢|被记录|记录下来)|all statements|whether or not)",
        r"(?i)(性能|开销|overhead|impact|影响|变慢|代价)",
    ],
    "Q23": [
        r"(?i)loops",
        r"(?i)(平均|每次(执行)?|per.?execution|乘(以)?|multiply|总(的)?(时间|耗时))",
    ],
    "Q24": [
        r"(?i)(不是|并非|不属于|not an? estimation error|不能算|无需)",
        r"(?i)(LIMIT|提前(停止|结束)|停(止|下)|stopped short|跑完|run to completion|取(够|满)|够了就)",
    ],
    "Q25": [
        r"(?i)(相关|correlat|独立(性)?假设|independent)",
        r"(?i)(CREATE STATISTICS|扩展统计|多元统计|multivariate|extended statistics)",
    ],
}

# 每条：题干前缀、结论方向相反（但事实词齐全）的答案、这段答案为什么是错的。
_CR066_WRONG_DIRECTION_ANSWERS = {
    "Q19": (
        "pg_stat_statements 的 query 列显示的语句",
        "不一定完全一致，取决于 search_path，常量原样保留不会被替换。",
        "否认了常量会被归一化成 $1，与卡片相反",
    ),
    "Q20": (
        "想从 pg_stat_statements 里挑出最该优化的语句",
        _CR066_Q20_REAL_ANSWER,
        "把单次极端延迟指向 total_exec_time（应看 mean/max）",
    ),
    "Q21": (
        "线上偶发的慢查询抓不到执行计划",
        "auto_explain 会记录所有超过阈值的慢语句，性能开销可以忽略。",
        "官方原文是 extremely negative impact，且计时发生在所有语句上",
    ),
    "Q22": (
        "EXPLAIN ANALYZE 输出里 cost 和 actual time",
        _CR066_Q22_COUNTEREXAMPLE,
        "官方明确说这类不符本身不代表计划有问题",
    ),
    "Q23": (
        "嵌套循环内层的 Index Scan 节点",
        "是的，actual time 只有 0.003 毫秒说明这个节点不耗时，"
        "loops 只是循环次数，平均值已经足够说明问题。",
        "actual time 是每次执行的平均值，要乘 loops 才是总时间",
    ),
    "Q24": (
        "计划里 Index Scan 估计返回 10 行",
        "这不是 LIMIT 的问题，而是统计信息不准，建议重新收集统计信息后再看计划。",
        "官方明确说这是显示方式的差异，不是估计错误",
    ),
    "Q25": (
        "两个 WHERE 条件涉及的列都刚 ANALYZE 过",
        "行数估计差是因为列之间相关，独立性假设不成立；"
        "再跑一次 ANALYZE 就能修好，也可以用 CREATE STATISTICS。",
        "重跑 ANALYZE 修不了跨列相关，规则统计天生测不到它",
    ),
}


@pytest.mark.parametrize("key", sorted(_PRE_CR066_KEYPOINTS))
def test_pre_cr066_keypoints_would_have_passed_the_wrong_direction_answer(key):
    """判别性基线：改判据之前，这些结论完全相反的答案两条关键点全中、
    `failures` 为空——也就是会被判 passed。Q20 用的是保存报告里的真实
    输出，不是构造的。（Q22 没有列在这里，因为它的两条正向关键点本轮
    一字未改，误放行发生在"正向全中但结论相反"这一层，见下一条测试。）
    """
    _prefix, answer, _why = _CR066_WRONG_DIRECTION_ANSWERS[key]
    hit, missed, forbidden, failures = _score(answer, _PRE_CR066_KEYPOINTS[key], [])
    assert not missed, f"旧关键点本应全部命中（这正是问题所在）：{missed}"
    assert hit == _PRE_CR066_KEYPOINTS[key]
    assert not failures, "旧规则下这个结论相反的答案没有任何 failures，会被判 passed"
    assert not forbidden


@pytest.mark.parametrize("key", sorted(_CR066_WRONG_DIRECTION_ANSWERS))
def test_cr066_wrong_direction_answers_are_no_longer_passed(key):
    """新规则：同样这些答案不能再是 `passed`——要么因为绑定式关键点没
    命中而 `failed`，要么因为命中负向约束而 `review_required`（交人工看）。
    两种结果都可接受，本项目要求的是"判据能阻断同类反向结论"，不是
    "必须自动判成 failed"（CR-054~060 已证明正则判不了极性）。
    """
    prefix, answer, why = _CR066_WRONG_DIRECTION_ANSWERS[key]
    status = _status_of(_spec(prefix), answer)
    assert status in ("failed", "review_required"), f"{key}（{why}）不应再被判 passed"


def test_cr066_q20_real_answer_is_flagged_by_the_new_forbid_pattern():
    """点名复现：保存报告里第 20 题的真实输出。它的两条绑定式关键点
    **仍然全中**（这个答案确实分别讲对了两个字段的定义），所以拦住它的
    只能是负向约束——正向正则在这里原理上无能为力。
    """
    spec = _spec("想从 pg_stat_statements 里挑出最该优化的语句")
    hit, missed, forbidden, failures = _score(
        _CR066_Q20_REAL_ANSWER, spec["expect_keypoints"], spec["forbid_patterns"]
    )
    assert not missed and not failures, "两条绑定式关键点对这个答案依然成立"
    assert forbidden, "'若关注单次极端延迟，可看 total_exec_time' 必须被负向约束标出"
    assert _status_of(spec, _CR066_Q20_REAL_ANSWER) == "review_required"


def test_cr066_q22_counterexample_is_flagged_by_the_new_forbid_pattern():
    """点名复现：审查方给的 Q22 反例原话。两条正向关键点全中且本轮未改，
    因此只可能被负向约束拦住。
    """
    spec = _spec("EXPLAIN ANALYZE 输出里 cost 和 actual time")
    hit, missed, forbidden, failures = _score(
        _CR066_Q22_COUNTEREXAMPLE, spec["expect_keypoints"], spec["forbid_patterns"]
    )
    assert not missed and not failures, "反例正是构造成两条正向关键点全中的"
    assert forbidden
    assert _status_of(spec, _CR066_Q22_COUNTEREXAMPLE) == "review_required"


# 按卡片正文写的"正确答案"样例：新判据不能严到连正确答案都判不过
# （只会变红的判据和只会变绿的判据一样没有信息量）。
_CR066_CORRECT_ANSWERS = {
    "pg_stat_statements 的 query 列显示的语句":
        "不是逐字一样。pg_stat_statements 会把仅有字面常量差异的语句归一化成同一条记录，"
        "常量显示为 $1，文本取的是该 queryid 第一条查询的代表文本。",
    "想从 pg_stat_statements 里挑出最该优化的语句":
        "total_exec_time 是跨全部调用的累计耗时，排在前面的是总时间花得最多的语句形状；"
        "mean_exec_time 反映单次执行的平均耗时，单次最坏要看 max_exec_time。",
    "线上偶发的慢查询抓不到执行计划":
        "log_analyze 打开后，所有语句都会做逐节点计时，而不只是够慢被记录下来的那些，"
        "对性能影响很大；可以关掉 log_timing 或调低 sample_rate 来减轻开销。",
    "EXPLAIN ANALYZE 输出里 cost 和 actual time":
        "cost 是任意单位，与毫秒本来就不可比，数值对不上不说明什么；"
        "该先看的是估计行数和实际行数的差距。",
    "嵌套循环内层的 Index Scan 节点":
        "不能这么看。内层节点的 actual time 是每次执行的平均值，要乘以 loops 才是它的总耗时。",
    "计划里 Index Scan 估计返回 10 行":
        "不是统计信息不准。LIMIT 让节点提前停止取行，而估计值按跑完整来显示，"
        "两者的差异只是显示方式不同。",
    "两个 WHERE 条件涉及的列都刚 ANALYZE 过":
        "再跑一次 ANALYZE 也没用：规则统计是逐列的，天生测不到跨列相关性，"
        "planner 仍按条件独立假设估算。要用 CREATE STATISTICS 建扩展统计对象，"
        "再跑一次 ANALYZE 才会真正收集数据。",
    "已经建了包含查询所有列的覆盖索引":
        "因为可见性还得回堆里确认：index-only scan 会查 visibility map 的 all-visible 位，"
        "位没置上就必须访问 heap，跟普通索引扫描相比就没有优势了。",
}


# ---- CR-067：CR-066 漏掉的三题（先讲对机制、后反转结论）----
#
# codex R61 复审指出：CR-066 给 Q20/Q22/Q23 配了负向约束，却把
# Q19/Q24/Q25 当成"绑定式正向关键点就够了"的例外。审查方用当前规则
# 独立复现，三题分别 2/2、2/2、3/3 命中、无 forbidden_hit、
# `_case_status()` 全返回 `passed`——正是 R61 自己定义为"正向正则判不了
# 结论方向、应进三态人工复核"的那种形状。下面三条例句逐字取自
# `code-review.md` 的 CR-067 记录。

_CR067_REVERSED_AFTER_CORRECT = {
    "Q19": (
        "pg_stat_statements 的 query 列显示的语句",
        "语句会归一化，常量显示 $1，但 query 仍逐字等于原文。",
        "前半句讲对了归一化，后半句否认它",
    ),
    "Q24": (
        "计划里 Index Scan 估计返回 10 行",
        "不是统计信息不准；LIMIT 会提前停止，但这依然说明统计信息不准。",
        "先答对再反转回「统计信息不准」",
    ),
    "Q25": (
        "两个 WHERE 条件涉及的列都刚 ANALYZE 过",
        "列相关，独立性假设不成立；可以用 CREATE STATISTICS，"
        "再跑 ANALYZE 没用，但总体重跑 ANALYZE 就能解决。",
        "三条 keypoint 都被前半句满足，结论在后半句反转",
    ),
}


@pytest.mark.parametrize("key", sorted(_CR067_REVERSED_AFTER_CORRECT))
def test_pre_cr067_rules_would_have_passed_the_reversed_conclusion(key):
    """判别性基线：只用 CR-066 那版规则（正向关键点，Q19/Q24/Q25 当时
    没有任何负向约束），这三个例句全部 `passed`——复现审查方的结论，
    不是采信它。
    """
    prefix, answer, _why = _CR067_REVERSED_AFTER_CORRECT[key]
    spec = _spec(prefix)
    hit, missed, forbidden, failures = _score(answer, spec["expect_keypoints"], [])  # 无负向约束
    assert not missed, f"正向关键点本应全中（这正是问题所在）：{missed}"
    assert not failures
    case = ScenarioCase(
        question=spec["q"], expect_keypoints=spec["expect_keypoints"], expect_sources=[],
        forbid_patterns=[], answer_text=answer, keypoints_hit=hit, forbidden_hit=[], failures=[],
    )
    assert _case_status(case, []) == "passed", "旧规则下这个反转结论会被判通过"


@pytest.mark.parametrize("key", sorted(_CR067_REVERSED_AFTER_CORRECT))
def test_cr067_reversed_conclusion_is_no_longer_passed(key):
    """新规则：三题各自的负向约束必须把它拦成 `review_required`
    （正向关键点仍然全中——反转发生在下一句，这正是为什么只能靠
    负向通道）。
    """
    prefix, answer, why = _CR067_REVERSED_AFTER_CORRECT[key]
    spec = _spec(prefix)
    hit, missed, forbidden, failures = _score(
        answer, spec["expect_keypoints"], spec.get("forbid_patterns", [])
    )
    assert not missed and not failures, "正向关键点依然全中，说明拦住它的确实是负向约束"
    assert forbidden, f"{key}（{why}）必须被负向约束标出"
    assert _status_of(spec, answer) == "review_required"


# "X 不 X" 叠字（能不能 / 是不是 / 会不会 / 要不要 / 需不需要 / 行不行…）是
# 中文正反问句最稳定的结构特征，比逐个往词表里加词可靠——CR-082 正是
# 词表漏了"能不能"漏出来的。
_A_NOT_A = re.compile(r"(.)不\1")
# 叠字覆盖不到、但同样带明确结论方向的问法，只能显式列。
_POLARITY_PHRASES = ("吗", "是否", "可否", "能否", "有没有",
                     "有什么不同", "有什么代价", "先看哪个")


def _is_yes_no_question(text: str) -> bool:
    return bool(_A_NOT_A.search(text)) or any(m in text for m in _POLARITY_PHRASES)


def test_cr067_every_yes_no_question_has_a_negative_constraint():
    """结构性约束：凡是有明确结论方向的是非题都必须配 `forbid_patterns`——
    CR-066/067 两轮的共同教训是"漏配一题就漏一题"，靠人记不住，写成测试。

    **两次修补，两种不同形状的同一个后门**：

    1. **R71（2026-09-09）扩到全量**：原来写的是 `_QUESTIONS_YAML[-8:]`，
       只检查最近一批。那是个**滑动窗口**——R71 加完第五张卡片后一数，
       前三张卡片的 Q4/Q5/Q6/Q7/Q10/Q17 六道是非题一条负向约束都没有，
       而门禁一直是绿的。
    2. **R72（CR-082）改判定方式**：遍历全量之后仍然漏，因为 `markers`
       是一份**手工维护的词表**，里面没有"能不能"——Q8/Q13/Q15 三道
       直接是非题照样在不判定的空间里。词表这种形式的问题和滑动窗口
       一样：它在写下的那一刻是对的，之后每加一种问法就默默漏一次，
       且没有任何信号。改用中文正反问句的结构特征"X 不 X"叠字
       （`_A_NOT_A`）作为主判据，词表退化为叠字覆盖不到的补充项。

    **这条判据能做到什么、做不到什么**（按 CR-082 的要求写清楚，不作
    全称承诺）：它识别的是**问法形式**，不是"这题有没有结论方向"。
    形如"用 X 做 Y 好不好"会被识别；而"X 和 Y 哪个更合适"这类没有
    正反结构、也不含上列短语的选择题不会被识别，仍然要靠出题时自觉。

    豁免必须逐条写在 `exempt` 里并说明理由，改动时会被这条测试逼着重新
    解释：
      - Q26：""为什么"型问题，事实性关键点本身就承载方向；
      - Q34：问的是"怎么确认"，"是不是内存超限"只是宾语从句；
      - Q14：问的是"为什么会出现超卖"，"判断是否还有余量"是题干里
        描述的那段错误代码在做什么，不是这题的结论方向。
    后两条都属于中文字面误伤：句子里出现了正反结构，但它不在问句的
    主干上。
    """
    exempt = {
        "已经建了包含查询所有列的覆盖索引",
        "容器隔一阵就没了、应用日志里什么都没留下",
        "秒杀场景下，先 GET 库存判断是否还有余量",
    }
    for q in _QUESTIONS_YAML:
        if any(q["q"].startswith(e) for e in exempt):
            continue
        if _is_yes_no_question(q["q"]):
            assert q.get("forbid_patterns"), f"有结论方向的题必须配负向约束: {q['q'][:40]}"


def test_cr082_detector_recognizes_the_phrasings_it_claims_to():
    """判别性：CR-082 的根因是判定方式识别不了"能不能"。这条把它钉住——
    也顺带记录这条判据**不**声称能识别的形状。
    """
    for text in ("能不能用它实现死信队列", "这样配会不会更慢", "是不是所有异常都重试",
                 "需不需要再做幂等", "这样算不算超卖", "值不值得加索引",
                 "这么写行不行", "有没有更好的做法", "能否保证恰好一次",
                 "是否需要重建索引", "副本数会跟着回滚吗"):
        assert _is_yes_no_question(text), f"应识别为是非题: {text}"
    # 明确不声称能识别的：没有正反结构、也不含上列短语的选择/开放题。
    for text in ("Kafka 和 RabbitMQ 哪个更适合订单场景", "为什么执行计划走了顺序扫描"):
        assert not _is_yes_no_question(text), f"不应被识别（判据只看问法形式）: {text}"


@pytest.mark.parametrize("prefix", sorted(_CR066_CORRECT_ANSWERS))
def test_cr066_new_rules_still_pass_a_correct_answer(prefix):
    """反向判别性：按卡片正文写的正确答案在新判据下必须仍然 `passed`，
    否则说明新规则是"只会变红"的坏判据（`AGENTS.md` §5.2 的对偶情形）。
    """
    assert _status_of(_spec(prefix), _CR066_CORRECT_ANSWERS[prefix]) == "passed"


# ---- R71（2026-09-09）：负向约束的判别性 ----
#
# `test_cr067_every_yes_no_question_has_a_negative_constraint` 本轮从
# `[-8:]` 扩到全量后，前三张卡片的 Q4/Q5/Q6/Q7/Q10/Q17 补齐了负向约束，
# 第五张卡片的 Q28/Q31/Q33/Q35 随卡片一起写。这一组测试成对固定两件事：
#   1. 每条负向约束都能命中它自己那句"结论反过来"的构造反例；
#   2. 它不会命中同一题**真实跑出来的**那次回答（逐字取自
#      bench/reports/eval-scenarios-20260909T220321Z.json，未删改）。
# 第 2 点是必须的：CR-054~060 的整段教训就是负向正则最容易误伤正确的
# 否定表述，只测"能抓反例"会重演那个坑。

_R71_NEGATIVE_CONSTRAINT_CASES = {
    "Q4": (
        'Outbox 表的结构有什么讲究',
        'Outbox 表随便加字段都行，SMT 会自动忽略多余的列 [1]。',
        'Outbox 表必须符合 Debezium Outbox Event Router 的预期结构，字段需支持 JSON 字段的嵌套与扩展，且字段命名需与 SMT 配置中的 `predicates.IsOutboxTable.pattern` 一致 [1]。  \n若使用 MongoDB 连接器，不可使用通用 Outbox Event Router，应改用 MongoDB 特有的 Outbox Event Router SMT [4]。  \n字段类型需支持 JSON 内容的扩展，如 `partition` 字段用于分区，但其值不能为别名，否则无效 [2]。  \n\n**关键前提**：Outbox 表结构必须与 `predicates.IsOutboxTable.pattern=outbox.event.*` 匹配，且字段需能被 SMT 正确解析，否则消息无法路由 [1][4]。  \n**风险**：若字段类型不兼容（如非 JSON 或无足够字段空间），可能导致消息丢失或解析失败 [5]。',
        '把「结构有约束」反转成「随便加都行」',
    ),
    "Q5": (
        '用了 Outbox 模式之后',
        '用了 Outbox 之后消费者不需要再做幂等，重复消息已经在写入侧被消除 [1]。',
        '是，消费者仍需处理重复消息 [1]。  \n需基于消息的唯一键（如 outbox row id）实现幂等处理 [1][3]。  \n若使用 JSON 转换器，需确保 payload 中包含可识别的唯一标识 [2]。',
        '把「仍需幂等」反转成「不需要幂等」',
    ),
    "Q6": (
        'Outbox 模式解决的是消息发布的哪一半问题',
        'Outbox 同时也解决了消费端的重复消费问题，两者加起来就等于恰好一次 [1]。',
        'Outbox 模式解决的是消息发布写入侧的可靠性，确保已提交的业务变更最终会生成事件并进入 Kafka [1]。Kafka 自身的投递保证（如持久化副本存活）解决的是消费侧的事件可见性与不丢失问题，但不保证“恰好一次”交付 [2]。两者分别解决发布与消费两端的可靠性，互不替代 [1]。  \n\n关键前提：Outbox 仅保证事件被发布，不保证消费端只消费一次；若需“恰好一次”消费，需额外设计消费者幂等性或状态管理 [1]。',
        '把「各解决一半」反转成「加起来就是 exactly once」',
    ),
    "Q7": (
        '如果 Outbox 的转发进程挂了一段时间',
        '转发进程挂了就会丢数据，因为事件只在内存里排队 [1]。',
        '不会丢，因为 Outbox Event Router 仅转发事件到指定 topic（如 `outbox.event.*`），原始数据已由 Debezium Connector 完整捕获并持久化到源数据库，转发进程挂掉不会导致数据丢失 [1][3]。  \n若转发进程挂起，可配置 `transforms.outbox.predicate=IsOutboxTable` 保证仅转发匹配的事件 [1]。  \n若需恢复，应确保 Kafka 消费者或下游系统具备重试机制，避免因短暂中断导致数据丢失 [2]。  \n\n关键前提：Debezium Connector 已启用持久化（如 binlog 或 WAL），且未关闭 `enable.auto.commit` 或 `commit.offsets` 配置。  \n风险：若下游系统未消费或处理，事件可能堆积，导致延迟或重复处理。',
        '把「不会丢」反转成「会丢」',
    ),
    "Q10": (
        '在 Spring Kafka 的 DefaultErrorHandler 里',
        '在 DefaultErrorHandler 里所有异常都会被重试，直到达到次数上限 [1]。',
        '不是，所有异常类型都不应被重试 [2]。  \n`DefaultErrorHandler` 默认跳过 `Error` 类型异常（如 `DeserializationException`、`ClassCastException` 等）且不重试 [2]。  \n若需控制重试行为，应通过 `addNotRetryableExceptions()` 显式配置 [2]。  \n\n关键前提：`Error` 异常直接终止消费者线程，无重试、无死信投递，可能导致消息丢失且应用看似健康 [2]。',
        '把「不是所有异常都该重试」反转成「所有异常都重试」',
    ),
    "Q17": (
        '每次执行 Redis 的 Lua 脚本',
        '每次执行都要把脚本内容重新发送给服务器，所以并发高时会明显变慢 [1]。',
        '不会，每次执行 Lua 脚本时不会重新发送脚本内容。Spring Data Redis 通过缓存脚本的 SHA1 哈希值，仅在首次执行时编译脚本并缓存其哈希，后续调用直接使用缓存的哈希值执行 [2]。  \n若配置了 `DefaultRedisScript` 单例，可避免每次运行时重复计算脚本的 SHA1，提升性能 [2]。  \n风险：若未正确缓存，可能导致脚本每次被重新编译，造成网络往返开销和延迟，尤其在高并发场景下 [1]。  \n\n建议配置：  \n```java\n@Bean\npublic RedisScript<Boolean> checkAndSetScript() {\n    return new DefaultRedisScript<>("checkandset.lua", RedisScript.Boolean.class);\n}\n```  \n前提：脚本内容在应用启动时已加载，且 Redis 服务版本支持 Lua 脚本执行（如 Redis 6+）[1]。',
        '把「靠 SHA1 缓存、不必重发」反转成「每次重发全文」',
    ),
    "Q28": (
        '发布卡了十几分钟一直没推进',
        '超过 progressDeadlineSeconds 之后，Kubernetes 会自动回滚到上一个版本 [1]。',
        '不会，Kubernetes 不会主动回滚到上一个版本 [2]。若 rollout 无进展，需手动检查 `Progressing` 状态为 `False` 后执行回滚 [2]。  \n使用 `kubectl rollout undo deployment/<name> --to-revision=N` 回退到指定版本 [3]。  \n风险：若新版本 Pod 启动失败，回滚后服务可能仍不可用，需确认旧版本健康 [3]。',
        '把「只报状态、不回滚」反转成「会自动回滚」',
    ),
    "Q31": (
        '容器压根没配 readiness probe',
        '容器没有配置 readiness probe 时，探针检查一直不通过，Pod 会一直不就绪 [1]。',
        '不会，容器未配置 readiness probe 时，kubelet 会默认认为其处于就绪状态，不会阻断流量 [2]。  \n若容器未配置 readiness probe，初始阶段结果为 `Failure`，但后续无探针则不会持续标记为不就绪 [2]。  \n关键前提：若容器未配置 readiness probe，kubelet 会将其视为就绪，Pod 会持续接收流量，除非配置了其他机制（如 liveness 或 startup）影响状态 [4]。',
        '把「没配就按 Success」反转成「没配就一直不就绪」',
    ),
    "Q33": (
        'Pod 状态是 CrashLoopBackOff',
        'Pod 处于 CrashLoopBackOff 说明应用代码有 bug，改代码即可 [1]。',
        '不能，CrashLoopBackOff 仅表明 Pod 启动失败且持续重启，具体原因需结合日志和容器退出码判断 [1]。  \n若 Pod 持续处于该状态，应检查容器日志（如 `kubectl logs <pod> --previous`）和资源限制（如内存溢出）[4]。  \n若使用 Pod Disruption Budget（PDB），需确认是否启用 `AlwaysAllow` 策略导致误驱逐 [2]。',
        '把「状态只是退避」反转成「状态就等于应用有 bug」',
    ),
    # Q35 的"反例"不是构造的：本轮本地后端真的答成了"是，会把副本数
    # 恢复到上一版本"，逐字取自同一份报告；"不该命中"的那一侧才是构造的
    # 正确答案。第一版负向约束写成"恢复**回**"，恰好打不中这句真实反转
    # （实际是"恢复**到**"），已按 AGENTS.md §5.3 收紧——一条抓不住自己
    # 动机例句的负向约束等于没有。
    "Q35": (
        "上一次发布顺手把副本数从 3 调到 10",
        '是，执行 `kubectl rollout undo` 会将副本数恢复到上一版本的配置值（即 3）[1][3][5]。  \n若需指定具体版本（如回退到 revision=1），使用 `--to-revision=1` [4]。  \n**前提**：该 Deployment 的 `spec.revisionHistoryLimit` 需大于 0，否则无历史可回退 [4]。',
        "不会。rollout undo 只回滚 .spec.template，副本数不受影响；"
        "扩缩容不会产生新的 revision [1]。",
        "把「只回滚 Pod 模板」反转成「副本数也一起回滚」",
    ),
}


# ---- R72（2026-09-09，CR-082）：三道"能不能"型是非题的负向约束 ----
#
# 与上面 R71 那组同样成对固定"能抓反例"和"不误伤真实回答"。Q8 的反例
# 不是构造的：本地后端真的答成了"可实现端到端恰好一次"，逐字取自同一
# 份报告；它"不该命中"的那一侧才是构造的正确答案。
_R72_NEGATIVE_CONSTRAINT_CASES = {
    "Q8": (
        "Kafka Streams 或事务型生产者能不能让 Outbox",
        'Kafka Streams 和事务型生产者可实现 Outbox 事件端到端恰好一次处理 [1][2]。  \n需在消费者使用 `read-committed` 隔离级别，并启用事务性生产者（`transactional.id` 配置）[1]。  \n若使用非事务性生产者或默认 `read_uncommitted`，将导致重复处理或丢失 [2]。  \n\n关键前提：必须启用事务性生产者（`transactional.id` 配置）且消费者使用 `read-committed` 隔离级别，否则无法保证端到端恰好一次 [1][2]。',
        "不能。Kafka 的事务只覆盖到写入侧，端到端不保证恰好一次，"
        "消费端仍需按业务键幂等 [1]。",
        "把「端到端做不到 exactly once」反转成「可实现」（真实答错的一次生成）",
    ),
    "Q13": (
        "批量消费的 Kafka Listener 能不能用",
        "可以。批量监听器同样能用 @RetryableTopic 实现死信队列 [1]。",
        '不能。`@RetryableTopic` 不支持批量监听器（batch listener）的死信队列功能 [1][2]。需使用 `DefaultErrorHandler` 配合 `DeadLetterPublishingRecoverer` 实现 [2]。  \n配置方式：  \n```java\n@Bean\npublic DefaultErrorHandler defaultErrorHandler() {\n    return new DefaultErrorHandler(\n        new DeadLetterPublishingRecoverer(kafkaTemplate)\n    );\n}\n```  \n关键前提：批量监听器的重试在原消费者线程内进行，无法解耦，存在阻塞风险 [2]。',
        "把「不支持批量监听器」反转成「可以用」",
    ),
    "Q15": (
        "用 Spring 的 @Transactional 把 Redis",
        "能。用 @Transactional 把 GET 和 DECR 包在一个事务里就能解决超卖 [1]。",
        '不能。Redis 事务（如 Spring Data Redis 的 `checkandset.lua` Lua 脚本）依赖 Redis 服务器层面的原子性，而 Spring 的 `@Transactional` 仅对 JDBC 或 JPA 等数据库操作提供事务控制，无法跨 Redis 命令实现读-写原子性 [1][2]。  \n\n正确做法是使用 Redis Lua 脚本将 `GET` 和 `DECR` 包装为单个原子操作 [2]。  \n若使用 Spring Data Redis，应配置 `valueOperations` 通过 `RedisTemplate` 执行 Lua 脚本，例如：  \n```java\nString lua = "if (redis.call(\'GET\', KEYS[1]) >= ARGV[1]) then return redis.call(\'DECRBY\', KEYS[1], ARGV[1]) else return 0 end";\nObject result = redisTemplate.execute(new DefaultRedisScript<>(lua, Long.class, "stock:product:123", "10"));\n```  \n关键前提：Redis 节点必须支持 Lua 脚本执行，且无网络延迟或重试机制干扰 [1][2]。  \n\n风险：若 Lua 脚本执行失败或被中断，可能造成库存不一致，需配合幂等和重试策略 [3][4]。',
        "把「解决不了」反转成「就能解决超卖」",
    ),
}

# ---- R78（2026-09-10）：第六张卡片（订单/对账）5 道是非/选择题的负向约束 ----
#
# 与前两批同样的两侧判别性：既要命中结论反转，又不得误伤这题真实跑出来的
# 那次回答。本批里 Q38 和 Q40 的第一版都是被"不得误伤真实回答"这一侧当场
# 抓出来改掉的（详见 scenario_questions.yaml 里各自的校准注释）：
#   - Q38 第一版被真实答案"事务将不会回滚"误伤——`(?<!不)` 只挡住了紧邻
#     "将"之前的否定，否定词落在"将"和"回滚"中间的间隔里照样匹配；
#   - Q40 第一版**一条都没打中**它的动机例句——那次真实回答给出的正是
#     反转结论（"则事务会生效"），却因为词序（"事务会生效"而非
#     "生效…事务"）和距离而漏掉。
# 因此 Q40 这条用真实答错的那次生成当反转例、另写一份按卡片正文的正确
# 回答当"不得误伤"的样本（写法沿用 R72 的 Q8/Q13/Q15）。
_R78_NEGATIVE_CONSTRAINT_CASES = {
    "Q38": (
        "在 @Transactional 方法里把异常 catch 住",
        "把异常 catch 住只记日志之后，事务仍然会自动回滚 [1]。",
        "不会回滚 [2]。若在 @Transactional 方法中捕获异常后仅记录日志，未调用 "
        "`status.setRollbackOnly()`，事务将不会回滚 [2]。  \n需确保在异常处理中显式调用 "
        "`status.setRollbackOnly()` 才能触发回滚 [1][3]。",
        "把「catch 住就不回滚」反转成「仍然会回滚」",
    ),
    "Q40": (
        "外层 placeOrder 已经标了 @Transactional",
        "会跑在新事务里 [3]。  \n配置 `@Transactional(REQUIRES_NEW)` 时，会创建独立于外层"
        "事务的全新事务 [3]。  \n若外层事务已提交或回滚，内层事务不受影响，需确保事务"
        "管理器支持跨事务传播 [5]。",
        "不会。同一个类里的自调用不经过代理，`writeLedger` 上的 `@Transactional(REQUIRES_NEW)` "
        "不生效，不会新开事务 [1]。这次写操作仍然跑在 `placeOrder` 开的那个外层事务里，"
        "跟着它一起提交或回滚，它自己声明的传播行为、隔离级别都不作数 [1]。要真拿到独立"
        "事务，得把 `writeLedger` 挪到另一个 bean，或改用 AspectJ 模式 [1]。",
        "把「不会新开事务、仍在外层事务里」反转成「会新开一个事务」（CR-089 改题后真实答错的一次生成）",
    ),
    "Q42": (
        "对账方法上标了 readOnly",
        "会生效。方法上标的隔离级别会按注解生效，Spring 会为它单独开一个事务 [1]。",
        "readOnly和隔离级别设置在被上层事务调用时**不生效**：  \n- Spring事务配置中"
        "`read-only`和`isolation`仅对`REQUIRED`或`REQUIRES_NEW`传播生效，且仅作用于"
        "当前事务边界 [2]；  \n- 若上层事务已管理事务上下文，底层方法的"
        "`@Transactional(readOnly = true, isolation = ...)`将被覆盖或忽略 [2][4]。",
        "把「参与外层事务时被静默忽略」反转成「按注解生效」",
    ),
    "Q44": (
        "REQUIRES_NEW 和 NESTED 有什么不同",
        "NESTED 的内层提交能在外层回滚之后独立保留下来，所以审计记录应该用 NESTED [1]。",
        "`REQUIRES_NEW` 创建独立事务，内层提交可存活于外层回滚；`NESTED` 依赖单事务"
        "多保存点，内层失败可回滚但不持久 [1][2]。审计记录需在主事务回滚后留存，"
        "应选 `REQUIRES_NEW` [1]。",
        "把「NESTED 是保存点、不独立提交」反转成「能独立留下来」",
    ),
    "Q46": (
        "对账任务在一个事务里先统计订单再统计支付",
        "在同一个事务里两次查询一定一致，默认隔离级别下就能保证 [1]。",
        "否，两次查询看到的数据不一致 [3][4]。  \n在默认隔离级别（如 Read Committed）下，"
        "统计支付时可能读取到已提交但未同步的订单变更 [3]。",
        "把「默认级别下两次查询可能不同」反转成「一定一致」",
    ),
}

# ---- R79（CR-089）：把"内层注解不生效"与"完全没有事务"分开 ----
#
# CR-089 指出卡片标题与原 Q40 把两种局面合并成一个结论。改题之后这一对题
# 才真正可判别：**同一个把两者混为一谈的答案，必然在这两题里至少错一题**。
# 因此除了常规的两侧判别性，下面另有一条专门的交叉用例。
_R79_NEGATIVE_CONSTRAINT_CASES = {
    # CR-091 改题后，这一条的两个样本都换成了新问法下的说法：反转是"代理会
    # 拦截并为它开启新事务"，"不得误伤"的那句只说到注解本身。
    "Q41": (
        "外层 placeOrder 没标",
        "会建立。`writeLedger` 上标了 `@Transactional`，代理会拦截并为它开启一个新事务 [1]。",
        "不会建立。代理只拦截经过代理进来的外部调用，同一个类里的自调用不生效 [1]。",
        "把「自调用注解不建立它声明的边界」反转成「代理会拦截并开启新事务」",
    ),
}

# 这段话是 CR-089 描述的那种答案：前半句（代理只拦外部调用、自调用不生效）
# 完全正确，后半句把它推成"完全没有事务"。**改题之前它在 Q40 上是 passed**
# ——本轮用 `git show HEAD:knowledge/eval/scenario_questions.yaml` 取改题前的
# 判据实跑过，关键点 2/2、负向约束 0 命中（不手抄历史判据，避免再造一份
# 与实现脱钩的复刻件，那是 R76/R77 已经登记过的问题）。
_CR089_CONFLATED_ANSWER = (
    "不生效。代理只拦截经过代理进来的外部调用，同一个类里的自调用不生效 [1]，"
    "所以 `writeLedger` 里的写操作完全没有事务，出错也不会回滚 [1]。"
)


def test_cr089_conflating_no_inner_boundary_with_no_transaction_is_caught():
    """CR-089 的核心判别性：一个把"内层注解不生效"直接推成"完全没有事务"的
    答案，在**外层已有事务**那道题上必须被负向约束标出。

    这正是改题之前判据抓不到的那种错误——旧 Q40 只问"生不生效"，这段话
    回答得"对"，却会让读者以为订单写入根本没有被任何事务保护。
    """
    spec = _spec("外层 placeOrder 已经标了 @Transactional")
    hits = [p for p in spec["forbid_patterns"] if re.search(p, _CR089_CONFLATED_ANSWER)]
    assert hits, "把「无内层边界」说成「完全无事务」必须被标出"
    assert _status_of(spec, _CR089_CONFLATED_ANSWER) != "passed"


def test_cr091_the_conflated_sentence_is_not_a_free_pass_on_the_other_question_either():
    """**这条是 R79 那条断言的反面，改动理由记在这里。**

    R79 写的是"同一段话在外层没有事务那道题上是正确答案，不得被标出"，
    并断言它 `passed`。CR-091 指出那个期望本身就是外推：Spring 的自调用规则
    只说**被自调用的那个注解**不会建立它声明的边界，`writeLedger()` 随后调用
    另一个 bean（例如 repository）时，那次调用经过的是**那个 bean 的代理**，
    下游照样可以在那里建立事务边界，程序式事务（`TransactionTemplate`）同理。
    所以"这次写操作完全没有事务"在**两道题上都**是没有依据的说法。

    CR-089 要分开的那两种局面仍然分得开——靠的是正向关键点（Q40 要求说出
    "仍在外层事务里"），不再靠"完全没有事务"这半句在某一题上被判对。
    """
    for prefix in ("外层 placeOrder 已经标了 @Transactional", "外层 placeOrder 没标"):
        spec = _spec(prefix)
        hits = [p for p in spec["forbid_patterns"] if re.search(p, _CR089_CONFLATED_ANSWER)]
        assert hits, f"「完全没有事务」在「{prefix[:12]}…」这题上也必须被标出"
        assert _status_of(spec, _CR089_CONFLATED_ANSWER) != "passed"


# CR-091 点名要求必须**通过**的那种回答：既说明自调用注解不生效，又说明下游
# 仍可能另建事务。它是这条收窄的反向判别性——判据收窄之后不能变成"只会变红"。
_CR091_FAITHFUL_ANSWER = (
    "不会。同一个类里的自调用不经过代理，`writeLedger` 上的 `@Transactional` 不会建立"
    "它声明的事务边界 [1]。但这只说明这一次调用没有开启事务：如果 `writeLedger` 接着"
    "去调用别的 bean（比如一个 repository），那次调用会经过那个 bean 的代理，下游仍然"
    "可能在那里开自己的事务；用 `TransactionTemplate` 也可以 [1]。"
)

# 只说到注解本身、不往下游推的回答，同样必须通过。
_CR091_MINIMAL_ANSWER = (
    "不会建立。代理只拦截经过代理进来的外部调用，同一个类里的自调用不生效 [1]。"
)


@pytest.mark.parametrize("answer", [_CR091_FAITHFUL_ANSWER, _CR091_MINIMAL_ANSWER])
def test_cr091_answers_that_stop_at_the_annotation_still_pass(answer):
    spec = _spec("外层 placeOrder 没标")
    hits = [p for p in spec["forbid_patterns"] if re.search(p, answer)]
    assert not hits, f"停在证据边界上的回答被误伤：{hits}"
    assert _status_of(spec, answer) == "passed"


def test_cr091_extending_to_the_whole_call_chain_is_flagged():
    """判别性：把"这个注解不建边界"推广成"整条调用链都没有事务、没人回滚"
    必须被标出。**改题前这句话是 `passed`**——本轮用
    `git show HEAD:knowledge/eval/scenario_questions.yaml` 取旧判据实跑确认。
    """
    answer = ("不会。自调用不经过代理，注解不生效 [1]，所以整条调用链都没有事务，"
              "没有任何东西会回滚它 [1]。")
    spec = _spec("外层 placeOrder 没标")
    assert [p for p in spec["forbid_patterns"] if re.search(p, answer)]
    assert _status_of(spec, answer) != "passed"


# ---- R80（CR-090）：另一侧的过度外推——"没有事务" ≠ "写库必然已落盘" ----
#
# CR-089 收窄标题之后，卡片仍然断言"外层没有事务时，写操作会自行提交"。
# 审查方指出这同样没有证据：被引用的那一页只说明自调用不会建立内层注解声明的
# 事务边界，管不着数据访问 API 在无事务时的行为——transaction-scoped 的 JPA
# `EntityManager` 在没有活动事务时 `persist`/`merge` 要抛
# `TransactionRequiredException`（Jakarta Persistence 3.1 §3.3）。本机八个来源的
# 语料里 `TransactionRequiredException` 与 `transaction-scoped` **各 0 块**，
# 两种结果都拿不出可引用的出处，因此卡片改成不承诺具体持久化结果。
#
# 判据这一侧的分工要说清楚：这条负向约束**不判它必错**（纯 JDBC 自动提交下
# "会提交"可能是对的），判的是"这份证据支撑不了这个断言"，因此走人工复核通道。
_CR090_OVERREACHING_ANSWERS = {
    "会自动提交": "保不住。自调用不经过代理，注解不生效，所以这次写库会自动提交，"
                  "出错也不会回滚 [1]。",
    "会直接落库": "不能保住。没有事务，写入会直接落库，无法回滚 [1]。",
}

# 两句都按卡片改后的正文写，都必须**不被**任何负向约束命中。第二句是这次
# 重新校准的直接动机：它里面的"没有任何东西会回滚它"曾被 R79 那条裸的
# "(会|仍然|…)…回滚" 误伤——定长逆序断言挡不住出现在句子更早处的"没有"。
# CR-091 之后这两句**都被换掉了**，原因值得记：R80 写的"忠实回答"是
# "所以根本没有事务被开启""这条调用链里没有任何东西会回滚它"——按 CR-091，
# 那两句同样是外推（下游的 repository 代理或 `TransactionTemplate` 都可能
# 另建事务）。也就是说上一轮我用来证明"没有误伤"的样本，本身就是下一条
# 意见要抓的东西。现在的样本只说到注解本身与代理机制。
_CR090_FAITHFUL_ANSWERS = (
    "不会建立。`placeOrder` 自己没标 `@Transactional`，同一个类里的自调用不经过代理，"
    "`writeLedger` 上的注解不生效 [1]。",
    "不会。代理只拦截经过代理进来的外部调用，所以这个注解不会建立它声明的事务边界 [1]；"
    "至于后面还会不会有事务，得看 `writeLedger` 接下来调用了什么 [1]。",
)


@pytest.mark.parametrize("key", sorted(_CR090_OVERREACHING_ANSWERS))
def test_cr090_claiming_the_write_just_commits_is_flagged(key):
    """判别性，**两句的旧行为不一样，如实分开写**（用
    `git show HEAD:knowledge/eval/scenario_questions.yaml` 取旧判据实跑，不手抄）：

    - "会自动提交"那句在旧判据下**关键点 2/2、负向约束 0 命中，判定 passed**
      ——这是这条约束真正补上的洞；
    - "会直接落库"那句当时也被标出了，但命中的是讲**回滚**的那个分支，而且是
      靠"无法回滚"里的"无"从间隔类里漏过去的**误命中**，不是因为它断言了提交。
      也就是说这一句的旧"拦下"是巧合，不能算判别性证据。

    现在两句都必须因为"断言了证据支撑不了的持久化结果"而被标出、不得判过。
    """
    spec = _spec("外层 placeOrder 没标")
    answer = _CR090_OVERREACHING_ANSWERS[key]
    assert [p for p in spec["forbid_patterns"] if re.search(p, answer)], \
        f"「{key}」这种无证据的持久化断言必须进人工复核通道"
    assert _status_of(spec, answer) != "passed"


@pytest.mark.parametrize("answer", _CR090_FAITHFUL_ANSWERS)
def test_cr090_constraint_spares_answers_that_stop_where_the_evidence_stops(answer):
    """反向判别性：只说到证据支持的那一步（没有事务被开启、这条链路里没有
    任何东西回滚它）的答案不得被标出，否则这条约束就成了"只会变红"。
    """
    spec = _spec("外层 placeOrder 没标")
    hits = [p for p in spec["forbid_patterns"] if re.search(p, answer)]
    assert not hits, f"忠实于证据的回答被负向约束误伤：{hits}"
    assert _status_of(spec, answer) == "passed"


_NEGATIVE_CONSTRAINT_CASES = {**_R71_NEGATIVE_CONSTRAINT_CASES,
                              **_R72_NEGATIVE_CONSTRAINT_CASES,
                              **_R78_NEGATIVE_CONSTRAINT_CASES,
                              **_R79_NEGATIVE_CONSTRAINT_CASES}


@pytest.mark.parametrize("key", sorted(_NEGATIVE_CONSTRAINT_CASES))
def test_r71_negative_constraint_catches_the_reversed_conclusion(key):
    """判别性：每题的负向约束必须命中它对应的反转结论。"""
    prefix, inverted_answer, _real, why = _NEGATIVE_CONSTRAINT_CASES[key]
    spec = _spec(prefix)
    pats = spec.get("forbid_patterns", [])
    assert pats, f"{key} 应当已配负向约束"
    assert [p for p in pats if re.search(p, inverted_answer)], f"{key}（{why}）应被负向约束标出"


@pytest.mark.parametrize("key", sorted(_NEGATIVE_CONSTRAINT_CASES))
def test_r71_negative_constraint_spares_the_real_answer(key):
    """反向判别性：同一条约束不得命中这题真实跑出来的那次回答。"""
    prefix, _inverted, real_answer, _why = _NEGATIVE_CONSTRAINT_CASES[key]
    spec = _spec(prefix)
    hits = [p for p in spec.get("forbid_patterns", []) if re.search(p, real_answer)]
    assert not hits, f"{key} 的负向约束误伤了真实回答：{hits}"
