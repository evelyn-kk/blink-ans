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


@pytest.mark.parametrize("prefix", sorted(_CR066_CORRECT_ANSWERS))
def test_cr066_new_rules_still_pass_a_correct_answer(prefix):
    """反向判别性：按卡片正文写的正确答案在新判据下必须仍然 `passed`，
    否则说明新规则是"只会变红"的坏判据（`AGENTS.md` §5.2 的对偶情形）。
    """
    assert _status_of(_spec(prefix), _CR066_CORRECT_ANSWERS[prefix]) == "passed"
