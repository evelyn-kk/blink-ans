"""场景评测判据自身的回归（CR-053~060）。

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
- **CR-054~060**：`forbid_patterns` 上线后，又花了六轮（CR-054 触发词
  太窄漏判"需使用" → CR-055 加否定语境过滤又误伤"不是只能用 Lua" →
  CR-056 窗口跨命题泄漏 → CR-057 分句边界不认转折连词 → CR-057 "零容忍
  紧邻"矫枉过正 → CR-059 分句边界不认加合连词 → CR-060 单命中场景下
  加合连词接无关否定仍会泄漏）反复调整"命中之后要不要自动判断这次命中
  是否被否定"这条逻辑，每次修一个反例就暴露一个新反例。根因是这条
  启发式想用"这段前文有没有否定词"回答一个本质上是句法结构的问题，而
  中文/英文的连接词是一个开放集合，枚举永远追不完。

**CR-060 之后的决定**：不再尝试自动判断否定极性。`forbid_patterns`
命中只记入 `forbidden_hit`，**不再计入 `failures`、不影响 `ok`**——
交给人工判断这次命中到底是真实的越界断言还是已被正确否定的表述。这不
是放弃 CR-053 想要的东西（"已知错误论断不能被正向关键点掩盖而悄悄放
过"）：`forbidden_hit` 依然会被打印、写进报告，只是不再自动定性。

这里只测纯判定函数 `_score`，不加载模型/索引，因此毫秒级，可进快速
门禁。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from packages.evaltools.run_scenarios import _score  # noqa: E402

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
# CR-060 之后不再自动判断极性，这句话依然会被记入 forbidden_hit（因为
# 字面包含"只能…Lua"），这是本轮接受的代价，用测试如实固定下来，不假装
# 它会被自动放过。
_CLEANLY_NEGATED_ANSWER = "不能直接防止超卖，但这不是只能用 Lua 脚本才能解决，也可以采用其他机制。"


def test_forbid_pattern_hit_is_recorded_but_does_not_fail_the_case():
    """CR-060 之后的核心行为：命中 forbid_patterns 记入 forbidden_hit，
    但不再出现在 failures 里——即使正向关键点全部命中，命中禁止模式也
    不会让这道题被判定为失败。
    """
    hit, missed, forbidden, failures = _score(_CR053_ANSWER, _POSITIVE_KEYPOINTS, _FORBID_PATTERNS)
    assert hit == _POSITIVE_KEYPOINTS
    assert not missed
    assert forbidden, "真实存在的越界断言仍应被记入 forbidden_hit，供人工复核"
    assert not failures, "forbid_patterns 命中不应再自动计入 failures（CR-060 之后的设计）"


def test_forbid_pattern_hit_is_visible_without_forbid_patterns_configured():
    """对照组：不传 forbid_patterns 时，同一条含越界断言的答案完全没有
    任何痕迹（这正是 CR-053 最初指出的洞——只是现在的修法不再是"自动判
    失败"，而是"至少要能看见"）。
    """
    hit, missed, forbidden, failures = _score(_CR053_ANSWER, _POSITIVE_KEYPOINTS, [])
    assert hit == _POSITIVE_KEYPOINTS
    assert not missed
    assert not forbidden
    assert not failures


def test_missing_keypoints_still_fail_the_case_independent_of_forbid_patterns():
    """确认 CR-060 的改动没有连带削弱正向关键点判据——缺关键点依然记入
    failures，这条判据完全不受 forbid_patterns 语义变化的影响。
    """
    answer = "这段回答完全跑题，什么都没提到。"
    hit, missed, forbidden, failures = _score(answer, _POSITIVE_KEYPOINTS, _FORBID_PATTERNS)
    assert not hit
    assert missed == _POSITIVE_KEYPOINTS
    assert not forbidden
    assert failures and all("未命中关键点" in f for f in failures)


def test_all_historical_negation_repros_are_now_flagged_for_review():
    """CR-054~060 六轮反例，不管否定语境是否成立，字面上都包含真实的
    forbid_patterns 触发词组——CR-060 之后不再尝试自动判断极性，这六句
    都应该被记入 forbidden_hit（供人工复核），不会再有任何一句像
    CR-055/056/057/059/060 那样被自动逻辑错误放行、完全没有痕迹。
    """
    for label, answer in _HISTORICAL_REPROS.items():
        _, _, forbidden, failures = _score(answer, [], _FORBID_PATTERNS)
        assert forbidden, f"{label}: 应至少命中一条 forbid_patterns 供人工复核"
        assert not failures, f"{label}: forbid_patterns 命中不应自动计入 failures"


def test_cleanly_negated_answer_is_still_flagged_not_silently_dropped():
    """如实固定 CR-060 之后接受的代价：一句明确否定排他性主张、且没有
    独立真实违规的正确表述，字面依然包含触发词组，依然会被记入
    forbidden_hit——这不是 bug，是"不再自动判断极性"这个设计决定的
    直接后果，用测试防止未来有人"顺手"把这种情况悄悄改回自动放行
    （那样就是在重新引入 CR-054~060 已经证明治不好的那套机制）。
    """
    _, _, forbidden, failures = _score(_CLEANLY_NEGATED_ANSWER, [], _FORBID_PATTERNS)
    assert forbidden, "即使语境上是正确的否定表述，字面命中依然应记入 forbidden_hit"
    assert not failures, "但依然不应自动计入 failures——由人工复核决定这次命中是否真的有问题"
