"""场景评测判据自身的回归（CR-053）。

为什么单独测：判据是场景评测结果的门禁，判据本身漏判会让"通过/不通过"
这个数字失去意义——就像 `test_probe_gate.py` 对排序探针门禁（CR-015）
做的事一样。CR-053 就是这么一个洞：`run_scenarios.py` 原先只看正向的
`expect_keypoints`，一条答案哪怕混进一句**已知错误的越界断言**，只要
凑巧命中了两条比较宽泛的正向关键点（"不能"/"无法…判断"），照样被记成
"关键点 2/2 全部命中"、判为通过。

具体案例（真实保存输出，非构造）：`bench/reports/
eval-scenarios-20260907T055103Z.json` 第 18 题，模型断言 `RedisAtomicLong`
"仅支持原子增减，无法实现读-改-写完整逻辑"（越界——CR-052 已指出公开
API 还有 `compareAndSet` 等方法，只是这些细节不在已入库语料里，卡片
不该断言"只能"）且"需使用 Lua 脚本"（越界的排他性结论）。

这里只测纯判定函数 `_score`，不加载模型/索引，因此毫秒级，可进快速门禁。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from packages.evaltools.run_scenarios import _score  # noqa: E402

# 逐字取自 bench/reports/eval-scenarios-20260907T055103Z.json 第 18 题的
# answer_text，未做任何删改。
_CR053_ANSWER = (
    "不能。RedisAtomicLong 仅支持原子增减，无法实现读-改-写完整逻辑，"
    "无法判断库存是否充足 [1]。需使用 Lua 脚本实现读取+判断+扣减的原子操作 [1]。  \n"
    "配置 `RedisTemplate` 时应使用 `DefaultRedisScript` 缓存 Lua 脚本 SHA1，"
    "避免每次调用重复传输 [2]。  \n"
    "风险：若未使用 Lua 脚本，多个请求并发时可能触发超卖"
    "（如库存为 1，两个请求同时读取为 1 后扣减） [1]。"
)

# 事发时 `knowledge/eval/scenario_questions.yaml` 里这道题的两条正向
# expect_keypoints（CR-053 之前的版本，内容未改）。
_POSITIVE_KEYPOINTS = [
    r"(?i)(不能|不行|不足以|does.{0,10}not|cannot)",
    r"(?i)(单个?命令|single command|(没有|无法|不(能|会)).{0,8}(条件|判断|检查|阈值)|no.{0,10}(condition|check))",
]

# CR-053 之后新增的 forbid_patterns。
_FORBID_PATTERNS = [
    r"(?i)(仅支持.{0,6}(原子)?(增减|递增|递减|increment)|only support.{0,10}(atomic )?increment(/decrement)?)",
    r"(?i)((必须|只能|唯一(的)?(方法|方式|办法)|only way).{0,10}(Lua|脚本|script))",
]


def test_old_positive_only_keypoints_wrongly_pass_the_cr053_answer():
    """判别性基线：旧逻辑（只看正向 keypoint，没有 forbid_patterns）
    确实会把这条含越界断言的真实答案判成 2/2、没有任何失败记录——
    这正是 CR-053 指出的洞，不是构造出来的假想场景。
    """
    hit, missed, forbidden, failures = _score(_CR053_ANSWER, _POSITIVE_KEYPOINTS, [])
    assert hit == _POSITIVE_KEYPOINTS
    assert not missed
    assert not forbidden
    assert not failures


def test_forbid_patterns_catch_the_cr053_overclaim():
    """新逻辑：正向关键点依旧全部命中（没有削弱既有判据），但
    forbid_patterns 命中"仅支持原子增减"这句越界断言，判定失败。
    """
    hit, missed, forbidden, failures = _score(
        _CR053_ANSWER, _POSITIVE_KEYPOINTS, _FORBID_PATTERNS
    )
    assert hit == _POSITIVE_KEYPOINTS
    assert not missed
    assert forbidden, "应命中至少一条 forbid_patterns"
    assert failures, "命中 forbid_patterns 必须记入 failures，不能被正向判据掩盖"


def test_forbid_patterns_do_not_reject_the_narrowed_correct_answer():
    """避免误伤：CR-052 收窄后卡片支持的正确答案（只说包装的是无条件命令、
    不含阈值检查，不断言"仅支持增减"或"必须用 Lua"）不应被 forbid_patterns
    误判。
    """
    correct_answer = (
        "不能直接防止超卖。它包装的是 Redis 的无条件自增/自减命令，"
        "这个命令本身没有阈值检查，无法判断扣减后是否还够，"
        "需要额外的条件逻辑才能补上这一步，比如用脚本把读取和判断放进同一次调用。"
    )
    hit, missed, forbidden, failures = _score(
        correct_answer, _POSITIVE_KEYPOINTS, _FORBID_PATTERNS
    )
    assert not forbidden, f"不应误判为越界断言，实际命中: {forbidden}"
    assert not failures


def test_forbidden_hit_fails_the_case_even_when_all_keypoints_matched():
    """端到端确认判据组合的净效果：不是"forbid_patterns 命中"和"正向
    keypoint 缺失"两件独立的事，而是任一发生都必须让整道题判失败——
    这正是 CR-053 要堵住的"正向全中就判过"的漏洞。
    """
    _, _, forbidden, failures = _score(_CR053_ANSWER, _POSITIVE_KEYPOINTS, _FORBID_PATTERNS)
    case_ok = not failures
    assert forbidden and not case_ok
