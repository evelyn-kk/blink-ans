"""场景评测判据自身的回归（CR-053、CR-054）。

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

CR-054 是 CR-053 修复上线后 R50 复审又发现的两个同类问题：
1. Lua 排他性 forbid pattern 当时只认"必须/只能/唯一/only way"，不认
   "需/需要"——本轮真实重新校准的输出恰好用的就是"需使用 Lua 脚本"这个
   说法，漏判了。
2. 同一条 forbid pattern 不识别否定语境——"不是只能用 Lua，也可以采用
   其他机制"是明确推翻排他性主张的**正确**表述，字面却包含"只能…Lua"
   这个触发词组，裸正则会把它也判成命中，误伤正确答案。

CR-055 是 CR-054 修复上线后 R51 复审又发现的问题：`_is_negated()` 原来
直接量前 15 个字符找否定标记，不管中间是否跨了标点分句——"不能；没有
条件检查，但**不只是**先读再写，仍然**只能用 Lua 脚本**。"里，"不只是"
否定的是前一分句"先读再写"这个完全不同的命题，却落在"只能用 Lua 脚本"
前 15 字符窗口内，被误当成后者的否定标记，放过了一条真实的排他性断言。
当时的修复：否定标记表去掉"不只/不仅/不止/not only"，并按标点/分句
边界截断否定标记的搜索范围，只在触发词所在的同一分句内找否定标记。

CR-056 是 CR-055 修复上线后 R52 复审又发现的问题：按标点分句边界截断
治标不治本——"但是/不过"这类转折连词并不产生标点分句边界，"并非只能
用 Lua **但是**仍然只能用 Lua 脚本。"里，"并非只能用 Lua"和"但是仍然
只能用 Lua 脚本"因为中间没有标点，仍被判定成同一个分句，前一命题的
"并非"照样泄漏给了后一个独立的真实排他性断言。当时判断"分句边界"这个
方向在打不完的补丁，换成了更严格的判定：否定标记必须**紧邻**触发词
本身（`str.endswith`，只留几个字符的宽松余量）。

CR-057 是 CR-056 修复上线后 R53 复审又发现的问题：CR-056 的"紧邻"矫枉
过正——"并不一定只能用 Lua 脚本"（否定词"并不一定"和触发词"只能"之间
没有隔别的命题，只是隔着"一定"这个自然修饰语）、英文
"definitely not the only way to use Lua"（"not"和"only way"之间隔着
"the"）这类**明确的否定语境**，同样会被"零容忍紧邻"拒之门外，误判成
真实违规。真正需要的不是"零窗口"也不是"无限窗口"：把 CR-055 的"按分句
边界截断"和 CR-056 指出的"边界要认得出转折连词，不能只认标点"一起用，
边界内再留一个不大的窗口容纳"一定/the"这类修饰语，否定标记词表也要
覆盖"不一定/并不一定"这类更松散的否定构式。

这里只测纯判定函数 `_score`（及其内部用到的 `_is_negated`），不加载
模型/索引，因此毫秒级，可进快速门禁。
"""

from __future__ import annotations

import re
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

# CR-053 上线时、CR-054 修复前的 forbid_patterns 快照——Lua 排他性那条
# 只认"必须/只能/唯一/only way"，是 CR-054 指出的漏判来源。仅用于下面
# 的判别性基线测试，不代表当前生产配置。
_FORBID_PATTERNS_PRE_CR054 = [
    r"(?i)(仅支持.{0,6}(原子)?(增减|递增|递减|increment)|only support.{0,10}(atomic )?increment(/decrement)?)",
    r"(?i)((必须|只能|唯一(的)?(方法|方式|办法)|only way).{0,10}(Lua|脚本|script))",
]

# CR-054 修复后、`knowledge/eval/scenario_questions.yaml` 里第 18 题
# 当前实际使用的 forbid_patterns。
_FORBID_PATTERNS = [
    r"(?i)(仅支持.{0,6}(原子)?(增减|递增|递减|increment)|only support.{0,10}(atomic )?increment(/decrement)?)",
    r"(?i)((必须|只能|需要?|唯一(的)?(方法|方式|办法)|only way|must|need).{0,10}(Lua|脚本|script))",
]

# CR-054 案例一：孤立出"需使用 Lua 脚本"这句，不掺杂 CR-053 那句"仅支持
# 原子增减"——用来单独验证 Lua 排他性这条 forbid pattern 本身的判别力，
# 不依赖另一条 pattern 顺带兜底。取自 R50 复审给出的具体构造例句。
_CR054_ANSWER_NEEDS_LUA = "不能。无条件递减没有库存阈值检查，需使用 Lua 脚本。"

# CR-054 案例二：明确推翻"只能用 Lua"这个排他性主张的正确表述——字面
# 包含"只能…Lua"这个触发词组，但前面紧跟"不是"这个否定标记。
_CR054_ANSWER_NEGATED_EXCLUSIVITY = (
    "不能直接防止超卖，因为无条件递减没有阈值检查。"
    "但这不是只能用 Lua 脚本才能解决，也可以采用其他机制来补上这个条件判断。"
)

# CR-055 复现句（取自 codex R51 复审给出的具体构造例句）：前一分句"不只是
# 先读再写"否定的是一个完全不同的命题，不应该豁免后一分句"只能用 Lua
# 脚本"这句独立的真实排他性断言。
_CR055_ANSWER_CROSS_CLAUSE = "不能；没有条件检查，但不只是先读再写，仍然只能用 Lua 脚本。"

# CR-055 附加案例：同一个 forbid pattern 在答案里出现两次，第一次被同一
# 分句内的"不是"正确豁免，第二个独立分句里是一条真实的排他性断言——
# 用来验证 `re.finditer` + 逐命中判极性的实际承诺（第一次被否定不等于
# 整条 pattern 都被豁免）。
_CR055_ANSWER_NEGATED_THEN_REAL = (
    "不是只能用 Lua 脚本这一种方式；不过对库存扣减这类场景，"
    "确实必须用 Lua 才能保证原子性。"
)


def _is_negated_pre_cr055(answer: str, match_start: int, *, window: int = 15) -> bool:
    """CR-054 刚上线时 `_is_negated()` 的行为快照：不按分句边界截断，
    且把"不只/不仅/不止/not only"也当否定标记。仅用于下面的判别性基线
    测试，不是生产代码的一部分。
    """
    markers = ("不是", "并非", "不只", "不仅", "不止", "not only", "isn't", "is not")
    prefix = answer[max(0, match_start - window):match_start]
    return any(marker in prefix for marker in markers)


# CR-056 复现句（取自 codex R52 复审给出的具体构造例句）：没有标点、只用
# "但是"这个转折连词分隔两个命题——"并非只能用 Lua"是对第一个命题的
# 正确否定，"但是仍然只能用 Lua 脚本"是完全独立的第二个命题，是一条
# 真实的排他性断言，不应该被前一个命题的"并非"连带豁免。
_CR056_ANSWER_TRANSITION_WITHOUT_PUNCTUATION = (
    "不能；没有条件检查，并非只能用 Lua 但是仍然只能用 Lua 脚本。"
)

# CR-056 附加案例：只有第一个命题、没有转折出的第二个命题——纯粹的
# "并非只能用 Lua" 不该被误判为排他性断言。
_CR056_ANSWER_ONLY_THE_NEGATED_CLAIM = (
    "不能直接防止超卖，并非只能用 Lua，也可以考虑其它机制来补上条件判断。"
)


def _clause_boundary_negation_pre_cr056(
    answer: str, match_start: int, *, window: int = 15
) -> bool:
    """CR-055 修复后、CR-056 修复前 `_is_negated()` 的行为快照：按标点
    分句边界截断，但不识别"但是/不过"这类无标点的转折连词。仅用于下面
    的判别性基线测试，不是生产代码的一部分。
    """
    markers = ("不是", "并非", "isn't", "is not")
    clause_boundary = re.compile(r"[，。；！？、,.;!?\n]")
    clause_start = 0
    for m in clause_boundary.finditer(answer, 0, match_start):
        clause_start = m.end()
    prefix = answer[max(clause_start, match_start - window):match_start]
    return any(marker in prefix for marker in markers)


def _is_negated_pre_cr057(answer: str, match_start: int, *, adjacency: int = 6) -> bool:
    """CR-056 刚上线时 `_is_negated()` 的行为快照：要求否定标记恰好紧邻
    （`str.endswith`）触发词本身，容不下"一定/the"这类自然修饰语。仅用于
    下面的判别性基线测试，不是生产代码的一部分。
    """
    markers = ("不是", "并非", "isn't", "is not")
    prefix = answer[max(0, match_start - adjacency):match_start]
    return any(prefix.endswith(marker) for marker in markers)


# CR-057 复现句一（取自 codex R53 复审给出的具体构造例句）：否定词
# "并不一定"和触发词"只能"之间隔着"一定"这个自然修饰语，不是隔着别的
# 命题——这是明确的否定语境，不应被判为真实排他性断言。
_CR057_ANSWER_LOOSE_ZH = "不能；没有条件检查，并不一定只能用 Lua 脚本，也可以采用其他机制。"

# CR-059 复现句（取自 codex R54 复审给出的具体构造例句）：同一个 pattern
# 命中两次——"并非只能用 Lua"（真的被否定）和"且必须使用 Lua 脚本"（中间
# 隔着加合连词"且"，是独立的真实排他性断言）。"且"没有被列进
# `_CLAUSE_BOUNDARY` 的连词词表（本轮故意不加，见 CR-059 说明），修复靠
# 的是"后一次命中不能越过前一次命中的结尾"这条结构性约束。
_CR059_ANSWER = "不能；没有条件检查，并非只能用 Lua 且必须使用 Lua 脚本。"


def _is_negated_pre_cr059(answer: str, match_start: int, *, window: int = 12) -> bool:
    """CR-057 修复后、CR-059 修复前 `_is_negated()` 的行为快照：按分句
    边界（含标点与转折连词）截断否定标记搜索范围，但不知道"同一个
    pattern 上一次命中在哪里结束"，因此后一次命中的否定检索能越过前一次
    命中，一路找到更早的、属于不同命题的否定标记。仅用于下面的判别性
    基线测试，不是生产代码的一部分。
    """
    markers = ("不是", "并非", "不一定", "isn't", "not")
    clause_boundary = re.compile(
        r"[，。；！？、,.;!?\n]|但是|不过|然而|却|仍然|but|however|though|yet"
    )
    clause_start = 0
    for m in clause_boundary.finditer(answer, 0, match_start):
        clause_start = m.end()
    prefix = answer[max(clause_start, match_start - window):match_start]
    return any(marker in prefix for marker in markers)


# CR-057 复现句二（英文版本）：否定词"not"和触发短语"only way"之间隔着
# "the"。
_CR057_ANSWER_LOOSE_EN = (
    "cannot; no condition check; it is definitely not the only way to use Lua; "
    "other mechanisms work."
)


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


def test_pre_cr054_lua_forbid_pattern_missed_the_need_phrasing():
    """判别性基线：CR-053 刚上线时的 Lua forbid pattern 只认"必须/只能/
    唯一/only way"，"需使用 Lua 脚本"同样是 CR-052 已撤回的排他性表述，
    却因为不含这几个词而完全没被拦截——两条正向 keypoint 照样全中，
    这正是 CR-054 指出的漏判，不是构造出来的假想场景。
    """
    hit, missed, forbidden, failures = _score(
        _CR054_ANSWER_NEEDS_LUA, _POSITIVE_KEYPOINTS, _FORBID_PATTERNS_PRE_CR054
    )
    assert hit == _POSITIVE_KEYPOINTS
    assert not missed
    assert not forbidden, "这正是 CR-054 之前的漏洞：本该拦截却没拦截"
    assert not failures


def test_cr054_lua_forbid_pattern_catches_the_need_phrasing():
    """CR-054 修复：补上"需/需要"触发词后，同一条答案被正确拦截。"""
    hit, missed, forbidden, failures = _score(
        _CR054_ANSWER_NEEDS_LUA, _POSITIVE_KEYPOINTS, _FORBID_PATTERNS
    )
    assert hit == _POSITIVE_KEYPOINTS
    assert not missed
    assert forbidden, "应命中修复后的 Lua 排他性 forbid pattern"
    assert failures


def test_naive_regex_without_negation_check_would_misfire_on_negated_claim():
    """判别性基线：不做否定语境处理、只靠裸正则命中，会把"不是只能用
    Lua，也可以用其他机制"这句明确推翻排他性主张的正确表述也判成命中——
    直接用 `re.search`（不经过 `_score`/`_is_negated`）验证这一点，
    证明极性处理不是可有可无的装饰，而是必须的一步。
    """
    lua_pattern = _FORBID_PATTERNS[1]
    assert re.search(lua_pattern, _CR054_ANSWER_NEGATED_EXCLUSIVITY), (
        "裸正则本该命中'只能…Lua'这个字面串——这正是需要否定语境处理的原因，"
        "如果这个断言本身不成立，说明触发词组已经变了，下面的负向测试就没有意义"
    )


def test_forbid_patterns_do_not_misfire_on_explicitly_negated_exclusivity_claim():
    """CR-054 修复：`_score()` 内置的否定语境过滤应放过这句正确表述，
    不能因为字面包含触发词组就误判为越界断言。
    """
    hit, missed, forbidden, failures = _score(
        _CR054_ANSWER_NEGATED_EXCLUSIVITY, _POSITIVE_KEYPOINTS, _FORBID_PATTERNS
    )
    assert not forbidden, f"不应误判为越界断言，实际命中: {forbidden}"
    assert not failures


def test_pre_cr055_negation_check_wrongly_exempts_a_cross_clause_violation():
    """判别性基线：CR-054 刚上线时的否定检测（不按分句边界截断、且把
    "不只/不仅/不止"也当否定标记）会把前一分句"不只是先读再写"这个
    完全不同命题的否定词，错误地当成后一分句"只能用 Lua 脚本"这句
    真实排他性断言的否定标记——这正是 CR-055 指出的洞，复现句取自
    codex R51 复审给出的具体构造例句，不是臆造的场景。
    """
    lua_pattern = _FORBID_PATTERNS[1]
    matches = list(re.finditer(lua_pattern, _CR055_ANSWER_CROSS_CLAUSE))
    assert matches, "复现句必须包含一次真实的 Lua 排他性字面命中"
    assert all(
        _is_negated_pre_cr055(_CR055_ANSWER_CROSS_CLAUSE, m.start()) for m in matches
    ), "旧的否定检测应该（错误地）把这次命中判定为已被否定——这正是 CR-055 的洞"


def test_score_no_longer_exempts_the_cross_clause_negation_repro():
    """CR-055 修复：按分句边界截断后，"不只是"不再能豁免后一独立分句里
    真实存在的"只能用 Lua 脚本"这句排他性断言。
    """
    hit, missed, forbidden, failures = _score(
        _CR055_ANSWER_CROSS_CLAUSE, _POSITIVE_KEYPOINTS, _FORBID_PATTERNS
    )
    assert hit == _POSITIVE_KEYPOINTS
    assert not missed
    assert forbidden, "应正确识别出这条真实的排他性断言，不被前一分句的否定词豁免"
    assert failures


def test_finditer_polarity_check_still_catches_a_later_independent_violation():
    """验证 `re.finditer` + 逐命中判极性的实际承诺：同一个 forbid pattern
    第一次命中被同一分句内的"不是"正确豁免，但后一个独立分句里"确实
    必须用 Lua"是另一条真实的排他性断言，不能因为第一次命中被豁免就
    连带放过整条 pattern。
    """
    _, _, forbidden, failures = _score(
        _CR055_ANSWER_NEGATED_THEN_REAL, _POSITIVE_KEYPOINTS, _FORBID_PATTERNS
    )
    assert forbidden, "第二个独立分句里的真实排他性断言不应被第一次的豁免连带放过"
    assert failures


def test_pre_cr056_negation_check_wrongly_exempts_a_transition_word_violation():
    """判别性基线：CR-055 刚上线时的否定检测按标点分句截断，但"但是"这类
    转折连词不产生标点分句边界——"并非只能用 Lua"和"但是仍然只能用 Lua
    脚本"因为中间没有标点，被判定成同一个分句，前一命题的"并非"泄漏
    给了后一个独立的真实排他性断言，两次命中都被错误判定为已否定。
    复现句取自 codex R52 复审给出的具体构造例句，不是臆造的场景。
    """
    lua_pattern = _FORBID_PATTERNS[1]
    matches = list(re.finditer(lua_pattern, _CR056_ANSWER_TRANSITION_WITHOUT_PUNCTUATION))
    assert len(matches) == 2, (
        "复现句应包含两次字面命中：'并非只能用 Lua' 与 '但是仍然只能用 Lua 脚本'"
    )
    assert all(
        _clause_boundary_negation_pre_cr056(
            _CR056_ANSWER_TRANSITION_WITHOUT_PUNCTUATION, m.start()
        )
        for m in matches
    ), "旧的按标点分句的否定检测应该（错误地）把两次命中都判定为已被否定——这正是 CR-056 的洞"


def test_score_no_longer_exempts_the_transition_word_violation():
    """CR-056 修复：否定标记必须紧邻触发词本身，不再因为"但是"这类转折
    连词没有产生标点分句边界，就把前一命题的否定词泄漏给后一个独立的
    真实排他性断言。
    """
    hit, missed, forbidden, failures = _score(
        _CR056_ANSWER_TRANSITION_WITHOUT_PUNCTUATION, _POSITIVE_KEYPOINTS, _FORBID_PATTERNS
    )
    assert hit == _POSITIVE_KEYPOINTS
    assert not missed
    assert forbidden, "应正确识别出'但是仍然只能用 Lua 脚本'这条独立的真实排他性断言"
    assert failures


def test_forbid_patterns_still_pass_a_purely_negated_claim_without_a_real_violation():
    """避免矫枉过正：只有"并非只能用 Lua"这一个被否定的命题、后面没有
    独立的真实排他性断言时，不应该被误判为命中——修复 CR-056 不能把
    否定检测收得太紧，连正常的否定语境都识别不出来。
    """
    _, _, forbidden, _ = _score(
        _CR056_ANSWER_ONLY_THE_NEGATED_CLAIM, _POSITIVE_KEYPOINTS, _FORBID_PATTERNS
    )
    assert not forbidden, f"纯粹被否定的表述不应误判为越界断言，实际命中: {forbidden}"


def test_forbid_patterns_do_not_misfire_on_negated_atomic_only_claim():
    """R52 复审自陈第 3 点指出："仅支持原子增减"这条 forbid pattern 一直
    没有专门的否定语境判别性测试，只手工验证过"不是仅支持原子增减，还有
    其他操作"不会误判——本轮把这个手工验证固化成一条真正的回归测试，
    确认 `_is_negated()` 的紧邻检查对这条 pattern 同样有效，不是只对
    Lua 排他性那条 pattern 生效。
    """
    correct_answer = "不是仅支持原子增减，还有其他操作，只是这些细节不在已入库语料里。"
    _, _, forbidden, _ = _score(
        correct_answer, _POSITIVE_KEYPOINTS, _FORBID_PATTERNS
    )
    assert not forbidden, f"不应误判为越界断言，实际命中: {forbidden}"


def test_pre_cr057_negation_check_wrongly_rejects_a_loose_zh_negation():
    """判别性基线：CR-056 刚上线时的"紧邻"否定检测（`str.endswith`，
    几乎零容忍）会把"并不一定只能用 Lua 脚本"这句夹着"一定"这个自然
    修饰语的否定表述，错判成没有被否定——复现句取自 codex R53 复审
    给出的具体构造例句，不是臆造的场景。
    """
    lua_pattern = _FORBID_PATTERNS[1]
    matches = list(re.finditer(lua_pattern, _CR057_ANSWER_LOOSE_ZH))
    assert matches, "复现句必须包含一次真实的 Lua 排他性字面命中"
    assert not any(
        _is_negated_pre_cr057(_CR057_ANSWER_LOOSE_ZH, m.start()) for m in matches
    ), "旧的紧邻检测应该（错误地）把这次命中判定为未被否定——这正是 CR-057 的洞"


def test_score_recognizes_the_loose_zh_negation():
    """CR-057 修复：分句边界内留出的窗口能容纳"一定"这类自然修饰语，
    正确识别出这是被否定的表述，不判为真实排他性断言。
    """
    hit, missed, forbidden, failures = _score(
        _CR057_ANSWER_LOOSE_ZH, _POSITIVE_KEYPOINTS, _FORBID_PATTERNS
    )
    assert hit == _POSITIVE_KEYPOINTS
    assert not missed
    assert not forbidden, f"应识别为否定语境，不应误判为越界断言，实际命中: {forbidden}"
    assert not failures


def test_pre_cr057_negation_check_wrongly_rejects_a_loose_en_negation():
    """同上，英文版本："not"和"only way"之间隔着"the"，CR-056 的零容忍
    紧邻检测同样会错判成未被否定。
    """
    lua_pattern = _FORBID_PATTERNS[1]
    matches = list(re.finditer(lua_pattern, _CR057_ANSWER_LOOSE_EN))
    assert matches, "复现句必须包含一次真实的 'only way' 字面命中"
    assert not any(
        _is_negated_pre_cr057(_CR057_ANSWER_LOOSE_EN, m.start()) for m in matches
    ), "旧的紧邻检测应该（错误地）把这次命中判定为未被否定——这正是 CR-057 的洞"


def test_score_recognizes_the_loose_en_negation():
    """CR-057 修复：英文版本同样能正确识别否定语境。"""
    hit, missed, forbidden, failures = _score(
        _CR057_ANSWER_LOOSE_EN, _POSITIVE_KEYPOINTS, _FORBID_PATTERNS
    )
    assert hit == _POSITIVE_KEYPOINTS
    assert not missed
    assert not forbidden, f"应识别为否定语境，不应误判为越界断言，实际命中: {forbidden}"
    assert not failures


def test_pre_cr059_negation_check_wrongly_exempts_a_second_chained_violation():
    """判别性基线：CR-057 修复后、CR-059 修复前的否定检测不知道"同一个
    pattern 上一次命中在哪里结束"，因此"并非只能用 Lua"里"并非"的否定
    效力能越过"且"这个未被列进连词词表的加合连词，泄漏给后面"必须使用
    Lua 脚本"这句独立的真实排他性断言——复现句取自 codex R54 复审给出的
    具体构造例句，不是臆造的场景。
    """
    lua_pattern = _FORBID_PATTERNS[1]
    matches = list(re.finditer(lua_pattern, _CR059_ANSWER))
    assert len(matches) == 2, (
        "复现句应包含两次字面命中：'并非只能用 Lua' 与 '且必须使用 Lua 脚本'"
    )
    assert all(
        _is_negated_pre_cr059(_CR059_ANSWER, m.start()) for m in matches
    ), "旧逻辑应该（错误地）把两次命中都判定为已被否定——这正是 CR-059 的洞"


def test_score_no_longer_exempts_the_second_chained_violation():
    """CR-059 修复：同一个 pattern 后一次命中的否定检索不能越过前一次
    命中的结尾——不需要认识"且"是连词，就能正确识别出这是两个独立命题。
    """
    hit, missed, forbidden, failures = _score(
        _CR059_ANSWER, _POSITIVE_KEYPOINTS, _FORBID_PATTERNS
    )
    assert hit == _POSITIVE_KEYPOINTS
    assert not missed
    assert forbidden, "应正确识别出'且必须使用 Lua 脚本'这条独立的真实排他性断言"
    assert failures
