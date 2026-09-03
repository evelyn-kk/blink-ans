"""问答编排的判定逻辑测试。

这些断言不加载模型，只验证充分性判定、证据选取与引用统计——
它们是决定"何时该拒答"的核心逻辑，必须能快速回归。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from services.orchestrator.answering import (  # noqa: E402
    AnswerConfig, AnswerRequest, Orchestrator, Sufficiency, _select_evidence_greedy, assess,
    citation_coverage, declined, select_evidence,
)
from services.retrieval.search import Hit  # noqa: E402

CFG = AnswerConfig()


def hit(dist=0.65, kw=1, tokens=200, i=1, score=0.03) -> Hit:
    return Hit(
        rowid=i, text=f"证据正文 {i}" * 10, title_path=f"A › B{i}",
        source_url=f"https://example.com/{i}.html", source_project="kafka",
        version_or_commit="abc", retrieved_at="2026-08-31T00:00:00+00:00",
        technology="kafka", content_type="prose", token_estimate=tokens,
        score=score, keyword_rank=kw, vector_rank=1, vector_distance=dist,
    )


# ---------- 充分性判定 ----------

def test_empty_hits_are_insufficient():
    a = assess([], CFG)
    assert a.level is Sufficiency.INSUFFICIENT


def test_close_evidence_is_sufficient():
    """实测：语料覆盖的问题 top1 距离在 0.60–0.70。"""
    assert assess([hit(dist=0.66)], CFG).level is Sufficiency.SUFFICIENT


def test_far_evidence_is_insufficient():
    """实测：语料不覆盖的问题 top1 距离在 0.75–0.83。

    判定不能靠空结果——FTS 用 OR 查询，几乎任何中文提问都会返回内容（I1 实测）。
    """
    assert assess([hit(dist=0.83, kw=0)], CFG).level is Sufficiency.INSUFFICIENT


def test_middle_distance_is_limited_regardless_of_keyword():
    """中间档不依赖关键词命中。

    技术名词被剔除出检索词后（tokenize.PROJECT_TERMS），中文提问打英文语料
    关键词命中常为 0。若中间档要求 kw>0，该档永不生效，边缘的有效问题会被一律误拒——
    50 题回归中「Spring Boot 怎么配置日志级别」（实测 0.7241）就因此被错判为超范围。
    """
    assert assess([hit(dist=0.74, kw=1)], CFG).level is Sufficiency.LIMITED
    assert assess([hit(dist=0.74, kw=None)], CFG).level is Sufficiency.LIMITED


def test_borderline_valid_question_is_not_refused():
    """实测值回归：0.7241 属有效问题，不得判为证据不足。"""
    assert assess([hit(dist=0.7241, kw=None)], CFG).level is not Sufficiency.INSUFFICIENT


def test_assessment_uses_closest_not_first():
    """命中列表按 RRF 排序，最相关的未必排在首位。"""
    a = assess([hit(dist=0.80, i=1), hit(dist=0.62, i=2)], CFG)
    assert a.level is Sufficiency.SUFFICIENT
    assert a.top_distance == pytest.approx(0.62)


def test_missing_vector_signal_degrades_to_limited():
    """只有关键词命中时无法评估语义相关性，保守处理而非直接作答。"""
    h = hit(); h.vector_distance = None
    assert assess([h], CFG).level is Sufficiency.LIMITED


# ---------- 证据选取 ----------

def test_selection_respects_token_budget():
    """预算约束来自 I0 实测：超出即首 token 时延不达标。"""
    cfg = AnswerConfig(evidence_budget=500, max_evidence=10)
    ev = select_evidence([hit(tokens=200, i=i) for i in range(1, 6)], cfg)
    assert sum(len(e.text) for e in ev) > 0
    assert len(ev) == 2


def test_selection_skips_oversized_and_keeps_filling():
    """放不下的跳过、继续找更小的——宁可多带一条短证据也不让预算空着。"""
    cfg = AnswerConfig(evidence_budget=300, max_evidence=10)
    ev = select_evidence([hit(tokens=900, i=1), hit(tokens=100, i=2), hit(tokens=100, i=3)], cfg)
    assert [e.index for e in ev] == [1, 2]
    assert all("证据正文 1" not in e.text for e in ev)


def test_selection_caps_count():
    cfg = AnswerConfig(evidence_budget=10_000, max_evidence=3)
    assert len(select_evidence([hit(tokens=50, i=i) for i in range(1, 9)], cfg)) == 3


def test_greedy_evidence_selection_is_suboptimal():
    """T-030 判别性回归：证明"按融合分顺序贪心装填"（旧实现，现仍留存为
    _select_evidence_greedy 防御性回退）在构造场景下选不到预算内总分最大的组合，
    而 select_evidence 的 0/1 背包 DP 能选到。

    构造：候选 A(score=10, cost=650) 排序最靠前、单独恰好装满预算；
    B、C(各 score=9, cost=325) 排序靠后，二者合计 650 同样刚好装满预算、
    但总分 18 高于单独装 A 的 10。旧贪心一遇到"当前这条能装下"就装，装满 A
    后 B、C 都放不下，只能选到总分 10 的次优解；DP 按"预算内总分最大化"求解，
    应该选到 {B, C}，总分 18。
    """
    budget = 650
    cfg = AnswerConfig(evidence_budget=budget, max_evidence=5)
    hits = [
        hit(tokens=650, i=1, score=10.0),
        hit(tokens=325, i=2, score=9.0),
        hit(tokens=325, i=3, score=9.0),
    ]

    old = _select_evidence_greedy(hits, cfg)
    assert [e.citation for e in old] == [hits[0].citation]  # 只选到 A，总分 10——次优

    new = select_evidence(hits, cfg)
    assert {e.citation for e in new} == {hits[1].citation, hits[2].citation}  # 选到 {B, C}，总分 18——最优
    assert sum(1 for h in hits if h.citation in {e.citation for e in new}) == 2


def test_evidence_indices_are_sequential_from_one():
    """编号是引用的锚，必须从 1 连续——模型输出 [2] 时客户端据此对应来源卡片。"""
    cfg = AnswerConfig(evidence_budget=300, max_evidence=10)
    ev = select_evidence([hit(tokens=900, i=1), hit(tokens=100, i=2), hit(tokens=100, i=3)], cfg)
    assert [e.index for e in ev] == list(range(1, len(ev) + 1))


# ---------- 引用覆盖率 ----------

def test_citation_coverage_extracts_used_indices():
    assert citation_coverage("结论：见 [1] 与 [3]。", 4) == [1, 3]


def test_citation_coverage_ignores_out_of_range():
    """模型编造超出范围的编号时不得计入，否则覆盖率虚高。"""
    assert citation_coverage("参考 [9] 和 [2]", 3) == [2]


def test_citation_coverage_empty_when_no_evidence():
    assert citation_coverage("任何文本 [1]", 0) == []


def test_uncited_answer_reports_zero():
    """无引用的技术论断无法追溯，对本产品等同于不可用，必须能被观测到。"""
    assert citation_coverage("结论：应当调大线程池。", 3) == []


class FakeEmbedder:
    def encode_one(self, _question):
        return [0.0]


class FakeEngine:
    """T-028 前的假引擎：只测判定逻辑，不测路由。"""

    count_tokens = staticmethod(lambda text: len(text) // 4)

    def stream(self, *_args, **_kwargs):
        yield {"type": "delta", "text": "答案 [1]"}
        yield {"type": "done", "ttft_s": 0.1, "prompt_tokens": 10,
               "prefilled_tokens": 10, "prefix_reused": True, "decode_tps": 20.0}


class FakeCloudBackend:
    """`_evidence_budget()` 只需要 `available()`——足够用的最小假货。"""

    name = "claude"

    def available(self) -> bool:
        return True


class FakeRouter:
    """T-028：把 FakeEngine 适配成 Router 的最小接口（count_tokens + generate）。

    不用真正的 `Router`，是因为这批测试要断言的是"编排层是否正确转发了
    项目边界过滤条件"，与路由决策本身无关——用真 Router 只会多引入一个
    需要被 mock 的 GenerationBackend，对测试意图没有帮助。真正的路由决策
    （云端/本地切换、断网降级）由 test_router.py 覆盖。
    """

    def __init__(self, engine: FakeEngine, served_by: str = "local") -> None:
        self._engine = engine
        self._served_by = served_by
        # T-022：Orchestrator._evidence_budget() 读这两个属性来预判走哪条路径
        # （与真 Router 的判断条件保持一致，见 answering.py 的注释）。
        self.offline = False
        self.cloud = FakeCloudBackend() if served_by == "claude" else None

    def count_tokens(self, text: str) -> int:
        return self._engine.count_tokens(text)

    def generate(self, content, *, max_tokens, system_override=None, cloud_allowed=True):
        for ev in self._engine.stream(content, max_tokens=max_tokens, system_override=system_override):
            if ev["type"] == "done":
                ev = {**ev, "served_by": self._served_by}
            yield ev


def test_answering_forwards_project_boundary_to_both_retrieval_paths(monkeypatch):
    """T-103：HTTP/编排层不能在传递过程中丢掉项目隔离条件。"""
    captured = {}

    def fake_search(*_args, **kwargs):
        captured.update(kwargs)
        return [hit()]

    monkeypatch.setattr("services.orchestrator.answering.hybrid_search", fake_search)
    events = list(Orchestrator(None, FakeEmbedder(), FakeRouter(FakeEngine())).answer(AnswerRequest(
        "怎么预留库存", project_id="orders", module="checkout", symbol="reserve_stock",
    )))
    assert any(e["type"] == "sources" for e in events)
    assert {k: captured[k] for k in ("project_id", "module", "symbol")} == {
        "project_id": "orders", "module": "checkout", "symbol": "reserve_stock",
    }


def test_answering_forwards_served_by_into_done_event(monkeypatch):
    """T-028：路由决策的 served_by 必须原样透传进 done 事件（architecture.md §7）。"""
    monkeypatch.setattr(
        "services.orchestrator.answering.hybrid_search", lambda *a, **kw: [hit()]
    )
    events = list(Orchestrator(None, FakeEmbedder(), FakeRouter(FakeEngine(), served_by="claude"))
                  .answer(AnswerRequest("怎么预留库存")))
    done = next(e for e in events if e["type"] == "done")
    assert done["served_by"] == "claude"


class FakeCloudEngine(FakeEngine):
    """T-029 回归：模拟云端 done 事件里带 cache_read_tokens/cache_write_tokens/
    cost_usd（真实实现见 services/inference/claude_backend.py 的 stream()）。
    """

    def stream(self, *_args, **_kwargs):
        yield {"type": "delta", "text": "答案 [1]"}
        yield {
            "type": "done", "ttft_s": 0.1, "prompt_tokens": 10,
            "prefilled_tokens": 10, "prefix_reused": True, "decode_tps": 20.0,
            "cache_read_tokens": 123, "cache_write_tokens": 45, "cost_usd": 0.000789,
        }


def test_generate_forwards_cache_tokens_and_cost_into_done_event(monkeypatch):
    """T-029 CR 修复回归：Orchestrator._generate() 此前手工挑字段构造 done 事件时
    漏传了 cache_read_tokens/cache_write_tokens（只透传了 prefix_reused）。
    这里用假路由构造一个带这两个字段的 done 事件，断言最终 done 事件里确实出现。
    """
    monkeypatch.setattr(
        "services.orchestrator.answering.hybrid_search", lambda *a, **kw: [hit()]
    )
    events = list(
        Orchestrator(None, FakeEmbedder(), FakeRouter(FakeCloudEngine(), served_by="claude"))
        .answer(AnswerRequest("怎么预留库存"))
    )
    done = next(e for e in events if e["type"] == "done")
    assert done["cache_read_tokens"] == 123
    assert done["cache_write_tokens"] == 45
    assert done["cost_usd"] == 0.000789


def test_generate_defaults_cache_tokens_to_none_and_cost_to_zero_for_local(monkeypatch):
    """本地引擎的 done 事件里没有 cache_read_tokens/cache_write_tokens/cost_usd 这几个
    键（services/inference/engine.py 没有这个概念）——透传时不能伪造出数值，
    cache 字段缺省为 None，cost_usd 缺省为 0.0（本地生成成本恒为 0，这是项目
    一贯口径，不是"测不出来"的占位）。
    """
    monkeypatch.setattr(
        "services.orchestrator.answering.hybrid_search", lambda *a, **kw: [hit()]
    )
    events = list(
        Orchestrator(None, FakeEmbedder(), FakeRouter(FakeEngine(), served_by="local"))
        .answer(AnswerRequest("怎么预留库存"))
    )
    done = next(e for e in events if e["type"] == "done")
    assert done["cache_read_tokens"] is None
    assert done["cache_write_tokens"] is None
    assert done["cost_usd"] == 0.0


def test_project_cloud_ban_forces_cloud_allowed_false(monkeypatch):
    """架构决策（§6.4）：命中证据里只要有一条项目材料禁止云端，就必须强制本地。

    这里不重复测 Router 的降级逻辑（见 test_router.py），只断言编排层算出的
    `cloud_allowed` 确实随 `cloud_generation_allowed=False` 的命中翻转——
    路由层只能看到编排层传给它的这一个布尔值，算错了路由无从纠正。
    """
    banned_hit = hit()
    banned_hit.cloud_generation_allowed = False
    monkeypatch.setattr(
        "services.orchestrator.answering.hybrid_search", lambda *a, **kw: [banned_hit]
    )

    captured_cloud_allowed = {}

    class RecordingRouter(FakeRouter):
        def generate(self, content, *, max_tokens, system_override=None, cloud_allowed=True):
            captured_cloud_allowed["value"] = cloud_allowed
            yield from super().generate(
                content, max_tokens=max_tokens, system_override=system_override,
                cloud_allowed=cloud_allowed,
            )

    list(Orchestrator(None, FakeEmbedder(), RecordingRouter(FakeEngine()))
         .answer(AnswerRequest("怎么预留库存")))
    assert captured_cloud_allowed["value"] is False


# ---------- T-022：拒答标记（语言无关） ----------

def test_declined_matches_marker_at_start():
    assert declined("NO_EVIDENCE") is True
    assert declined("  NO_EVIDENCE\n") is True  # 允许前后空白


def test_declined_matches_marker_followed_by_more_text():
    """插入式使用（INSUFFICIENT_PROMPT 要求先出标记再补充说明）也算拒答。"""
    assert declined("NO_EVIDENCE\n\n本地知识库未覆盖……") is True


def test_declined_false_when_marker_not_at_start():
    """标记必须在开头；模型把它当收尾语气词粘在正文末尾，不该被判定为拒答——
    那种情况下正文本身可能是编造内容，展示来源反而更危险，需要人工可见。
    """
    assert declined("消费者未提交偏移量导致重复消费 [1]。\n\nNO_EVIDENCE") is False


def test_declined_false_for_normal_cited_answer():
    assert declined("结论：需要手动提交偏移量 [1]。") is False


def test_declined_ignores_translated_prose():
    """回归 T-022 之前的缺陷：中文散文正则在英文回答下必然不命中。

    换成固定标记后，中英文的"正常措辞"都不应被误判为拒答——
    这里用两种语言里含有"证据不足"字面含义的句子做反例。
    """
    assert declined("现有证据不足以支撑这个结论，建议查阅官方文档。") is False
    assert declined("The evidence is insufficient to support a conclusion.") is False


# ---------- T-022：双语请求的语言选择 ----------

def test_answer_rejects_unsupported_language(monkeypatch):
    """未知语言必须在检索前就被拒绝，不能悄悄落到某个默认语言上。"""
    called = {"retrieval": False}

    def fake_search(*_a, **_kw):
        called["retrieval"] = True
        return [hit()]

    monkeypatch.setattr("services.orchestrator.answering.hybrid_search", fake_search)
    events = list(Orchestrator(None, FakeEmbedder(), FakeRouter(FakeEngine()))
                  .answer(AnswerRequest("怎么预留库存", language="fr")))
    assert called["retrieval"] is False
    assert events == [{
        "type": "error", "stage": "request",
        "message": "不支持的语言 'fr'，仅支持 ('zh', 'en')",
    }]


class RecordingRouter(FakeRouter):
    """记录每次 `generate()` 调用收到的 content 与 system_override。"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.calls: list[dict] = []

    def generate(self, content, *, max_tokens, system_override=None, cloud_allowed=True):
        self.calls.append({"content": content, "system_override": system_override})
        yield from super().generate(
            content, max_tokens=max_tokens, system_override=system_override,
            cloud_allowed=cloud_allowed,
        )


def test_english_request_renders_english_user_message_labels(monkeypatch):
    monkeypatch.setattr(
        "services.orchestrator.answering.hybrid_search", lambda *a, **kw: [hit()]
    )
    router = RecordingRouter(FakeEngine())
    list(Orchestrator(None, FakeEmbedder(), router)
         .answer(AnswerRequest("How to reserve inventory", language="en")))
    assert "[Evidence]" in router.calls[0]["content"]
    assert "[Question]" in router.calls[0]["content"]
    assert "【证据】" not in router.calls[0]["content"]


def test_chinese_request_renders_chinese_user_message_labels(monkeypatch):
    monkeypatch.setattr(
        "services.orchestrator.answering.hybrid_search", lambda *a, **kw: [hit()]
    )
    router = RecordingRouter(FakeEngine())
    list(Orchestrator(None, FakeEmbedder(), router).answer(AnswerRequest("怎么预留库存")))
    assert "【证据】" in router.calls[0]["content"]
    assert "【问题】" in router.calls[0]["content"]


def test_system_override_is_none_when_request_language_matches_default(monkeypatch):
    """请求语言与本地常驻前缀预热语言一致时传 None，才能让本地引擎复用前缀。"""
    monkeypatch.setattr(
        "services.orchestrator.answering.hybrid_search", lambda *a, **kw: [hit()]
    )
    router = RecordingRouter(FakeEngine())
    list(Orchestrator(None, FakeEmbedder(), router, config=AnswerConfig(default_language="zh"))
         .answer(AnswerRequest("怎么预留库存", language="zh")))
    assert router.calls[0]["system_override"] is None


def test_system_override_is_explicit_when_request_language_differs_from_default(monkeypatch):
    """请求语言与常驻前缀语言不一致时必须显式传入对应语言提示词（T-027：
    本地遇到语言切换直接完整 prefill，不重新实现多槽常驻前缀）。
    """
    from packages.prompts.answer import system_prompt

    monkeypatch.setattr(
        "services.orchestrator.answering.hybrid_search", lambda *a, **kw: [hit()]
    )
    router = RecordingRouter(FakeEngine())
    list(Orchestrator(None, FakeEmbedder(), router, config=AnswerConfig(default_language="zh"))
         .answer(AnswerRequest("How to reserve inventory", language="en")))
    assert router.calls[0]["system_override"] == system_prompt("en")


# ---------- T-022：证据预算按路径分档 ----------

class OfflineRouter(FakeRouter):
    def __init__(self, *, offline=False, cloud=None):
        self.offline = offline
        self.cloud = cloud

    def count_tokens(self, text):
        return len(text) // 4


def test_router_might_use_cloud_true_when_cloud_available():
    orch = Orchestrator(None, FakeEmbedder(), OfflineRouter(cloud=FakeCloudBackend()))
    assert orch._router_might_use_cloud() is True


def test_router_might_use_cloud_false_when_offline():
    orch = Orchestrator(
        None, FakeEmbedder(), OfflineRouter(offline=True, cloud=FakeCloudBackend())
    )
    assert orch._router_might_use_cloud() is False


def test_router_might_use_cloud_false_when_no_cloud_backend_configured():
    orch = Orchestrator(None, FakeEmbedder(), OfflineRouter(cloud=None))
    assert orch._router_might_use_cloud() is False


def test_router_might_use_cloud_false_when_cloud_unavailable():
    class UnavailableCloud(FakeCloudBackend):
        def available(self) -> bool:
            return False

    orch = Orchestrator(None, FakeEmbedder(), OfflineRouter(cloud=UnavailableCloud()))
    assert orch._router_might_use_cloud() is False


def test_cloud_and_local_evidence_budgets_are_distinct_and_ordered():
    """云端档必须明显宽松于本地档，否则拆分没有意义（architecture.md §6.6 第 1 条：
    云端 prefill 不随上下文线性增长，本地仍受限）。
    """
    cfg = AnswerConfig()
    assert cfg.evidence_budget_cloud > cfg.evidence_budget


# ---------- CR-025：cloud_allowed 按实际选中的证据判定，不是全部候选 ----------

class CostRouter(OfflineRouter):
    """按 `hit.text -> token_estimate` 的映射精确控制每条候选的"真实" token
    成本，而不是依赖 `len(text)//4`（`hit()` 的 text 与 `tokens=` 参数无关，
    真实成本必须能被测试精确控制才能断言预算边界）。
    """

    def __init__(self, hits: list[Hit], **kw):
        super().__init__(**kw)
        self._costs = {h.text: h.token_estimate for h in hits}

    def count_tokens(self, text):
        return self._costs[text]


def test_select_evidence_uses_cloud_budget_and_allows_cloud_when_nothing_selected_is_banned():
    hits = [hit(i=1, score=0.05, tokens=100)]
    orch = Orchestrator(
        None, FakeEmbedder(), CostRouter(hits, cloud=FakeCloudBackend()),
    )
    evidence, cloud_allowed = orch._select_evidence_and_cloud_allowed(hits)
    assert cloud_allowed is True
    assert len(evidence) == 1


def test_select_evidence_forces_local_budget_when_selected_evidence_is_banned():
    """唯一候选就是禁云块——它必然被选中，必须强制本地并使用本地预算。"""
    banned = hit(i=1, score=0.05, tokens=100)
    banned.cloud_generation_allowed = False
    orch = Orchestrator(
        None, FakeEmbedder(), CostRouter([banned], cloud=FakeCloudBackend()),
    )

    evidence, cloud_allowed = orch._select_evidence_and_cloud_allowed([banned])

    assert cloud_allowed is False
    assert len(evidence) == 1


def test_candidate_banned_hit_not_selected_as_evidence_does_not_force_local():
    """CR-025 判别性场景：候选里有一条禁云块，但它分数最低、超出
    `max_evidence` 条数上限，从未被选为最终证据——不该因此强制本地。

    旧实现（按全部候选判断）会在这里错误地把 cloud_allowed 判成 False；
    修复后的实现只看实际选中的证据，这条从未参与选择的禁云块不该影响判定。
    """
    cfg = AnswerConfig(max_evidence=5, evidence_budget=650, evidence_budget_cloud=3200)
    allowed_hits = [hit(i=i, score=1.0 - i * 0.01, tokens=50) for i in range(1, 6)]
    banned_but_unselected = hit(i=6, score=0.01, tokens=50)  # 分数最低，且第 6 条超出 max_evidence
    banned_but_unselected.cloud_generation_allowed = False
    hits = allowed_hits + [banned_but_unselected]

    orch = Orchestrator(
        None, FakeEmbedder(), CostRouter(hits, cloud=FakeCloudBackend()), config=cfg,
    )
    evidence, cloud_allowed = orch._select_evidence_and_cloud_allowed(hits)

    assert len(evidence) == 5
    assert 6 not in {e.rowid for e in evidence}
    assert cloud_allowed is True


def test_cloud_allowed_is_recomputed_after_local_reselect():
    """CR-029 独立复现场景（用审查方给出的原始数字）：禁云块 rowid=1
    (score=10, cost=700) 与允许块 rowid=2 (score=1, cost=650)，云端预算 1000、
    本地预算 650。

    云端预算下只能装下 1 或 2 中的一个（合计 1350 > 1000），DP 选分数更高的
    块 1（禁云）→ 触发重选；本地预算 650 下块 1（cost 700）已经装不下，
    重选只能选块 2（不禁云）。最终返回的证据是 [2]，完全不含禁云块，
    `cloud_allowed` 必须重新算成 True——旧实现会继续沿用触发重选那一次算出的
    False，让本可以走云端的证据被无故强制本地。
    """
    banned = hit(i=1, score=10.0, tokens=700)
    banned.cloud_generation_allowed = False
    allowed = hit(i=2, score=1.0, tokens=650)
    hits = [banned, allowed]
    cfg = AnswerConfig(max_evidence=5, evidence_budget=650, evidence_budget_cloud=1000)

    orch = Orchestrator(
        None, FakeEmbedder(), CostRouter(hits, cloud=FakeCloudBackend()), config=cfg,
    )
    evidence, cloud_allowed = orch._select_evidence_and_cloud_allowed(hits)

    assert {e.rowid for e in evidence} == {2}
    assert cloud_allowed is True


def test_select_evidence_skips_cloud_budget_reselect_when_router_cannot_use_cloud():
    """路由层面已经确定走本地（离线）时，直接用本地预算选，不要先按云端
    预算选出超过本地 prefill 能力的证据——即使证据本身没有禁云限制。
    """
    cfg = AnswerConfig(max_evidence=5, evidence_budget=100, evidence_budget_cloud=3200)
    # 单条 200 token：超过本地预算(100)，若错误地按云端预算(3200)选中就会超本地预算。
    hits = [hit(i=1, score=1.0, tokens=200)]
    orch = Orchestrator(
        None, FakeEmbedder(),
        CostRouter(hits, offline=True, cloud=FakeCloudBackend()), config=cfg,
    )

    evidence, cloud_allowed = orch._select_evidence_and_cloud_allowed(hits)

    assert evidence == []  # 唯一候选超出本地预算，选不出证据——不会误用云端预算硬塞进去
    assert cloud_allowed is True  # 证据内容本身不禁云，只是路由层面已经不会走云端


# ---------- CR-024：双后端都失败时的"要点+来源"最小可用答案 ----------

class BothBackendsFailRouter(FakeRouter):
    """模拟云端与本地都已耗尽（Router 内部把最终失败原样抛出）。"""

    def generate(self, content, *, max_tokens, system_override=None, cloud_allowed=True):
        raise RuntimeError("both backends exhausted")
        yield  # pragma: no cover — 让这个方法仍是生成器，便于 for-in 调用形态一致


def test_double_backend_failure_yields_deterministic_fallback_with_sources(monkeypatch):
    """architecture.md §6.4：云端与本地仍失败则返回"要点+来源"的最小可用答案，
    不是一个只有 error、没有任何可用内容的空响应。
    """
    monkeypatch.setattr(
        "services.orchestrator.answering.hybrid_search", lambda *a, **kw: [hit(i=1, dist=0.6)]
    )
    events = list(Orchestrator(None, FakeEmbedder(), BothBackendsFailRouter(FakeEngine()))
                  .answer(AnswerRequest("怎么预留库存")))

    assert not any(e["type"] == "error" for e in events)  # 不是纯错误，有可用内容
    delta_text = "".join(e["text"] for e in events if e["type"] == "answer_delta")
    assert "[1]" in delta_text  # 证据编号，与 sources 事件的 index 对应

    done = next(e for e in events if e["type"] == "done")
    assert done["served_by"] == "none"
    assert done.get("fallback") is True
    assert "both backends exhausted" in done["fallback_reason"]

    sources = next(e for e in events if e["type"] == "sources")
    assert len(sources["items"]) == 1  # 真实证据来源仍然完整展示，不是空列表


def test_double_backend_failure_with_no_evidence_still_errors():
    """"证据不足"分支本身就没有证据可列——没有"来源"可给时没有更好的最小可用
    答案，只能报错，这与"有证据但生成失败"的场景不同，不应该混为一谈。
    """
    events = list(Orchestrator(None, FakeEmbedder(), BothBackendsFailRouter(FakeEngine()))
                  ._generate(
        "问题", max_tokens=100, started=0.0, sufficiency=Sufficiency.INSUFFICIENT,
    ))
    assert len(events) == 1
    assert events[0]["type"] == "error"


def test_midstream_cloud_failure_is_not_treated_as_double_backend_failure(monkeypatch):
    """已经吐出过部分正文的云端中途失败（T-028 既有场景）与 CR-024 的
    "双后端全无产出"是两回事——前者已经在 Router 里处理成 error 事件转发，
    不应该在这里被 evidence 触发再包一层最小可用答案，那样才是真的拼接。
    """
    class MidstreamFailRouter(FakeRouter):
        def generate(self, content, *, max_tokens, system_override=None, cloud_allowed=True):
            yield {"type": "delta", "text": "云端开了个头"}
            yield {"type": "error", "stage": "generation_cloud_midstream", "message": "boom"}

    monkeypatch.setattr(
        "services.orchestrator.answering.hybrid_search", lambda *a, **kw: [hit(i=1, dist=0.6)]
    )
    events = list(Orchestrator(None, FakeEmbedder(), MidstreamFailRouter(FakeEngine()))
                  .answer(AnswerRequest("怎么预留库存")))

    deltas = [e["text"] for e in events if e["type"] == "answer_delta"]
    assert deltas == ["云端开了个头"]  # 没有被最小可用答案的文本追加在后面
    assert any(e["type"] == "error" and e["stage"] == "generation_cloud_midstream" for e in events)
    assert not any(e["type"] == "done" for e in events)


# ---------- CR-002: 未加载时的错误路径不依赖 MLX ----------

def test_stream_before_load_raises_runtime_error_without_mlx():
    """未加载模型时必须抛 RuntimeError，而不是导入 MLX 失败。

    代码评审 CR-002：错误路径若先导入 mlx_lm，在没有 Metal 的环境
    （如 CI）会抛 ImportError，纯逻辑测试就无法作为门禁运行。
    """
    from services.inference.engine import InferenceEngine

    eng = InferenceEngine()
    with pytest.raises(RuntimeError, match="尚未加载"):
        next(eng.stream("测试"))
