"""ClaudeBackend 的非网络回归（T-028）。

**不发起任何真实请求**——只测 `available()`（纯环境变量判断）与端点/超时/重试
这些构造参数是否按预期传给了 SDK 客户端。真正的流式调用行为已经在
`bench/bench_llm_remote.py`（T-026）里用真实凭据验证过；这里只保证生产代码
没有在搬运那份参考实现时漏掉关键参数。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from services.inference.claude_backend import (  # noqa: E402
    CLAUDE_API_BASE_URL, ClaudeBackend, DEFAULT_MODEL, DEFAULT_TIMEOUT_S,
)


def test_unavailable_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert ClaudeBackend("system").available() is False


def test_available_with_api_key_set(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake")
    assert ClaudeBackend("system").available() is True


def test_empty_api_key_is_not_available(monkeypatch):
    """空字符串在 dotenv 语义里等价于"未设置"（见 packages/config/env.py）。"""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    assert ClaudeBackend("system").available() is False


def test_client_uses_explicit_endpoint_key_timeout_and_disables_retries(monkeypatch):
    """T-109/CR 修过的坑：不能静默继承同机可能存在的 Kimi 兼容网关环境变量。

    显式传 base_url/api_key 就是防这个的手段；同时 max_retries=0 防止 SDK
    默认的指数退避重试悄悄把一次超时拖成三次，吃掉降级到本地的时间预算。
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake")
    # 同机可能配置的 Kimi 兼容网关环境变量——必须被显式覆盖，不能被继承。
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://kimi.example.com")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "not-a-claude-token")

    backend = ClaudeBackend("system", timeout_s=1.23)
    client = backend._client()

    assert str(client.base_url).rstrip("/") == CLAUDE_API_BASE_URL
    assert client.api_key == "sk-test-fake"
    assert client.max_retries == 0


def test_default_model_matches_t026_benchmarked_model():
    """T-026 报告用的型号（bench/reports/llm-remote-*.json）必须是这里的默认值，
    否则路由决策（architecture.md §6.4）依据的实测数据和生产实际调用的模型对不上。
    """
    assert DEFAULT_MODEL == "claude-opus-5"


def test_default_timeout_matches_cloud_generation_sub_budget():
    """architecture.md §6.5：generation_started→first_answer_text 云端子预算 3.6s。"""
    assert DEFAULT_TIMEOUT_S == 3.6
