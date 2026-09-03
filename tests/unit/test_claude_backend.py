"""ClaudeBackend 的非网络回归（T-028）。

**不发起任何真实请求**——只测 `available()`（纯环境变量判断）与端点/超时/重试
这些构造参数是否按预期传给了 SDK 客户端。真正的流式调用行为已经在
`bench/bench_llm_remote.py`（T-026）里用真实凭据验证过；这里只保证生产代码
没有在搬运那份参考实现时漏掉关键参数。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from services.inference.claude_backend import (  # noqa: E402
    CLAUDE_API_BASE_URL, ClaudeBackend, DEFAULT_MODEL, DEFAULT_TIMEOUT_S,
    PRICE_PER_MTOK_CACHE_READ_USD, PRICE_PER_MTOK_CACHE_WRITE_USD,
    PRICE_PER_MTOK_INPUT_USD, PRICE_PER_MTOK_OUTPUT_USD, compute_cost_usd,
    probe_network_floor,
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


# ---------- T-029：cost_usd 计价 ----------

def test_pricing_constants_match_2026_06_24_official_rate_card():
    """费率表本身（$5/$25 每 MTok，缓存写 1.25x/读 0.1x）核对，不是循环论证——
    这几个数字来自 Anthropic 官方费率表（2026-06-24），改动费率表要先改这里。
    """
    assert PRICE_PER_MTOK_INPUT_USD == 5.00
    assert PRICE_PER_MTOK_OUTPUT_USD == 25.00
    assert PRICE_PER_MTOK_CACHE_WRITE_USD == 6.25
    assert PRICE_PER_MTOK_CACHE_READ_USD == 0.50


def test_cost_usd_pure_input_output_no_cache():
    """1M 输入 + 1M 输出，无缓存：按费率表手算 = $5.00 + $25.00 = $30.00。"""
    cost = compute_cost_usd(prompt_tokens=1_000_000, output_tokens=1_000_000)
    assert cost == 30.0


def test_cost_usd_pure_cache_read_and_write():
    """1M 缓存读 + 1M 缓存写，无普通输入/输出：手算 = $0.50 + $6.25 = $6.75。"""
    cost = compute_cost_usd(
        prompt_tokens=0, output_tokens=0,
        cache_read_tokens=1_000_000, cache_write_tokens=1_000_000,
    )
    assert cost == 6.75


def test_cost_usd_mixed_small_amounts():
    """四个桶都非零的真实量级：手算 = 1000/1e6*5 + 500/1e6*25 + 200/1e6*0.5 + 100/1e6*6.25
    = 0.005 + 0.0125 + 0.0001 + 0.000625 = 0.018225。
    """
    cost = compute_cost_usd(
        prompt_tokens=1000, output_tokens=500,
        cache_read_tokens=200, cache_write_tokens=100,
    )
    assert cost == pytest.approx(0.018225, abs=1e-9)


def test_cost_usd_missing_cache_tokens_defaults_to_zero_not_error():
    """cache_read_tokens/cache_write_tokens 省略（本地场景不会调用这里，但函数
    本身对 None 要能正常算，不是必须显式传 0）：手算 = 100/1e6*5 = 0.0005。
    """
    cost = compute_cost_usd(prompt_tokens=100, output_tokens=0)
    assert cost == pytest.approx(0.0005, abs=1e-9)


# ---------- T-029：网络地板探测 ----------

def test_probe_network_floor_host_matches_api_base_url():
    """host 从 CLAUDE_API_BASE_URL 剥离协议头得到，不是另外硬编码一份。"""
    assert CLAUDE_API_BASE_URL == "https://api.anthropic.com"


def test_probe_network_floor_captures_connection_failure_without_raising(monkeypatch):
    """网络不通/超时时必须返回带 error 字段的 dict，不能向上抛异常——
    调用方 /healthz 依赖这一点才能不因为这一项测不出来就整体 500。
    """
    import socket

    def _boom(*_args, **_kwargs):
        raise OSError("network unreachable (simulated)")

    monkeypatch.setattr(socket, "create_connection", _boom)
    result = probe_network_floor(timeout_s=0.5)

    assert result["error"] is not None
    assert result["tcp_connect_s"] is None
    assert result["tcp_tls_s"] is None
    assert result["host"] == "api.anthropic.com"


def test_probe_network_floor_success_shape(monkeypatch):
    """连接成功时两个耗时字段都应是非负浮点数，error 为 None——
    用一个假的 socket/ssl 上下文验证形状，不依赖真实网络（沙箱可能没有出网权限）。
    """
    import socket
    import ssl

    class _FakeSock:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class _FakeSSLSock:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(socket, "create_connection", lambda *a, **kw: _FakeSock())
    monkeypatch.setattr(
        ssl.SSLContext, "wrap_socket", lambda self, sock, server_hostname=None: _FakeSSLSock()
    )

    result = probe_network_floor(timeout_s=0.5)

    assert result["error"] is None
    assert result["host"] == "api.anthropic.com"


def test_probe_network_floor_does_not_count_local_context_setup_as_network_time(monkeypatch):
    """CR-028 判别性回归：`tcp_connect_s`/`tcp_tls_s` 只能测网络往返，不能把本机
    证书库初始化（`ssl.create_default_context()`）的耗时算进去——那会把本机负载
    误报成网络地板。

    构造：`create_default_context()` 人为延迟 50ms，socket/TLS 握手都设为立即
    返回（0 延迟）。计时正确时两个字段应接近 0；旧实现（`t0` 在建 context 之前）
    会把这 50ms 也计进去，报出 ≈0.05s。
    """
    import socket
    import ssl
    import time as time_module

    class _FakeSock:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class _FakeSSLSock:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def _slow_create_default_context(*a, **kw):
        time_module.sleep(0.05)
        return ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)

    monkeypatch.setattr(ssl, "create_default_context", _slow_create_default_context)
    monkeypatch.setattr(socket, "create_connection", lambda *a, **kw: _FakeSock())
    monkeypatch.setattr(
        ssl.SSLContext, "wrap_socket", lambda self, sock, server_hostname=None: _FakeSSLSock()
    )

    result = probe_network_floor(timeout_s=0.5)

    assert result["error"] is None
    # 留够浮点噪声余量，但必须远小于人为注入的 50ms 延迟——否则说明本机初始化
    # 耗时又被算进"网络"数字了。
    assert result["tcp_connect_s"] < 0.02
    assert result["tcp_tls_s"] < 0.02
    assert isinstance(result["tcp_connect_s"], float) and result["tcp_connect_s"] >= 0
    assert isinstance(result["tcp_tls_s"], float) and result["tcp_tls_s"] >= 0
