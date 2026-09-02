# 本机性能基准

这些脚本回答 I0 的选型问题：**在这台 Mac 上，5 秒响应预算能撑起多大的模型和多长的证据上下文。**

`bench_llm_remote.py` 是 2026-09-02 范围澄清后新增的：生成阶段允许调用商用模型 API，
于是"本地还是云端更快"重新变成一个待测问题（T-026）。它刻意复用 `bench_llm.py`
的同一份语料与同一道题，把逐字相同的提示词发给各家，两组数字才可直接对比。

## 环境

```bash
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

（此前这里写的是 `bench/requirements.txt`，该文件从未存在；依赖一直在仓库根的 `requirements.txt` 里。）

跑远端基准还需要凭据，缺哪家就跳过哪家：

```bash
export ANTHROPIC_API_KEY=...   # Claude
export MOONSHOT_API_KEY=...    # Kimi（月之暗面）
```

## 运行

```bash
cd bench
../.venv/bin/python bench_llm.py   --model mlx-community/Qwen3-4B-Instruct-2507-4bit
../.venv/bin/python bench_asr.py   --model mlx-community/whisper-large-v3-turbo
../.venv/bin/python bench_embed.py --model mlx-community/bge-m3-mlx-8bit

# 远端（T-026）。先只测网络地板确认链路与凭据，再跑分档，避免白花钱
../.venv/bin/python bench_llm_remote.py --providers claude --rtt-only
../.venv/bin/python bench_llm_remote.py --providers claude kimi
```

报告写入 `bench/reports/*.json`（不入版本控制）。每次更换模型、量化位宽或运行时都复跑同一套脚本，结果记入 `progress.md`。

## 怎么读这些数字

| 指标 | 来自 | 对应预算 | 判据 |
| --- | --- | --- | --- |
| `ttft_s` 按上下文分档 | `bench_llm` | 生成首段 2.5s | 决定检索能塞几段证据——这是最关键的一个数 |
| `decode_tps` | `bench_llm` | 首段之后的流式体验 | 低于 15 tok/s 时长答案读起来会明显卡顿 |
| `rtf`（实时率） | `bench_asr` | 转写 0.7s | 整段实时率不足 3x，流式转写就压不进预算 |
| `latency_s`（单条） | `bench_embed` | 检索 0.8s | 查询侧嵌入应远低于 100ms |
| `chunks_per_second` | `bench_embed` | 离线成本 | 决定 I1 全量建索引要跑多久 |
| `peak_memory_gb` | 全部 | 16GB 统一内存 | 三者常驻之和须为浏览器和编辑器留出余量 |
| `p95.ttft_text_s` 按上下文分档 | `bench_llm_remote` | 生成首段 3.0s | **云端看 P95 不看中位数**：本地时延由算力决定、方差小，云端由网络抖动和对方排队决定 |
| `network_rtt.tcp_connect_s` | `bench_llm_remote` | — | **真正的网络地板**，≈ 一次往返。SDK 复用连接，所以这才是稳态每请求要付的网络成本。不花钱、不发模型请求 |
| `network_rtt.tcp_tls_s` | `bench_llm_remote` | — | TCP+TLS 完整握手，**冷连接**一次性成本。keep-alive 生效后不再付，**不能按请求数乘** |
| `min_request.p95.ttft_text_s` | `bench_llm_remote` | — | 最小请求的端到端 TTFT。减去 `network_rtt` 才是服务端固定开销（排队+最小 prefill+首字） |
| `cache.status` | `bench_llm_remote` | — | `hit` / `miss` / **`unverified`**。三者必须分开读：`unverified` 是「这家没报字段，测不了」，**不是**没命中。只有 `hit` 才允许把缓存收益写进成本或时延模型 |
| `first_event_s` vs `ttft_text_s` | `bench_llm_remote` | — | 思考型模型先流式吐 thinking，两者的差就是被思考吃掉的时间。产品指标是后者 |

`cold` 是首次运行（含缓存未命中），`median` 是重复测量的中位数。**服务端模型常驻，因此以 `median` 为准**；`cold` 只用来估算冷启动和一键启动后的首次可用时间。

### 三条容易读错的地方

1. **各家的 `prompt_tokens` 不可直接横比。** 可比的只有证据正文那一段——
   本地基准会套一层 Qwen chat template，各家 API 也各自拼模板，模板 token 不在我们手里。
   报告里同时记了 `qwen_evidence_tokens` 与各家自报的 `provider_prompt_tokens`，
   差值就是模板与分词器的膨胀，下结论前必须先剥掉。

2. **`network_rtt` 与 `min_request` 是两个东西。** 早先只有一个叫「RTT 地板」的指标，
   实际发的却是一次完整的流式生成，里面含排队、prefill 和首字生成。
   混成一个数，看到 P95 超标就分不清该换网络还是该换供应商。

3. **prompt cache 不命中不报错**，只是变慢变贵。因此判定用两阶段断言：
   首轮必须有写入，后续轮必须有读取。另外要记住**最小可缓存前缀是
   512–4096 token 且随模型而变**，短于它会静默不缓存——
   生产里那条稳定前缀（系统提示词，实测 341 token）很可能整个落在门槛以下。

### 一条已经量到的结论（2026-09-02）

网络地板差距比预想大：同一台机器同一时刻，
`api.anthropic.com` TCP 中位 **0.018s**，`api.moonshot.cn` TCP 中位 **0.240s**、
TLS 完整握手中位 **0.794s**。单次往返差 13 倍。
3.0 秒生成预算下这不是可忽略的量，但也别急着下结论——
稳态下只付 TCP 那一份，且实际影响要等分档数据出来才能定。

## 已知局限

- `bench_asr` 的音频用 macOS `say` 合成，比真实口语干净。这些数字只用于时延选型，**准确率必须在 I4 用真实录音重新评估**。
- 未测多组件同时常驻时的内存竞争与降频，该场景在 I4 端到端联调时补测。
