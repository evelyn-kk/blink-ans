# 本机性能基准

这些脚本回答 I0 的选型问题：**在这台 Mac 上，5 秒响应预算能撑起多大的模型和多长的证据上下文。**

## 环境

```bash
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r bench/requirements.txt
```

## 运行

```bash
cd bench
../.venv/bin/python bench_llm.py   --model mlx-community/Qwen3-4B-Instruct-2507-4bit
../.venv/bin/python bench_asr.py   --model mlx-community/whisper-large-v3-turbo
../.venv/bin/python bench_embed.py --model mlx-community/bge-m3-mlx-8bit
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

`cold` 是首次运行（含缓存未命中），`median` 是重复测量的中位数。**服务端模型常驻，因此以 `median` 为准**；`cold` 只用来估算冷启动和一键启动后的首次可用时间。

## 已知局限

- `bench_asr` 的音频用 macOS `say` 合成，比真实口语干净。这些数字只用于时延选型，**准确率必须在 I4 用真实录音重新评估**。
- 未测多组件同时常驻时的内存竞争与降频，该场景在 I4 端到端联调时补测。
