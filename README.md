# blink-ans

本地部署的语音输入、文字输出编程问答服务，面向 Java / Spring 云原生生产问题。

- 推理与转写全部在本机 Apple Silicon 上运行，不依赖云端大模型。
- 回答优先引用可追溯的本地知识库；时效性内容才检索官方来源。
- 桌面与手机共用一套 PWA 客户端，通过局域网访问本机服务。

## 开发状态

**文本问答已经可用**：提问 → 混合检索 → 带引用的流式回答，一条链路已经跑通。
语音输入、PWA 与局域网访问尚未接入（见 `progress.md` 的任务看板）。

| 能力 | 状态 |
| --- | --- |
| 知识同步与索引（5 个官方来源，约 1.19 万块） | 可用 |
| 混合检索（jieba 关键词 + 向量，RRF 融合） | 可用 |
| 文本问答 API 与 SSE 流式输出 | 可用 |
| 带版本与锚点的引用 | 可用 |
| 语音输入 | 未接入 |
| PWA 与局域网访问控制 | 未接入 |
| 官方来源实时检索 | 未接入 |

## 环境要求

- Apple Silicon Mac（本项目在 M4 / 16 GB / macOS 26.6 上开发与实测）。
  MLX 只支持 Apple Silicon，其他平台无法运行推理部分。
- Python 3.13（mlx 在 3.14 上生态尚不完整）。
- Homebrew 的 Python：需要 `sqlite3.enable_load_extension`，
  macOS 自带的 Python 关闭了这个能力，sqlite-vec 无法加载。
- 约 10 GB 磁盘：模型权重约 8.6 GB，语料稀疏克隆约 50 MB，索引约 80 MB。

## 快速开始

```bash
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r requirements.txt

./infra/scripts/kb sync      # 拉取语料、建索引、跑回归、激活（首次约 51 分钟，之后约 20 秒）
./infra/scripts/dev.sh       # 启动网关，打开 http://127.0.0.1:8080
```

首次同步要为全部切块计算嵌入（实测 12.9 块/秒），因此慢。
之后的同步按正文校验和复用向量，只有正文真的变了才重算。

模型在首次启动时从 Hugging Face 下载，冷加载约 60 秒；
加载完成前 `/healthz` 返回 `status=loading`。

## 常用命令

```bash
./infra/scripts/kb sources                             # 来源与许可状态
./infra/scripts/kb sync                                # 全量重建并激活
./infra/scripts/kb sync --only kafka --mode verify     # 只建这个来源的索引验证，不激活
./infra/scripts/kb sync --only kafka --mode merge      # 以当前索引为底座换掉这个来源后激活
./infra/scripts/kb search "PostgreSQL 慢查询 执行计划"   # 检索
./infra/scripts/kb stats                               # 索引概况
./infra/scripts/kb verify-links --check-anchors        # 抽样核对引用链接与锚点是否真实存在
./infra/scripts/freeze.sh                              # 重新生成 requirements.lock.txt
```

## 测试

```bash
.venv/bin/python -m pytest tests/unit tests/retrieval -q   # 快速门禁，约 0.5s，不加载模型
.venv/bin/python -m pytest tests/ -q                       # 全量，约 12s，需要模型与 Metal
.venv/bin/python packages/evaltools/run_basic.py --check-links   # 50 题回归，约 8 分钟
```

`tests/unit` 与 `tests/retrieval` 刻意不依赖 MLX 与 Metal，可以在任何机器上跑。

## 目录结构

| 路径 | 内容 |
| --- | --- |
| `apps/` | PWA 客户端、API 网关与 `kb` 命令行 |
| `services/` | 转写、编排、检索、推理、同步服务 |
| `packages/` | 共享类型、提示词模板、评测工具 |
| `knowledge/` | 来源注册表、术语映射、回归查询、场景卡片、评测集 |
| `infra/` | 本地启动、配置模板、运维脚本 |
| `bench/` | Mac 本机性能基准脚本与报告 |
| `docs/` | 使用、运维与故障排查手册 |
| `tests/` | 单元、集成、端到端、检索与性能测试 |

## 知识来源

五个入库来源全部固定在发布版本上——引用显示的版本必须和用户点开的页面是同一版本。

| 来源 | 版本 | 许可 |
| --- | --- | --- |
| PostgreSQL | REL_17_STABLE | PostgreSQL |
| Kubernetes | release-1.36 | CC-BY-4.0 |
| Kafka | 4.3.1 | Apache-2.0 |
| Spring Boot | v4.1.1 | Apache-2.0 |
| Spring Data Redis | 4.1.1 | Apache-2.0 |

许可在同步时从仓库内的许可文件实测校验，不信注册表里的声明。
Redis 官方文档实测为 CC-BY-NC-SA-4.0（禁止商用），正文不入库，只做链接检索。

## 不入版本控制

模型权重、向量索引、原始语料、音频、日志、凭据，以及本项目的 AI 开发文档。详见 `.gitignore`。
