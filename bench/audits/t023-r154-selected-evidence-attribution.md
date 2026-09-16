# T-023 R154：R153 实际 selected evidence 的只读归因审计

## 范围与身份

- 运行报告：`bench/reports/eval-basic-20260916T064408Z.json`（M4、n=1、中文、`--offline --check-links`）。
- 报告运行期实现：`704a3e8b7b3f1c7a326fabb17a18081cfd4a6f95`；报告 index=17080、词典
  `eba42d53742f`。审计时只读打开的 `data/index/current.db` 同为 17080/`eba42d53742f`。
- 方法：对 7 个失败 case 的每一条 `selected_evidence`，以报告的 `chunk_id` 查询 DB；严格要求
  DB `source_url == report.url` 且 `SHA-256(db.text.encode("utf-8")) == report.text_sha256`，任一缺行、URL
  或 hash 不同即停止、不读取正文。35/35 均通过。下面每条的 citation、URL、chunk id 来自报告事件原序；
  `hash` 已逐条按上述完整 64 位值核对（报告保存完整值），正文不复制到此审计。
- “直接支持”只说实际进入 prompt 的块是否包含登记 keypoint 所需事实；它不由 `sufficient` 推导，也不把
  DB 中未选的其他块拿来补。答案是否写入则单独按 R153 JSON 的 `answer_text` 和同一条登记正则检查。

## 逐题结果

### `k8s-eviction`

目标是 `pressure` 与 `memory/disk` 的关联；答案未写入该关联。已选块只有 #5516 的 “Node resource
pressure”，没有 memory/disk，故**证据不支持：检索/选证据缺口**，不能称生成遗漏。

| chunk | citation / URL | 短旁证与结论 |
| --- | --- | --- |
| 5390 | Kubernetes · Pod disruption budgets · [URL](https://v1-36.docs.kubernetes.io/docs/concepts/workloads/pods/disruptions/#pod-disruption-budgets) | PDB/health/drain；目标词缺失。 |
| 3902 | Kubernetes · API-initiated Eviction · [URL](https://v1-36.docs.kubernetes.io/docs/concepts/scheduling-eviction/api-eviction/) | Eviction API/PDB/grace period；目标词缺失。 |
| 6398 | Kubernetes · Safely Drain a Node › The Eviction API · [URL](https://v1-36.docs.kubernetes.io/docs/tasks/administer-cluster/safely-drain-node/#eviction-api) | 程序化 eviction/PDB；目标词缺失。 |
| 5516 | Kubernetes · Pod QoS › Burstable · [URL](https://v1-36.docs.kubernetes.io/docs/concepts/workloads/pod-qos/#burstable) | “eviction due to Node resource pressure”，无 memory/disk。 |
| 5452 | Kubernetes · Pod Lifecycle › Pod lifetime · [URL](https://v1-36.docs.kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/#pod-lifetime) | scheduling gates；目标词缺失。 |

### `k8s-hpa`

已选证据跨块给出 custom `metric`（#5028）和 CPU utilization `target`（#5041）；答案也写了“指标”和
“CPU 利用率目标”。登记正则要求二词 40 字以内，因此本次未命中，但这不是“证据不支持”或“答案遗漏”的
因果结论，而是**当前文字距离判据未捕获已表达的事实**。

| chunk | citation / URL | 短旁证与结论 |
| --- | --- | --- |
| 5067 | Kubernetes · Autoscaling Workloads · [URL](https://v1-36.docs.kubernetes.io/docs/concepts/workloads/autoscaling/#scaling-workloads-horizontally) | HPA 按 CPU/memory observed utilization 调整副本。 |
| 5041 | Kubernetes · HPA in kubectl · [URL](https://v1-36.docs.kubernetes.io/docs/concepts/workloads/autoscaling/horizontal-pod-autoscale/#support-for-horizontalpodautoscaler-in-kubectl) | `--cpu=80%` 创建 target CPU utilization=80%。 |
| 7647 | Kubernetes · HPA Walkthrough · [URL](https://v1-36.docs.kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale-walkthrough/#creating-the-autoscaler-declaratively) | 声明式创建；不单独给目标关联。 |
| 7630 | Kubernetes · HPA Walkthrough · [URL](https://v1-36.docs.kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale-walkthrough/#run-and-expose-php-apache-server) | 示例 Deployment；不单独给目标关联。 |
| 5028 | Kubernetes · Scaling on custom metrics · [URL](https://v1-36.docs.kubernetes.io/docs/concepts/workloads/autoscaling/horizontal-pod-autoscale/#scaling-on-custom-metrics) | `autoscaling/v2` 可按 custom metric 缩放。 |

### `k8s-pod-lifecycle`

目标是 Pod phase 的 `Pending/Running/Succeeded/Failed`。答案只列 conditions；实际选中 #5429 的
`PodResizePending` 会碰巧匹配 `Pending` 子串，但不是目标 phase，#5456 只说明 phase 是高阶摘要而未列值。
故目标事实未被实际证据直接支持，答案也未写入，属**检索/选证据缺口**。

| chunk | citation / URL | 短旁证与结论 |
| --- | --- | --- |
| 5429 | Kubernetes · Pod Conditions · [URL](https://v1-36.docs.kubernetes.io/docs/concepts/workloads/pods/pod-condition/#built-in-pod-conditions) | `PodScheduled`/`Ready` conditions；`PodResizePending` 非 phase。 |
| 4690 | Kubernetes · Networking on Windows · [URL](https://v1-36.docs.kubernetes.io/docs/concepts/services-networking/windows-networking/#network-modes) | 网络流；目标缺失。 |
| 5434 | Kubernetes · Pod Conditions › Other conditions · [URL](https://v1-36.docs.kubernetes.io/docs/concepts/workloads/pods/pod-condition/#other-pod-conditions) | 非 normal lifecycle progression；目标 phase 缺失。 |
| 7373 | Kubernetes · Pod failure policy · [URL](https://v1-36.docs.kubernetes.io/docs/tasks/job/pod-failure-policy/#usage-scenarios) | Job failure policy；目标缺失。 |
| 5456 | Kubernetes · Pod Lifecycle › Pod phase · [URL](https://v1-36.docs.kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/#pod-phase) | phase 是高阶摘要；本块未列目标四个值。 |

### `postgres-seq-scan`

目标为 index 与 selectivity/statistics 的关联。已选块是 Seq Scan/EXPLAIN 示例或无关 SE-PostgreSQL；没有这项
关联，答案虽建议索引却也未写入 selectivity/statistics。属**检索/选证据缺口**。

| chunk | citation / URL | 短旁证与结论 |
| --- | --- | --- |
| 12336 | PostgreSQL · Row Estimation Examples · [URL](https://www.postgresql.org/docs/17/row-estimation-examples.html#ROW-ESTIMATION-EXAMPLES) | `EXPLAIN SELECT *` 的 Seq Scan；关联缺失。 |
| 12353 | PostgreSQL · Row Estimation Examples · [URL](https://www.postgresql.org/docs/17/row-estimation-examples.html#ROW-ESTIMATION-EXAMPLES) | `stringu1='xxx'` 的 Seq Scan；关联缺失。 |
| 12348 | PostgreSQL · Row Estimation Examples · [URL](https://www.postgresql.org/docs/17/row-estimation-examples.html#ROW-ESTIMATION-EXAMPLES) | `stringu1='CRAAAA'` 的 Seq Scan；关联缺失。 |
| 12101 | PostgreSQL · EXPLAIN Basics · [URL](https://www.postgresql.org/docs/17/using-explain.html#USING-EXPLAIN-BASICS) | Seq Scan cost 示例；关联缺失。 |
| 14121 | PostgreSQL · SE-PostgreSQL Regression · [URL](https://www.postgresql.org/docs/17/sepgsql.html#SEPGSQL-REGRESSION) | SELinux regression；目标缺失。 |

### `postgres-explain-analyze`

目标为 actual 与 time/rows 的同一事实。实际证据提到 EXPLAIN ANALYZE 的 run-time statistics/per-plan-node
timing（#9924/#9923/#9921）及 auto_explain timing（#7954），但未直接给出 actual time/rows；答案也未写
actual/rows。故对登记的具体 keypoint 是**检索/选证据缺口**，不以 `sufficient` 推定生成原因。

| chunk | citation / URL | 短旁证与结论 |
| --- | --- | --- |
| 9924 | PostgreSQL · FDW Routines for EXPLAIN · [URL](https://www.postgresql.org/docs/17/fdw-callbacks.html#FDW-CALLBACKS-EXPLAIN) | EXPLAIN ANALYZE run-time statistics；actual time/rows 缺失。 |
| 7954 | PostgreSQL · auto_explain parameters · [URL](https://www.postgresql.org/docs/17/auto-explain.html#AUTO-EXPLAIN-CONFIGURATION-PARAMETERS) | per-plan-node timing 代价；actual/rows 缺失。 |
| 9923 | PostgreSQL · FDW Routines for EXPLAIN · [URL](https://www.postgresql.org/docs/17/fdw-callbacks.html#FDW-CALLBACKS-EXPLAIN) | run-time statistics；actual time/rows 缺失。 |
| 9921 | PostgreSQL · FDW Routines for EXPLAIN · [URL](https://www.postgresql.org/docs/17/fdw-callbacks.html#FDW-CALLBACKS-EXPLAIN) | run-time statistics；actual time/rows 缺失。 |
| 13894 | PostgreSQL · Materialized Views · [URL](https://www.postgresql.org/docs/17/rules-materializedviews.html#RULES-MATERIALIZEDVIEWS) | 仅引出 EXPLAIN ANALYZE；目标缺失。 |

### `postgres-btree-index`

#10937 直接说 plan-node cost 乘以 `selectivity estimate`；这是实际 prompt 证据对目标的直接支持。答案写了
统计信息/成本估算，却没有 selectivity/function/cast 的登记 keypoint，故是**生成遗漏**（不说明模型为什么遗漏）。

| chunk | citation / URL | 短旁证与结论 |
| --- | --- | --- |
| 10937 | PostgreSQL · Examining Index Usage · [URL](https://www.postgresql.org/docs/17/indexes-examine.html#INDEXES-EXAMINE) | “per-row costs … times the selectivity estimate”；直接支持。 |
| 8090 | PostgreSQL · Bloom Examples · [URL](https://www.postgresql.org/docs/17/bloom.html#BLOOM-EXAMPLES) | bloom vs btree 示例；目标词缺失。 |
| 8146 | PostgreSQL · B-Tree Implementation · [URL](https://www.postgresql.org/docs/17/btree.html#BTREE-IMPLEMENTATION) | 实现说明；目标词缺失。 |
| 10935 | PostgreSQL · Examining Index Usage · [URL](https://www.postgresql.org/docs/17/indexes-examine.html#INDEXES-EXAMINE) | index usage 概览；目标词缺失。 |
| 12024 | PostgreSQL · Locking and Indexes · [URL](https://www.postgresql.org/docs/17/locking-indexes.html#LOCKING-INDEXES) | B-tree 与非标量类型；目标词缺失。 |

### `spring-graceful-shutdown`

#1545 直接说 graceful shutdown “occurs as part of closing the application context”；目标事实在实际 prompt 中，答案
提 graceful shutdown 却未写 application context，故为**生成遗漏**。#386 也将 SIGTERM 后的 graceful shutdown
写为容器生命周期的一部分。

| chunk | citation / URL | 短旁证与结论 |
| --- | --- | --- |
| 1547 | Spring Boot · Disabling Graceful Shutdown · [URL](https://docs.spring.io/spring-boot/4.1/reference/web/graceful-shutdown.html#web.graceful-shutdown.disabling-graceful-shutdown) | `server.shutdown=immediate`；context 关联缺失。 |
| 1545 | Spring Boot · Graceful Shutdown · [URL](https://docs.spring.io/spring-boot/4.1/reference/web/graceful-shutdown.html) | “occurs as part of closing the application context”；直接支持。 |
| 386 | Spring Boot · Kubernetes Container Lifecycle · [URL](https://docs.spring.io/spring-boot/4.1/how-to/deployment/cloud.html#howto.deployment.cloud.kubernetes.container-lifecycle) | SIGTERM 后 graceful shutdown 开始。 |
| 1548 | Spring Boot · Reactive Web Applications · [URL](https://docs.spring.io/spring-boot/4.1/reference/web/reactive.html) | WebFlux auto-configuration；目标词缺失。 |
| 1539 | Spring Boot · Structuring Your Code · [URL](https://docs.spring.io/spring-boot/4.1/reference/using/structuring-your-code.html) | 代码结构建议；目标缺失。 |

## 结论与边界

- 按这次**已验证的实际 prompt 证据**：生成遗漏为 B-tree selectivity、Spring application context；证据不支持的
  是 eviction memory/disk pressure、Pod phase 值、Seq Scan 的 index-selectivity/statistics、EXPLAIN ANALYZE
  的 actual time/rows。HPA 是已选证据与答案均表达指标/目标，但当前 40 字文字距离断言未命中，不能塞进前两类。
- 这些是 R153 的单次、特定 prompt/输出归因，不是对检索、模型或充分性阈值的一般性因果结论；尤其没有用
  `sufficient` 标签推导任何原因。未运行 Embedder、Router、LLM、网络或英文评测，也未修改 DB、报告、代码、
  prompt、检索或基线。
