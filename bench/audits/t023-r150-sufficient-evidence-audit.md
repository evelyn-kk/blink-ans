# T-023 R150：7 条 `sufficient` keypoint 失败的只读证据审计

## 可复现身份与边界

- 报告：`bench/reports/eval-basic-20260916T062018Z.json`（该报告被忽略；本审计只读取它）。
- 报告运行期实现：`5b828106f2dde13e0e490aabbef7b408db35531c`。审计产物提交
  `3b7ff6d3a775fea0be46f1fb566d25b988d2bb29` 是该实现的后代（可用
  `git merge-base --is-ancestor 5b82810 3b7ff6d` 复核），**不是同一 HEAD**；没有倒填旧报告。
- 报告/index 身份：`index_chunks=17080`、`dictionary_version=eba42d53742f`、
  `template_version=b89dcecf0dcd`、`offline_mode=true`、`served_by={local:47, policy:3}`。
- DB：`data/index/current.db`。只读 SQL：
  `SELECT id,title_path,text FROM chunks WHERE source_url=? ORDER BY id`；未调用 Embedder、检索、
  Router 或 LLM，也没有改写 DB、报告或代码。
- **留证缺口**：R149 报告的 `run_basic.run_case()` 只保存 `sources` 数量与每个 source item 的
  `url`，没有保存 item 的 `chunk_id`/`citation`/evidence text。虽然生产 `sources` 事件有
  `chunk_id`，该值在评测报告生成时已被丢弃。因此 URL 对应多行时，不能从本报告唯一恢复当时
  进入 prompt 的 rowid；下表绝不从同 URL 的候选行猜一行。

`直接含目标事实` 的 `no` 是对已唯一定位的完整 chunk 应用报告中登记的 keypoint 正则；
`unknown` 意味着实际 rowid 未留存，不是“该 URL 的任意候选行都不含”。短摘录只证明对应的唯一
chunk 内容，不能替代未留存的实际 prompt 证据。

## 逐题审计

### `k8s-eviction`

预期：`pressure` 与 `memory/disk` 在 40 字内；报告答案遗漏该 keypoint。

|报告 evidence 位置|URL|实际 rowid / 标题|直接含目标事实|
|---:|---|---|---|
|1|`https://v1-36.docs.kubernetes.io/docs/concepts/workloads/pods/disruptions/#pod-disruption-budgets`|unknown；候选 `5388,5389,5390`，均为 `Disruptions › Pod disruption budgets`|unknown|
|2|`https://v1-36.docs.kubernetes.io/docs/concepts/scheduling-eviction/api-eviction/`|`3902`，`API-initiated Eviction`|no；“request eviction … `kubectl drain`”|
|3|`https://v1-36.docs.kubernetes.io/docs/tasks/administer-cluster/safely-drain-node/#eviction-api`|`6398`，`Safely Drain a Node › The Eviction API`|no；“finer control over the pod eviction process”|
|4|`https://v1-36.docs.kubernetes.io/docs/concepts/workloads/pods/pod-qos/#burstable`|`5516`，`Pod Quality of Service Classes › … › Burstable`|no；“lower-bound resource guarantees … request”|
|5|`https://v1-36.docs.kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/#pod-lifetime`|unknown；候选 `5451,5452`，均为 `Pod Lifecycle › Pod lifetime`|unknown|

结论：**证据不支持/不确定**。两条实际 evidence 的 rowid 缺失；其余三个唯一块未命中登记事实，
不能据 `sufficient` 推断为生成遗漏。

### `k8s-hpa`

预期：`metric/指标` 与 `target/目标` 在 40 字内；报告答案遗漏该 keypoint。

|位置|URL|实际 rowid / 标题|直接含目标事实|
|---:|---|---|---|
|1|`https://v1-36.docs.kubernetes.io/docs/concepts/workloads/autoscaling/#scaling-workloads-horizontally`|`5067`，`Autoscaling Workloads › … › Scaling workloads horizontally`|no；“automatically scale a workload horizontally … HPA”|
|2|`https://v1-36.docs.kubernetes.io/docs/concepts/workloads/autoscaling/horizontal-pod-autoscale/#support-for-horizontalpodautoscaler-in-kubectl`|`5041`，`Horizontal Pod Autoscaling › Support … in kubectl`|no；“create … `kubectl create`”|
|3|`https://v1-36.docs.kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale-walkthrough/#creating-the-autoscaler-declaratively`|`7647`，`… › Creating the autoscaler declaratively`|no；“Instead of `kubectl autoscale` … manifest”|
|4|`https://v1-36.docs.kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale-walkthrough/#run-and-expose-php-apache-server`|`7630`，`… › Run and expose php-apache server`|no；“start a Deployment … expose it”|
|5|`https://v1-36.docs.kubernetes.io/docs/concepts/workloads/autoscaling/horizontal-pod-autoscale/#scaling-on-custom-metrics`|`5028`，`Horizontal Pod Autoscaling › Scaling on custom metrics`|no；“configure … scale based on a custom metric”|

结论：**证据不支持/不确定**。五条 URL 均唯一定位，且完整 chunk 都不满足登记的 metric–target
邻近事实；这是 prompt 证据不足的直接旁证，但不把 `sufficient` 标签当成其因果解释。

### `k8s-pod-lifecycle`

预期：`Pending/Running/Succeeded/Failed` 等 phase；报告答案遗漏该 keypoint。

|位置|URL|实际 rowid / 标题|直接含目标事实|
|---:|---|---|---|
|1|`https://v1-36.docs.kubernetes.io/docs/concepts/workloads/pods/pod-condition/#built-in-pod-conditions`|`5429`，`Pod Conditions › Built-in Pod conditions`|正则命中但**非目标 phase**：`PodResizePending`；不能算 phase 事实支持|
|2|`https://v1-36.docs.kubernetes.io/docs/concepts/services-networking/windows-networking/#network-modes`|unknown；候选 `4685–4690`，`Networking on Windows › Network modes`|unknown|
|3|`https://v1-36.docs.kubernetes.io/docs/concepts/workloads/pods/pod-condition/#other-pod-conditions`|`5434`，`Pod Conditions › Other Pod conditions`|no；“not part of the normal Pod lifecycle progression”|
|4|`https://v1-36.docs.kubernetes.io/docs/tasks/job/pod-failure-policy/#usage-scenarios`|`7373`，`… › Usage scenarios`|no；“Avoiding unnecessary Pod retries”|
|5|`https://v1-36.docs.kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/#pod-phase`|unknown；候选 `5456–5459`，`Pod Lifecycle › Pod phase`|unknown（候选包含 row `5457`，但不能断言它被选）|

结论：**证据不支持/不确定**。`PodResizePending` 显示该宽 OR 正则本身可假命中；真正 phase URL
的实际 rowid 没有留证，不能判为“证据已支持但生成遗漏”。

### `postgres-seq-scan`

预期：index 与 selectivity/statistics；报告答案遗漏该 keypoint。

|位置|URL|实际 rowid / 标题|直接含目标事实|
|---:|---|---|---|
|1–3|`https://www.postgresql.org/docs/17/row-estimation-examples.html#ROW-ESTIMATION-EXAMPLES`|unknown；每项候选 `12335–12378`（44 行），`Planstats › … › Row Estimation Examples`|unknown|
|4|`https://www.postgresql.org/docs/17/using-explain.html#USING-EXPLAIN-BASICS`|unknown；候选 `12098–12117`（20 行），`Perform › … › EXPLAIN Basics`|unknown|
|5|`https://www.postgresql.org/docs/17/sepgsql.html#SEPGSQL-REGRESSION`|unknown；候选 `14119–14121`，`Sepgsql › … › Regression Tests`|unknown|

结论：**证据不支持/不确定**。五个 selected-rowid 均不可从 URL 反推，不能用该来源页里存在的
其他 chunk 推断 prompt 已含 selectivity/statistics。

### `postgres-explain-analyze`

预期：actual 与 time/rows；报告答案遗漏该 keypoint。

|位置|URL|实际 rowid / 标题|直接含目标事实|
|---:|---|---|---|
|1、3、4|`https://www.postgresql.org/docs/17/fdw-callbacks.html#FDW-CALLBACKS-EXPLAIN`|unknown；每项候选 `9921–9924`，`Fdwhandler › … › FDW Routines for EXPLAIN`|unknown|
|2|`https://www.postgresql.org/docs/17/auto-explain.html#AUTO-EXPLAIN-CONFIGURATION-PARAMETERS`|unknown；候选 `7953–7959,17071`，含 `Auto Explain › …` 与场景卡片标题|unknown|
|5|`https://www.postgresql.org/docs/17/rules-materializedviews.html#RULES-MATERIALIZEDVIEWS`|unknown；候选 `13885–13901`，`Rules › … › Materialized Views`|unknown|

结论：**证据不支持/不确定**。所有实际 rowid 未留存，不能声称 `actual time/rows` 已进入 prompt，
也不能把来源页候选当作实际选中证据。

### `postgres-btree-index`

预期：selectivity/function/cast；报告答案遗漏该 keypoint。

|位置|URL|实际 rowid / 标题|直接含目标事实|
|---:|---|---|---|
|1、4|`https://www.postgresql.org/docs/17/indexes-examine.html#INDEXES-EXAMINE`|unknown；每项候选 `10935–10937`，`Indices › Indexes › Examining Index Usage`|unknown|
|2|`https://www.postgresql.org/docs/17/bloom.html#BLOOM-EXAMPLES`|unknown；候选 `8085–8094`，`Bloom › … › Examples`|unknown|
|3|`https://www.postgresql.org/docs/17/btree.html#BTREE-IMPLEMENTATION`|`8146`，`Btree › B-Tree Indexes › Implementation`|no；“implementation details … advanced users”|
|5|`https://www.postgresql.org/docs/17/locking-indexes.html#LOCKING-INDEXES`|unknown；候选 `12023,12024`，`Mvcc › … › Locking and Indexes`|unknown|

结论：**证据不支持/不确定**。唯一可定位块不含登记事实，其他 evidence 行不能唯一恢复。

### `spring-graceful-shutdown`

预期：graceful shutdown 与 application context；报告答案遗漏该 keypoint。

|位置|URL|实际 rowid / 标题|直接含目标事实|
|---:|---|---|---|
|1|`https://docs.spring.io/spring-boot/4.1/reference/web/graceful-shutdown.html#web.graceful-shutdown.disabling-graceful-shutdown`|`1547`，`Graceful Shutdown › Disabling Graceful Shutdown`|no；“`server.shutdown: immediate`”|
|2|`https://docs.spring.io/spring-boot/4.1/reference/web/graceful-shutdown.html`|`1545`，`Graceful Shutdown`|正则 no；但直接短证据为“Graceful shutdown … occurs as part of closing the application context”，语义上相关，不能仅以正则 no 判无支持|
|3|`https://docs.spring.io/spring-boot/4.1/how-to/deployment/cloud.html#howto.deployment.cloud.kubernetes.container-lifecycle`|unknown；候选 `385,386`，`Cloud › … › Kubernetes Container Lifecycle`|unknown|
|4|`https://docs.spring.io/spring-boot/4.1/reference/web/reactive.html`|`1548`，`Reactive › Reactive Web Applications`|no；“auto-configuration for Spring Webflux”|
|5|`https://docs.spring.io/spring-boot/4.1/reference/using/structuring-your-code.html`|`1539`，`Structuring Your Code`|no；“does not require any specific code layout”|

结论：**证据不支持/不确定**。row `1545` 是“证据可能支持而答案遗漏”的旁证，但第三项实际 rowid
缺失，故不能把整题归类为确定的生成遗漏。

## 总结与下一步

所有 7 条报告 case 都有 `sufficiency=sufficient`、5 sources、`retrieval_miss=false`，但这些字段只描述
当时的判定/结果，**不是**“关键事实已在 prompt”的因果证明。HPA 的五条 URL 唯一块均未命中登记事实；
其余六题至少一条 evidence URL 映射多行，无法从报告恢复完整 selected rowid 集合。故本审计没有把任何题
定性为“证据已支持但生成遗漏”；均暂列“证据不支持/不确定”，其中 Spring `1545` 是需在未来带 rowid 的
评测留证中复核的相关旁证。

若后续要区分检索/证据选择/生成，应使评测报告保留每条 `sources` item 的 `chunk_id`、citation 和实际
evidence text（或不可变摘要）；在相同实现/索引身份下再审计。该建议不在 R150 的只读范围内。
