# T-023 R187：英文独有 7 条失败的只读归因

R184/R185 把中英两臂拉到同一实现、同一索引各跑一遍 50 题：中文 44/50、英文 41/50，
失败集合只重叠 2 条。中文侧的失败有 R150/R154 两份审计，**英文独有的 7 条一直没做归因**。
这份审计补上它，并连带把两臂共有的 2 条、中文独有的 4 条一起放进同一张表——
只看英文那 7 条，会把"英文更差"当成前提而不是结论。

## 一、身份与方法

| 项 | 值 |
| --- | --- |
| 中文报告 | `bench/reports/eval-basic-20260920T055551Z.json` |
| 英文报告 | `bench/reports/eval-basic-20260920T055038Z.json` |
| 两份报告的实现 commit | `0d2f5037fe11adc0a7d66147ca8aea31239da28f`（同一个） |
| 索引指纹 | `fe11b6b41ded3d60f422768727c67a9aefe4dac8ebc83dac625efc6b1d176305`（17080 块） |
| 嵌入模型 / 词典 / 模板 | `mlx-community/bge-m3-mlx-8bit` / `eba42d53742f` / `07e2d41f1aeb` |
| 机读产物 | `bench/audits/t023-r187-english-only-failure-attribution.json` |
| 判别性实验 | `bench/audits/t023-r187-mutation-discriminativeness.json` |

复跑（前四项身份逐项相同才继续，不同即中止）：

```bash
.venv/bin/python bench/bench_language_failure_attribution.py \
    --zh-report bench/reports/eval-basic-20260920T055551Z.json \
    --en-report bench/reports/eval-basic-20260920T055038Z.json \
    --json bench/audits/t023-r187-english-only-failure-attribution.json \
    --expect-index-fingerprint fe11b6b41ded3d60f422768727c67a9aefe4dac8ebc83dac625efc6b1d176305 \
    --probe-retrieval        # 这一项需要 Metal；不带它其余部分照常产出
```

**读到的都是当时那条证据**：13 道题两臂共 124 条 `selected_evidence`，逐条按 `chunk_id`
回查索引，要求 `source_url` 与报告一致、正文 SHA-256 与报告登记的 `text_sha256` 一致；
任一条对不上即中止、不读正文（R154 的口径）。124/124 通过。

**脚本只产出机械信号，"原因"是本文件里的人工判断**（§5.3）。三个信号必须分开读：

- `in_answer`：判据正则在**答案文本**上是否匹配；
- `in_selected_evidence`：同一条正则在**实际进入 prompt 的那 5 块**上匹配了哪几块；
- `in_corpus_chunk_count`：同一条正则在**整个 17080 块索引**里能匹配多少块。

三者分开，才能把"语料里就没有"、"语料里有但这次没取到"、"取到了但模型没写"
分成三件事；只看进入 prompt 的那 5 块，它们长得完全一样。

## 二、先看整臂：44 vs 41 的差额不在 keypoint 上

| 整臂统计（各 n=1） | 中文 | 英文 |
| --- | ---: | ---: |
| 失败题数 | 6 | 9 |
| 其中 keypoint 未命中 | 6 | 6 |
| 其中来源不符 | 0 | 1 |
| 其中该拒答而未拒答 | 0 | 1 |
| 其中正文无证据编号 | 0 | 1 |
| 登记 keypoint 总条数 / 未命中 | 44 / 6 | 44 / 6 |
| 已答题答案长度中位数（字符） | 268 | 485 |
| `expand_terms()` 在题面上展开非空 | 28/50 | **0/50** |

**两臂各漏 6 条 keypoint，条数相同、题目不同**；44→41 的全部差额来自另外三类各 1 条。
因此"英文 keypoint 命中率更差"这个说法在这次数据上**不成立**——它错的地方在于
把三类失败合并成一个分数看。

`expand_terms()` 的 0/50 是结构性事实：`knowledge/term_map.yaml` 的键全是中文词，
英文题面一个也匹配不到，**整条查询扩展链路在英文侧不工作**。
但它是否就是英文侧取错块的原因，本轮探针给出的是**否定**的证据，见 §三·5。

`detect_technology()` 两臂只有 2 道不同（`k8s-eviction` 与 `postgres-explain-analyze`，
中文 `None`、英文分别认出 `kubernetes`/`postgresql`），**方向对英文有利**，
不构成英文侧的劣势来源。

## 三、英文独有的 7 条，逐条

### 1. `postgres-btree-index` — 检索名次：同一判据的证据块中文第 1、英文第 14

| 机械事实 | 值 |
| --- | --- |
| 两臂证据交集 | 2/5（`#8146`、`#12024`） |
| 判据在证据里 | 中文 `#10937`；英文 **无** |
| 判据在语料里 | 2610 块（该正则含 `function`，覆盖面很宽） |
| 匹配块的最好名次（top-50） | 中文 **1**（`#10937`）、英文 **14**（`#7839`） |

中文臂的 5 块里有 2 块是 `Indexes › Examining Index Usage`（`#10935`/`#10937`，
讲成本估算不准、统计信息过时），英文臂一块都没有——它拿到的是
`B-Tree Indexes › Implementation`、`Index Types` 三块与 `Locking and Indexes`，
讲的是 B-Tree 支持哪些查询、并发下的表现。
答案随证据走：中文写"统计信息/选择性"，英文写"非锚定 LIKE 模式与非标量类型"——
**两个答案对各自的证据都是忠实的**，只是英文那一份不含登记的那条事实。

**判断：检索侧。** 这是 7 条里唯一一条"该出现的块确实存在、只是在英文题面下排到了
候选窗口之外"的题，也是 7 条里最像"英文检索更弱"的一条。
名次 1 vs 14 是单次测量，不足以据此调任何共享参数。

### 2. `postgres-connections` — 两臂都没取到，中文靠模型先验写出而通过

| 机械事实 | 值 |
| --- | --- |
| 两臂证据交集 | 2/5 |
| 判据 `max_connections\|最大连接` 在证据里 | **两臂都无** |
| 判据在语料里 | 12 块，含 `Config › Connections and Authentication › Connection Settings`（`#8484`/`#8485`）——正是该问题对应的那一节 |
| 匹配块的最好名次 | 中文 **9**、英文 **20** |

两臂选中的 5 块都没有 `max_connections`：中文拿到 `Shared Memory and Semaphores`、
`SSL`、`Standby Servers`、`Parameter Interaction`；英文拿到 `Resource Limits`、
`WAL Configuration`、`Role Attributes`、`SSL`、`Standby Servers`。
中文答案照样写出"连接数上限由 `max_connections` 参数配置 [5]"，而它标的 [5] 是
`Parameter Interaction via the Configuration File`——那一块并不含这个参数名。
英文答案写的是 `CREATE ROLE ... CONNECTION LIMIT`，**有证据**（`#14837 Role Attributes`），
只是不是判据登记的那条事实。

**判断：这道题两臂都是检索缺口，只有英文臂被判出来。** 中文臂的"通过"来自
模型先验而非证据，按产品的引用契约那是一条无证据支撑的论断。
**把它记成"英文更差"是误读**——它测出来的是判据在中文侧恰好被先验补上了。

### 3. `postgres-partitioning` — 判据的中英备选项不等价

| 机械事实 | 值 |
| --- | --- |
| 判据 | `(?i)(PARTITION BY\|分区键\|分区表)` |
| 判据匹配**中文题面自身** | **是**（题面即"PostgreSQL 分区表怎么创建和使用"） |
| 判据在证据里 | 两臂都无 |
| 判据在语料里 | 23 块（`Table Partitioning › Overview`、`… › Example` 等），最好名次 中文 23、英文 22 |

中文备选项 `分区表`/`分区键` 是普通名词，**任何一份切题的中文答案都会写到**，
判据甚至在题面自己身上就匹配；英文备选项却只收 SQL 字面量 `PARTITION BY`。
英文答案写的是 "partition key column(s)"、"`CREATE TABLE ... PARTITION OF`"，
与证据 `#9245` 的原文用词一致（那一块写的正是 "the partition key"），
但它不含 `PARTITION BY` 这个串。

**判断：判据侧。** 同一条 keypoint 的两种语言备选项宽严不对等——中文收概念词、
英文收字面量。附带一条检索事实：含 `PARTITION BY` 的块（`#9243`、`#9249` 等）
与被选中的块同属 `Table Partitioning` 一节、就在隔壁，两臂都没取到。

### 4. `postgres-seq-scan` — 证据几乎相同，差在答案措辞

| 机械事实 | 值 |
| --- | --- |
| 两臂证据交集 | **4/5** |
| 判据在证据里 | 两臂都无；语料里 17 块，两臂 top-50 内**都没有** |
| 判据在答案里 | 中文命中（两词紧邻，窗口 0）；英文**任何距离上都没同时出现** |

两臂拿到的是同一批 `How the Planner Uses Statistics › Row Estimation Examples` 的
`EXPLAIN` 输出块。中文答案的"风险"段写了"索引选择性差"、"统计信息失效"，
英文答案只写了"建索引、必要时建部分索引"。

**判断：生成侧的措辞差异，叠加判据不可从证据满足。** 证据基本相同，
所以这条与检索无关；而判据要的"索引—统计/选择性"关联在两臂的证据里都没有，
中文那次同样是先验补写。

### 5. `redis-cache-annotations` — 来源构成的语言差异，机制未定位

| 机械事实 | 值 |
| --- | --- |
| 两臂证据交集 | **0/5** |
| 期望来源 | `spring-data-redis`；中文引用到了，英文全部来自 `spring-boot` |
| top-10 里 `spring-data-redis` 的条数 | 中文 **6**、英文 **1**（第 9 名） |
| 英文题面 + 中文侧展开词 后再测 | **0**（比不加更差） |
| 判据 `@Cacheable\|缓存注解` 在语料里 | **1 块**，两臂 top-50 内都没有 |

英文臂的 5 条证据里还混进了一块标题为 `Redirect` 的页面。
两份答案都写出了 `@Cacheable`/`@CacheEvict`，而这两个词在各自的证据里都没有。

**本轮否定了一个看起来很顺的解释**：既然 `expand_terms()` 在英文侧 0/50，
那把中文侧得到的展开词（`Cacheable`、`CacheEvict`、`cache annotation`、`cache manager`）
拼进英文题面，应当能把 `spring-data-redis` 拉回来——**实测没有**，
top-10 里该来源从 1 条变成 0 条。所以"缺展开词"不足以解释这道题，
差异来自整个查询表示（两路都变），**具体机制本轮未定位**。

**判断：检索侧，机制待查。** 这是 7 条里唯一一条真正的来源错误，
也是最值得继续追的一条。

### 6. `refuse-scifi-films` — R184 已定位的充分性档位

英文题面 `top_distance=0.7171 < sufficient_distance=0.72`，落在 `SUFFICIENT` 带，
而 `must_refuse_limited_out_of_scope()` 要求 `LIMITED` 带才触发，
**防线结构上够不到**（R184 记录，R185 的 `bench_refusal_band.py` 复跑一致）。
本轮新增的旁证：英文臂给这道题的 5 条证据全部来自 `postgresql`，
中文臂一条证据都没有。

**判断：拒答带的语言不对称，已有结论，本轮不重复归因，也未动 0.72/0.76。**

### 7. `spring-prometheus` — 内容对、证据对，就是没标编号

| 机械事实 | 值 |
| --- | --- |
| 判据在证据里 | 两臂都有，且是**同一块** `#683` |
| 判据在答案里 | 两臂都命中 |
| 证据编号数 | 中文 2、英文 **0** |
| 英文臂 44 道已答题里 `cited=0` 的 | **1 道**（就是这一道）；中文 0 道 |

英文答案内容正确（写了 `/actuator/prometheus`、默认不暴露、`scrape_config`），
证据也确实含这条事实，只是正文里一个 `[n]` 都没有，被"给出技术内容但未标注任何
证据编号"判失败。两版系统提示词都写了这条要求，中文那句用 `**` 加粗、
多一句"没有编号的论断视为无效"。

**判断：生成侧的格式遵从，n=1。** 一道题不足以说英文提示词的引用约束系统性更弱；
要下这个结论得跑多次或多题。**本轮不改提示词。**

## 四、顺带查出来的两条判据缺陷（不属于语言问题）

| 题 | 事实 |
| --- | --- |
| `spring-graceful-shutdown`（两臂共有失败） | 判据要求 `graceful shutdown` 与 `application context` 40 字内相邻，**整个 17080 块语料里 0 块能匹配**。这道题从证据出发**永远通不过**，与语言无关 |
| `spring-auto-configuration`（中文独有失败） | 判据同样 **0 块**可匹配；英文臂之所以"通过"，是模型自己写出了 `@Conditional`—`configuration` 的搭配 |

这两条不是本轮要归因的对象，但它们说明：**当前有 keypoint 在衡量"模型先验是否
恰好写出某个词串"，而不是"答案是否被证据支持"**。§三·2、3、4 三道英文独有失败
也落在同一个形状里。

另一条与语言相关、但本轮只出现一次的机械事实：`k8s-eviction` 英文答案写了
"pressure conditions defined by the kubelet, including memory"，两词相距 **46 字符**，
登记窗口是 40——**放宽窗口即可命中**。全部 13 道失败题里只有这一处属于
"窗口卡住"，且它在两臂共有失败里，不在英文独有的 7 条中。
英文答案长度中位数 485 字符、中文 268，**以字符计的邻近窗口对两种语言不等价**
这个担心是合理的，但这次数据只支持 1 例，不足以据此改窗口。

## 五、结论与不下的结论

**归因结果（7 条）**：

| 归因 | 条数 | 题 |
| --- | ---: | --- |
| 检索侧：证据块存在但英文名次靠后 | 1 | `postgres-btree-index` |
| 检索侧：来源构成不同，机制未定位 | 1 | `redis-cache-annotations` |
| 判据要的事实两臂都没取到，中文靠模型先验通过 | 2 | `postgres-connections`、`postgres-seq-scan` |
| 判据的中英备选项不等价 | 1 | `postgres-partitioning` |
| 拒答带（R184 已定位） | 1 | `refuse-scifi-films` |
| 生成侧未标证据编号 | 1 | `spring-prometheus` |

**不下的结论**：

1. **不说"英文检索系统性更差"**：两臂 keypoint 漏的条数相同；名次证据只有
   `postgres-btree-index` 一例（1 vs 14），`detect_technology()` 的两处差异反而对英文有利。
2. **不说"`term_map` 在英文侧失效导致了这些失败"**：0/50 是事实，但本轮探针
   在最像的那道题上**证伪**了这个因果。
3. **不改任何东西**：本轮未动阈值、题库、`term_map`、提示词与门禁基线。
   §三·3、§四指出的判据问题要改题库，那是独立决定，需要审查方复核后另起一轮。
4. **两臂各 n=1**。失败集合与 R184 逐题相同（两次不同 commit、不同时间）是很强的
   可复现旁证，但严格说仍是单次运行。
