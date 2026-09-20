# T-023 R188：`redis-cache-annotations` 的机制定位与 `spring`+`redis` 路由收敛

R187 把英文独有的 7 条失败逐条归因，只剩一条**机制未定位**：
`redis-cache-annotations` 两臂证据零交集、英文侧一条 `spring-data-redis` 都没引用到。
R187 只证伪了一个解释（把中文侧展开词拼进英文题面反而更差），没给出正解。本轮定位到机制并做了最小修复。

## 一、机制：这条信息在两处都被丢掉了

英文题面 `How are caching annotations used with Spring Data Redis?` 的实际 FTS 查询是

```
"how" OR "caching" OR "annotations" OR "data"
```

**`spring` 与 `redis` 两个词都不在里面。** 它们被 `PROJECT_TERMS` 剔除了——那条规则本身是对的
（项目名在对应语料里几乎每块都出现，区分度为零），但它的**前提**写在注释里：
「正确用法是把它们当作元数据过滤条件（technology / project），而非检索词」。

而这道题的 `detect_technology()` 返回 **`None`**：`spring` 与 `redis` 两个域同时命中，
`_COMBO_RESOLUTIONS` 里没有这一对，于是落回「跨域提问不过滤」。
**过滤没用上，检索词也被拿掉了**——同一条信息在两条路上各丢一次。

| 实测（改前） | 中文题面 | 英文题面 |
| --- | --- | --- |
| FTS 查询 | `"data" OR "的" OR "缓存" OR "注解" OR "缓存注解" OR "怎么" OR "用" OR "cacheable" OR "cacheevict" OR "cache" OR "annotation" OR "cache annotation" OR "manager" OR "cache manager"` | `"how" OR "caching" OR "annotations" OR "data"` |
| 关键词路 30 名内的 `spring-data-redis` | 11 块（第 2、7、8、9、10、11、12…） | **0 块** |
| 向量路 30 名内的 `spring-data-redis` | 25 块 | 23 块 |
| 融合 top-10 里的 `spring-data-redis` | 6 | 1（第 9 名） |

中文臂并不是"路由对了"，它同样丢了 `spring`/`redis`——只是 `缓存注解` 这个 `term_map` 键
展开出 `cacheable` / `cache annotation` 等词，**侥幸补回了区分度**。英文侧没有任何展开
（R187 实测 `expand_terms()` 英文 0/50），于是只剩 `how / caching / annotations / data` 去打分，
`Redirect` 这类垃圾块都能排进前 5。

## 二、改动：`spring` + `redis` 收敛到 `redis` 域（一行）

```python
frozenset({"spring", "redis"}): "redis",   # services/retrieval/tokenize.py
```

与 CR-041 的 `spring`+`kafka` → `spring-kafka` 同型：本产品的 redis 语料**只有
spring-data-redis 一家**（285 块，`technology='redis'`），所以同时提到这两个词的提问
说的就是这一个库，不是跨域提问。

## 三、举证

### 1. 排序探针 13 条 + 检索验证集 9 条：逐题名次**全部不变**

| 产物 | 改前 | 改后 |
| --- | --- | --- |
| 排序探针 | `t023-r188-routing-spring-redis-ranking-before.json` | `…-ranking-after.json` |
| 检索验证集 | `…-validation-before.json` | `…-validation-after.json` |

13 条探针逐题 0 变化（含已知缺口 DLQ 7/14 与自调用第 20 名，退出码前后同为 1、退步清单逐条相同）；
9 条 held-out 逐题 0 变化（含 `Spring Data Redis 怎么用 Lua 脚本`，前后均为第 1 名）。

**这两把尺子对本改动的分辨力有限**，必须说清：它们的题面绝大多数只命中一个技术域，
根本不走「两域同时命中」那条分支。**0 变化说明的是"没有误伤已覆盖的那些题"，
不是"这个改动没有风险"**。

### 2. 端到端 50×2（同一索引指纹 `fe11b6b4…7305`，实现 `2fbd23f`）

| 臂 | 改前（`0d2f503`） | 改后（`2fbd23f`） | 失败集合变化 |
| --- | --- | --- | --- |
| 中文 | 44/50 | **44/50** | **逐题完全相同** |
| 英文 | 41/50 | **41/50** | `redis-cache-annotations` 修好；`redis-pool` 新坏 |

中文侧另有两处只在来源构成上变化、判定不变：`redis-pool` 由
`spring-data-redis + spring-framework` 变为只引 `spring-data-redis`；
`redis-cache-annotations` 由 `spring-boot + spring-data-redis` 变为只引 `spring-data-redis`。

### 3. 英文 `redis-cache-annotations`：真修好了

改后 5 条证据全部来自 `spring-data-redis`（期望来源），含 `Redis Cache`；
答案写出 `RedisCacheConfiguration`、`RedisCacheManager` 并标了编号。
改前 5 条全部来自 `spring-boot`，其中一条是 `Redirect` 垃圾块。

### 4. 英文 `redis-pool`：新失败，但它的"通过"本来就不接地

这是本轮**最需要审查方裁定**的一条，因此把话说全：

- 登记判据是 `(pool|连接池).{0,40}(Lettuce|Jedis|client)|…`。
- **改前的 5 条证据里没有一块匹配这条判据**，改后的 5 条也没有。
  改前之所以通过，是模型自己写出了 "pooling is enabled by default if `commons-pool2` is on
  the classpath" ——那句话不在它引的任何一块证据里。这与 R187 记录的
  `postgres-connections` / `postgres-seq-scan` 是同一形状：**判据奖励的是模型先验**。
- 改后模型面对确实不含该事实的证据，**选择了拒答**，但格式不合契约——
  它在 `NO_EVIDENCE` 前多写了一句前言，而 `declined()` 要求整条回复就是那一行，
  于是被判成"给出技术内容但未标注任何证据编号"。**这是本轮顺带暴露的生成侧契约缺陷**，
  与路由无关（见 §五）。
- **语料里确实有能回答它的块**：`#15256 Drivers › Configuring the Lettuce Connector`
  （`spring-data-redis`，整个语料里匹配该判据的 3 块之一）。两次运行都没把它选进证据。
  本轮改动把它在英文题面下的名次从 **第 27 名提到第 9 名**（中文题面下由第 1 名降到第 4 名）。
  也就是说，剩下的缺口是**域内排序**，不是路由。

| `#15256` 的名次（top-50） | 改前 | 改后 |
| --- | ---: | ---: |
| 英文题面 | 27 | **9** |
| 中文题面 | 1 | 4 |

### 5. 判别性

- `test_spring_and_redis_together_resolves_to_redis`（3 条参数化）在改动前的实现上**失败**，
  失败原因是行为不同（返回 `None` 而非 `redis`），不是模块不存在。
- `test_redis_alone_is_unchanged` 与 `test_redis_has_no_technology_group_on_purpose`
  新旧都通过，**不判别新旧**，是防越界的安全断言。

## 四、被否决的变体（不得仅换参数重提）

**给 `redis` 配技术域分组** `(("redis", 1.0), ("spring", 0.4))`：

- 13 条探针、9 条验证集逐题名次与不配分组**完全相同**；
- 4 道 spring+redis 提问（含本轮两道 redis 评测题）的 top-5 **逐条相同**；
- 对 `redis-pool` 想要的那块 spring-boot 证据（`#797`）也**没有**把它拉回 top-10。

原因可算清楚：副域权重 0.4 下，副域第 1 名的 RRF 分 `0.4/(60+1)` 只相当于主域第 **92** 名，
而主域候选池深 30——**副域在主域候选充足时结构上进不来**。
所以这不是"权重调大一点就行"，调大它就是在为一道题调参，需要独立证据与完整验证，
不属于本轮。

## 五、顺带记录、本轮不修的两条

1. **拒答格式不合契约**：模型在 `NO_EVIDENCE` 前加了一句前言，`declined()` 的
   `startswith` 判定因此把它当成作答。改 `declined()` 是放宽判据（§5.4），
   改提示词会动模板版本、影响两臂全部结果——都需要单独一轮并由审查方复核。
2. **索引里有 224 块垃圾**：`spring-boot` 的 `redirect.html` 被切成 224 块纯 `xref:` 链接列表
   （`title_path='Redirect'`，全部来自同一个 URL）。它们含大量锚点名，BM25 下对各种提问都
   能拿到分——改前英文 `redis-cache-annotations` 的第 3 名和实际进入 prompt 的证据里都有它。
   清掉需要重新同步并重建索引（指纹会变，所有已记录报告的身份随之失效），本轮不做。
