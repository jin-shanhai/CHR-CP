# CHR-CP 项目说明文档

**项目名称**:Confidence-Gated Hierarchical Routing with Cache-Preserved Switching
**版本**:v0.1(Day 6 完成快照)
**目标会议**:CCF B / EI(主推 EMNLP / AAAI Findings / IJCAI / NLPCC)
**项目周期**:14 天(Day 1-6 已完成,Day 7-14 待执行)
**最后更新**:Day 6 结束

---

## 1. 项目定位与核心问题

### 1.1 问题域

LLM 多智能体系统(Multi-Agent System,MAS)能在复杂任务上提升准确率,但带来 **5-15 倍的 token 开销**。最近一年的研究开始用**异构模型路由**(简单子任务用便宜模型、复杂的用昂贵模型)来降本。代表性工作:AgentRouter (2025/10)、OI-MAS (2026)、SC-MAS (2026)、AgentCollab (2026)。

### 1.2 现有工作的盲点

四篇相关论文都假设**白盒部署条件**(本地部署 + 完整内部信号可访问),但生产级 MAS 应用 90% 通过 **商业 API** 调用模型。API 限制带来三个被忽视的挑战:

| 挑战 | 表现 | 现有方法的反应 |
|---|---|---|
| A. 不确定性信号缺失 | thinking 模式禁用 logprobs(DeepSeek 官方确认,Qwen-Max 返回 null) | 现有方法依赖 logprobs,API 场景下大面积失效 |
| B. 缓存机制由厂商定义 | DeepSeek prompt cache 是粗粒度的、跨厂商完全失效 | AgentCollab 提了问题但未解决 |
| C. 跨厂商上下文损失 | 不同模型对相同 prompt 的最优组织形式不同 | 无人系统处理 |

### 1.3 CHR-CP 的核心 claim

> CHR-CP 是**首个**针对纯 API 异构 MAS 设计的路由框架,联合优化路由决策与缓存效率。

**与四篇baseline的差异化定位**(画图比较):
                  细粒度
                    ↑
                    |
                    |   ★ CHR-CP (Ours)
                    |          ──────►
        AgentCollab(提了问题) ─────►   API-aware
                    |       /       │
          OI-MAS    |     /         │
                    |   /           ▼
        AgentRouter                白盒
                    |  SC-MAS
                    +──────────────────────►
                  粗粒度        考虑缓存切换成本

CHR-CP 占据"细粒度路由 + 显式 cache 优化 + API-aware"三重交集,目前是空白象限。

---

## 2. 核心创新点(论文卖点)

### 2.1 三层级路由架构(Section 3.1 of paper)

| 层 | 决策粒度 | 决策时机 | 输入 | 输出 |
|---|---|---|---|---|
| **L1** | 任务级 | 任务开始前(冷启动) | 任务描述 | agent池配置(数量+tier+拓扑) |
| **L2** | 步骤级 | 每步推理后 | VC²不确定性信号 | STAY / BRANCH / ESCALATE |
| **L3** | 切换级 | L2触发ESCALATE时 | 切换源/目标 | 缓存保护策略 |

**新颖性**:现有工作要么只做 L1(AgentRouter),要么只做 L2(OI-MAS),**没有任何工作做完整三层**。

### 2.2 VC²:API约束下的双信号不确定性估计(Section 3.3)

不依赖 logprobs,只用 API 输出可见的两个信号:

- **Verbalized Confidence (VC-V)**:模型在输出末尾给 `<confidence>X/10</confidence>` 自评分
- **Multi-Sample Consistency (VC-C)**:同 prompt K=3 次采样,答案一致性

融合公式:`U = α · U_consistency + (1 − α) · U_verbalized`,默认 α=0.6。

**Verbalized parse 失败时的降级策略**(实测发现的关键问题):
- `consistency_only`:忽略verbalized,U = U_consistency(默认)
- `neutral`:U_verbalized = 0.5(论文ablation使用)

### 2.3 三动作L2决策(Section 3.3.3)

不同于现有"用强 / 用弱"二元决策:
    U < τ_low:    STAY (当前tier继续)
τ_low ≤ U < τ_high:   BRANCH (同tier多采样投票)
U ≥ τ_high:   ESCALATE (升级到上一tier)

预算自适应阈值:`τ_adj = τ · (2 − budget_ratio)`,预算紧时阈值升高,趋向保守。

**Branch级联规则**:若 BRANCH 内样本一致性 < 0.3,强制升级到 ESCALATE(避免在弱tier卡住)。

### 2.4 三机制L3缓存保护(Section 3.4)

| 机制 | 名称 | 解决什么 |
|---|---|---|
| **M1** | Stable Prefix Engineering | 强制 prompt 分层结构,最大化同厂商内 cache 命中 |
| **M2** | Cross-Vendor Distillation | 跨厂商切换时用 T1 把历史压缩成结构化 JSON,降低目标 tier 的 prefill 成本 |
| **M3** | Speculative Cache Warming | L2 检测到可能升级时,异步给目标 tier 发轻量请求触发其 cache |

---

## 3. 项目代码结构

### 3.1 目录树(Day 6 状态)
mas_chrcp/
├── README.md
├── requirements.txt
├── .env.example                      # API key 模板(.env 文件不进 git)
├── .gitignore
├── docs/
│   └── PROJECT_OVERVIEW.md           # 本文档
├── configs/
│   └── models.yaml                   # 4档模型配置 + 定价
│
├── chr_cp/                           # 核心代码包
│   ├── init.py
│   ├── clients/                      # ▶ API客户端层
│   │   ├── init.py
│   │   ├── base_client.py            # 抽象基类 + 统一response结构
│   │   ├── deepseek_client.py        # DeepSeek API封装(T1/T2/T4)
│   │   ├── qwen_client.py            # Qwen API封装(T3)
│   │   └── client_pool.py            # 4档统一调度入口
│   │
│   ├── confidence/                   # ▶ 不确定性估计层(VC²)
│   │   ├── init.py
│   │   ├── verbalized.py             # VC-V:置信度tag解析
│   │   ├── consistency.py            # VC-C:多采样自洽性
│   │   └── vc2.py                    # 双信号融合
│   │
│   ├── prompts/                      # ▶ Prompt工程层
│   │   ├── init.py
│   │   ├── stable_prefix.py          # L3-M1:稳定前缀构造器
│   │   ├── role_templates.py         # 角色prompt + confidence footer
│   │   └── distillation.py           # L3-M2:压缩prompt + JSON解析
│   │
│   ├── routing/                      # ▶ 路由决策层
│   │   ├── init.py
│   │   ├── decisions.py              # 决策动作 + StepResult数据结构
│   │   ├── budget.py                 # 预算追踪 + 自适应阈值
│   │   ├── l1_coarse.py              # L1:任务级路由器(规则版)
│   │   ├── l2_step.py                # L2:步骤级路由器(三动作决策)
│   │   ├── l3_cache.py               # L3:M2 + M3 缓存保护
│   │   └── orchestrator.py           # 三层协调器(端到端入口)
│   │
│   └── utils/                        # ▶ 工具层
│       ├── init.py
│       ├── cost_tracker.py           # 成本+cache hit追踪
│       ├── text_sim.py               # 数值/文本相似度
│       └── ast_diff.py               # 代码AST相似度
│
└── tests/                            # ▶ 验证测试(每个Day一个)
├── init.py
├── test_clients.py               # Day 1
├── test_confidence.py            # Day 2-3
├── test_l2_routing.py            # Day 4
├── test_l1_routing.py            # Day 5
├── test_l3_cache.py              # Day 6
└── test_orchestrator.py          # Day 6 端到端

### 3.2 模块依赖关系
        ┌────────────────────┐
        │    Orchestrator    │  ← 论文 Section 3.5 算法伪代码对应这里
        └─────────┬──────────┘
                  │
   ┌──────────────┼──────────────┐
   ▼              ▼              ▼
┌───────┐    ┌─────────┐    ┌────────┐
│  L1   │    │   L2    │    │  L3    │
│ Rule  │    │ Router  │    │ Cache  │
│ Class │    │ 3-Action│    │ M2/M3  │
└───────┘    └────┬────┘    └────┬───┘
│              │
▼              │
┌───────────┐         │
│   VC²     │         │
│ Estimator │         │
└─────┬─────┘         │
│               │
┌────────────┼───────────────┘
▼            ▼               ▼
┌─────────┐ ┌──────────┐  ┌──────────┐
│Verbal-  │ │Consist-  │  │Distill-  │
│ized     │ │ency      │  │ation     │
└─────────┘ └────┬─────┘  └────┬─────┘
│              │
└──────┬───────┘
▼
┌─────────────┐
│ Client Pool │
└──────┬──────┘
│
┌────────────┼────────────┐
▼            ▼            ▼
┌────────┐  ┌─────────┐  ┌─────────┐
│DeepSeek│  │  Qwen   │  │(future)│
│Client  │  │ Client  │  │ extend  │
└────────┘  └─────────┘  └─────────┘
│            │
▼            ▼
DeepSeek API   Qwen API
(T1/T2/T4)      (T3)

---

## 4. 各模块功能详解

### 4.1 `chr_cp/clients/` — API客户端层

**职责**:屏蔽不同厂商 API 的差异,对上层提供统一接口。

**关键设计**:
- `BaseClient`:抽象基类,定义 `chat_completion()` 方法 + 重试逻辑
- `CompletionResponse`:**统一的响应数据结构**,包含 content、tokens、cache hit、cost、latency、logprobs(可空)、reasoning_content(可空)
- 子类通过实现 `_build_request_params()` 和 `_parse_response()` 适配各自 API 的差异
- `ClientPool`:从 YAML 配置加载所有 tier 的客户端,通过 `pool.invoke(tier, messages)` 统一调用

**当前模型池**:

| Tier | 模型 | 厂商 | 模式 | 价格 (in/out per M) | logprobs |
|---|---|---|---|---|---|
| T4 | deepseek-v4-pro | DeepSeek | thinking | $0.435 / $0.87 | ✗ |
| T3 | qwen-max | Alibaba | non-thinking | $2.4 / $9.6 | ✗ |
| T2 | deepseek-v4-flash | DeepSeek | thinking | $0.14 / $0.28 | ✗ |
| T1 | deepseek-v4-flash | DeepSeek | non-thinking | $0.14 / $0.28 | ✓ |

**测试覆盖**:`tests/test_clients.py`(Day 1 ✓ 通过)

### 4.2 `chr_cp/confidence/` — 不确定性估计层

**职责**:计算 VC² 信号,这是 L2 决策的输入。

**关键组件**:

- `VerbalizedConfidenceParser`:用 6 个正则模式串联解析 confidence。覆盖:
  - `<confidence>X/10</confidence>` 严格 tag
  - `<confidence>X</confidence>` 简化 tag
  - `<置信度>X/10</置信度>` 中文 tag
  - `confidence: X/10` 内联
  - `confidence: 85%` 百分比
  - `85% confident` 反向百分比

- `ConsistencyEstimator`:K=3 多温度采样(0.3/0.7/0.9),按任务类型选择相似度度量:
  - NUMERIC:精确数值匹配
  - MULTIPLE_CHOICE:字母匹配
  - CODE:Python AST 节点类型 cosine 相似度
  - OPEN_TEXT:ROUGE-L F1

- `VC2Estimator`:
  - **Full mode**:verbalized + consistency 全部跑
  - **Lite mode**:仅 verbalized(L2 在 STAY 区间深处可省采样成本)
  - 对外暴露 `UncertaintySignal` 数据结构,字段包括 `U`、`U_verbalized`、`U_consistency`、`alpha`、`fusion_method`

**测试覆盖**:`tests/test_confidence.py`(Day 2-3 ✓ 通过,VC² 信号方向已验证:U(hard) > U(easy))

### 4.3 `chr_cp/prompts/` — Prompt 工程层

**职责**:产生 cache-friendly 的 prompt 结构。

**StablePrefixBuilder**(L3-M1):

强制 prompt 5 层结构,前 3 层组成"稳定前缀",所有 agent 调用共享:
[SYSTEM-FIXED]    永不变 (cacheable)
[TASK-ANCHOR]     永不变 (cacheable)
[SHARED-CONTEXT]  只追加 (incrementally cacheable)
─── prefix boundary ───
[ROLE-DYNAMIC]    每agent变 (not cached)
[STEP-PAYLOAD]    每step变 (not cached)

实测验证:Day 1 第二次调用 cache_hit_tokens > 0,机制生效。

**RoleTemplate**:四个标准角色 — solver / verifier / aggregator / compressor。每个角色的 prompt 末尾自动追加 `CONFIDENCE_FOOTER`,**强制模型输出置信度 tag**(VC² 的硬依赖)。

**Distillation**(L3-M2):

- `build_distillation_messages(history)` → 压缩 prompt
- `parse_distilled(text)` → 鲁棒 JSON 解析(支持 markdown 围栏、文本前后缀干扰)
- `DistilledContext`:含 task_recap / completed_steps / verified_facts / pending_question + token 压缩比

### 4.4 `chr_cp/routing/` — 路由决策层

#### `decisions.py`

定义路由的"语言":
- `RoutingAction`:STAY / BRANCH / ESCALATE 枚举
- `RoutingDecision`:单步决策 + 完整 trace(uncertainty、thresholds、reason、budget_ratio 等)
- `StepResult`:一步的最终结果(决策列表 + 所有 API 调用 + 成本汇总)

#### `budget.py`

预算追踪 + 自适应阈值:
- `BudgetTracker`:接受总预算,追踪累计开销
- `adjust_thresholds()`:返回 `AdaptiveThresholds`,公式 `τ_adj = τ · (2 − budget_ratio)`
- `warmup_floor`:防止 budget 几乎耗尽时阈值变成 2.0(全部锁死 ESCALATE)

#### `l1_coarse.py`

任务分类器:
- 5 类标签:MATH / CODE / KNOWLEDGE_QA / LOGICAL_REASONING / OPEN_TEXT
- **rule 模式**(默认):用关键词 + 正则评分,Day 5 测得 ≥85% 准确率
- **learned 模式**(预留):MLP 训练版,作为 ablation 实验
- 类别 → 池配置映射表(可被 `L1Config.config_table` 覆盖)

#### `l2_step.py`

L2 主路由器,**项目最复杂的模块**。
- `execute_step()` 流程:
  1. 主调用(取得 primary response)
  2. VC² 评估(full mode)
  3. 应用 verbalized-failure 降级策略
  4. 预算自适应阈值
  5. 三动作决策
  6. 执行动作(STAY/BRANCH/ESCALATE)
  7. BRANCH 时复用 consistency 已有的 K 个样本做投票/共识
- **级联规则**:BRANCH 内一致性 < 0.3 时强制 ESCALATE
- **顶 tier 兜底**:T4 想 ESCALATE 但没上层时降级为 BRANCH

#### `l3_cache.py`

`L3CacheManager`:
- `is_cross_vendor()`:基于 `tier_to_provider` 映射检查
- `handoff()`:执行升级切换,跨厂商时触发 M2 蒸馏
- `maybe_warm_cache()`:M3 异步预热(daemon thread,< 100ms 启动)
- `wait_warmups()`:实验结束时回收预热线程,确保所有成本被记账

#### `orchestrator.py`

`CHRCPOrchestrator`:三层的"总指挥"。
- 输入:task 字符串
- 输出:`CHRCPResult`,包含 final_answer、所有 step 的 trace、L1 配置、动作分布、L3 触发次数
- 流程:
  1. L1 分类 → AgentPoolConfig
  2. 每个 agent:
     - 用 StablePrefixBuilder 构建 messages
     - L2 执行(可能 BRANCH/ESCALATE)
     - 跨厂商 ESCALATE → L3-M2
     - U 在 BRANCH 区且后续还有 agent → L3-M3 预热
  3. 最后一个 agent 的输出 = final_answer

### 4.5 `chr_cp/utils/` — 工具层

- `CostTracker`:实验级 API 调用追踪。每次调用记录 tier / tokens / cache hit / cost / latency / task_id / step_id / routing_action。可导出 JSON,后面画 Pareto 图、cache hit breakdown 都靠它。
- `text_sim.py`:数值答案抽取(支持 \boxed{} / "answer is X" 等多种格式)、numeric_match(带容差)、ROUGE-L、MC 字母抽取
- `ast_diff.py`:Python 代码 AST 节点类型 cosine 相似度;非 Python 代码 fallback 到 ROUGE-L

---

## 5. 论文实验设计映射

### 5.1 实验变量与代码模块的对应关系

| 论文实验 | 涉及模块 | 是否已就绪 |
|---|---|---|
| Main Table 1: 8 方法 × 6 benchmark | Orchestrator + Benchmark Runner(Day 7) | ⏳ Day 7-9 |
| Figure 2: Cost-Accuracy Pareto | CostTracker + analyze_results.py(Day 12) | ⏳ Day 12 |
| Figure 3: VC² 信号有效性 | confidence/* + 失败案例日志 | ✓ 数据可生产 |
| Figure 4: Cache hit rate breakdown | CostTracker.by_tier 字段 | ✓ 数据可生产 |
| Figure 5: 阈值敏感性热力图 | L2Config + experiments/run_sensitivity.py(Day 12) | ⏳ Day 12 |
| Figure 6: Case study 决策路径 | RoutingDecision trace | ✓ 数据可生产 |
| Table 2: 消融研究(11种变体) | L2Config / L3Config 的开关 | ✓ 接口已提供 |

### 5.2 消融研究的代码挂钩点

| Ablation 项 | 控制开关 |
|---|---|
| w/o L1 | `L1Config(mode="rule")` 替换为固定 pool |
| w/o L2 | 跳过 VC² 评估,随机选 tier |
| w/o L3-M1 | 不使用 StablePrefixBuilder,直接拼字符串 |
| w/o L3-M2 | `L3Config(enable_m2_distillation=False)` |
| w/o L3-M3 | `L3Config(enable_m3_warming=False)` |
| Only verbalized | `L2Config` 设 `verbalized_failure_strategy="neutral"` 强制走 verbalized |
| Only consistency | 设置 `alpha=1.0` |
| Two-action(无BRANCH) | 修改 L2Router 的 _decide(改 enum 控制),做一个 L2RouterTwoAction 子类 |

---

## 6. 当前状态与已验证能力

### 6.1 Day 1-6 完成情况

| Day | 任务 | 状态 | 关键产出 |
|---|---|---|---|
| 1 | 4档API统一客户端 + 成本追踪 + 稳定前缀 | ✅ 全过 | T1/T2/T3/T4 全部通畅,cache 机制生效 |
| 2-3 | VC² 双信号(verbalized + consistency) | ✅ 全过 | 已验证 U(hard) > U(easy),信号方向正确 |
| 4 | L2 三动作路由器 + 预算自适应阈值 | ✅ 全过 | 决策 trace 完整,STAY/BRANCH/ESCALATE 边界正确 |
| 5 | L1 任务级路由(规则版) | ⏳ 待跑 | 5 类分类、池配置映射表 |
| 6 | L3-M2 蒸馏 + L3-M3 预热 + Orchestrator | ⏳ 待跑 | 端到端 pipeline 闭合 |

### 6.2 已经可以从代码产出的论文素材

跑完 Day 6 的端到端测试后,**不需要任何额外开发**就能产出:

1. **Figure 3 (VC² 有效性)**:对每个测试样本记录 `(U_verbalized, U_consistency, U, is_correct)`,做密度分布图
2. **Figure 6 (Case Study)**:从 `CHRCPResult.step_results[].decisions[]` 直接画流程图
3. **Cache 命中率统计**:`CostTracker.by_tier[T1].cache_hit_rate` 等
4. **路由动作分布**:`CHRCPResult.num_stay / num_branch / num_escalate`

### 6.3 待开发组件

| 组件 | 用途 | 计划 Day |
|---|---|---|
| `benchmarks/` 模块 | 6 个 benchmark 加载器 + 评测器 | Day 7 |
| `baselines/` 模块 | 7 个对比方法实现 | Day 10 |
| `experiments/run_main.py` | 主实验入口(批量跑任务) | Day 7 |
| `experiments/run_ablation.py` | 消融实验入口 | Day 11 |
| `experiments/run_sensitivity.py` | 阈值敏感性扫描 | Day 12 |
| `experiments/analyze_results.py` | 画图 + 表格生成 | Day 12 |

---

## 7. 预期最终效果

### 7.1 学术指标(主实验目标值)

| 指标 | 目标 | 论据 |
|---|---|---|
| 6 benchmark 平均准确率 | ≥ Single-T4 的 95% | 不能靠廉价 tier 牺牲准确率 |
| 6 benchmark 平均成本 | ≤ Homo-MAS(T4 全程)的 30% | 主要节省来源于 STAY/BRANCH 取代 ESCALATE |
| Cost-Accuracy Pareto 位置 | 严格优于 OI-MAS、SC-MAS | 即在相同准确率下成本更低,或相同成本下准确率更高 |
| Cache hit rate(DeepSeek侧) | ≥ 60%(开启 L3-M1+M3) | 稳定前缀的直接收益 |
| 路由动作分布 | STAY:BRANCH:ESCALATE ≈ 60:25:15 | 验证三动作设计的必要性 |

### 7.2 论文结构对应(目标 9-10 页 + appendix)
Section 1 Introduction        → §2 三个被忽视挑战 + §1.3 CHR-CP claim
Section 2 Related Work        → 四篇baseline的盲点分析
Section 3 Method
3.1 Problem Formulation     → §1.1
3.2 L1 Coarse Router        → §4.4 l1_coarse
3.3 L2 Step Router with VC² → §2.2 + §2.3 + §4.2 + §4.4 l2_step
3.4 L3 Cache Mechanisms     → §2.4 + §4.4 l3_cache
3.5 Algorithm Pseudocode    → orchestrator.py 的简化版
Section 4 Experiments         → 待 Day 7-12 产出数据
Section 5 Conclusion          → 总结 + Limitations

### 7.3 投稿目标 venue 优先级

1. **首选**:EMNLP 2026 Findings / AAAI 2026 / IJCAI 2026
2. **次选**:NLPCC 2026 / PRICAI 2026 / ACL Workshop on Agents
3. **保底**:ICTAI / 国内核心期刊

---

## 8. 关键工程决策记录

记录这一项是为了答辩 / reviewer 提问时有据可查:

### 8.1 为什么 K=3 而非 K=5?

- K=3 已能区分明显不一致的样本(成本 3x)
- K=5 边际收益递减(成本 5x)
- 可在 ablation 中做 K ∈ {1, 3, 5, 7} 敏感性分析

### 8.2 为什么 α=0.6(consistency 权重略高于 verbalized)?

- Consistency 是基于实际行为的硬证据,verbalized 是模型自报(已知偏乐观)
- 但 verbalized 不能完全忽略,因为它是**唯一能在单次调用就拿到的信号**(K=1 时唯一可用)
- 0.6 是平衡点;ablation 中扫 α ∈ {0.0, 0.3, 0.5, 0.7, 1.0}

### 8.3 为什么默认阈值 (τ_low=0.3, τ_high=0.65)?

- 经验值,后续 Day 12 在 200 个验证样本上做 grid search 确认
- 阈值不会一刀切,会按 benchmark 微调(论文中报告每个 benchmark 的最优 τ)

### 8.4 为什么用规则模式做 L1 而不是训练分类器?

- 规则模式零训练数据 + 零依赖,2 周时间内必定能跑通
- 学习模式作为 ablation,展示"L1 设计本身就够鲁棒,L2 路由能在 L1 误分类时 graceful degradation"

### 8.5 为什么不直接用 logprobs 即使 T1 支持?

- 主流场景(T2/T3/T4)都不支持,统一接口必须不依赖
- T1 logprobs 在 ablation 中作为补充信号(论文 4.7 节专门讨论)

---

## 9. 已识别的限制与应对

### 9.1 已知限制(论文 Limitations 章节会写的)

| 限制 | 影响 | 应对 |
|---|---|---|
| Verbalized parse 失败率(实测可达 10-20%) | U 信号失真 | consistency_only 降级策略 + Section 4 数据报告 |
| Distillation parse 偶发失败 | M2 跳过该次蒸馏 | 论文 Limitations 提及 + 失败时直接传原始 history |
| Threshold 需校准 | 不同 benchmark 最优阈值不同 | Day 12 sensitivity heatmap 展示鲁棒性 |
| 仅 2 厂商 4 模型 | 跨厂商场景验证不充分 | 论文 Limitations 注明,future work 扩展到 4+ 厂商 |
| 同步代码,未做异步并发 | benchmark 跑得慢 | Day 7 加 ThreadPoolExecutor 并发 |

### 9.2 未来工作(论文 Conclusion 中提及)

- 学习版 L1 在更大数据集上的验证
- 自动化阈值在线调整(replace grid search)
- 扩展到多模态 MAS(VC² 的 verbalized 和 consistency 信号在视觉任务中的适配)

---

## 10. 给后续实验阶段的开发指南

### 10.1 Day 7+ 开发 do/don't

**Do**:
- 任何新功能优先添加到现有 `routing/` 或 `confidence/` 子模块,不破坏接口
- 新加的 baseline 放在独立的 `baselines/` 包下,不混入 `chr_cp/`
- 评测脚本统一输出到 `results/{benchmark}/{method}.json`
- 每跑完一个 benchmark 立刻保存 `CostTracker.save()`,防止意外中断丢数据

**Don't**:
- 不要修改 `BaseClient` 和 `CompletionResponse` 的字段(下游全部依赖)
- 不要在主实验跑到一半时改 `L2Config` 的默认值
- 不要 print 调试信息到 stdout(使用 `logger`,方便 grep)

### 10.2 实验数据的标准格式

每次实验产出的 JSON 结构:

```json
{
  "experiment_id": "main_chrcp_gsm8k_2026-05-08T14-30",
  "method": "CHR-CP",
  "benchmark": "gsm8k",
  "config": {
    "tau_low": 0.30,
    "tau_high": 0.65,
    "alpha": 0.6,
    "...其他超参": "..."
  },
  "summary": {
    "n_tasks": 500,
    "accuracy": 0.852,
    "total_cost_usd": 0.45,
    "avg_latency_seconds": 8.3,
    "cache_hit_rate": 0.71,
    "action_distribution": {
      "STAY": 0.62, "BRANCH": 0.24, "ESCALATE": 0.14
    }
  },
  "per_task": [
    {
      "task_id": "gsm8k_0", "correct": true, "cost": 0.0008,
      "trace": [/* RoutingDecision 序列 */]
    },
    ...
  ]
}
```

这个结构能直接喂给 `analyze_results.py` 画图。

---

## 11. 文档维护

每完成一个 Day 后,在本文档第 6 节的"完成情况"表更新状态。论文写作阶段(Day 13-14),本文档的 §2、§4、§5 直接转为论文 §3 (Method) 的初稿。

文档版本:v0.1 (Day 6 末)
下次更新:Day 9 (主实验跑完后)、Day 12 (消融完成后)、Day 14 (定稿)