# CHR-CP 项目完整说明

> Confidence-gated Hierarchical Routing with Cache-Preserved Switching
> 版本: 2026-05-23

---

## 一、项目概述

CHR-CP 是一个多层 LLM 路由系统，通过 4 层模型梯级（T1-T4）和 3 层决策系统（L0-L3），在保证准确率的前提下最小化推理成本。核心思想：简单题用便宜模型解决，难题才升级到强模型，升级时通过结构化交接避免重复推理。

---

## 二、目录结构

```
/home/aiagent/mas_workspace/code/
│
├── CHR_CP_项目说明.md          ← 本文档
├── configs/                     # 配置文件
│   └── models.yaml              # T1-T4 模型池配置（模型名/provider/模式/定价）
│
├── chr_cp/                      # 核心框架代码
│   ├── clients/                 # API 客户端（4 个 provider 的封装）
│   │   ├── base_client.py       # 抽象基类 + CompletionResponse 数据结构
│   │   ├── client_pool.py       # 多 tier 客户端池（根据 tier 路由到对应客户端）
│   │   ├── deepseek_client.py   # DeepSeek V4（T2/T3）客户端
│   │   ├── openai_client.py     # OpenAI GPT-5.5（T4）客户端
│   │   └── qwen_client.py       # Qwen-Turbo（T1）客户端
│   │
│   ├── confidence/              # 不确定性估计（VC²）
│   │   ├── consistency.py       # K 采样一致性估计 + verify-based BRANCH
│   │   ├── escalation_reason.py # 升级原因推断
│   │   ├── vc2.py               # VC² 融合（U = α×Uc + (1-α)×Uv）
│   │   └── verbalized.py        # 言语化置信度解析（<confidence>X/10</confidence>）
│   │
│   ├── prompts/                 # 提示词模板
│   │   ├── distillation.py      # M2 蒸馏（跨 tier 上下文压缩）
│   │   ├── handoff.py           # CTOR 交接包（HandoffPacket 数据结构 + 解析）
│   │   ├── role_templates.py    # Agent 角色模板（solver/verifier/escalator/verifier_corrector/fresh_solver）
│   │   └── stable_prefix.py     # M1 稳定前缀（DeepSeek 缓存优化）
│   │
│   ├── routing/                 # 路由决策系统
│   │   ├── budget.py            # CA²R 预算追踪器 + 自适应阈值
│   │   ├── cache_history.py     # 跨任务缓存历史（滑动窗口）
│   │   ├── decisions.py         # 路由决策数据结构
│   │   ├── difficulty_probe.py  # L0 难度探针（T1 单次调用评估任务难度）
│   │   ├── l1_coarse.py         # L1 粗分类器（规则匹配，决定 agent 链）
│   │   ├── l2_step.py           # L2 步级路由（STAY/BRANCH/ESCALATE 决策 + 难度自适应）
│   │   ├── l3_cache.py          # L3 缓存切换（M2 CADS 蒸馏 + M3 CTOR 交接）
│   │   └── orchestrator.py      # 主协调器（L0+L1+L2+L3 的完整执行流程）
│   │
│   ├── benchmarks/              # 评测基准
│   │   ├── base.py              # Benchmark 抽象基类 + 数据结构
│   │   ├── math500.py           # MATH-500（5 级难度数学题）
│   │   ├── aime.py              # AIME 2024+2025（竞赛数学）
│   │   ├── humaneval.py         # HumanEval（代码生成，Pass@1 执行测试）
│   │   ├── mmlu.py              # MMLU（多科目选择题，A-E 选项）
│   │   ├── mmlu_pro.py          # MMLU-Pro（10 选项研究生级别）
│   │   ├── gpqa.py              # GPQA Diamond（研究生科学题，A-D 选项）
│   │   ├── gsm8k.py             # GSM8K（小学数学文字题）
│   │   └── answer_verify.py     # 答案验证器（4 层级联：type_norm→math-verify→SymPy→LLM）
│   │
│   └── utils/                   # 工具函数
│       ├── cost_tracker.py      # 成本追踪器
│       └── text_sim.py          # 文本相似度 + 答案提取 + LaTeX 归一化
│
├── experiments/                 # 实验运行与监控
│   ├── run_main.py              # 主实验运行器（CLI 入口，支持所有方法+基准组合）
│   ├── progress_tracker.py      # JSONL 断点续跑
│   ├── sensitivity_grid_v2.py   # 参数灵敏度网格实验
│   ├── run_phase2.sh            # Phase 2 批量实验脚本
│   └── phase2_ctl.sh            # Phase 2 实验控制（start/stop）
│
├── tests/                       # 测试与诊断
│   ├── analyze_phase2.py        # Phase 2 实时监控面板（CHR-CP vs baselines 对比）
│   ├── test_evaluator_regression.py  # 评测器回归测试
│   ├── evaluator_regression_cases.jsonl  # 回归测试用例
│   └── add_regression_case.py   # 添加回归测试用例
│
├── results/phase2/              # 实验输出
│   ├── chrcp/                   # CHR-CP 方法结果
│   ├── single_t1/               # Single-T1 基线
│   ├── single_t4/               # Single-T4 基线
│   └── static_3agent/           # Static-3-agent 基线
│
├── logs/                        # 运行日志
│   └── phase2_logs/             # Phase 2 实验日志
│
└── mas_chrcp/cache/benchmarks/  # 基准数据缓存
    ├── math500.jsonl
    ├── gsm8k_test.jsonl
    ├── humaneval.jsonl
    └── gpqa_diamond.jsonl
```

---

## 三、模型梯级（T1-T4）

| Tier | Model | 提供商 | 模式 | 输入 $/M | 输出 $/M | 缓存命中 $/M |
|---|---|---|---|---|---|---|
| T1 | qwen-turbo | Qwen | non-thinking | 0.04 | 0.085 | — |
| T2 | deepseek-v4-flash | DeepSeek | thinking | 0.14 | 0.28 | 0.014 |
| T3 | deepseek-v4-pro | DeepSeek | thinking | 0.44 | 2.75 | 0.044 |
| T4 | gpt-5.5 | OpenAI | non-thinking | 5.00 | 30.00 | 1.25 |

**能力单调递增**: T1 < T2 < T3 < T4
**成本递增**: T1(极低) < T2(低) < T3(中) < T4(高)
**跨厂商边界**: T1-T2（不同）、T2-T3（相同）、T3-T4（不同）

---

## 四、整体框架流程

```
每个 task 的处理流程（Phase 2 最新版本）：

┌─ L0: 难度预判 ────────────────────────────────────────┐
│ T1 单次调用 (512t, $0.00004)，评估任务难度               │
│ 输出: domain, self_assessment, reasoning_depth, score   │
├─────────────────────────────────────────────────────────┤
│                   分支决策                               │
│  ┌─ cannot_solve + (medium|deep) → 直达 T4              │
│  └─ 其他 → 走阶梯                                       │
├─ L1: 阶梯模式 ─────────────────────────────────────────┤
│ 规则分类器 分配 agent 链: solver→verifier→escalator      │
│ 起步 tier = 探针建议 (T2/T3/T4，替代原 T1)              │
├─ 每步 L2 执行 ─────────────────────────────────────────┤
│ Primary call → verify-BRANCH (128t, "Is it correct?")   │
│ → VC² uncertainty → 难度自适应阈值                       │
│ → STAY / BRANCH / ESCALATE (跳级 U≥0.8→T4)              │
├─ ESCALATE → L3 交接 ──────────────────────────────────┤
│ M2 CADS: 蒸馏历史上下文 (T1 调用，~200t 压缩)            │
│ M3 CTOR: 结构化交接 (verifier-corrector, max_tokens=2048)│
└─────────────────────────────────────────────────────────┘
```

### 阈值表（难度自适应）

| 难度分 (score) | τ_low | τ_high | BRANCH | T4 跳级 |
|---|---|---|---|---|
| 1-3 (简单) | 0.10 | 0.50 | 启用 | U≥0.8 |
| 4-5 (中等) | 0.05 | 0.60 | 启用 | U≥0.8 |
| 6-7 (难) | 0.03 | 0.60 | **禁用** | U≥0.8 |
| 8-10 (极难) | 直达 T4 | — | — | — |

---

## 五、核心技术点详解

### 5.1 VC² 不确定性估计

```
U = α × Uc + (1-α) × Uv    (E2: α=0.7)

Uv (言语化): 解析 <confidence>X/10</confidence>，0 API 调用
Uc (一致性): K=5 verify 样本投票 "Is it correct? YES/NO"
```

| U 范围 | 决策 | 说明 |
|---|---|---|
| U < τ_low | STAY | 接受答案 |
| τ_low ≤ U < τ_high | BRANCH | 需要更多证据 |
| U ≥ τ_high | ESCALATE | 升级到更强 tier |
| U ≥ 0.8 | ESCALATE→T4 | **跳级**（当前 tier 完全无能为力）|

### 5.2 CA²R 自适应阈值

```
τ_low_adj  = τ_low  × (2 - r_budget) × (1 - β × h_current)
τ_high_adj = τ_high × (2 - r_budget) × (1 - β × h_target)

β = cache_sensitivity (0.3)
r_budget: 剩余预算比例
h_current: 当前 tier 缓存命中率（滑动窗口 50 条）
h_target:  目标 tier 缓存命中率
```

### 5.3 M1: Stable Prefix（稳定前缀）

- 输出严格为 `[system_message, user_message]` 两消息结构
- system_message 字节单调增长 → DeepSeek 前缀缓存命中
- SHARED-CONTEXT 滚动替换（只保留上一 agent 输出）
- 节省 70-80% prompt token 成本（串行 consistency 场景）

### 5.4 M2: CADS 蒸馏

- 升级时压缩历史上下文为结构化 JSON（~3:1 压缩比）
- T1 调用作为蒸馏器（$0.00004/次）
- 当前**所有升级**都触发蒸馏（不限跨厂商）

### 5.5 M3: CTOR 交接

- 从当前 tier 输出中提取 `candidate_answer + confidence + context`
- Target tier 收到 "verifier-corrector" 角色（非求解者）
- max_tokens=2048（限制重复推理）
- reasoning_effort=low（减少内部思考 token）

### 5.6 Verify-based BRANCH

```
Primary call 完成 → 提取 \boxed{answer} 作为 anchor
  → K=5 verify 调用 (128t): "Is {anchor} correct? YES/NO <answer>X</answer>"
  → 投票: majority vote → Uc = 1 - max_count/K
```

替代原来的 5× 完整推理一致性采样，节省 91% completion tokens。

### 5.7 AnswerVerifier（评测器）

4 层级联判定答案等价性：

```
Layer 1: type_norm      — 类型特定归一化
  → 字母精确匹配、整数精确匹配、集合比较
  → 剥离变量前缀 (x=5 → 5)

Layer 2: math-verify    — HF 符号数学库
  → 矩阵、分数、表达式的结构化等价判定

Layer 3: SymPy          — 符号化简
  → LaTeX 解析 + 差分为 0 → 等价

Layer 4: LLM judge      — T1 语义等价判断
  → 缓存结果,避免重复调用
```

### 5.8 难度预判（L0 Probe）

- 单次 T1 调用（512t, $0.00004）
- 输出 6 个结构化字段：domain, self_assessment, reasoning_depth, needs_expert, tentative_answer, difficulty_score
- 直达条件：`cannot_solve + (medium|deep)` → 跳过阶梯直接 T4
- 保守倾向：默认走阶梯，宁可漏判不可误判

---

## 六、实验方法

### CHR-CP (E2 配置)
- K=5, τ_low=0.10, τ_high=0.50, α=0.7
- T2(thinking) + T3(thinking) + 难度感知路由

### 基线方法
| 方法 | 说明 |
|---|---|
| Single-T1 | qwen-turbo 单独调用（成本下界） |
| Single-T4 | GPT-5.5 单独调用（精度上界） |
| Static-3-agent | T2→T3→T4 固定三级，无路由 |

### CLI 入口
```bash
python -m experiments.run_main \
    --method chrcp \           # chrcp|single_t1|single_t4|static_3agent
    --benchmark math \         # math|aime|humaneval|mmlu|gpqa
    --n_samples 300 \          # 样本数
    --seed 42 \                # 随机种子
    --concurrency 4 \          # 并发数
    --max_cost_usd 30.0 \      # 成本上限
    --enable_difficulty_routing true \  # 难度感知路由
    --ctor_mode self_compress  # CTOR 模式
```

### 监控面板
```bash
python -m tests.analyze_phase2    # 所有 benchmark × 所有方法对比
```

---

## 七、数据流与关键数据结构

### 核心类关系
```
CHRCPOrchestrator
  ├── L1Router (l1_coarse)        → AgentPoolConfig
  ├── L2Router (l2_step)          → RoutingDecision
  │   ├── VC2Estimator (vc2)      → UncertaintySignal
  │   └── ConsistencyEstimator    → ConsistencyResult
  ├── L3CacheManager (l3_cache)   → HandoffResult
  │   ├── M2 蒸馏                  → DistilledContext
  │   └── M3 CTOR                 → HandoffPacket
  └── BudgetTracker (budget)      → AdaptiveThresholds
```

### JSONL 输出记录
每条记录包含: `sample_id, correct, extracted_answer, final_answer, cost_usd, latency, final_tier, api_calls[], decisions[], ctor_handoffs[], difficulty_probe, routing_mode, start_tier...`

---

## 八、当前实验状态（截至 2026-05-23）

| Benchmark | CHR-CP acc | CHR-CP cost | Single-T4 acc | Single-T4 cost | CHR-CP 优势 |
|---|---|---|---|---|---|
| MATH | ~94% | ~$0.005 | ~98% | ~$0.017 | ✅ 便宜 3x |
| HumanEval | ~95% | ~$0.003 | ~94% | ~$0.008 | ✅ 便宜 3x |
| AIME | ~92% | ~$0.060 | ~83% | ~$0.058 | ✅ 准 9%+ |
| GPQA | ~85% | ~$0.083 | ~84% | ~$0.037 | ⚠️ 持平精度，稍贵 |
| MMLU | ~90% | ~$0.019 | ~90% | ~$0.005 | ⚠️ 持平精度，稍贵 |

---

## 九、废弃模块

以下模块已从框架中彻底移除：
- **UD-SCW 预热**（M3 预热）：替换为 CTOR 交接
- `l3_cache.py` 中：`maybe_warm_cache()`, `wait_warmups()`, `select_warming_targets()`, `_warm_cache_call()`
- `orchestrator.py` 中：UD-SCW 调用块、`enable_speculative_warming`、`l3.wait_warmups()`
- `run_main.py` 中：`--disable_m3_warming` 参数
- `CHRCPResult` 中：`warmups_triggered` 字段
