# CHR-CP 论文内容优化 Brief

## 一句话定位

建议把投稿版本定位为：

> CHR-CP is an API-aware heterogeneous multi-agent LLM routing framework that uses logprob-free uncertainty estimation and cache-preserved switching to reduce inference cost while preserving task accuracy.

中文理解：这不是“泛泛的多智能体协作”，而是把 **商业 API 约束、黑盒不确定性、跨厂商 cache 损失、成本-准确率权衡** 放到同一个系统里解决。

## 推荐标题

首选：

> CHR-CP: Confidence-Gated Hierarchical Routing with Cache-Preserved Switching for API-Based Multi-Agent LLM Systems

更短：

> Cache-Aware Confidence-Gated Routing for Cost-Efficient Multi-Agent LLM Systems

如果坚持 fuzzing 方向，需要改成另一个项目标题：

> CHR-Fuzz: Multi-Agent Directed Fuzzing with Confidence-Gated LLM Orchestration

但这个版本需要补 fuzzing 核心实验，不能只改标题。

## PRICAI Regular Paper 结构

按 12-16 页 regular paper 组织：

| 章节 | 页数 | 内容 |
|---|---:|---|
| Introduction | 1.5-2 | API-based MAS 的四个痛点；CHR-CP 概述；贡献列表 |
| Related Work | 1.5 | LLM routing, multi-agent systems, uncertainty estimation, prompt caching |
| Method | 4 | L0/L1/L2/L3；VC2；CA2R；CADS/CTOR；算法伪代码 |
| Experiments | 4-5 | setup, baselines, main results, ablation, sensitivity, case study |
| Discussion | 1 | 何时有效、何时不省钱、API economics 的启发 |
| Limitations | 0.75 | 已完成结果中暴露的 GPQA/MMLU 成本问题、vendor 数量、cache 可复现性 |
| Conclusion | 0.25 | 简短收束 |

## 贡献写法

建议贡献列表写成 3 条，不要堆太多缩写：

1. We formulate API-based heterogeneous MAS routing as a joint accuracy-cost-cache optimization problem, where provider-level prompt caching is treated as a first-class routing signal.
2. We propose a logprob-free confidence gate, VC2, that combines verbalized confidence and lightweight consistency checks to support black-box reasoning APIs.
3. We design CHR-CP, a hierarchical routing and switching framework with adaptive thresholds, structured handoff compression, and auditable routing traces.

## 摘要草稿

```text
Multi-agent large language model systems can improve reasoning accuracy, but their deployment through commercial APIs often incurs prohibitive cost and latency. Existing routing methods typically select models at the task level and treat prompt caching and provider switching as implementation details. We present CHR-CP, a confidence-gated hierarchical routing framework for API-based heterogeneous multi-agent LLM systems. CHR-CP combines a logprob-free uncertainty signal, built from verbalized confidence and lightweight consistency checks, with adaptive step-level routing over STAY, BRANCH, and ESCALATE actions. To reduce the cost of switching across model tiers and providers, CHR-CP further uses stable prompt prefixes and compressed structured handoffs that preserve cache value and reasoning context. Experiments on mathematical reasoning, code generation, and knowledge benchmarks show that CHR-CP can approach strong-model accuracy while substantially reducing average inference cost on tasks where uncertainty is concentrated in a subset of samples. Ablation and sensitivity analyses highlight when cache-aware routing helps, and when difficult or broad-domain benchmarks require stronger cost controls.
```

注意：最后一句故意留了边界，不把 GPQA/MMLU 说成全面胜利。审稿人会更信。

## 图表放置建议

| 图 | 文件 | 论文位置 |
|---|---|---|
| Figure 1 | `figure_01_system_architecture.svg` | Method overview |
| Figure 2 | `figure_07_tier_ladder_cache.svg` | Setup / model ladder |
| Figure 3 | `figure_03_cost_accuracy_map.svg` | Main results |
| Figure 4 | `figure_04_phase1b_sensitivity.svg` | Sensitivity analysis |
| Figure 5 | `figure_06_decision_trace_template.svg` | Case study |
| Optional | `figure_02_pricai_positioning.svg` | Introduction slides, not necessarily final paper |
| Optional | `figure_08_fuzzing_reframe_bridge.svg` | Internal decision figure, not final paper unless discussing future CHR-Fuzz |

## 目前最需要补强的证据

- 从 `paper_assets/data/results_summary.json` 自动生成所有表图。
- 主实验至少要有 CHR-CP、Single-T1、Single-T4、Static-3-agent 三类 baseline。
- 消融至少保留：w/o CA2R、w/o BRANCH、w/o CADS/CTOR 或 w/o cache feedback。
- 每个 benchmark 报告 `n`, accuracy, cost/sample, latency, final tier distribution, action distribution。
- GPQA/MMLU 如果成本不占优，建议如实放入 stress-test 表，而不是放在主 claim 第一行。

## 如果转为“多智能体定向模糊测试”

必须新增实验闭环：

1. 目标程序集：例如 LAVA-M、Magma、CGC、FuzzBench subset 或真实 C/C++ 工具。
2. Directed objective：给定目标行、目标函数、补丁位置、漏洞函数，定义 distance-to-target。
3. 多智能体角色：planner 解析目标路径，mutator 生成/改写种子，executor 运行覆盖，triage 分析 crash。
4. 指标：time-to-exposure, target coverage, unique crashes, edge coverage, bug reproduction rate。
5. baseline：AFLGo、AFL++、LibFuzzer、FairFuzz/Angora/NEUZZ 中至少 2-3 个。

没有这些，审稿人会认为“fuzzing”只是应用包装，而不是论文实质。
