# PRICAI 2026 投稿检查表

核对日期：2026-05-24（按当前本地环境）。PRICAI 2026 论文截稿是 **2026-06-13**，从 2026-05-24 算起约 **20 天**。

## 关键信息

| 项目 | 要求 |
|---|---|
| 会议 | 23rd Pacific Rim International Conference on Artificial Intelligence, PRICAI 2026 |
| 地点 | South China Normal University, Guangzhou, China |
| 主会日期 | 2026-11-17 至 2026-11-20 |
| 投稿截止 | 2026-06-13 |
| 接收通知 | 2026-08-08 |
| Camera-ready | 2026-08-22 |
| 模板 | Springer Lecture Notes in Artificial Intelligence (LNAI) |
| 投稿系统 | EasyChair: `https://easychair.org/conferences/?conf=pricai2026` |
| 页数 | Regular/long papers 12-16 页；short papers 6-11 页；投稿 PDF 不超过 16 页，均含 references |
| 审稿 | Double anonymous |
| 出版 | Springer LNAI proceedings |

来源：

- PRICAI CFP: https://2026.pricai.org/calls/call-for-papers
- PRICAI Submission: https://2026.pricai.org/submission
- Springer LNCS/LNAI author instructions: https://www.springer.com/gp/computer-science/lncs/conference-proceedings-guidelines

## 双匿名注意事项

- 首稿首页只能放 title 和 abstract，不能放作者姓名、单位、致谢。
- 正文、脚注、附录、artifact 链接都不能暴露作者身份或学校/实验室信息。
- 自引用要用第三人称，不写“our previous work”。未发表的自有材料不能作为可识别线索塞进 references。
- GitHub 链接不能直接使用当前 `jin-shanhai/CHR-CP.git`。若要提交代码补充材料，应做匿名压缩包，或使用匿名仓库/匿名 artifact 服务。

## AI 工具披露

PRICAI submission 页面写明：LLM 不满足 Springer 作者署名标准；如果使用 LLM 进行生成性工作，应在 Methods 或合适章节记录。单纯 copy editing 不需要声明，但最终文本必须由作者负责。

建议在论文末尾或 appendix 准备一句简短披露：

> We used large language models for language polishing and figure-draft assistance. All technical claims, experiments, analysis, and final text were reviewed and approved by the authors.

若实际使用 AI 生成实验代码、图表或初稿，需要把范围写得更具体。

## 主题匹配判断

当前仓库和技术文档最强的主线是：

> API-aware heterogeneous multi-agent LLM routing with uncertainty-gated escalation and cache-preserved switching.

它非常适合 PRICAI 的 **Agents**, **Large Language Models**, **Machine Learning & Models**, **Intelligent Systems & Applications** 方向。

当前还不适合直接声称是“多智能体定向模糊测试”，原因是仓库内没有 fuzzing 论文通常必须有的核心证据：

- seed corpus / mutation operators
- target program / target site / distance-to-target
- coverage guidance / branch distance
- crash or bug oracle
- 与 AFLGo、AFL++、LibFuzzer、Angora、NEUZZ 等 fuzzing baseline 的比较

结论：如果 2026-06-13 前不能补齐真实 fuzzing 实验，建议 PRICAI 主投按 **多智能体 LLM 路由/成本优化系统论文** 写；“定向模糊测试”可作为下一版 CHR-Fuzz 的扩展方向。

## 截稿前 20 天行动表

| 时间 | 必做 |
|---|---|
| D-20 至 D-16 | 确定论文定位，冻结 title/claim/section outline |
| D-15 至 D-11 | 汇总已完成的主实验结果并生成图表 |
| D-10 至 D-7 | 完成 main results、ablation、sensitivity 三组表图 |
| D-6 至 D-4 | 写完 Method/Experiments/Limitations，处理双匿名 |
| D-3 至 D-2 | LNAI 模板排版、页数压缩、引用检查 |
| D-1 | EasyChair 元数据、PDF 检查、匿名补充材料检查 |
