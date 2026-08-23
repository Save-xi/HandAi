# 单右手控制意图预测：第一轮正式报告

日期：2026-08-23

## 1. 本轮做了什么

本轮只研究现有单右手虚拟预览链路中的一个可选模块：

`最近 30 帧 svh_preview.target_positions -> 预测未来 50 / 100 / 150 ms 的 9 通道目标`

没有把项目改成双手，也没有接入真机。现有
`摄像头/视频 -> MediaPipe 单右手 -> control_representation -> svh_preview -> Unity UDP`
主链和 Unity 工程均未被本实验修改。

完成项：

- 解压并审计 H2O 四名受试者的 pose-only 数据；
- 将右手 21 点姿态复用现有映射，自动转换为 9 通道控制序列；
- 建立严格跨受试者 train/val/test 切分；
- 比较 hold-last、Linear、Kalman、GRU、TCN、Transformer；
- 输出 checkpoint、整体/分预测距离/分通道指标、训练耗时和单窗口延迟；
- 增加按“已观测历史运动强度”分层的测试，避免静止帧掩盖动态性能；
- 将 CUDA 训练放进独立环境，并修复模型子集运行时 seed 漂移和 Transformer 非确定性 attention 风险。

## 2. 数据与切分

H2O pose-only 原始检查结果：

| 项目 | 数量 |
|---|---:|
| take | 184 |
| pose 文件 | 114,329 |
| 无效帧 | 933 |
| 解析拒绝帧 | 0 |
| 最终连续序列 | 217 |
| 最终有效帧 | 113,322 |

严格跨受试者切分：

| split | subject | 序列 | 帧 | 训练/评测窗口 |
|---|---|---:|---:|---:|
| train | subject1 + subject2 | 105 | 54,243 | 25,364 |
| val | subject3 | 60 | 28,631 | 13,310 |
| test | subject4 | 52 | 30,448 | 14,352 |

三组 `sequence_id` 交集均为 0。阈值、早停和模型选择只使用 train/val；test 只用于最终报告。

pose-only 压缩包不含相机内参。本轮对每帧使用腕点为原点、腕到中指 MCP 为单位掌长的
相机平面标准化，再交给现有 9 通道映射。该处理适合当前主要由角度和距离比构成的映射，
但不能等价于摄像头检测误差或真实 SVH 关节真值。

## 3. 正式结果

下表中 Transformer 使用关闭 flash/memory-efficient SDP、启用 math SDP 的严格确定性复跑。
正数“相对 hold 改善”表示误差降低。

| 模型 | 参数量 | 训练轮数 | 记录的模型阶段耗时 | 单窗 p50 | MAE | RMSE | P95 | MAE 改善 | RMSE 改善 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hold-last | 0 | — | — | — | **0.019952** | 0.055138 | 0.075735 | 0.00% | 0.00% |
| Linear | 0 | — | — | — | 0.026122 | 0.066519 | 0.109300 | -30.92% | -20.64% |
| Kalman-CV | 0 | — | — | — | 0.027071 | 0.069681 | 0.115393 | -35.68% | -26.38% |
| GRU | 172,699 | 18 | 16.77 s | **0.60 ms** | 0.021956 | 0.052936 | 0.074328 | -10.04% | +3.99% |
| TCN | 186,555 | 22 | 5,891.45 s | 7.65 ms | 0.021693 | **0.052681** | **0.073839** | -8.72% | **+4.46%** |
| Transformer（严格） | 422,811 | 26 | 94.35 s | 15.40 ms | 0.022543 | 0.053818 | 0.076467 | -12.99% | +2.39% |

旧版报告中的“模型阶段耗时”包含优化、checkpoint 保存和批量测试预测；后续代码已将纯优化耗时和
总阶段耗时拆开记录。耗时和延迟只代表本机 RTX 4060 Laptop、PyTorch 2.7.0 + CUDA 12.6 与本次确定性设置，
不是硬件无关的模型属性。TCN 的 dilated Conv1D 在严格确定性模式下吞吐很差，是本轮明确发现的工程风险。

## 4. 为什么 hold-last 的整体 MAE 最好

50 / 100 / 150 ms 都是很短的预测距离，H2O 中又包含大量低运动窗口。对这类窗口，保持最后值
本来就是很强的基线。Linear 和常速度 Kalman 会把 9 通道映射中的帧间抖动当成速度继续外推，
运动越强时反而越容易过冲。

为了避免只看静止帧，本轮用验证集历史运动分数的第 90 百分位作为固定阈值，再评估测试集中
超过该阈值的 1,310 个窗口。阈值只由已观测历史和 val 决定，没有使用测试未来真值。

| 模型 | 高运动 MAE 改善 | 高运动 RMSE 改善 | 高运动 P95 |
|---|---:|---:|---:|
| hold-last | 0.00% | 0.00% | 0.324722 |
| Linear | -79.37% | -39.01% | 0.396616 |
| Kalman-CV | -92.67% | -49.79% | 0.435906 |
| GRU | +0.06% | +7.35% | 0.297588 |
| TCN | +0.22% | **+8.46%** | **0.288972** |
| Transformer（严格） | -1.26% | +5.42% | 0.310001 |

结论不是“深度模型没有用”，而是：当前深度模型主要减少少量大误差，所以 RMSE 和动态尾部改善；
它们还没有在所有帧的平均绝对误差上击败 persistence。

## 5. 当前推荐

下一轮以“hold-last 安全回退 + GRU 实验候选”为主：

- GRU 的动态 RMSE 改善已接近 TCN，但记录的模型阶段只需约 17 秒，单窗 p50 约 0.60 ms；
- TCN 保留为离线精度上界对照，不值得以约 98 分钟模型阶段换取当前这点额外收益；
- Transformer 参数最多、严格推理最慢，且当前指标没有优势，暂不优先；
- Linear/Kalman 暂不接入主链，它们证明了对当前 9 通道直接做速度外推会放大抖动。

建议第二轮先做一个残差/门控 GRU：默认输出 hold-last，只在观测到明确运动时预测相对最后值的
`delta`。损失函数对高运动窗口和大误差适度加权。目标是保住低运动窗口的 MAE，同时继承 GRU
在高运动尾部的优势。

第二轮离线准入建议：

- 整体 RMSE 相对 hold-last 至少降低 3%；
- 高运动 q90 子集 RMSE 至少降低 5%，P95 同时下降；
- 输出越界率为 0；
- 单窗口 p95 推理延迟小于 10 ms；
- 仍使用跨受试者切分，调参不得查看 test 指标。

按这些条件，当前 GRU 已具备成为下一轮候选的资格，但尚不能宣称整体预测全面优于 hold-last。

## 6. 证据边界与剩余风险

- 本轮验证的是公开 H2O 姿态序列上的 9 通道短期预测，不是摄像头检测、Unity Play Mode 或端到端延迟验收；
- 9 通道是由现有规则映射生成的代理目标，不是传感器测得的真实 SVH 关节位置；
- 当前 test 只有 subject4。若论文需要更强统计证据，优先对 GRU 做四折 leave-one-subject-out，暂不重复训练慢速 TCN；
- 用户当前没有可用摄像头，因此自录域适配与真实视频闭环推迟，不影响本轮离线结论；
- 首份完整报告中的 Transformer 曾触发非确定性 attention 警告，正式表格已由同窗口、同 seed 的严格复跑替换；
- Unity 和真机均不在本轮验证范围内。

## 7. 关键产物

- 实验说明：`experiments/intent_prediction/README.md`
- 正式配置：`experiments/intent_prediction/configs/h2o_first_round.json`
- 快速迭代配置：`experiments/intent_prediction/configs/h2o_quick_iteration.json`
- 预处理 manifest：`D:/VR/HandAi/datasets/H2O/processed/cross_subject_v1/manifest.json`
- 完整比较报告快照：`experiments/intent_prediction/reports/first_round/20260823T063728_024071Z_report.json`
- 严格 Transformer 报告快照：`experiments/intent_prediction/reports/first_round/20260823T082140_217836Z_strict_transformer_report.json`
- 统一运动分层审计快照：`experiments/intent_prediction/reports/first_round/20260823T063728_024071Z_motion_audit_strict_transformer.json`

正式引用数值时，应以“统一运动分层审计”为入口；其中 GRU/TCN 来自完整比较报告，
Transformer 来自严格确定性复跑。
