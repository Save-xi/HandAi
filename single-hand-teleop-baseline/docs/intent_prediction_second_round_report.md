# 单右手控制意图预测：第二轮离线报告

## 1. 结论先行

第二轮离线门槛 **6/6 通过**。按事先冻结的规则，subject3 validation 选中：

```text
residual_unweighted
  = 零残差初始化的 residual GRU
  + validation 拟合的连续运动门控
  + 每个预测距离 alpha = 0.75
```

在 subject4 test 上，相对 hold-last：

- 整体 MAE 改善 **3.06%**；
- 整体 RMSE 改善 **3.60%**；
- 整体 P95 绝对误差改善 **5.60%**；
- validation-q90 动态子集 RMSE 改善 **5.73%**；
- validation-q90 动态子集 P95 改善 **6.70%**；
- 预测越界率为 **0**；
- RTX 4060 Laptop GPU 单窗口推理 P95 为 **1.503 ms**。

因此，这个方案可以晋级为下一步的 **离线回放/影子模式候选**。这不是 Unity、摄像头、
UDP 整链路或真实 SVH 已验收的结论。

## 2. 本轮到底改了什么

输入和输出仍保持第一轮口径：用过去 30 帧的单右手
`svh_preview.target_positions`，预测未来 `50 / 100 / 150 ms` 的 9 通道目标。

新增了三项机制：

1. `residual_gru` 不直接从零猜绝对位置，而是以最后一帧（hold-last）为基准预测有界残差；
   最后一层权重和偏置全零初始化，所以训练前严格等于 hold-last。
2. 两个候选在训练 loss 中提高历史运动较强窗口的权重；运动强度只来自已经观测到的历史帧。
3. validation 门控在 hold-last 与模型预测之间连续插值；候选集合包含 `alpha=0`，验证失败时
   可以精确退化为 hold-last。

正式候选和种子在运行前写入
`experiments/intent_prediction/configs/h2o_second_round.json`，没有根据 test 结果临时增删。

## 3. 防止本轮 test 调参的证据

程序严格按这个顺序执行：

```text
加载 train + validation
  -> 训练全部候选并只在 validation 拟合门控
  -> 写入 selection.json 并计算 SHA-256
  -> 才加载 subject4 test
  -> 只评估 hold-last 和已选中的 residual_unweighted
```

冻结选择文件：

- `selection.json` SHA-256：
  `9cc711e18d911907c8e678bb984f97f2318dd66484079844494b1c832014a231`
- 文件内固定记录 `test_loaded=false` 和 `test_metrics_available=false`；
- 配置文件 SHA-256：
  `833870e34a56f138d284b1494777834f90526571b7a97fd502168e4eede5d38a`
- 数据 manifest SHA-256：
  `f5b66550e9c689b9d6c3c1579c4477aa3e144aa89b91e3342eb3aa352621d237`
- 选中 checkpoint SHA-256：
  `4b22201ae0eba6c2e8f6920fefbd03fd932dcfc690c8df7d2496271685339ad3`

上述四个哈希已在运行后从文件重新计算，并与报告记录一致。

## 4. 数据和切分

| split | subject | 序列数 | 窗口数 |
|---|---|---:|---:|
| train | subject1 + subject2 | 105 | 25,364 |
| validation | subject3 | 60 | 13,310 |
| test | subject4 | 52 | 14,352 |

三组 `sequence_id` 交集均为 0。窗口仍以 stride=2 从各自序列内部生成，没有跨越无效帧或
序列边界。

## 5. Validation 选型

hold-last 在 validation 上的 MAE / RMSE / P95 为
`0.02006692 / 0.05679992 / 0.07866473`。

| 排名 | 候选 | gated MAE | gated RMSE | gated P95 | 目标值（hold=1） |
|---:|---|---:|---:|---:|---:|
| 1 | residual_unweighted | 0.01970361 | 0.05513979 | 0.07594440 | 0.975531 |
| 2 | residual_motion2 | 0.01974062 | 0.05500693 | 0.07643035 | 0.976561 |
| 3 | residual_motion4 | 0.01975014 | 0.05507491 | 0.07628692 | 0.976943 |
| 4 | absolute_gru | 0.01995596 | 0.05533510 | 0.07610781 | 0.983333 |

选中方案相对 validation hold-last 改善：MAE **1.81%**、RMSE **2.92%**、P95
**3.46%**。运动加权的两个版本没有胜过未加权残差模型，所以本轮证据支持的是“残差基线
设计”，不支持“运动加权 loss 一定更好”。前三名目标值非常接近，仍需多种子或跨人员复验。

冻结的门控参数：

- 最近历史帧：8；
- 运动阈值：validation 第 25 百分位，数值 `0.00245038`；
- sigmoid 温度：`0.00373491`；
- `50 / 100 / 150 ms` 的 alpha：`0.75 / 0.75 / 0.75`。

## 6. Subject4 一次测试结果

| 方法 | MAE | RMSE | P95 绝对误差 | 越界率 |
|---|---:|---:|---:|---:|
| hold-last | 0.01995203 | 0.05513830 | 0.07573530 | 0 |
| 冻结的 residual + gate | **0.01934165** | **0.05315119** | **0.07149785** | 0 |

逐预测距离 RMSE：

| 距离 | hold-last | residual + gate | 改善 |
|---:|---:|---:|---:|
| 50 ms | 0.03935071 | 0.03846335 | 2.26% |
| 100 ms | 0.05727858 | 0.05561370 | 2.91% |
| 150 ms | 0.06550864 | 0.06247268 | 4.63% |

更远的 150 ms 改善更明显，符合预测器主要补偿持续运动的预期。

动态子集阈值只从 validation 的已观测历史运动强度取得。q90 阈值为 `0.01943288`，在
subject4 test 中覆盖 1,310 个窗口（9.13%）：

| q90 动态子集 | hold-last | residual + gate | 改善 |
|---|---:|---:|---:|
| MAE | 0.05953730 | 0.05696130 | 4.33% |
| RMSE | 0.12035698 | 0.11345640 | 5.73% |
| P95 | 0.32472156 | 0.30297103 | 6.70% |

模型共有 172,699 个参数，checkpoint 约 681 KiB。在当前 GPU 上单窗口推理 P50 / P95 /
最大值为 `1.028 / 1.503 / 1.812 ms`。这里只测了预测模型，不含 MediaPipe、payload 处理、
门控调度、UDP 和 Unity 帧循环。

## 7. 预先写入配置的离线门槛

| 条件 | 要求 | 实测 | 结果 |
|---|---:|---:|---|
| 整体 RMSE 改善 | >= 3% | 3.60% | PASS |
| q90 RMSE 改善 | >= 5% | 5.73% | PASS |
| q90 P95 改善 | >= 0% | 6.70% | PASS |
| 整体 MAE 退化 | <= 0.5% | 改善 3.06% | PASS |
| 输出越界率 | <= 0 | 0 | PASS |
| 单窗口 P95 | <= 10 ms | 1.503 ms | PASS |

前两项只分别高出门槛 0.60 和 0.73 个百分点，属于“通过但余量不大”，不能夸大成大幅领先。

## 8. 一个必须保留的结果纪律

选中模型的 **raw** test RMSE 为 `0.05259274`，反而优于冻结门控后的 `0.05315119`；但
raw/gated 的选择没有在 test 前预注册为二次选择，所以不能看到这个结果后再把正式方案改成
raw。这项数据只保留为诊断：下一轮应在新的 validation/CV 协议中重新比较 raw 和 gated，
不能用本次 subject4 结果做回头选择。

## 9. 仍然不能声称什么

1. subject4 已经用于第一轮诊断，本轮架构正是受第一轮结果启发；所以它不是整个项目历史上
   从未看过的盲测集。本报告是内部工程复验，不是最终无偏论文测试。
2. H2O 的 9 通道由公开三维姿态经过当前映射生成，不包含本机坏摄像头、MediaPipe 抖动、
   遮挡、光照和用户动作域偏移。
3. 大量窗口在同一序列内重叠。跨 subject 切分避免了人员泄漏，但论文置信区间应按序列或
   subject 聚合，不能把 14,352 个窗口当作完全独立样本。
4. 每个候选本轮只有一个固定种子；候选之间很接近，尚未证明排序对随机种子稳定。
5. 没有验证 Unity Play Mode、UDP 抖动、端到端延迟或真实机械手安全。

## 10. 建议的下一步

当前最合适的推进方式是：把选中 checkpoint 接入 Python 侧 **默认关闭的影子模式**。影子
模式只并行计算并记录 `hold / raw / gated`，不改变现有 `svh_preview` 和 UDP 输出；任何
checkpoint、输入质量或时序异常都回退到原链路。这样在没有摄像头时可以先用 H2O/JSONL
回放做接口和时序验收，等有可用摄像头后再录少量个人数据验证域偏移。

影子模式应直接扩展仓库现有的 `timing v1`、JSON/JSONL contract 和 Unity 诊断，不能新增
一套与现有帧号和时间戳无法对应的平行日志。

若要把结果写成更严肃的论文结论，先做多种子和 leave-one-subject-out/subject bootstrap，
并保留一批新的自录数据作为真正未参与设计的最终测试集。

计划书中的重模型、双手数据集、头显边缘部署与当前实现之间的关系，见
`docs/ai_scope_and_proposal_alignment.md`。其中涉及算法或部署目标变更的内容，必须先与
指导老师确认，不能用本次离线结果代替项目口径审批。

## 11. 证据文件

- 正式机器可读报告：
  `experiments/intent_prediction/reports/second_round/20260823T092028_938598Z_report.json`
- 冻结选择：
  `experiments/intent_prediction/reports/second_round/20260823T092028_938598Z_selection.json`
- 正式中文解释：`docs/intent_prediction_second_round_report.md`
- validation 候选表：
  `experiments/intent_prediction/reports/second_round/20260823T092028_938598Z_validation_candidates.csv`
- 正式配置：`experiments/intent_prediction/configs/h2o_second_round.json`
- 运行入口：`experiments/intent_prediction/scripts/run_second_round.py`
