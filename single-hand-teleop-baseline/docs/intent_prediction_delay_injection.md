# 单右手意图预测：延迟、抖动、丢包冻结回放报告

日期：2026-08-29
状态：正式矩阵已完成，retention gate **4/6，未通过**
控制决策：继续以 `hold-last` 为控制参考；`residual_motion4` 只保留影子研究

## 1. 为什么先做这个实验

第二轮 v2 模型能预测未来 50/100/150 ms，但“未来姿态误差略低”不等于“网络延迟下的
接收端显示更准”。真正需要回答的问题是：

> 接收端在某个时刻只能拿到较旧的源帧时，用冻结模型预测到当前时刻，是否稳定优于直接
> 保持最后一帧？

因此本轮没有先扩大模型、调 test 门槛或做多 seed 排名，而是冻结同一个模型、gate、数据、
网络矩阵和保留门槛，直接检验模型是否值得继续。

## 2. 运行前冻结的协议

配置：
[`delay_injection_v1.json`](../experiments/intent_prediction/configs/delay_injection_v1.json)

配置文件运行前 SHA-256：

`09d63d9696118350f2bd0a27b7f12d3d266ba96a1e4c251c1c63b40a018bf538`

冻结内容：

| 项目 | 冻结值 |
|---|---|
| 模型 | `residual_motion4` |
| checkpoint SHA | `0146d8ecdc117d9ac9d49a22f087199ca0fe1f60173fb10a2908cabf3043fee6` |
| history | 30 帧、30 Hz |
| horizons | 50/100/150 ms |
| 延迟 | 0/20/50/100 ms |
| 抖动 | 0/5/10 ms，确定性均匀分布 `[-jitter,+jitter]`，负延迟夹到 0 |
| 丢包 | 0%/1%/5%，固定 seed |
| 场景数 | 4 × 3 × 3 = 36 |
| 接收策略 | 每个接收 tick 使用已到达包中 source frame 最新的一帧 |
| 预测对齐 | 在 0 ms hold 与 50/100/150 ms 冻结预测间线性插值 |
| 超 horizon | 夹到 150 ms，并单独报告比例 |

H2O test 使用 52 个 subject4 连续序列、28,680 个逐帧发送窗口。这里使用 stride=1，
因为网络模拟需要每个 30 Hz 源帧都能发包；没有重新选择模型或拟合 gate。

primary retention 场景预先限定为 50/100 ms 的全部抖动和丢包组合，共 18 个场景。

## 3. H2O v2 正式结果

primary 场景累计 513,742 个接收时刻：

| 指标 | hold-last | raw | gated | gated 相对 hold |
|---|---:|---:|---:|---:|
| MAE | 0.021030 | 0.020863 | 0.020625 | 改善约 1.92% |
| RMSE | 0.069828 | 0.068241 | 0.068465 | **改善 1.95%** |
| P95 absolute error | 0.071720 | 0.067005 | 0.067819 | **改善 5.44%** |
| 越界率 | 0 | 0 | 0 | 持平 |

validation 预先确定的 q90 动态子集有 52,365 个接收时刻：

- raw RMSE 改善 `2.78%`；
- gated RMSE 改善 `2.67%`；
- gated P95 改善 `3.46%`。

六项 retention gate：

| 门槛 | 结果 |
|---|---|
| 聚合 RMSE 至少改善 3% | **失败：1.95%** |
| 聚合 P95 不回退 | 通过：改善 5.44% |
| q90 RMSE 至少改善 5% | **失败：2.67%** |
| 任一 primary 场景 RMSE 回退不超过 1% | 通过：最差回退 0% |
| 越界率为 0 | 通过 |
| 预测可用覆盖率至少 95% | 通过：100% |

最终为 **4/6，retention gate 未通过**。

这不是“模型完全无效”：36 个场景中没有出现 gated RMSE 回退，尾部误差也有改善；但平均收益
小于运行前门槛，不能据此承担控制复杂度。

## 4. 真实摄像头 JSONL 探索性回放

输入：`outputs/session_20260828_210132.jsonl`，原文件 SHA-256：

`e69846c4951f7c2739f9c2eaef8290b1fdf7c5c80aa3d80a306f25eaca330cc2`

该日志共 808 帧：

| 状态 | 帧数 |
|---|---:|
| invalid | 96 |
| warming up | 254 |
| predicted | 458 |
| valid preview | 712 |
| 连续有效片段 | 25 |

同一 primary 场景矩阵累计 12,103 个接收时刻：

| 指标 | hold-last | raw | gated | gated 相对 hold |
|---|---:|---:|---:|---:|
| RMSE | 0.179923 | 0.176771 | 0.176588 | **改善 1.85%** |
| P95 absolute error | 0.479371 | 0.471148 | 0.472022 | **改善 1.53%** |

动态 q90 gated RMSE 改善 `5.12%`，说明快速动作段确有预测信号；但两个工程问题更关键：

- 预测可用覆盖率只有 `65.17%`；warming up、丢检和短片段会退回 hold-last；
- `30.55%` 的接收时刻帧龄超过 150 ms，只能夹到最远 horizon；帧龄 P95 约 171.87 ms。

真实日志里的“真值”仍是之后时刻的 MediaPipe/SVH 映射输出，不是实体手关节传感器真值，
所以这部分只作跨域趋势检查，不参与 H2O retention gate。

## 5. 当前决策

```text
Unity / 后续控制参考：hold-last
residual_motion4：默认关闭、仅 prediction shadow
延迟补偿有效：不得宣称
继续扩大模型/多 seed：暂缓
```

暂缓多 seed 的原因不是算力不足，而是当前最核心的结构性结论已跨 H2O 和真实日志出现：
整体收益约 2%，真实域预测覆盖仅约 65%。多 seed 可以估计波动，却不能解决低帧率、丢检、
history warming 和 horizon 不足这些更上游的问题。

## 6. 下一步建议

1. 录制固定右手视频集，保留真实时间戳，覆盖 open/fist/pinch、慢/快屈伸、遮挡和出入画面。
2. 优先改善输入连续性并测量 coverage：确认稳定 20–30 FPS 时 predicted 覆盖能否显著提高。
3. 若未来把网络目标定在 100–200 ms，再决定是否增加 200 ms horizon；不能用当前结果直接外推。
4. 只有覆盖率和真实域收益先达到冻结门槛，才值得做 3–5 seed 或 LOSO 稳定性训练。
5. 无论后续模型结果如何，仍不授权其直接进入实体 SVH。

## 7. 复现

```powershell
conda activate handai-intent-prediction
python -X utf8 experiments\intent_prediction\scripts\run_delay_injection.py `
  --config experiments\intent_prediction\configs\delay_injection_v1.json `
  --data-root D:\VR\HandAi\datasets\H2O\processed\cross_subject_v2_open_release `
  --output-root experiments\intent_prediction\outputs\delay_injection_v1 `
  --runtime-jsonl outputs\session_20260828_210132.jsonl `
  --runtime-config configs\unity_udp_preview.yaml
```

正式机器可读证据：

- [完整 JSON 报告](../experiments/intent_prediction/reports/delay_injection/20260829T112008_630685Z_report.json)
- [36 场景汇总 CSV](../experiments/intent_prediction/reports/delay_injection/20260829T112008_630685Z_scenario_metrics.csv)
- [H2O 逐序列 CSV](../experiments/intent_prediction/reports/delay_injection/20260829T112008_630685Z_sequence_metrics.csv)
- [真实日志场景 CSV](../experiments/intent_prediction/reports/delay_injection/20260829T112008_630685Z_runtime_scenario_metrics.csv)
- [真实日志逐片段 CSV](../experiments/intent_prediction/reports/delay_injection/20260829T112008_630685Z_runtime_sequence_metrics.csv)
