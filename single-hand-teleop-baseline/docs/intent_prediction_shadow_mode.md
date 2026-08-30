# 单右手 9 通道意图预测影子模式

当前版本：2026-08-29 Phase 1.5

## 1. 当前结论

当前加载的是依据 open-release 映射重新标注、重新训练的 v2
`residual_motion4` residual GRU。它只能以**默认关闭、只观察、不控制**的方式运行。

模型在 H2O v2 test 上相对 hold-last 改善了 MAE、RMSE 和 P95，但总体 RMSE 与高运动 RMSE
没有达到预注册门槛，因此 `offline_gate_passed=false`。这不影响用它学习和记录影子诊断，
但禁止把它描述成“延迟补偿已有效”，更不能发送给 Unity 或真机。

完整风险清理和指标见 [Phase 1.5 二轮风险加固完成记录](phase15_risk_hardening.md)。

## 2. 实时结构

```text
当前 canonical baseline/control/svh_preview
  -> 原始 frozen payload 立即发送给本机 Unity UDP
  -> 非阻塞提交到容量 1 的 latest-only worker
  -> baseline last JSON / session JSONL 正常写出

后台 worker
  -> 检查 svh_9ch、frame index、timestamp
  -> 按 timestamp 重采样到固定 30 Hz / 30 帧 / 966.67 ms
  -> 计算 hold-last / raw residual / frozen gated residual
  -> 写独立 latest_prediction_shadow.json
  -> 可选写独立 prediction_session_*.jsonl
```

因此当前实现保证：

- 模型只处理单右手 9 通道时序，不预测 21 个关键点；
- 不改写 `svh_preview.target_positions`；
- baseline UDP 永远不含 `prediction_diagnostics`；
- 模型推理不阻塞下一帧采集或 UDP；
- 输入队列和结果队列都有上限，不会用排队累计实时延迟；
- 缺模型、SHA/映射不匹配、输入 invalid、历史不足或推理错误时，baseline 与 Unity 继续运行。

## 3. 输出文件

显式加 `--prediction-shadow --save-jsonl` 后：

| 文件 | 内容 |
|---|---|
| `outputs/latest_svh_9ch.json` | 最近一帧原始 baseline/control/preview，不含预测 |
| `outputs/session_*.jsonl` | 完整原始 baseline 会话，不含预测 |
| `outputs/latest_prediction_shadow.json` | 最近一个已完成的影子结果，含诊断 |
| `outputs/prediction_session_*.jsonl` | 后台完成的影子结果，用 source frame 对齐 baseline |

对齐关系：

```text
prediction_diagnostics.source_frame_index == frame_index
prediction_diagnostics.source_timestamp_unix_ms == timestamp * 1000
```

预测日志可能因 latest-only 丢帧而少于 baseline 日志；这是为了不让慢模型拖累实时链。

## 4. 运行方法

激活包含 PyTorch 和 baseline 依赖的环境：

```powershell
cd D:\VR\HandAi\single-hand-teleop-baseline
conda activate handai-intent-prediction
```

摄像头 + Unity 预览 + 影子诊断：

```powershell
python -X utf8 src\main.py `
  --config configs\unity_udp_preview.yaml `
  --camera-index 0 `
  --prediction-shadow `
  --save-jsonl
```

视频文件、无 GUI：

```powershell
python -X utf8 src\main.py `
  --config configs\svh_9ch_preview.yaml `
  --video-file D:\path\to\right_hand_video.mp4 `
  --prediction-shadow `
  --headless `
  --save-jsonl
```

不要用 `configs/default.yaml` 验收预测。它有意关闭 SVH preview，影子层会正确输出
`invalid_input`。

## 5. 冻结身份与映射闸门

当前证据文件：

- selection：`experiments/intent_prediction/reports/second_round/20260829T024400_143036Z_selection.json`
- report：`experiments/intent_prediction/reports/second_round/20260829T024400_143036Z_report.json`
- checkpoint：`experiments/intent_prediction/outputs/second_round_v2_open_release/20260829T024400_143036Z/checkpoints/residual_motion4.pt`

checkpoint SHA-256：

```text
0146d8ecdc117d9ac9d49a22f087199ca0fe1f60173fb10a2908cabf3043fee6
```

mapping contract：

```text
version = svh9-label-v2-open-release
sha256 = 453c18c3ae851c325cd3b882409800c074f97f7f7ad898865d1d4895b2e318a0
```

加载时会同时核对：

1. selection schema 与 validation-only 冻结顺序；
2. report 引用的 selection SHA；
3. selection 与 checkpoint SHA；
4. H2O manifest/selection/checkpoint `data_contract`；
5. 当前运行配置的 mapping contract version/hash；
6. checkpoint 模型名、历史长度与 horizon 数。

任何一项不一致都返回 `initialization_error`。旧 v1 checkpoint 不会被静默兼容。

## 6. 主要配置

| 配置 | 作用 |
|---|---|
| `prediction_shadow_enabled` | 总开关，默认 false |
| `prediction_shadow_selection_path` | 冻结 selection |
| `prediction_shadow_checkpoint_path` | 本机 checkpoint |
| `prediction_shadow_report_path` | 含离线 acceptance 的正式报告 |
| `prediction_shadow_require_offline_gate` | true 时 gate 未通过就拒绝加载；当前 false 只允许影子学习 |
| `prediction_shadow_device` | `auto`、`cpu` 或 `cuda` |
| `prediction_shadow_horizon_ms` | `[50, 100, 150]` |
| `prediction_shadow_target_fps` | 模型输入重采样频率，当前 30 Hz |
| `prediction_shadow_max_frame_gap_ms` | 原始有效帧间隔超过该值就清空历史，当前 100 ms |
| `prediction_shadow_result_queue_size` | 已完成结果队列上限 |
| `prediction_shadow_worker_shutdown_timeout_s` | 正常退出时有界等待在途推理 |

门控参数不在实时数据上重新拟合，直接读取 validation 冻结的 selection。

## 7. 诊断状态

| 状态 | 含义 | 行为 |
|---|---|---|
| `initialization_error` | 依赖、selection、report、checkpoint、SHA 或映射契约失败 | 不推理，只记录原因 |
| `invalid_input` | 当前不是有效 9 通道 preview，或时戳/帧号非法 | 清空历史 |
| `warming_up` | 原始有效时间跨度不足以覆盖 966.67 ms | 记录准备进度 |
| `predicted` | 30 Hz 重采样、推理和门控均成功 | 只写独立本机日志 |
| `inference_error` | 模型输出或运行异常 | 本次运行永久停用模型 |

`frame_index`/timestamp 倒序，或原始相邻有效帧间隔超过 100 ms，都会重置历史。

## 8. 当前验证证据

- `handai-intent-prediction` 环境 Python 全量测试：`160 passed`。
- 新 checkpoint CUDA 30 帧 smoke：`status=predicted`，固定 966.67 ms 历史窗。
- 2026-08-28 真实日志回放：808 帧中 458 predicted；原始帧率 p50 约 12.49 Hz；
  CUDA inference p50/p95/max 约 1.01/1.34/1.62 ms。
- 2026-08-29 摄像头无界面 60 帧短跑：baseline 60 行、prediction 60 行，worker 输入/结果
  丢弃均为 0；当时右手未入镜，60 行均为 `invalid_input`，所以这只证明运行链，不证明手势效果。
- 注入 50 ms 慢推理的单元测试：实时线程提交小于 20 ms，推理在后台完成。

## 9. 扰动实验之后仍然没有证明什么

- 0/20/50/100 ms × 0/5/10 ms 抖动 × 0/1/5% 丢包已经完成，但 retention gate 仅 4/6；
- H2O primary gated RMSE 只改善 1.95%，未证明 prediction 达到预注册的保留门槛；
- 没有个人固定视频 holdout、多 seed/LOSO、sequence bootstrap；
- 没有 ONNX/INT8、头显、5G、双手或真机安全链。

当前正确结论是继续使用 hold-last 作为控制参考，模型只保留为影子诊断；详见
[延迟、抖动、丢包冻结回放报告](intent_prediction_delay_injection.md)。
