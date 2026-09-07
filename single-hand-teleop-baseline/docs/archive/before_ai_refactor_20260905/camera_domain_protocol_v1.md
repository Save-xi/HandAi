# 真实摄像头域评测协议（开发草稿 v1）

- 日期：2026-08-31
- 状态：**development 草稿，尚未冻结盲测门槛**
- 适用范围：单右手视觉主链、`control_representation`、`svh_9ch` Unity 虚拟预览与默认关闭的预测影子
- 不适用：双手、实体 SVH、串口/RS485、远端网络或预测结果写入 UDP

## 1. 目的与结论边界

当前 v2 模型来自 H2O pose-only 代理标签，retention gate 为 4/6，只有一条真实摄像头日志。
本协议先回答两件彼此独立的问题：

1. **算法效用**：在固定真实视频的源时间轴上，gated prediction 是否比 hold-last 更接近后续
   `svh_preview`；
2. **实时能力**：现有电脑运行真实摄像头时，检测、映射和异步 prediction worker 的吞吐与覆盖如何。

未来 `svh_preview` 只是视觉映射的伪真值，不是实体关节真值，也不是用户主观意图真值。
重叠的 receiver tick 不能当作独立统计样本，本阶段不做显著性夸大。

## 2. 为什么不能直接使用旧的视频运行命令

现役 `src/main.py --video-file` 的顶层 `timestamp` 来自处理 wall-clock。视频被快读或慢读时，
`observed_fps`、30 帧历史和 50/100/150 ms horizon 会随电脑速度改变，不能代表录制视频时间。

本协议改用独立评测脚本：

- 优先读取容器 `CAP_PROP_POS_MSEC`；
- PTS 缺失、重复或倒退时，按 `frame_index / nominal_fps` 回退；
- 所有媒体时间戳必须严格递增；
- 离线算法预测同步重放 baseline JSONL，不使用 latest-only worker；
- 处理耗时另记为 `offline_processing_capacity`，绝不冒充源视频 FPS。

该脚本不创建 `JsonExporter`，不发送 UDP，不修改 Unity payload。

## 3. 两阶段视频集

### 3.1 开发集 D（现在采集，不能用于最终准入结论）

固定机位、720p 以上、建议 30 FPS；右手正对镜头。输入是否镜像必须全程一致，并在命令中明确。

| ID | 内容 | 建议时长 |
|---|---|---:|
| V1 | open → fist 慢速循环 ×3 | 20 s |
| V2 | open → pinch → open 慢速循环 ×3 | 20 s |
| V3 | 半握 → 全握连续渐变 ×3 | 20 s |
| V4 | 快速屈伸，镜头保持固定 | 15 s |
| V5 | 遮挡右手约 3 s 后恢复 | 15 s |
| V6 | 右手出画 → 入画 ×3 | 15 s |
| V7 | 连续随意动作，贴近预期操作 | 30 s |

开发集用途：检查时间轴、输入连续性、失败分桶，并据此提出但**不回填到开发结果**的盲测门槛。

### 3.2 盲测集 B（开发结束并冻结协议后另录）

本节只保留开发阶段如何形成候选门槛的历史说明，不提供正式盲测操作。正式 B1–B7 的采集、身份、共享 campaign 状态和命令仅以 `camera_domain_blind_protocol_v1.md` 与 `camera_domain_blind_v1.json` 为准；不得把本开发配置或本文 SHA 当作正式盲测凭据。

当前机器配置仍是 `protocol_stage=development`、`blind_policy.enabled=false`，程序会主动拒绝
`--role blind`，因此还不能误跑成正式准入。

## 4. 开发集运行命令

```powershell
conda activate handai-intent-prediction
cd D:\VR\HandAi\single-hand-teleop-baseline

python -X utf8 experiments\intent_prediction\scripts\run_camera_domain_eval.py `
  --role development `
  --video V1=D:\HandAiVideos\V1.mp4 `
  --video V2=D:\HandAiVideos\V2.mp4 `
  --video V3=D:\HandAiVideos\V3.mp4 `
  --video V4=D:\HandAiVideos\V4.mp4 `
  --video V5=D:\HandAiVideos\V5.mp4 `
  --video V6=D:\HandAiVideos\V6.mp4 `
  --video V7=D:\HandAiVideos\V7.mp4 `
  --input-not-mirrored
```

如果录制软件保存的是自拍镜像画面，把最后一项改为 `--input-mirrored`。不得通过观察结果后再切换。

只为检查安装是否正常时可以少量视频配合 `--allow-partial`；这种运行永远不能作为开发集完整结果，
盲测也禁止使用该参数。

## 5. 输出与可复现清单

每次运行在 `experiments/intent_prediction/outputs/camera_domain_eval_v1/<UTC>/` 生成：

- `baseline_jsonl/V*.jsonl`：媒体时间轴上的 canonical baseline；
- `camera_domain_report.json`：完整机器可读报告；
- `camera_domain_report.md`：人类可读摘要；
- `video_summary.csv`：逐视频输入、时间轴、吞吐与模型指标；
- `scenario_metrics.csv`、`sequence_metrics.csv`：冻结网络矩阵明细。

报告自动记录视频/JSONL/config/protocol/checkpoint/selection SHA、Git revision、环境、PTS 回退次数、
输入镜像选项及安全边界。

显式运行 `src/main.py ... --save-jsonl` 时，Python 还会用同一个 `run_id` 生成三件旁路证据：

- `session_<run_id>.jsonl`：实际发往 Unity 前冻结的 baseline 帧；
- `prediction_session_<run_id>.jsonl`：latest-only worker 真正完成并落盘的影子结果；
- `runtime_session_<run_id>.json`：原子冻结的配置 SHA、精确路径、日志 SHA/字节数/行数、首末帧、
  模型身份与 worker 提交/完成/丢弃计数。

`run_id` 只存在于文件名和 manifest，不进入 canonical payload 或 UDP，因而不改变 Unity contract。

## 6. 三类指标必须分开解释

### 6.1 源视频与输入连续性

- `nominal_fps`：容器声明帧率；
- `median_interval_fps`：媒体时间戳相邻差的中位数换算；
- `duration_based_fps`：`(帧数 - 1) / (末 PTS - 首 PTS)`，表示整段首末时长均值；
- `effective_fps`：仅为兼容旧 JSON 读取方保留的 `median_interval_fps` 别名，新报告不再把它笼统写成“源 FPS”；
- `timestamp_source_counts`：容器 PTS 与回退帧数量；
- `detected_fraction`、`control_ready_fraction`、`svh_valid_fraction`；
- 连续片段数、warming_up/invalid_input 状态数。

### 6.2 离线算法效用

- hold-last、raw、gated 的 RMSE/MAE/P95；
- gated 相对 hold-last 的总体与 validation-q90 动态段改善；
- 条件 prediction availability、receiver coverage、端到端 prediction coverage；
- 越界率。

这些指标全部基于媒体时间轴上的同步预测重放。

### 6.3 真实异步运行能力

真实摄像头运行仍使用：

```powershell
python -X utf8 src\main.py --config configs\unity_udp_preview.yaml `
  --camera-index 0 --prediction-shadow --save-jsonl
```

停止 Python 后，`runtime_session_<run_id>.json` 会从 `running` 原子冻结为 `completed`（Ctrl+C 为
`interrupted`）。退出 Unity Play 后，Console 会打印 `unity_timing_*.json` 的实际路径；该文件默认写到
`Application.persistentDataPath/HandAiDiagnostics/`，每项指标最多保留最近 4096 个样本，同时记录总样本数。

把同一 `run_id` 的两份 JSONL 与 Unity 摘要传给评测脚本：

```powershell
python -X utf8 experiments\intent_prediction\scripts\run_camera_domain_eval.py `
  --role development `
  --video V1=D:\HandAiVideos\V1.mp4 `
  --video V2=D:\HandAiVideos\V2.mp4 `
  --video V3=D:\HandAiVideos\V3.mp4 `
  --video V4=D:\HandAiVideos\V4.mp4 `
  --video V5=D:\HandAiVideos\V5.mp4 `
  --video V6=D:\HandAiVideos\V6.mp4 `
  --video V7=D:\HandAiVideos\V7.mp4 `
  --input-not-mirrored `
  --live-baseline-jsonl D:\...\session_<run_id>.jsonl `
  --live-prediction-jsonl D:\...\prediction_session_<run_id>.jsonl `
  --live-unity-timing-json C:\...\HandAiDiagnostics\unity_timing_*.json
```

manifest 默认从 baseline 同目录按 `run_id` 推导；也可显式传 `--live-session-manifest`。评测器会拒绝
跨 run_id、重复帧、缺帧引用、逐帧时间戳错配、模型 SHA 漂移及日志 SHA/行数不符，不再对两份日志
做宽松交集后假装配对成功。通过严格配对后只分析：

- source timestamp 的中位间隔/首末时长 FPS；
- Python `source.read()` 返回后的处理与 UDP send attempt 耗时；
- worker 结果覆盖；
- predicted 覆盖；
- 推理耗时与状态计数；
- 可选的 UDP delivery、Unity 主线程排队及 `source.read()` 返回后到目标应用的 P50/P95/max；
- Unity accepted/overwritten/frame-gap/rejected/stale/watchdog 计数及三项硬件关闭边界。

这里的 timing 起点是输入源 `read()` 已经返回，不包含相机曝光、此前等待、显示刷新、人体反应或
实体机械响应；Unity 的“target apply”也不等于画面已经渲染。

它不代替 6.2 的算法效用结果。

## 7. 开发集不设上线门槛

开发集报告固定返回：

`development_only_no_release_decision`

原草稿中的“FPS ≥20、条件覆盖 ≥90%、端到端覆盖 ≥85%、动态改善 ≥3%”只有一条日志参考，
当前不作为正式门槛。开发结果完成后，门槛必须结合：

- 每段强制约 1 s warming 的理论上限；
- 输入连续片段长度与丢检结构；
- 多段视频的逐段分布，而不是只看样本加权总数；
- 项目能够接受的实际失败成本；
- 仍不可放宽的零越界与 shadow-only 约束。

正式判定转入 `camera-domain-blind-v1` campaign 后，按 `camera_domain_blind_protocol_v1.md` 的本机流程性单次纪律执行；本开发草稿不提供 WORM 或跨机器一次性保证。

## 8. 三分支决策逻辑

1. **时钟或输入不健康**：PTS/回退异常、有效片段过短、频繁丢检或真实 worker 覆盖不足。
   先修采集/连续性，不评价模型，也不把问题归因于 v2/v3。
2. **输入健康但 v2 无效或回退**：允许立项一次预注册 v3 实验——normalized-perspective +
   顺序 `GestureStabilizer` 重标、升级 contract、重新训练；不覆盖 v2。
3. **v2 已稳定有益**：继续保持 default-off shadow。v3 只做可选消融，不因“新版本”而强行替换。

只有 v3 在冻结的盲测集上改善，才值得投入多 seed/LOSO；否则如实保留 hold-last 是合格结论。

## 9. 不可突破的纪律

- prediction 永不进入 Unity UDP；
- `range_violation_rate` 必须为 0；
- 不使用开发视频回选最终盲测模型或修改盲测门槛；
- 不把伪真值写成真实意图、实体关节精度或已实现延迟补偿；
- 不增加 200 ms horizon，除非先测得目标系统实际延迟；
- 真机 SVH 必须另走协议、标定、限位、watchdog、急停与断线验收。

## 10. 当前参考值（不参与未来盲测判定）

2026-08-28 单条摄像头日志曾得到：条件预测率约 65.17%、接收覆盖约 94.97%、端到端预测覆盖
约 61.90%、中位处理 FPS 约 12.5。由于该日志使用 wall-clock 且只有一个 session，只保留为历史诊断，
不能用于冻结最终结论。

2026-09-01 的真实 Play 运行严格配对 1,093 帧：source 时长均值 22.78 FPS、worker 结果覆盖 100%、
有效控制帧 predicted 覆盖 85.12%、Unity 应用 1,090 帧、frame gap 3、rejected/stale 均为 0，
source.read 返回后至 Unity 应用 P95 53.79 ms。用户另行确认充分张开、握拳/捏合跟随和遮挡后
安全张开三项视觉检查均通过。完整证据见
[2026-09-01 实时验收记录](live_runtime_acceptance_20260901.md)。这些值仍只作为 development 参考，
不参与未来 B1–B7 的门槛回填。
