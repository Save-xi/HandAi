# AI 部分与计划书口径对齐说明

## 1. 为什么需要这份说明

当前工程已经形成一条可运行、可离线验收的单右手路线，但它与计划书中列举的完整技术路线
并不等价。为了避免中期检查或结题时把“工程上选择了更合适的实现”误写成“计划书中的所有
算法和设备均已完成”，本文件把事实、选择和待确认变更分开记录。

本文件不修改计划书，也不代替指导老师或项目管理单位的确认。

## 2. 当前已经完成并有证据的范围

```text
单右手图像/视频
  -> MediaPipe 21 点检测与右手筛选
  -> control_representation
  -> svh_preview 9 通道
  -> JSON / JSONL / 可选 Unity UDP 预览
```

此外，AI 支线已经完成：

- H2O pose-only 数据到单右手 9 通道序列的可复现转换；
- hold-last、线性、Kalman、GRU、TCN、Transformer 第一轮比较；
- 零残差初始化 GRU、validation 门控和第二轮冻结选型；
- subject1+2 / subject3 / subject4 跨人员切分与自动验收门槛；
- 明确披露 subject4 已被第一轮使用，第二轮不是项目历史上的全新盲测。

这些证据只覆盖公开姿态序列上的离线预测，不覆盖摄像头域偏移、头显部署、5G、Unity
端到端时延或真实机械手安全。

## 3. 计划书算法与当前实现的关系

| 计划书相关方向 | 当前事实 | 正确表述 |
|---|---|---|
| Cascade R-CNN / ResNet-101 等重模型 | 当前主链未实现，也没有相应训练与量化证据 | 不能写成已完成；如结题口径要求，应做小规模统一数据、统一指标的对照实验 |
| MMPose / MediaPipe 方案选择 | 当前选择 MediaPipe，并已完成单右手离线与 Unity preview 证据 | 可写成阶段性工程选型，但应保留选择依据和指标，不应宣称所有备选方案均完成 |
| InterHand2.6M 或双手方向 | 当前项目主线明确为单右手 | 不因计划书出现双手数据集就把主线悄悄改成双手 |
| VR 头显边缘意图预测 | 当前预测器只在本机近端 GPU/PC 离线运行 | 不能写成头显部署已完成；改为近端边缘 PC 需要先与指导老师确认口径 |
| HoloLens / Kinect / 5G / 真机 | 当前不属于已验收链路 | 只能列为后续实验室联调阶段 |

建议在下一次和指导老师沟通时形成一页书面决定：

1. MediaPipe 是否作为正式主方案，重模型是否只保留为小规模对照；
2. 意图预测最终部署目标是 VR 头显还是近端边缘 PC；
3. 单右手是否作为本项目可验收主线，双手是否明确降为扩展项；
4. 论文/结题使用哪些指标以及哪一批数据作为真正最终测试集。

## 4. timing v1 与影子诊断对齐约束

仓库已经有逐帧 timing 字段、JSON/JSONL contract 和 Unity UDP 诊断。后续预测器影子模式
必须复用现有 `frame_index`、顶层 `timestamp` 和 Unix epoch 毫秒时钟。Phase 1.5 为避免
后台推理打乱完整 baseline 会话，使用独立 prediction JSONL，但每条仍携带完整 canonical
payload 并按 source frame 对齐，不能形成无法 join 的平行日志。

实现审计后确认 `timing v1` 是严格冻结的 9 字段对象，因此不直接增加键。本次新增可选
`prediction_diagnostics v1`，并强制：

- `source_frame_index == frame_index`；
- `source_timestamp_unix_ms == timestamp * 1000`；
- 当前帧 UDP 先发送，主线程只做非阻塞提交；
- 模型在 latest-only 后台 worker 推理；
- 预测诊断只写独立本机 prediction JSON/JSONL，不进入 baseline 日志或 Unity 数据报。

该对象记录：

- 历史窗口数量、覆盖时长与观察帧率；
- 模型推理开始/结束时间；
- 原始、门控和 hold-last 方案标识；
- 历史 span、观察帧率、运动分数和门控系数；
- selection/checkpoint SHA 与确定性回退原因。

schema、两类 JSONL、测试和说明已同步。后续延迟注入的释放、丢弃、覆盖和乱序诊断也沿用
同一帧号/时钟，不得破坏 timing v1。预测器异常时仍使用当前原始 preview 路径，不能让
可选扩展成为 baseline 启动前提。

## 5. 后续八周建议里程碑

| 周次 | AI 侧任务 | 可验收交付物 | 需要协作方 |
|---:|---|---|---|
| 1 | 与导师确认算法和部署口径 | 一页确认纪要；冻结指标与最终测试策略 | 指导老师、项目负责人 |
| 2（已完成） | 默认关闭的预测影子模式 | 不改变 UDP/baseline 的独立 hold/raw/gated prediction JSONL；30 Hz 重采样与 checkpoint smoke | Unity/接口成员确认字段 |
| 3（已完成） | H2O 与现有 JSONL 确定性回放 | replay 配置、输入哈希和回归测试 | 无需摄像头 |
| 4（已完成） | 0/20/50/100 ms 延迟、抖动和丢包注入 | 36 场景、逐序列 CSV 与 retention gate；结果 4/6 | 无需摄像头 |
| 5–6 | 多种子、跨 subject 或 subject bootstrap | 均值、方差、置信区间和失败案例 | 指导老师确认论文口径 |
| 7 | 根据目标设备决定 ONNX FP32/INT8 | 模型大小、误差变化、P50/P95 延迟 | 设备/平台成员 |
| 8 | 实验室联调前冻结 | 模型哈希、配置、回退规则、接口清单 | Unity、通信、控制成员 |

如果目标设备和计划书变更尚未确认，第 7 周不得提前宣称“头显轻量化部署完成”。

## 6. 下一阶段准入条件

当前 v2 第二轮和网络扰动 retention gate 均未全部通过，只允许默认关闭的影子学习；现阶段
明确不改变 UDP。未来若产生新模型并重新申请准入，至少应满足：

- 影子模式对 baseline 输出零影响；
- timing v1 保持冻结，prediction diagnostics、schema、JSONL 和测试逐帧对齐；
- checkpoint 缺失、输入无效、历史不足或推理异常时确定性回退；
- 延迟注入实验在事先冻结的指标上通过全部 retention gate（当前 4/6，未满足）；
- 使用新的 validation/CV 协议决定 raw 与 gated，不回看 subject4 重选；
- 有可用摄像头后完成个人视频域验证；
- 真实硬件接入仍需独立的限位、watchdog、急停和协议验收。
