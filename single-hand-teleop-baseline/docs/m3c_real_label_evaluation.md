# M3-C：新会话采集与人工真值评测工具

2026-09-26。工具和合成样例已可运行；**新人员/会话真实视频、人工标注、独立效果结论尚未完成**。V1–V7、mirror_check、v1_pilot 仅有历史开发身份，不进入新留出。原始视频与标注默认留在 ignored 的 `outputs/datasets/`，不上传 GitHub。

## 实际采集安排

本轮预先固定 **independent_person** 切分：P01 为开发，P02/P03 为留出。每人 S01/S02 两次独立会话；每次录五段，每段约 20–30 秒。可读计划在 [camera_m3c_plan.json](../experiments/intent_prediction/configs/camera_m3c_plan.json)，它不是已经取得的数据清单。

| 片段名 | 内容 | 应覆盖 |
|---|---|---|
| `01_static_rotate` | 静态张手及旋转 | 原图尺寸、左右手、角度 |
| `02_fast_flex` | 快速屈伸 | 加速、减速、恢复 |
| `03_pinch_transition` | 捏合及转换 | open/fist/pinch 的连续边界 |
| `04_side_view` | 侧视 | 自遮挡、检测质量 |
| `05_occlusion_recovery` | 遮挡/出框/恢复 | 漏检、无输出与预热 |

第一批先提供 P01/S01 五段。空目录已建于 `D:\VR\HandAi\single-hand-teleop-baseline\outputs\datasets\camera_m3c_raw\P01\S01`；这里只是存放入口，未生成视频或人员身份记录。录制时确认右手在画面中、保留原始帧率/尺寸/音视频时间戳，不裁掉失败时段。每段身份、会话、用途与类别由实际采集人确认。禁止同会话跨开发/留出；如看过留出结果后按它调参，该留出须改记为开发诊断，并另留新人员/会话。

## 数据契约与入口

使用已有 Conda 环境。以下是 Windows CMD 中可复制的基本命令；PowerShell 可将 `cd /d` 改为 `Set-Location -LiteralPath`。

```bat
conda activate handai-intent-prediction
cd /d D:\VR\HandAi\single-hand-teleop-baseline
python -X utf8 experiments\intent_prediction\scripts\run_camera_m3c.py import --video outputs\datasets\camera_m3c_raw\P01\S01\01_static_rotate.mp4 --person-id P01 --session-id S01 --clip-id P01_S01_01_static_rotate --scenario static_rotate --role development --split-policy independent_person
python -X utf8 experiments\intent_prediction\scripts\run_camera_m3c.py annotate --clip-id P01_S01_01_static_rotate
python -X utf8 experiments\intent_prediction\scripts\run_camera_m3c.py evaluate
```

`annotate` 只监听 `127.0.0.1:8765`，显示的逐帧 JPEG 由原视频按帧号解码；视频播放器仅供预览，浏览器不支持编码也能逐帧标注。每帧保留原图宽高、帧号、容器 PTS 或显式 FPS/连续性回退来源。若源视频无有效名义 FPS、帧不可解码或导入/评测时帧号、尺寸、PTS 不一致，拒绝评测。原始素材路径保留在 manifest，导入不复制、不改动视频。

数据集 `dataset.json` 记录人员、会话、片段、类别、用途及 real/synthetic 身份。各片段 `manifest.json`、`timeline.jsonl`、`annotations.json` 保存来源、时间和真值。人工关键点每帧明确 `sampling_purpose`：`uniform`、`failure_targeted`、`dense_transition`。建议每段先均匀选约五帧，独立于模型是否检出；专项失败另选并分表。21 个点各有原图像素 `x/y` 及人工 `visible`；未处理点不可保存为完成 GT，人工不可见点不计入定位分母。右手不存在/出框用 `presence=absent`，无记录是未标注。约五帧/段只支持稀疏二维检测评测；50/100/150ms 未来二维轨迹误差需要转换附近局部密集点标签，目前不声称有该真值。

手势在 `gesture_intervals` 按原始 PTS 保存连续半开区间 `[start_ms,end_ms)`，类别为 open/fist/pinch/unknown/absent。unknown 是人工确认的未知手势；absent 是无右手；区间外是未标注。额外的 `transition_events` 记录转换起止 PTS、from/to，可标时间不确定窗口。事件预测落入窗口记 0ms，早于起点记负滞后，晚于终点记正滞后；同类事件在 ±250ms 内求一对一最大匹配。没有显式事件时报告可从相邻连续手势区间边界生成零宽候选，但人工事件窗口更准确；未标注孔洞或右手离场都隔断事件，不跨段匹配。

## 评测与结论边界

`evaluate` **默认只读 development**。在开发部分确定配置、20px PCK 阈值和固定三方法后，另行执行：

```bat
conda activate handai-intent-prediction
cd /d D:\VR\HandAi\single-hand-teleop-baseline
python -X utf8 experiments\intent_prediction\scripts\run_camera_m3c.py evaluate --role heldout
```

评测重用 `HandPipeline`、M3-A 图像宽度几何/连续释放以及 M3-B 的保持通道、保持姿态和已选定 `linear_pose_w2`。其计算成本分别跑纯算法、nominal、jitter_drop、compute_load；50/100/150ms 为同一媒体 PTS 上的离散重放，不是墙钟并发或设备测试。基础流逾期与预测逾期分别计数，源/队列丢弃、预热、回退、无输出都留在源分母。报告 `evaluation_config.json` 记录真实运行配置、阈值、FPS、方法和场景。
每次运行还复制各片段的 `annotations.json`、`manifest.json`、`timeline.jsonl` 到该次报告目录，防止后续改标注后旧报告失去可读的真值快照；原始视频仍按 manifest 指向的本机路径读取。

二维姿态全 GT PCK@20px 的分母是人工可见点，原始右手检测缺失计失败；另报仅完整有限 21 点检测条件下的像素误差、覆盖率、GT 无手时误检。原始检测完整率独立于后续控制门控。手势与未来手势按完整人工区间的 target PTS 评分，四类逐类 precision/recall/F1、混淆矩阵将 `__no_output__` 另列，absent 另列误激活；unknown+无输出是漏报。按人员/会话汇总原始计数再计算比率，不将片段或帧假装独立受试者。没有足够独立组时不制造置信区间。二维标签不能生成公制 3D，也不能造假 Z 输入 9 通道当真值；M3-B 检测轨迹代理分数与这里的人工二维/手势分表。

失败样例含帧号、PTS、GT/预测、质量原因、基础/预测逾期和待人工复核原因。历史线索 V6 源帧 111→112（PTS 3743.933→3775.967ms），100ms 目标 3843.933→3875.967ms，食指近节通道跳变约 0.803；附近 109→110→111 也有 open→fist→open。它们尚无人工 GT，不能断言是真实快速动作、检测错误或规则边界。新数据的失败归因再决定是否修 M3-A 规则、开展 M4 感知对照或考虑小预测模型；本轮不自动启动 M4/M5。

合成工具样例：

```bat
conda activate handai-intent-prediction
cd /d D:\VR\HandAi\single-hand-teleop-baseline
python -X utf8 experiments\intent_prediction\scripts\make_m3c_synthetic_sample.py --dataset outputs\datasets\camera_m3c_synthetic_demo_v2
python -X utf8 experiments\intent_prediction\scripts\run_camera_m3c.py --dataset outputs\datasets\camera_m3c_synthetic_demo_v2 evaluate
```

该样例只有编号空画面及人工构造的 absent 标签，只验证导入、标注读写、流水线和报告；不提供关键点精度或真实效果证据。

## 已执行的工程验证

本机使用 `handai-intent-prediction` 环境、上述子项目运行目录，主审独立复跑全套 pytest **282 passed、0 skipped**，Ruff 全范围及 compileall 均通过。测试记录保存在 `D:\VR\HandAi\single-hand-teleop-baseline\outputs\m3c_parent_review\release_tests.xml`。

合成空画面端到端运行报告在 `D:\VR\HandAi\single-hand-teleop-baseline\outputs\datasets\camera_m3c_synthetic_demo_v2\reports\20260926T140702_867946Z\report.json`，同目录有配置与各片段标注、来源、时间轴快照。独立复核另用两段各 12 帧的编号视频及注入的有效 open 姿态，完整走提取、HandPipeline、四场景、三方法、三时距和评分；24 个源帧、全 GT PCK 为 11/12，默认开发评测未打开虚设留出片段。复核摘要在 `D:\VR\HandAi\single-hand-teleop-baseline\outputs\m3c_parent_review\positive_integration\parent_verification.json`。本地标注器完成了逐点可见性、失败专项用途、连续手势与人工转换区间的保存和重新读回；截图在 `D:\VR\HandAi\single-hand-teleop-baseline\outputs\m3c_parent_review\final_annotation_roundtrip.png`。这些均为合成/注入验证，不能替代新人员视频的人工 GT 和独立效果判断。
