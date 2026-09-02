# 单右手控制意图预测

这一实验支线研究：根据最近一段 `svh_preview.target_positions`，预测未来
`50 / 100 / 150 ms` 的 9 通道目标。

它不会改变现有主链：

`摄像头/视频 -> MediaPipe 单右手 -> control_representation -> svh_preview -> Unity UDP 预览`

第一轮比较以下方法：

- `hold_last`
- `linear`
- `kalman`
- `gru`
- `tcn`
- `transformer`

## 1. H2O 下载范围

从 H2O 官方下载页只下载：

- `subject1_pose_v1_1.tar.gz`
- `subject2_pose_v1_1.tar.gz`
- `subject3_pose_v1_1.tar.gz`
- `subject4_pose_v1_1.tar.gz`
- `label_split.zip`
- `object.zip`

不需要 `h2o_CASA.tar`、`manolabel_v1.1.tar.gz`、`subject*_ego_*`、完整
`subject*_v1_1.tar.gz` 或 `test_obj_labels.zip`。

本机约定目录：

```text
D:\VR\HandAi\datasets\H2O\
  downloads\   # 原始压缩包
  raw\         # 统一解压后包含 subject1 ... subject4
  processed\   # 本实验生成的右手9通道序列
```

H2O 数据仅限学术、非商业使用；不要把原始数据提交到 Git 或重新分发。

## 2. 为什么不需要人工逐帧标注

每个 `hand_pose/*.txt` 有 128 个数：

- 左手：有效位 + `21 x 3`
- 右手：有效位 + `21 x 3`

转换器读取右手半段，再复用当前项目的
`hand_features -> control_representation -> svh_adapter` 自动得到 9 通道。
pose-only 压缩包通常不含 `cam_intrinsics.txt`。现役 v2 为了可复现，显式冻结旧的相机平面
`x/y` 路径；它不是针孔投影。新实验可以显式选择
`--pose-only-projection normalized_perspective_wrist_origin_palm_scale`，但必须生成新数据版本并
重新训练，不能覆盖 v2。若某个发行版带相机内参，转换器会使用内参投影。无效帧和不连续帧
会切断序列，不会被模型跨越。

默认采用严格的跨人员切分：

- train：subject1、subject2
- val：subject3
- test：subject4

## 3. 预处理

在 baseline 环境中运行：

```powershell
conda run -n single-right-hand-baseline python `
  experiments\intent_prediction\scripts\preprocess_h2o.py `
  --h2o-root D:\VR\HandAi\datasets\H2O\raw `
  --output-root D:\VR\HandAi\datasets\H2O\processed\cross_subject_v1
```

输出按连续有效片段保存为压缩 NPZ，并生成带文件哈希、切分计数和拒绝帧诊断的
`manifest.json`。

## 4. 模型环境

神经网络使用独立环境，避免污染已经验收的 Phase 1 环境：

```powershell
conda env create -f experiments\intent_prediction\environment.yml
conda activate handai-intent-prediction
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu126
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

上面的 PyTorch 安装命令针对本机 NVIDIA 显卡与 Windows；如果
`torch.cuda.is_available()` 为 `False`，模型仍能在 CPU 跑，只会更慢。GPU 安装问题应在
这个独立环境中处理，不要改 baseline 环境。若换到没有 NVIDIA 显卡的电脑，可从 PyTorch
官方安装页选择 CPU 版本。

## 5. 先做 smoke，再做真实实验

确定性 synthetic smoke 只证明代码能跑，报告会强制写入
`research_claims_allowed=false`：

```powershell
python experiments\intent_prediction\scripts\run_first_round.py --synthetic-smoke
```

真实 H2O 实验：

```powershell
python experiments\intent_prediction\scripts\run_first_round.py `
  --data-root D:\VR\HandAi\datasets\H2O\processed\cross_subject_v1
```

日常改代码或筛超参数时，先用较小但切分口径完全相同的快跑配置。它默认只比较
`hold_last + GRU`；需要时再通过 `--models` 临时加入其他方法：

```powershell
python experiments\intent_prediction\scripts\run_first_round.py `
  --config experiments\intent_prediction\configs\h2o_quick_iteration.json `
  --data-root D:\VR\HandAi\datasets\H2O\processed\cross_subject_v1
```

快跑报告会强制写入 `research_claims_allowed=false`，只用于迭代，不和正式配置的结果混在
同一张论文表格中。正式运行每轮还会追加
`training_progress.jsonl`，即使中途停止也能看到已经完成到哪个模型、哪个 epoch。

每次运行都会新建时间戳目录，包含：

- `report.json`
- `metrics.csv`
- `training_progress.jsonl`
- `checkpoints/gru.pt`
- `checkpoints/tcn.pt`
- `checkpoints/transformer.pt`

报告同时记录每种方法的 MAE、RMSE、P95 绝对误差、各预测距离误差、各通道误差、
相对 hold-last 的改善、参数量、训练耗时和单窗口推理延迟。

## 6. 解释边界

- FreiHAND 继续用于 Phase 1 静态感知验收；它不是时序训练集。
- H2O 提供的是手部姿态，不直接提供 SVH 9 通道；9 通道由当前映射自动生成。
- 第一轮预测输入暂定为过去 30 帧的 9 通道，保证所有方法比较口径一致。
- H2O 结果能证明公开姿态序列上的预测能力，但最终仍需少量自录视频做域适配和真实链路验收。

第一轮正式结果、运动分层审计和下一轮建议见
`docs/intent_prediction_first_round_report.md`。

## 7. 第二轮：validation 选型后 test 一次评估

第二轮不改 Unity，也不需要摄像头。它比较原始 GRU、零残差初始化 GRU 和两档运动加权
残差 GRU；每个候选只在 subject3 validation 上拟合运动门控。门控允许所有预测距离的
`alpha=0`，因此最坏可以精确退化为 hold-last。

正式运行：

```powershell
python experiments\intent_prediction\scripts\run_second_round.py `
  --data-root D:\VR\HandAi\datasets\H2O\processed\cross_subject_v1
```

代码自检：

```powershell
python experiments\intent_prediction\scripts\run_second_round.py --synthetic-smoke
```

第二轮会先把 validation 选型结果写入不可含 test 指标的 `selection.json` 并计算哈希，之后
才加载 subject4 test。输出还包括 `report.json`、`report.md`、
`validation_candidates.csv`、训练日志和候选 checkpoints。

边界：subject4 已在第一轮用过，所以第二轮的顺序隔离能防止“本轮拿 test 重选”，但它
不是整个项目历史上的全新盲测。第二轮结论只决定是否值得进入离线回放/影子模式，不代表
Unity Play Mode、摄像头、UDP 实时延迟或真机安全已经验收。

旧 v1 第二轮结果保留在 `docs/intent_prediction_second_round_report.md`。open-release 加入后，
已重新生成 H2O v2 标签和重训；现役 v2 的诚实结果与风险边界见
`docs/phase15_risk_hardening.md`。

## 8. 第三步：默认关闭的实时影子接入

与当前映射契约一致的 v2 `residual_motion4` checkpoint 已接入主链，但不会替换现有命令。
它的预注册离线 gate 未全部通过，只允许默认关闭的研究诊断：

```text
baseline/control/svh_preview -> 当前帧 UDP 先发
                             -> 非阻塞提交到 latest-only worker
                             -> 30 Hz 重采样 + residual GRU + frozen gate
                             -> 独立 prediction JSON/JSONL，不回写 baseline/UDP
```

无摄像头 smoke：

```powershell
conda activate handai-intent-prediction
python -X utf8 scripts\run_prediction_shadow_smoke.py `
  --device auto `
  --output outputs\prediction_shadow_smoke\latest.json
```

有视频时：

```powershell
python -X utf8 src\main.py `
  --config configs\svh_9ch_preview.yaml `
  --video-file D:\path\to\right_hand_video.mp4 `
  --prediction-shadow --headless --save-jsonl
```

默认配置均为 `prediction_shadow_enabled: false`。缺 PyTorch/checkpoint、selection/report/SHA
或 mapping contract 不匹配、输入 invalid、历史不足、断帧或推理异常只会产生明确诊断，
不会中断 baseline。完整 contract、状态表和证据边界见
`docs/intent_prediction_shadow_mode.md`。

这一实现仍只是观察层。冻结的延迟/抖动/丢包回放已经完成：H2O 50/100 ms primary 场景
gated RMSE 聚合改善 1.95%、q90 改善 2.67%，retention gate 为 4/6；真实摄像头 JSONL
总体改善 1.85%，但 prediction 覆盖率只有 65.17%。因此控制参考继续使用 hold-last，模型只
保留为默认关闭的影子诊断。完整结论见
`docs/intent_prediction_delay_injection.md`。

正式复现：

```powershell
python -X utf8 experiments\intent_prediction\scripts\run_delay_injection.py `
  --config experiments\intent_prediction\configs\delay_injection_v1.json `
  --data-root D:\VR\HandAi\datasets\H2O\processed\cross_subject_v2_open_release `
  --runtime-jsonl outputs\session_20260828_210132.jsonl `
  --runtime-config configs\unity_udp_preview.yaml
```

`outputs/` 保存本机完整运行目录并默认忽略；提交到 Git 的正式机器可读快照位于
`experiments/intent_prediction/reports/`，其中不包含 H2O 原始数据或模型 checkpoint。

## 9. 第四步：真实摄像头视频域开发评测

固定视频不能继续沿用 `src/main.py --video-file` 的处理 wall-clock 作为算法时间轴。新的评测
入口优先使用容器 PTS，并在 PTS 不可用时按 `frame_index / nominal_fps` 回退；模型在 baseline
JSONL 生成后同步重放，因此不会因离线快读触发 latest-only 丢帧。离线算法效用、离线处理
吞吐和真实摄像头异步 worker 覆盖会分别报告。

完整开发集命令、V1–V7 动作清单和证据边界见
`docs/camera_domain_protocol_v1.md`；正式 B1–B7 的 task-aware 输入门、P95/最差视频门、
Git/模型/Conda 环境锁、内容寻址只读视频副本、确定性 attempt 与固定一次性 receipt 见
`docs/camera_domain_blind_protocol_v1.md`。
开发冒烟示例：

```powershell
python -X utf8 experiments\intent_prediction\scripts\run_camera_domain_eval.py `
  --role development --allow-partial `
  --video V1=D:\path\to\V1.mp4 `
  --input-not-mirrored
```

默认 `camera_domain_eval_v1.json` 仍是 `development`，固定返回
`development_only_no_release_decision`。正式盲测必须改用
`camera_domain_blind_v1.json`，并显式提供合并后生成的 freeze manifest、manifest 外部 SHA、
配置 SHA 与协议 SHA；缺任一身份都会在输出目录和视频解码之前拒绝。
