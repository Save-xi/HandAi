# AI 时序预测实验

这里保留 H2O 数据转换、预测模型训练、validation 选型、误差统计和摄像头域评测。每次实验生成独立结果目录，可以正常重复运行。

当前支持既有 9 通道 `svh_preview`、M1 的 H2O 原生 21 点预测、M2 的 A/B/C 共同参考，以及 M3-A/B 的摄像头几何、简单预测和计时。下一步按 [AI 路线图](../../docs/ai_roadmap.md) 做 M3-C 新数据和人工标签效果验证。

## 环境与数据

```bat
conda activate handai-intent-prediction
cd /d D:\VR\HandAi\single-hand-teleop-baseline
```

以下命令均从这个子项目目录执行。已有数据无需重复下载：

- H2O：`D:\VR\HandAi\datasets\H2O`。
- 当前 v2 预处理数据：`D:\VR\HandAi\datasets\H2O\processed\cross_subject_v2_open_release`。
- 摄像头开发视频：`D:\HandAiVideos\camera_domain_dev_v1\v1.mp4` 至 `v7.mp4`。

H2O 数据清单记录来源、分人切分、帧率、投影方式及明文映射参数。默认 train=subject1+2、validation=subject3、test=subject4；subject4 已被历史实验使用。

## 21 点任务：M0 数据核对

[M0 结果与缺口](../../docs/m0_data_definition.md)已完成；[keypoint_m0.json](configs/keypoint_m0.json) 保存任务定义，由 M1 运行配置引用，不能直接作为训练配置。现有 `landmarks_3d` 是以米为单位的 H2O 相机 3D；`timestamps_s` 来自帧号除以 30，不是传感器采集时间。

需要重新核对数据时运行（可重复运行，只写报告）：

```bat
python -X utf8 experiments\intent_prediction\scripts\audit_keypoint_data.py --video-root D:\HandAiVideos\camera_domain_dev_v1 --output outputs\m0\data_audit.json
```

审计逐帧比较原始 21 点与 NPZ，检查人员切分、重复帧、有限值、时间、旧 XY 公式与片段排除；尺度下限仅用训练集计算。`--video-root` 可省略，加入时只读取视频容器信息。本命令不启动摄像头或模型。

## M1：21 点训练与推理

[M1 实现和实测结果](../../docs/m1_keypoint_prediction.md)已完成。默认小样本配置对每人最多取 4 段、每段最多取 256 个锚点，只在 validation 选择常速度窗口、GRU epoch 和方法；subject4 保持历史复评角色。

```bat
python -X utf8 experiments\intent_prediction\scripts\run_second_round.py --config experiments\intent_prediction\configs\keypoint_m1.json --synthetic-smoke --output-root outputs\m1_synthetic
python -X utf8 experiments\intent_prediction\scripts\run_second_round.py --config experiments\intent_prediction\configs\keypoint_m1.json --data-root D:\VR\HandAi\datasets\H2O\processed\cross_subject_v2_open_release --output-root outputs\m1_h2o_pilot
```

结果目录包含 `report.json/md`、`metrics.csv`、`window_sample.npz`、`model.json` 与 `checkpoints/residual_gru.pt`。`model.json` 始终保存本轮 GRU 候选供加载验证，即使 validation 选中简单基线；本轮常速度实际优于 GRU。保存加载后的原始坐标推理可用 [predict_keypoint_sequence.py](scripts/predict_keypoint_sequence.py)，完整可复制命令见 M1 文档。

M1 坐标不接受 `[0,1]` 截断，缺失输入历史跳过并计数，未来标签按点加掩码。输出未来 5 步及 50/100/150ms 查询结果；暂不接未来手势、映射或实时设备消费。

## M2：A/B/C 共同参考对照

[M2 实验定义与结果](../../docs/m2_representation_comparison.md)比较 A（9→9）、B（21→9）、C（21→21 后映射）。共同目标由原生姿态先插值、连续手势状态后映射生成；不沿用旧 NPZ 通道标签。B 单独接收当前映射作为残差，C 只沿自己的预测轨迹推进未来状态。

```bat
python -X utf8 experiments\intent_prediction\scripts\run_second_round.py --config experiments\intent_prediction\configs\representation_m2.json --synthetic-smoke --output-root outputs\m2_synthetic
python -X utf8 experiments\intent_prediction\scripts\run_second_round.py --config experiments\intent_prediction\configs\representation_m2.json --data-root D:\VR\HandAi\datasets\H2O\processed\cross_subject_v2_open_release --output-root outputs\m2_h2o
```

默认真实配置使用全部 217 段、每段最多 64 个固定抽样窗口和三个训练种子。前期建窗与生成参考可能一两分钟没有 epoch 日志；结果保存九个 checkpoint、共同目标样例、逐序列/分层/覆盖率指标以及研发继续条件。M2 合成数据额外弯曲手指，保证映射后的学习目标非恒定。

M2 权重使用实验批量重载接口，不能直接传给旧 `export_prediction_model.py` 或 M1 的原始坐标 API。H2O 代理结果不代表摄像头泛化或实体关节精度；既有实时配置保持原路径。

## M3-B：摄像头简单预测与成本重放

[M3-B 交付](../../docs/m3b_camera_prediction.md)使用 `camera_mediapipe_geometry_xyz_v1`，不加载 H2O checkpoint。[配置](configs/camera_m3b.json)事先指定 V1–V4 开发选型、V5–V7 历史复评，人员/会话为未知。已知会话不能跨用途；未知来源只能明确标为历史开发，不能宣称独立验证。

```bat
python -X utf8 experiments\intent_prediction\scripts\run_camera_m3b.py
```

每帧保存原始点、几何点、PTS/来源、尺寸、有效性、段号与感知耗时。每个方法共享源帧和参考，先按 nominal 场景的 50/100ms 序列等权惩罚 RMSE 选择速度窗口与简单赢家，再做复评。产生纯算法、正常成本、抖动丢帧及计算拥塞四组结果；每个阶段只有一个等待槽，丢弃与逾期均进入覆盖分母。姿态先插值再映射，分数查询不更新手势/释放状态。

保存的观测可重复评分，无需再次检测：

```bat
python -X utf8 experiments\intent_prediction\scripts\run_camera_m3b.py --observations-root outputs\m3b\实际结果目录
```

复用时检查坐标/映射参数、数值、连续时间和段号；报告明记检测耗时来自缓存。本轮预测耗时仍重新测量。时间是从源读回之后开始的 PC 成本重放，不是曝光到设备的实测。

## 旧 9 通道训练

先用合成小数据检查训练到报告的完整路径：

```bat
python -X utf8 experiments\intent_prediction\scripts\run_second_round.py --synthetic-smoke --output-root outputs\training_smoke
```

使用已有数据训练：

```bat
python -X utf8 experiments\intent_prediction\scripts\run_second_round.py --config experiments\intent_prediction\configs\h2o_second_round.json --data-root D:\VR\HandAi\datasets\H2O\processed\cross_subject_v2_open_release --output-root experiments\intent_prediction\outputs\second_round
```

模型结构、训练参数、候选及评测阈值在 JSON 配置中。保留 train/validation/test 分离：用 validation 选择模型和门控，随后评测 test。选中神经网络时，结果目录同时生成 `model.json`，供运行时直接加载；若选择 hold-last，则没有神经网络配置需要导出。

既有第一轮 `run_first_round.py` 仍可比较 hold-last、线性、Kalman、GRU、TCN、Transformer。合成自检用于验证代码，不作为算法效果结论。

## 将已有训练结果导出给运行时

```bat
python -X utf8 scripts\export_prediction_model.py --report experiments\intent_prediction\reports\second_round\20260829T024400_143036Z_report.json --output models\my_prediction.json
python -X utf8 scripts\run_prediction_shadow_smoke.py --config configs\ai.yaml --model models\my_prediction.json
```

运行时只需模型 JSON 与 checkpoint；原始评测报告不再参与启动核验。现有默认模型是 `models/residual_motion4.json`，旧 checkpoint 原样使用。

## 真实视频评测

```bat
python -X utf8 experiments\intent_prediction\scripts\run_camera_domain_eval.py --video V1=D:\HandAiVideos\camera_domain_dev_v1\v1.mp4 --video V5=D:\HandAiVideos\camera_domain_dev_v1\v5.mp4
```

可用任意不重复 ID；加 `--skip-prediction` 可仅评测姿态处理。脚本按容器 PTS 构建时间轴，失效时用源帧率连续补齐；不会把机器处理速度当作视频帧率。生成 JSON/Markdown 报告、视频指标和预测场景 CSV。

预测结果与后续视觉映射的代理标签比较，报告 hold-last/raw/gated 误差及条件覆盖率、总体覆盖率。新视频的实验用途由数据划分决定，不再靠文件身份或一次性收据管理。

## 延迟、抖动和丢包

```bat
python -X utf8 experiments\intent_prediction\scripts\run_delay_injection.py --data-root D:\VR\HandAi\datasets\H2O\processed\cross_subject_v2_open_release
```

保留原有网络扰动矩阵、随机种子和误差口径，方便与既有结果比较。这里评估软件注入扰动下的算法效果；通信和设备联调由对应成员处理。

## 数据重新预处理

修改标签或投影方式时，使用 `scripts/preprocess_h2o.py` 指向实际解压目录和新的输出目录。不要覆盖已用模型的标签；新数据清单直接保存参数，便于知道模型学的是什么。详细参数可用 `--help` 查看。

历史机器结果继续保留在 `reports/` 和被 Git 忽略的 `outputs/`。早期报告中的哈希字段仅是历史记录。
