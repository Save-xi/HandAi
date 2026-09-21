# M1：21 点预测链路与真实小样本结果

完成日期：2026-09-18。M1 已获批并完成：非空 21 点数据、时间采样、保持/常速度基线、残差 GRU、训练选型、指标、保存加载及原始坐标预测已经贯通。本文保留 M1 当时结果；2026-09-20 后续表示对照见 [M2](m2_representation_comparison.md)。

本轮的实际结论是：**常速度基线优于这次小样本 GRU**。这验证了可运行的关键点实验链路，并未证明神经网络优于简单方法。摄像头仍使用既有 9 通道影子预测；新候选没有启用到设备输出。

## 1. 完成了哪些接线

| 环节 | 本轮实现 |
|---|---|
| 配置与 CLI | [keypoint_m1.json](../experiments/intent_prediction/configs/keypoint_m1.json) 引用 M0 的坐标任务定义，由已有 `run_second_round.py` 分派到关键点实验 |
| 数据 | [keypoint_data.py](../experiments/intent_prediction/intent_prediction/keypoint_data.py)：读非空 `landmarks_3d` 和 `timestamps_s`，核对人员/序列切分，按时间建窗，保留掩码、包围时间和锚点变换 |
| 简单基线 | 复用 [baselines.py](../experiments/intent_prediction/intent_prediction/baselines.py)，显式采用 63 维带符号坐标；验证集在 2/4/8 帧速度拟合窗口中选择 |
| 模型 | 复用 [models.py](../experiments/intent_prediction/intent_prediction/models.py) 的 residual GRU；本任务为 63→63、未来 5 步，无 tanh 幅度限制、无 `[0,1]` 截断；初始输出为保持上一帧 |
| 训练 | 复用 [training.py](../experiments/intent_prediction/intent_prediction/training.py)：有效点 Smooth L1 损失、validation 选型、早停与 checkpoint；旧通道路径保持默认参数 |
| 评测 | [keypoint_metrics.py](../experiments/intent_prediction/intent_prediction/keypoint_metrics.py)：归一化/毫米 MPJPE、腕部与指形误差、P95、逐序列/人员结果、覆盖率、最大误差锚点 |
| 加载与调用 | [keypoint_model_loader.py](../src/prediction/keypoint_model_loader.py)：校验任务、单位、点序、采样及归一化参数；返回原生相机米坐标及目标时间 |

任务仍是 `h2o_camera3d_right21`。输入为 30 个 30Hz 观测，输出未来第 1–5 步；在 50/100/150ms 查询预测轨迹并与相同目标时刻的标注比较。H2O 的时间来自帧号/30，规则与单位来源见 [M0](m0_data_definition.md)。时间插值也通过了不规则时间轴的数值测试，不能据此称为已验证实时采集/就绪时间。

所有历史和未来点共用当前锚点的腕点、掌长；未来腕部位移保留。尺度下限仅从所有训练序列计算，先于小样本抽样：54,243 个正掌长，中位数 `0.09531036054196106 m`，下限为其 1%，与 M0 一致。

## 2. 缺失、断段与指标的处理

NPZ 可携带 `landmark_mask[T,21]`、`frame_valid[T]`、`segment_ids[T]`；没有这些字段时用有限坐标形成点有效位，不伪造逐点置信度。非递增时间直接报错；原始帧号缺口、超过 `2/30 s` 的时间间隔、显式无效帧和段号变化均断段。

M1 采用保守输入规则：历史必须完整，否则跳过窗口并记录原因。未来监督按点和时距加掩码；没有未来标签的片尾锚点保留在覆盖率分母中，训练时不把它们当作零目标。部分有效监督只对有效点计算损失。历史插值在实现上只读取锚点及之前的数组，目标插值不能越过无效段。

主指标对每个锚点的有效关节求欧氏误差均值，再按序列等权汇总；另报按帧汇总及有效标签数量。选择分数为 50/100ms 归一化 MPJPE 的等权平均。毫米误差使用同一锚点尺度还原；相对指形误差分别减去预测和目标自己的未来腕点，对其余 20 点计算，不把未来腕点送入模型。

未来 5 步保存了监督掩码和时间包围点，供下一阶段构建连续手势参考；本轮没有实现未来手势状态机、A/B/C 共同映射或运行门控。

## 3. 真实 H2O 小样本实验

随机种子 `20260918`。每人随机取最多 4 个保留序列，每段随机取最多 256 个历史完整锚点；抽样方案预先写在配置中。训练 8 段 / 1,406 窗口，验证 4 段 / 1,018 窗口，subject4 历史复评 4 段 / 1,007 窗口。序列名单和抽样前分母见机器报告；这些窗口不是相互独立的参与者样本。

49,531 参数、单层 64 hidden residual GRU，最多 12 epochs，patience 4。验证集最佳为第 8 epoch，第 12 epoch 停止。常速度窗口和方法选择都在 validation 完成，之后才构建并预测 subject4 复评窗口。

历史复评的结果如下，均按序列等权：

| 方法 | 50ms MPJPE mm | 100ms MPJPE mm | 150ms MPJPE mm | 50/100ms 平均归一化误差 |
|---|---:|---:|---:|---:|
| 保持上一帧 | 6.711 | 13.051 | 18.839 | 0.105104 |
| 常速度（验证集选 2 帧） | **3.140** | **7.052** | **12.092** | **0.054208** |
| residual GRU | 6.407 | 12.698 | 18.615 | 0.101608 |

验证集对应选择分数为 0.129932 / **0.063192** / 0.122645，排序一致。历史复评中，常速度相对保持上一帧改善 48.42%，GRU 改善 3.33%；这些百分比仅属于这次单种子、小样本、已平滑标注的数据域，不能作为 M2 正式收益结论。

三个方法使用相同目标掩码。抽样后的完整目标覆盖率在 50/100/150ms 分别为 99.70% / 99.30% / 98.81%，均以历史完整的 1,007 窗口为分母。它是标签覆盖率，不是摄像头预测成功率或按时送达率。抽样前，所选复评序列共 2,021 帧，其中 116 帧处于片段预热区，1,905 个锚点历史充足，1,885 个具有完整未来路径。

失败样本已记录。例如 `subject4/h1/4` 原始帧 260 的 150ms MPJPE：保持约 126.3mm、常速度约 84.1mm、GRU 约 73.7mm；单个片段里 GRU 可以更好，但整体仍落后于常速度。`subject4/h1/2` 原始帧 520 的 GRU 150ms 误差约 81.3mm。后续应检查这些轨迹和动作变化，不以单例取代总体比较。

环境为 `handai-intent-prediction`，Python 3.10.20、PyTorch 2.7.0+cu126、RTX 4060 Laptop GPU。记录的单窗口 forward P50/P95 约 0.458/0.800ms，仅包含设备驻留输入的网络计算，不含检测、数据搬运、归一化、输出或网络延迟。

## 4. 验证与可复核文件

新增 13 项测试覆盖不规则时间、历史无未来泄漏、归一化往返、未来腕部位移、缺帧与显式无效段、部分监督、退化尺度、带符号输出、损失掩码、统计单位、训练集尺度及跨坐标域拒绝。全量 **190 项测试通过**，全项目 Ruff 与相关 Python 编译检查通过。

默认 30 帧非空合成任务已通过同一 CLI 完成训练、评测、保存、加载。真实实验的相同批量重载预测最大差为 0；批量离线结果与单条原始坐标调用的最大差为 `3.540e-6 m`（0.00354mm），低于本轮 0.01mm 数值容差。独立预测 CLI 已从真实帧 520 生成未来 5×21×3 点和三个查询时距的 JSON。

| 证据 | 位置 |
|---|---|
| 真实小样本完整报告 | [JSON](../experiments/intent_prediction/reports/m1/20260918_h2o_pilot_report.json)、[结果表](../experiments/intent_prediction/reports/m1/20260918_h2o_pilot_report.md)、[CSV](../experiments/intent_prediction/reports/m1/20260918_h2o_pilot_metrics.csv) |
| 默认非空合成运行 | [报告](../experiments/intent_prediction/reports/m1/20260918_synthetic_smoke_report.json) |
| 测试代码 | [test_keypoint_prediction.py](../tests/test_keypoint_prediction.py) |
| 本机候选与权重 | [model.json](../outputs/m1_h2o_pilot/20260918T092000_416314Z/model.json)、[residual_gru.pt](../outputs/m1_h2o_pilot/20260918T092000_416314Z/checkpoints/residual_gru.pt) |
| 本机窗口与调用输出 | [window_sample.npz](../outputs/m1_h2o_pilot/20260918T092000_416314Z/window_sample.npz)、[prediction_example.json](../outputs/m1_h2o_pilot/20260918T092000_416314Z/prediction_example.json) |

`outputs` 下的文件是本机产物，继续保留，可按下面命令重建；代码与精简报告放在正常项目目录。`model.json` 指向 GRU 研究候选，即使 validation 选中常速度也会保存，便于复核加载链路，不能将它解读为已选用 GRU。

## 5. 复现命令与下一步

```bat
conda activate handai-intent-prediction
cd /d D:\VR\HandAi\single-hand-teleop-baseline
python -X utf8 experiments\intent_prediction\scripts\run_second_round.py --config experiments\intent_prediction\configs\keypoint_m1.json --synthetic-smoke --output-root outputs\m1_synthetic
python -X utf8 experiments\intent_prediction\scripts\run_second_round.py --config experiments\intent_prediction\configs\keypoint_m1.json --data-root D:\VR\HandAi\datasets\H2O\processed\cross_subject_v2_open_release --output-root outputs\m1_h2o_pilot
```

每次正常创建新的结果目录并打印报告位置。`keypoint_m0.json` 保留为历史任务定义；直接把它传给训练 CLI 会提示使用 `keypoint_m1.json`。

使用本次已生成的模型：

```bat
conda activate handai-intent-prediction
cd /d D:\VR\HandAi\single-hand-teleop-baseline
python -X utf8 experiments\intent_prediction\scripts\predict_keypoint_sequence.py --model outputs\m1_h2o_pilot\20260918T092000_416314Z\model.json --sequence ..\datasets\H2O\processed\cross_subject_v2_open_release\sequences\test\subject4_h1_2_cam4_segment000.npz --anchor-frame-id 520 --output outputs\m1_prediction_example.json
```

调用只消费指定锚点及以前的观测，输出相机米坐标和目标时间。省略 `--anchor-frame-id` 时使用最后一帧；无需未来真值。完整历史不足时明确报错。Python 接口和字段见 [AI 对接入口](ai_interfaces.md#原生-21-点预测m1)。

M1 当时建议下一步开展 M2 的 A/B/C 共同比较，并把常速度保留为强基线。该后续实验现见 [M2 文档](m2_representation_comparison.md)。摄像头域、人工手势评测、计算就绪时间、MMPose 对照及 HoloLens2 导出继续留在 M3–M5。
