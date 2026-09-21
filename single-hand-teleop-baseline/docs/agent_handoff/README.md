# AI 交接：已核实状态与整改依据

更新：2026-09-21。已合并 Claude 对项目书的审核与本机二审意见。[M0 数据定义](../m0_data_definition.md)、[M1 关键点预测](../m1_keypoint_prediction.md)及 [M2 H2O 三种子表示对照](../m2_representation_comparison.md)已完成；三条神经路线均未稳定胜过本轮预选简单基线。本文记录事实与缺口，下一步 M3 首轮范围及 M4/M5 计划统一放在 [AI 路线图](../ai_roadmap.md)。

日常运行以 [项目指令](../../AGENTS.md) 和 [README](../../README.md) 为准。阶段规划、交接或证据复核时参考本文，不把这份快照作为每次普通修改前的检查关卡。已有数据、权重和历史报告继续保留。

## 1. 项目书的 AI 目标

依据仓库根目录本机保存的 `项目.pdf`，2026-09-15 已核对原文及 PDF 第 5、8 页图文；以下页码按 PDF 页面计数。

| 要求 | 原文位置 | AI 侧含义 |
|---|---|---|
| 检测关键点 → 预测关键点 → 根据预测关节位置判断手势 → 选择映射 | 第 5 页，总技术路线 | 当前 9 通道预测不能等同于完成了关键点预测；需检验未来姿态到手势/映射的链路 |
| 手部姿态估计模型验证集准确率达到 90% | 第 8 页，技术指标 | 对象是姿态估计，不是手势分类；PCK 的阈值、分母和数据集仍需明确 |
| 手部检测时间不超过 50 ms | 第 8 页 | 指定设备、输入尺寸、手选择策略与计时范围后测量 |
| 遥操作响应延迟不超过 100 ms | 第 8 页 | 各模块共同验收；AI 只报告自身分段耗时 |
| VR 头显边缘端的轻量化姿态估计、预测与前馈补偿 | 第 8 页，4.1 | 软件预测、可调用输出与实际消费分开验证 |
| MediaPipe / MMPose 根据精度与速度择优 | 第 6 页，3.1.2a | 安排一个同口径预训练候选对照 |
| InterHand2.6M、Cascade R-CNN 手框检测、ResNet-101 特征提取 | 第 6–7 页，3.1.2b/c | 是项目书明确路线；若调整，记录实测依据，不能只靠模型名称声称完成 |

项目书指定的边缘目标为 HoloLens2；SDK、系统版本、运行后端和设备可用性由协作方确认。实验室为 SVH 右手版，具有 9 个驱动器，因此单右手及 9 通道消费接口有项目依据；这不自动验证当前代理映射公式或实体关节精度。

## 2. 当前执行链

```text
图像 / 设备关键点
  → HandPipeline → 单右手 21 点 → 几何特征与规则手势
  → control_representation → 9 通道 svh_preview → JSON/JSONL/可选 UDP
                                      └→ 9 通道预测 → 独立影子结果
```

| 能力 | 当前状态 | 实现位置 |
|---|---|---|
| MediaPipe 检测与设备关键点输入 | 可运行；设备输入不要求加载 MediaPipe/PyTorch | [pipeline.py](../../src/pipeline.py)、[perception/base.py](../../src/perception/base.py) |
| open / fist / pinch / unknown 手势 | 规则与去抖已实现；尚缺带人工标签的系统精度评测 | [rule_based_gesture.py](../../src/gesture/rule_based_gesture.py) |
| 9 通道映射 | 可运行的预览代理，不是实体关节测量 | [svh_adapter.py](../../src/svh/svh_adapter.py) |
| 50/100/150 ms 预测 | residual GRU 可运行；既有离线门槛未通过，默认关闭、影子运行 | [模型配置](../../models/residual_motion4.json)、[shadow_predictor.py](../../src/prediction/shadow_predictor.py) |
| 姿态与耗时评测 | 工具和真实历史结果均存在，见下一节 | [FreiHAND 评测](../../experiments/freihand_eval/README.md) |
| H2O 原生 21 点未来预测 | M1 已实现训练、基线、评测与加载；小样本常速度优于 GRU | [M1 交付](../m1_keypoint_prediction.md) |
| H2O 预测姿态后判手势/映射 | M2 已实现因果状态和共同参考；C 优于 A/B，但未通过继续条件 | [M2 对照](../m2_representation_comparison.md) |
| 摄像头域 21 点预测、计算就绪时间与人工手势评测 | 尚未实现 | M3 |
| MMPose 对照、InterHand2.6M 适配、自训练检测网络 | 尚未实现 | 后续路线 |
| ONNX/量化、边缘端运行 | 尚未实现 | 后续路线 |

2026-09-05 重构记录为 171 项 pytest 通过，并有指定视频片段的重构前后数值一致性检查；Claude 环境的 164 passed / 7 skipped 来自缺少 PyTorch。M1 完成 190 项测试；M2 增至 198 项，并完成三种子真实实验和九个 checkpoint 重载一致性检查。测试通过不等于算法效果成立。参见 [重构记录](../ai_refactor_20260905.md)、[M1](../m1_keypoint_prediction.md)与 [M2 报告](../m2_representation_comparison.md)。

## 3. 已有实验及适用范围

### 3.1 姿态与检测耗时

2026-08-24 的 FreiHAND evaluation 记录包含 3,960 张样本。2026-09-15 使用当前评分脚本、保存的预测与原 GT 重算，全部指标与原报告一致；没有重新运行检测器。

| 指标 | 结果 | 统计范围 |
|---|---:|---|
| 有效 21 点预测 | 3592 / 3960 | 368 条无有效关键点预测 |
| 21 点完整率 | 90.707% | 不等于定位准确率 |
| PCK@20px（全 GT） | 84.097% | 缺失预测计入分母 |
| PCK@20px（仅有效预测） | 92.713% | 条件指标，不能代替全 GT 口径 |
| 2D MPJPE（有效预测） | 8.713 px | 平均关节位置误差 |
| 检测器加手选择 P95 / P99 | 23.338 / 28.094 ms | 重新汇总历史 PC 耗时 |

本机原始报告位于 `experiments/freihand_eval/outputs/phase1_acceptance/20260824T022030_928281Z_a5a53da_clean/`；复算位于 `outputs/claude_review_20260915/freihand_recomputed_metrics.json`。这些产物被 Git 忽略，其他 checkout 可能没有；已入库的数字与范围见 [历史完成记录](../archive/before_ai_refactor_20260905/project_completion_manual.md)。

这批数据为 PC、静态图像、`prefer_any_hand=true`，不是实时右手筛选或 HoloLens2 测试。全 GT PCK@20 未到 90%；项目书也未指定 PCK@20 为官方准确率定义。下一步应整理已有证据并补测适用范围，而非重新认定“从未实测”。

### 3.2 预测效果

| 历史实验 | 结果 | 可支持的结论 |
|---|---|---|
| 第二轮 H2O 9 通道 | 总 RMSE 改善 2.5296%，q90 动态子集改善 3.7548%，原离线判据未通过 | 这些候选与标签条件下收益不足 |
| H2O 延迟注入 | gated 总 RMSE 改善 1.95%，retention 判据通过 4/6 | 保持简单参考，模型留在研究旁路 |

来源：[第二轮机器报告](../../experiments/intent_prediction/reports/second_round/20260829T024400_143036Z_report.json)、[延迟回放报告](../../experiments/intent_prediction/reports/delay_injection/20260829T112008_630685Z_report.md)。

这些结果没有证明“9 通道收益上限低于 3%”，也未证明改变预测位置必然改善效果。低 hold-last 误差可能来自短时连续性，不能单独推出通道接近常数。暂停无目的扩大模型搜索是资源建议，9 通道仍应保留为对照。

### 3.3 实时分段耗时

[2026-09-01 运行记录](../archive/before_ai_refactor_20260905/live_runtime_acceptance_20260901.md)已经报告 CUDA 推理 P95 6.92 ms、source.read 返回后至 UDP P95 25.91 ms，并有 Unity 应用时刻统计。因此“从未统计 timing”不准确。

这些是历史开发会话结果：source.read 返回不是相机曝光起点，Unity 应用时刻不是实体响应或已渲染画面。不能据此验收完整系统 ≤100 ms。旧单次摄像头回放的 65.17% 预测覆盖率和 30.55% 超时距比例也只能归属于那次输入与口径，不能作为当前所有视频的结论。

## 4. 已接受的整改依据

### 数据与参考目标

H2O 预处理确实保存了 21×3 和 21×2 数组；本机 v2 清单有 217 个序列。2026-09-17 [M0 全量核对](../m0_data_definition.md)已确认：933 帧无右手标注，3 个短段共 74 帧被舍弃，其余有效原始帧全部保留，未发现额外映射筛选排除。旧 XY 仍是替代坐标，不能当作真实图像投影。

H2O 相机 3D 与当前 MediaPipe 图像归一化 XY、腕部相对 Z 含义不同。腕点/掌长归一化不能自动统一它们。即使先做 `X/Z、Y/Z`，当 `fx/W != fy/H` 时仍不同于图像归一化投影；缺内参时应明确使用替代坐标，不能声称已完成跨域校正。[MediaPipe 输出定义](https://chuoling.github.io/mediapipe/solutions/hands.html#output)

[标签诊断](../../experiments/intent_prediction/reports/label_semantics/20260830_h2o_v2_label_semantics_audit.json)中的 95.09% 帧变化和平均每通道差 0.05191，是两个代理映射之间的差异，不能解释为相对真值的已测偏差。raw 与连续去抖还使 0.90% 帧的标签不同。新实验需明确共同目标、连续状态、归一化锚点和缺帧处理；目标改变后分别算出的 RMSE 不直接比较。

H2O subject4 与 V1–V7 已用于开发或诊断。后续保留其内部复评角色；新的泛化结论需要独立人员/会话。正常复现实验允许重复运行，看过测试结果后调整方案要如实标注，不能通过改名恢复独立性。

### 21 点任务的完整接线

| 位置 | M1 后的状态 |
|---|---|
| [sequence_data.py](../../experiments/intent_prediction/intent_prediction/sequence_data.py)、[keypoint_data.py](../../experiments/intent_prediction/intent_prediction/keypoint_data.py) | 保留旧 9 通道窗口；新增带锚点变换、实际时间包围点、标注掩码与断段检查的关键点窗口 |
| [models.py](../../experiments/intent_prediction/intent_prediction/models.py) | residual GRU 显式支持 63 维带符号、无界残差；旧模型默认维度、残差上限和输出 clamp 保留 |
| [baselines.py](../../experiments/intent_prediction/intent_prediction/baselines.py)、[keypoint_metrics.py](../../experiments/intent_prediction/intent_prediction/keypoint_metrics.py) | 保持/常速度支持显式 63 维无截断；关键点用 MPJPE、腕部/指形误差及标注掩码；没有逐点可见性真值 |
| [gating.py](../../experiments/intent_prediction/intent_prediction/gating.py) | 现有门控假设 9 通道历史/输出并执行 `[0,1]` clip；关键点首轮不复用这一门控，后续按新任务验证 |
| [run_second_round.py](../../experiments/intent_prediction/scripts/run_second_round.py)、[keypoint_runner.py](../../experiments/intent_prediction/intent_prediction/keypoint_runner.py)、[training.py](../../experiments/intent_prediction/intent_prediction/training.py) | CLI 依配置分派，复用训练循环；validation 选型后才构建历史复评窗口；关键点模型配置和加载器校验坐标任务 |
| 合成自检 | 旧 9 通道 smoke 保留；指定 M1 配置时生成非空 `[T,21,3]` 并完成训练、保存、加载 |
| 未来手势与映射 | M2 的 H2O 实验已实现：从预测关键点重算特征，复制当前状态，仅沿规则时刻推进；摄像头域和人工手势评测留在 M3 |

具体归一化、最小模型与继续条件见路线图。已有 `max_delta=0.35` 属于旧通道模型，不能直接当作新坐标的合理运动上限。关键点加载还要验证点顺序、任务维数、单位/归一化、时间距及有限值；这些是算法必要检查。

### 评测与轻量化

延迟收益应同时给出纯算法比较和计入检测、排队、推理就绪时间的比较。使用相同源帧和逐帧扰动，统计回退、无输出、丢弃及超出时距的情况；不把共同有效时刻误差当作全程覆盖率。

HoloLens2 是目标设备，ONNX 是候选交换格式，运行后端仍需核验。Barracuda 3.0.1 官方已标记弃用，支持列表未列 GRU；不能默认当前 GRU 导出后能直接运行。Windows ML 的模型/opset 支持也与系统版本相关。[Unity 支持列表](https://github.com/Unity-Technologies/barracuda-release/blob/release/3.0.1/Documentation~/SupportedOperators.md)、[Microsoft 版本说明](https://learn.microsoft.com/en-us/windows/ai/windows-ml/onnx-versions)

FP32/量化需要在明确平台上比较精度、模型大小和实际耗时；PC 结果不能称为头显实测，单个预测头导出不能称为整条姿态与预测链路已部署。

## 5. 分工与后续入口

AI 侧负责感知适配、姿态/手势、预测、训练评测、模型导出和调用接口。设备采集、时钟同步、网络、Unity、串口、实体手与机械臂由协作方负责。预测进入实验消费接口也不等同于获准驱动实体设备。

`HandPipeline` 返回的扩展诊断目前只进日志、未进入规范 payload，可在需要逐帧定位降级原因时补可选字段；不是新路线的前置大工程。两次 `prepare_frame_payload` 分别处理 pipeline 输出与主循环补充 timing 后的输出，不因重复调用就判定为缺陷。

下一步只维护三处入口：[阶段路线](../ai_roadmap.md)、[AI 接口](../ai_interfaces.md)、[训练评测命令](../../experiments/intent_prediction/README.md)。文件哈希、AST 指纹、Git 身份核验、冻结收据和一次性运行限制不再恢复。
