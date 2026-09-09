# AI 交接说明：不足之处与整改方案

面向接手 AI 部分的 agent。先读第 1–3 节确认事实，再按第 4 节制定整改方案。

- 审核基线：`main` = `bbe8884`（Merge PR #18，AI 主线重构之后）
- 审核日期：2026-09-09
- 分工前提：本文只覆盖 AI 侧（感知、手势、时序预测、轻量化、评测）。设备采集、Unity、5G、串口、SVH/AUBO 驱动由协作方负责，AI 侧只维护输入输出适配入口。

## 0. 这份文档怎么用

1. **不要把本文当作需求书。** 权威顺序是 `AGENTS.md` > 仓库根目录 `项目.pdf` > `docs/ai_roadmap.md` > 本文。本文是一次代码审核的结论快照，会过期。
2. **每条结论都附了证据路径。** 动手前先自己打开证据文件确认，尤其是报告类结论——它们是历史机器运行的产物，不会随代码自动更新。
3. **第 3 节是"不足"，第 4 节是"建议"。** 第 3 节的事实应当尊重；第 4 节的优先级和方案可以推翻，但推翻时要说明依据。
4. **第 6 节是禁止事项。** 那里列的是已经验证过没有收益、或会破坏既有数值可比性的做法。

### 本次审核的能力边界（诚实声明）

| 项 | 状态 |
|---|---|
| 源码、配置、测试、历史报告 | 已完整阅读 |
| `pytest` / `ruff` | 已在本机实际运行 |
| `项目.pdf` | **未读到**，被根目录 `.gitignore` 排除。所有"项目书要求"均转引自 `docs/ai_roadmap.md` |
| H2O 数据集、V1–V7 摄像头视频 | **不在环境中**，无法重跑训练或评测 |
| PyTorch / CUDA | **未安装**，涉及模型推理的测试被跳过 |

因此：凡涉及"模型效果"的结论，全部来自 `experiments/intent_prediction/reports/` 下的历史报告，不是本次重新测得的。

## 1. 复核基线（可复现）

```bash
cd single-hand-teleop-baseline
python -m pip install -r requirements.txt
python -m pytest -q
python -m ruff check src tests scripts experiments examples/use_ai_api.py
```

本次实测结果：

| 检查 | 结果 |
|---|---|
| pytest | 171 收集：**164 passed, 7 skipped** |
| 跳过原因 | 全部是缺 PyTorch（`test_intent_prediction_first_round` ×5、`test_intent_prediction_second_round` ×1、`test_prediction_shadow_checkpoint` ×1） |
| ruff | All checks passed |

`docs/ai_refactor_20260905.md` 记录的"171 passed"与此一致——那台机器装了 torch，所以 7 个没被跳过。**工程层面没有需要返工的东西**，本文后续所有问题都在算法与评测层，不在工程层。

## 2. 当前真实能力边界

写清楚这一节，是为了避免接手方把"代码里有"误当成"已经验收"。

| 能力 | 状态 | 证据 |
|---|---|---|
| 单右手 21 点检测（MediaPipe） | 可运行 | `src/perception/mediapipe_hand.py` |
| 统一 AI 入口，可换姿态模型 | 可运行 | `src/pipeline.py` `HandPipeline`；`src/perception/base.py` `HandDetector` |
| 设备关键点直接入口（不依赖摄像头） | 可运行 | `HandPipeline.process_detections` |
| 规则手势 open/fist/pinch/unknown | 可运行，**从未评测** | `src/gesture/rule_based_gesture.py` |
| 9 通道 `svh_preview` 代理表示 | 可运行 | `src/svh/svh_adapter.py` |
| 9 通道未来 50/100/150 ms 预测（影子模式） | 可运行，**离线验收未通过** | `models/residual_motion4.json` 中 `offline_gate_passed: false` |
| FreiHAND 二维姿态评测工具链 | 可运行，**无真实结果入库** | `experiments/freihand_eval/`；`examples/sample_report.md` 自述为展示用数值 |
| 21 点未来轨迹预测 | **未实现** | 见 3.A1 |
| MMPose / Cascade R-CNN / InterHand2.6M | **未实现** | 全仓库无相关代码 |
| ONNX / INT8 / 边缘推理 | **未实现** | 全仓库 grep 不到 onnx/quantize/tflite/tensorrt |
| 双手 | 明确的后续任务，当前不做 | `AGENTS.md` 第 2 条 |

## 3. 不足之处

按"是否阻塞后续工作"分级。A 级不解决，后面做什么都是在错误的任务上优化。

---

### A 级：阻塞

#### A1. 预测任务定义与项目书不一致，且已触到上限

**现象。** 当前学习目标是**映射之后**的 9 通道 `svh_preview.target_positions`，而不是项目书要求的 21 点关键点未来轨迹。两轮完整训练之后，模型打不过 hold-last 基线，验收判据不通过。

**证据。** `experiments/intent_prediction/reports/second_round/20260829T024400_143036Z_report.json`：

| 判据 | 要求 | 实测 | 结论 |
|---|---:|---:|---|
| `overall_rmse_improvement_percent` | ≥ 3.0 | **2.5296** | 不通过 |
| `q90_rmse_improvement_percent` | ≥ 5.0 | **3.7548** | 不通过 |
| `q90_p95_improvement_percent` | > 0 | 5.5600 | 通过 |
| `overall_mae_regression_percent` | ≤ 0.5 | −2.2018 | 通过 |
| `range_violation_rate` | = 0 | 0.0 | 通过 |
| `single_window_p95_ms` | ≤ 10.0 | 0.9533 | 通过 |

`acceptance.offline_gate_passed = false`。

延迟注入回放给出同样结论——`reports/delay_injection/20260829T112008_630685Z_report.md`：`retention gate: false`，决策字段为 `retain_hold_last_as_control_reference_and_keep_model_shadow_only`，聚合 RMSE 改善仅 1.95%。

**原因分析（推断，需接手方验证）。** 9 通道信号是几何特征经过参考点归一、手势上下文分支、以及 `clip(0,1)` 之后的产物，运动学结构在映射中被大幅压缩。旁证：同一份报告里 `finger_spread` 通道的 hold MAE 只有 0.0068，接近常数——对高度饱和的信号，persistence 天然极强。

**建议的验证方法**（一次性脚本，不需要训练）：统计已预处理 NPZ 中 `controls_9ch` 逐通道处于边界（≈0 或 ≈1）的帧占比，以及逐通道方差。若大比例贴边，则饱和假设成立。

**影响。** 在这个任务上继续调超参不会产生可交付结论。项目书要的 21 点轨迹预测目前是 0 实现。

**注意。** `docs/ai_roadmap.md` 已明确写过："当前 9 通道模型继续作为代理任务参考，不把它称为已实现项目书的 21 点轨迹预测"。汇报材料里不要混淆这两者。

---

#### A2. 训练标签的几何口径与运行时几何不一致，且偏差量级大于待验收收益

**现象。** H2O pose-only 发布包中绝大多数 take 缺少 `cam_intrinsics.txt`，预处理因此走了 `legacy_camera_xy_wrist_origin_palm_scale` 回退路径——直接取相机坐标 x/y 构造 2D 几何，**没有做透视除法**。而运行时 MediaPipe 输出的是真正的透视归一化坐标。

**证据。** `experiments/intent_prediction/reports/label_semantics/20260830_h2o_v2_label_semantics_audit.json`，`groups.all`：

- `projection_label_different_fraction` = **0.9509**（subject1 = 0.9988）
- `projection_mean_channel_abs_error_over_all_frames` = **0.0519**
- `projection_max_channel_abs_error` = 0.9624

对照量级：模型自身总 MAE 是 0.0217（`validation_hold_metrics.mae`），追求的改善约 2%。

代码位置：`experiments/intent_prediction/intent_prediction/h2o_adapter.py:32` 定义回退策略，`:293` 将其作为默认值。

**影响。** 训练标签所处的几何与部署时看到的几何在 95% 的帧上不同，平均每通道差 0.052——**标签口径偏差比整个待验收收益大一个量级**。在此基础上比较 2%–3% 的模型改善，结论不稳健。

**注意。** 现有 checkpoint 与 `models/residual_motion4.json` 就是在这份 v2 标签上训练的。按 `AGENTS.md` 第 7 条，既有数据和 checkpoint 保留、不覆盖；整改应产出**新的**数据目录与新模型，而不是改写旧的。

---

### B 级：项目书要求但零实现

引自 `docs/ai_roadmap.md` 的对照表。以下每一条当前都是 0 行代码。

#### B1. 手势分类没有任何评测（优先级最高的缺口）

- **现象**：`src/gesture/rule_based_gesture.py` 是纯规则 + 手调阈值，从未与任何标签对齐过。
- **证据**：全仓库 `grep -rniI "precision|recall|f1_score|confusion" --include=*.py` **零命中**。
- **影响**：项目书"验证集准确率 90%"这条指标，目前**一个数字都拿不出来**。这是答辩中最容易被追问的一条。
- **补充**：H2O 的 `action_labels`（`h2o_adapter.py:216` 读入，写进 NPZ）是**动作类别**，不是 open/fist/pinch，不能直接当手势标签用。

#### B2. 没有真实的检测延迟数据

- **现象**：`experiments/freihand_eval/examples/sample_report.md` 里的 19.134 ms / 21.137 ms 是**格式示例**，文件开头自己写明"表格里的数值是展示用口径，不代表真实评估结果"。
- **影响**：项目书"手部检测 ≤50 ms"没有可引用的实测证据。
- **好消息**：工具链已就绪（`experiments/freihand_eval/scripts/run_current_pipeline_predictions.py` 已记录 `detect + 手选择` 单样本耗时，`freihand/report.py` 已能出 P95/P99），只差跑一次真实数据。

#### B3. 轻量化 / ONNX / INT8 零实现

- **证据**：`grep -rniI "onnx|quantiz|int8|tflite|tensorrt|openvino"` 只在 `docs/ai_roadmap.md` 的计划文字里命中。
- **影响**：项目书 4.1 节的交付物（模型大小、FP32/INT8 数值误差、推理 P50/P95）全部缺失。
- **预期管理**：待导出模型只有 **172,699 参数**（报告 `validation_candidates[*].training.parameter_count`）。量化收益很可能有限，但对照表本身就是 4.1 要的交付物，**负结论也算完成**。

#### B4. MMPose 对照零实现

- **现象**：只有 MediaPipe 一条检测路径。
- **好消息**：`src/perception/base.py` 的 `HandDetector` 协议已经留好接口，只需 `detect(frame) -> list[HandDetection]` 与 `close()`；`freihand_eval` 已定义 `predictions.json` 交换格式，评测脚本可直接复用。

#### B5. 预测 horizon 在真实摄像头下覆盖不足

- **证据**：`reports/delay_injection/20260829T112008_630685Z_report.md` 真实摄像头 JSONL 回放段：`prediction 覆盖率 = 0.651739`，`超 150 ms horizon 比例 = 0.305544`。
- **影响**：50/100/150 ms 这套 horizon 对真实端到端延迟可能定短了，约 30% 的帧落在最大预测距离之外。整改 21 点预测时应重新论证 horizon 取值，而不是照抄。

---

### C 级：阻碍上述整改的代码耦合

这些不是 bug，是当前代码把"9 通道"写死了。做 A1 整改前必须先解绑。

| 编号 | 位置 | 问题 |
|---|---|---|
| C1 | `experiments/intent_prediction/intent_prediction/sequence_data.py:73` | `_iter_sequence_windows` 硬编码只读 `data["controls_9ch"]`；`:37`、`:39` 的形状校验绑定 `len(SVH_CHANNEL_NAMES)` |
| C2 | `experiments/intent_prediction/intent_prediction/models.py:233,238,243,248,254,259,264,272` | 四个 builder 全部硬编码 `input_size=9, output_size=9` |
| C3 | `experiments/intent_prediction/intent_prediction/models.py:54,165,217` | 三处 `forward` 以 `torch.sigmoid(...)` 收尾。对 [0,1] 通道正确，**对关键点坐标（相机系 3D，单位米）是错的** |
| C4 | `experiments/intent_prediction/intent_prediction/metrics.py:16,66` | 形状校验绑定 9 通道；`:94` `range_violation_rate` 假定值域 [0,1] |
| C5 | `experiments/intent_prediction/intent_prediction/baselines.py:11` | `_validate` 硬编码 `shape[-1] != 9`；`:34`、`:76` 强制 `np.clip(...,0,1)` |

**注意 C3 的替换方式**：`ResidualGRUForecaster` 已有 `max_delta` 参数（配置里为 0.35），改成 `tanh(·) × max_delta` 的有界残差是最自然的替换，能保留"初始状态严格等于 persistence"这个设计意图。不要简单删掉激活函数了事。

---

### D 级：轻微，不阻塞

| 编号 | 位置 | 说明 |
|---|---|---|
| D1 | `src/pipeline.py:237` | `_apply_extension_chain` 返回的 `diagnostics` 被直接丢弃，只有 `tests/test_main_runtime_modes.py`、`tests/test_integration_pipeline.py` 在消费返回值。扩展失败只进日志、不进 payload/JSONL，离线回放时分不出哪帧是降级帧。建议挂到 payload 上（需同步更新 `schemas/frame_payload.schema.json` 与 `output/frame_payload_contract.py` 的可选字段列表） |
| D2 | `src/pipeline.py:243` 与 `src/main.py:457` | `prepare_frame_payload` 被调用两次。**不是缺陷**——主循环在 pipeline 返回后才补 `timing`，必须重新规范化。仅记录，不需要改 |

## 4. 整改方案

优先级依据：先把任务定义摆正（否则后续优化都在错误目标上），再补最便宜的可测指标，最后补交付物类工作。

---

### P0 — 21 点关键点轨迹预测

**目标**：把学习目标从映射后的 9 通道换成项目书要求的 21 点未来轨迹，在同一套 train/val/test 协议下给出与 hold-last、线性外推的对照。

**关键前提（省一大笔工作）**：21 点数据**已经在盘里且从未被使用**。`h2o_adapter.py:255-256` 每条序列都写入了 `landmarks_3d`（21×3 相机坐标）和 `landmarks_2d`（21×2）。除写入方外，全仓库没有任何训练/评测代码读取它们（`sequence_data.py:188-189` 只在合成数据里写空数组占位）。**不需要重新预处理即可开工**。

**改动清单**

1. `sequence_data.py`
   - `_iter_sequence_windows` / `build_window_split` 增加 `target_key` 参数（`controls_9ch` | `landmarks_3d` | `landmarks_2d`），21×3 展平为 63 维。
   - `WindowSplit.__post_init__` 的形状校验改为按实际通道数校验，不再绑定 `SVH_CHANNEL_NAMES`。
   - **保持默认值为 `controls_9ch`**，确保既有 9 通道实验可原样复现。
2. **先定坐标系（不要跳过这一步）**
   - 建议：腕点为原点 + 掌长归一化，与 `h2o_adapter.canonicalize_h2o_camera_xy` 的口径保持一致；全局位移（腕点绝对轨迹）单独作为一路预测目标。
   - 理由：这个决定直接决定指标能否跨人、跨视角比较。定完写进数据 manifest，不要只留在代码里。
3. `models.py`：`build_model` 参数化 `input_size` / `output_size`；按 C3 的说明替换 sigmoid。
4. `metrics.py`：新增关键点口径——MPJPE（mm 或 px）、逐关节误差、按时间距分解。**不要沿用逐通道 MAE 汇报关键点结果**。`range_violation_rate` 对关键点无意义，应改为可选或按目标类型分支。
5. `baselines.py`：解绑 9 通道；**务必把线性外推纳入对照**——在关键点任务上常速度外推是很强的基线，只比 hold-last 会高估模型收益。
6. 复用 `second_round.py` 已有协议（train → validation 选型冻结 → test 一次），**不要重写**。可参考 `configs/h2o_second_round.json` 新建一份关键点任务配置。
7. 顺带处理 A2：用 `preprocess_h2o.py --pose-only-projection normalized_perspective_wrist_origin_palm_scale` 重跑一份新数据到**新的空目录**（`preprocess_h2o.py` 已支持该参数，见 `:40-41`）。旧数据保留不动。

**验收判据**
- 合成 smoke 跑通：`python -X utf8 experiments/intent_prediction/scripts/run_second_round.py --synthetic-smoke --output-root outputs/keypoint_smoke`
- 真实数据上给出 21 点任务的 MPJPE 对照表：hold-last / 线性外推 / 神经网络，按 50/100/150 ms（或重新论证的 horizon）分解。
- 9 通道旧路径回归不变：既有 9 通道测试全部仍然通过。

**"失败也算完成"的条件**：若 21 点任务上神经网络同样打不过线性外推，**如实记录并说明原因**即为有效交付。按 `AGENTS.md` 第 6 条，真实误差与失败案例必须如实记录。

**规模估计**：第 1、3 步是纯接线，一天内可跑通合成 smoke。第 2 步的坐标系设计需要认真想，别赶。

---

### P1 — 手势分类评测

**目标**：让项目书"验证集准确率 90%"这条指标**从零证据变成有数字**。

**为什么排第二**：投入产出比最高。当前是完全空白，补上就是从 0 到 1。

**改动清单**

1. 写一个最简标注脚本，从 V1–V7 摄像头视频抽帧，标 open / fist / pinch / unknown 四类，每类约 500 帧。
   - 数据划分按视频/人切分，不要让同一视频的相邻帧落进不同集合（`AGENTS.md` 第 6 条、`ai_roadmap.md` 推进顺序第 3 条）。
2. 评测脚本输出：逐类 precision / recall / F1 + 混淆矩阵 + 整体准确率。
3. 阈值扫描：`rule_based_gesture.py` 的八个阈值全部来自 cfg，可直接在开发集上扫，**在测试集上只评一次**。
4. 先明确"准确率"的定义再报数（`ai_roadmap.md` 指标表已提出这个要求）。

**验收判据**：一张混淆矩阵 + 一张逐类 P/R/F1 表 + 明确的数据划分说明。

**规模估计**：标注以小时计，脚本半天。

---

### P2 — 轻量化导出与真实延迟

**目标**：补齐项目书 4.1 节交付物，以及"检测 ≤50 ms / 总响应 ≤100 ms"的实测支撑。

**改动清单**

1. `torch.onnx.export` 导出选中模型；onnxruntime FP32 vs 动态 INT8 对照，报告：数值误差、模型文件大小、CPU 推理 P50/P95。
2. 补一次**真实**的 MediaPipe 检测耗时：跑 `experiments/freihand_eval/scripts/run_current_pipeline_predictions.py` + `evaluate_predictions.py`，把 `examples/sample_report.md` 那份展示用数值替换为实测结果，并注明输入尺寸、硬件、批大小（`ai_roadmap.md` 指标表的要求）。
3. AI 侧只交付"采集 + 检测 + 预测/输出"三段耗时；完整遥操作链路指标由各模块计时共同组成，**不要单方面声称满足 100 ms**。

**验收判据**：一份 FP32/INT8 对照表 + 一份带硬件与输入尺寸说明的检测耗时表。

**注意**：模型只有 17 万参数，量化收益可能很小。如实报告即可。

---

### P3 — MMPose 对照

**目标**：完成项目书 3.1.2 的 MediaPipe / MMPose 比较。

**改动清单**

1. 新增 `src/perception/mmpose_hand.py`，实现 `HandDetector` 协议（`detect` + `close`）。注意 `HandDetection` 的字段约定见 `docs/ai_interfaces.md`：`landmarks_2d` 为 21×2 图像归一化坐标，关节顺序 wrist → thumb → index → middle → ring → little；`handedness` 必须是已校正镜像的真实左右手标签。
2. 预测结果导出为 `experiments/freihand_eval/README.md` 定义的 `predictions.json` 格式，直接复用现有评测脚本。
3. 对照必须统一：同一数据、同一输入分辨率、同一点顺序、**同一计时范围**（当前 freihand 脚本的计时口径是 `detector.detect + 手选择`，不含读图与后续映射）。

**验收判据**：同口径的 PCK / MPJPE / 完整率 / 延迟对照表，能直接进汇报材料。

---

### 未纳入优先级的项目书条目

`Cascade R-CNN / ResNet-101 / InterHand2.6M`（`ai_roadmap.md` 第 3 行）当前无任何实现。按路线图的定位，这些是"候选路线，依据精度/耗时比较选择"，且 InterHand2.6M 的样本"先用于检测评测，不自动变成双手控制"。建议在 P3 拿到 MediaPipe/MMPose 对照数据之后再决定是否投入，避免过早开第三条检测路线。

## 5. 验收判据总表

| 编号 | 交付物 | 判据 |
|---|---|---|
| P0 | 21 点轨迹预测对照 | MPJPE 表覆盖 hold-last / 线性外推 / 神经网络；合成 smoke 通过；9 通道旧路径回归不变 |
| P1 | 手势分类评测 | 混淆矩阵 + 逐类 P/R/F1 + 数据划分说明；准确率定义已明确 |
| P2 | 轻量化与延迟 | FP32/INT8 数值误差与模型大小对照；带硬件说明的检测耗时 P50/P95 |
| P3 | MMPose 对照 | 同数据同分辨率同计时口径的 PCK / MPJPE / 延迟表 |

通用要求（`AGENTS.md` 第 5、6 条）：保留模型维度、通道、采样率、数据值域与时序连续性检查；数据切分、validation 选型、真实误差与失败案例如实记录。

## 6. 禁止事项

1. **不要在 9 通道任务上继续调超参或加模型。** 两轮已证明上限在 3% 以下（见 3.A1）。
2. **不要重新引入 SHA / AST 指纹、Git 身份核验、冻结收据、一次性运行限制。** `AGENTS.md` 第 5 条明确禁止；这些在 `e92092a` 中被移除，重新加回等于回退。
3. **不要覆盖既有数据、checkpoint 或机器报告。** `AGENTS.md` 第 7 条。新实验一律写新目录——`preprocess_h2o.py` 本身也强制输出目录为空。
4. **不要把 9 通道预测称为"已实现项目书的 21 点轨迹预测"。** `ai_roadmap.md` 已就此专门提醒。
5. **不要用合成 smoke 的指标作为算法效果结论。** 数据 manifest 里 `research_claims_allowed: false` 就是这个用途。
6. **不要单方面声称满足"总响应 ≤100 ms"。** AI 侧只交付自己那几段耗时。
7. **不要把可选扩展变成 AI 启动前提**（`AGENTS.md` 第 9 条）。PyTorch 目前是延迟导入的可选依赖，`process_detections` 路径不导入 MediaPipe/PyTorch 也能产出规范输出——保持这个性质。
8. **不要动 `integrations/` 下的外部 Unity 工程和协作方接口语义。**

## 7. 关键路径速查

**代码**

| 用途 | 路径 |
|---|---|
| 单帧 AI 主入口 | `src/pipeline.py` → `HandPipeline` |
| 姿态模型替换接口 | `src/perception/base.py` → `HandDetector` / `HandDetection` |
| 输出协议 | `schemas/frame_payload.schema.json`、`src/output/frame_payload_contract.py` |
| 影子预测运行时 | `src/prediction/shadow_predictor.py`、`shadow_worker.py`、`model_loader.py` |
| 当前模型配置 | `models/residual_motion4.json` |
| 训练/选型/评测 | `experiments/intent_prediction/intent_prediction/second_round.py` |
| 数据预处理 | `experiments/intent_prediction/scripts/preprocess_h2o.py` |
| 姿态评测工具链 | `experiments/freihand_eval/` |
| 最小调用样例 | `examples/use_ai_api.py` |

**文档**

| 用途 | 路径 |
|---|---|
| AI 开发约定（权威） | `AGENTS.md` |
| 路线图 | `docs/ai_roadmap.md` |
| 对接入口说明 | `docs/ai_interfaces.md` |
| 重构记录 | `docs/ai_refactor_20260905.md` |
| 训练与评测命令 | `experiments/intent_prediction/README.md` |
| 历史文档 | `docs/archive/` |

**关键证据文件**

| 结论 | 路径 |
|---|---|
| 离线验收未通过 | `experiments/intent_prediction/reports/second_round/20260829T024400_143036Z_report.json` |
| 延迟注入 retention gate 未通过 | `experiments/intent_prediction/reports/delay_injection/20260829T112008_630685Z_report.md` |
| 投影口径不一致 | `experiments/intent_prediction/reports/label_semantics/20260830_h2o_v2_label_semantics_audit.json` |

## 8. 给接手 agent 的第一步建议

如果只做一件事：先做 P0 改动清单的第 1、3 步（`sequence_data` 加 `target_key`、`models` 解绑 9 通道并替换 sigmoid），跑通合成 smoke。这是纯接线工作，风险低，且是后面所有算法工作的前置。

动手前请确认三件事：

1. 已读 `AGENTS.md` 全部九条。
2. 已亲自打开第 3 节引用的三份报告，确认结论没有过期。
3. 已确认新实验写入新目录，不覆盖 `experiments/intent_prediction/reports/` 与既有 checkpoint。
