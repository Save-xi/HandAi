# AI 交接说明：不足之处与整改方案

面向接手 AI 部分的 agent。先读第 1–4 节确认事实，再按第 5 节制定整改方案。

- 审核基线：`main` = `bbe8884`（Merge PR #18，AI 主线重构之后）
- 初版：2026-09-09；**本版：2026-09-09 修订，已核对 `项目.pdf` 原文**
- 分工前提：本文只覆盖 AI 侧（感知、手势、时序预测、轻量化、评测）。设备采集、Unity、5G、串口、SVH/AUBO 驱动由协作方负责，AI 侧只维护输入输出适配入口。

> **修订说明**：初版的"项目书要求"全部转引自 `docs/ai_roadmap.md`。本版已读到 `项目.pdf` 原文，修正了三处实质性偏差——技术指标"90% 准确率"的对象、流水线顺序、以及预测在控制链路中的位置。详见第 3 节各条目下的「修订」标注。

## 0. 这份文档怎么用

1. **不要把本文当作需求书。** 权威顺序是 `AGENTS.md` > `项目.pdf` > `docs/ai_roadmap.md` > 本文。本文是一次代码审核的结论快照，会过期。
2. **每条结论都附了证据路径。** 动手前先自己打开证据文件确认，尤其是报告类结论——它们是历史机器运行的产物，不会随代码自动更新。
3. **第 3、4 节是"事实与不足"，第 5 节是"建议"。** 前者应当尊重；后者的优先级可以推翻，但推翻时要说明依据。
4. **第 7 节是禁止事项。** 那里列的是已验证没有收益、或会破坏既有数值可比性的做法。

### 本次审核的能力边界（诚实声明）

| 项 | 状态 |
|---|---|
| 源码、配置、测试、历史报告 | 已完整阅读 |
| `pytest` / `ruff` | 已在本机实际运行 |
| `项目.pdf` | **已读到原文**（11 页）。该文件被根目录 `.gitignore` 排除，不在仓库内 |
| H2O 数据集、V1–V7 摄像头视频、FreiHAND | **不在环境中**，无法重跑训练或评测 |
| PyTorch / CUDA | **未安装**，涉及模型推理的测试被跳过 |

凡涉及"模型效果"的结论，全部来自 `experiments/intent_prediction/reports/` 下的历史报告，不是本次重新测得的。

## 1. 复核基线（可复现）

```bash
cd single-hand-teleop-baseline
python -m pip install -r requirements.txt
python -m pytest -q
python -m ruff check src tests scripts experiments examples/use_ai_api.py
```

本次实测：

| 检查 | 结果 |
|---|---|
| pytest | 171 收集：**164 passed, 7 skipped** |
| 跳过原因 | 全部是缺 PyTorch（`test_intent_prediction_first_round` ×5、`test_intent_prediction_second_round` ×1、`test_prediction_shadow_checkpoint` ×1） |
| ruff | All checks passed |

`docs/ai_refactor_20260905.md` 记录的"171 passed"与此一致——那台机器装了 torch。**工程层面没有需要返工的东西**，本文后续所有问题都在算法与评测层。

## 2. 项目书对 AI 侧的原文要求

以下摘自 `项目.pdf`，**只保留技术内容**；学校、成员、经费等申报信息按该文件被 gitignore 的原意有意省略。摘录于此是为了让后续 agent 不必依赖本机 PDF。

### 2.1 总技术路线（第 5 页，第 3 节开头）

> 操作者根据任务需要做出手势动作，VR头戴设备和深度相机采集该手势动作并将其传入手部姿态估计网络，**使用轻量化AI对手部关键点的位置信息进行预测**，**根据预测得到的关节位置信息判断手势分类**，选择相应的映射方法计算得到合适的SVH控制信息，使用Socket通信将控制信息传输给在Unity中搭建的虚拟手控制平台，对虚拟手进行操控验证，同时将信息经5G网络低延时传输给实物灵巧手。

**这句话定义了 AI 侧的流水线顺序**：

```
关键点检测 → 【预测未来关键点】 → 【由预测关键点判手势】 → 选映射方法 → SVH 控制信息
```

### 2.2 技术指标（第 8 页，共三条）

> —手部姿态估计模型在验证集上的准确率达到90%；
> —手部检测时间不超过50ms;
> —灵巧手遥操作响应时间延迟不超过100ms

**注意指标 1 的对象是"手部姿态估计模型"，不是手势分类。** 第 7 页另有一处相关表述：Cascade R-CNN"对手部的识别精度可达到90%以上"。

指标 3 的预算分解：项目书第 8 页称工业 5G 基站"将时延控制在10ms以内"，指标 2 给检测 50 ms，**留给预测、映射、渲染、串口的合计余量约 40 ms**。

### 2.3 姿态估计选型（第 6–7 页，3.1.2）

> a) 算法库的选择：本项目有MMPose和MediaPipe两种手部姿态估计框架可供选择，**后期可根据运算精度与速度择优选取**，以实现灵巧手控制的低延迟。
>
> b) 数据集的选择：考虑到SVH灵巧手具有20自由度，且在现实装配过程中操作者通常需要双手交互，且容易产生遮挡现象，因此本文选择关节划分方式为2号的左右手交互数据集**InterHand2.6M**。……标注主要是对单手21个关键节点进行标注。
>
> c) 手部识别与特征提取：选择**Cascade R-CNN作为手部识别网络，用来获得图像中的手部包围框**；选择**ResNet-101作为手部图像的特征提取网络**。

### 2.4 创新点 4.1（第 8 页）

> 创新性地在**边缘端（VR头显）**部署轻量化手部姿态估计与意图预测算法。通过在本地提前预测操作者的动作意图，**对灵巧手进行前馈控制**，从算法层面主动补偿网络抖动带来的非确定性，验证了高精度AI模型在资源受限设备上运行的可行性。

第 1.1 节另有："通过**模型压缩与量化**技术验证算法可在资源受限设备上运行。"

**边缘设备是 HoloLens2**（第 6、10 页：骁龙 850 + Adreno 630，Windows 全息操作系统），不是 Jetson、不是 PC。

### 2.5 与当前实现相符、可以放心保留的部分

| 项目书表述 | 出处 | 对当前实现的意义 |
|---|---|---|
| SVH"运动指骨带有**9个驱动器**"（20 自由度、欠驱动） | 第 11 页 | **9 通道表示不是臆造的**，它对应真实执行器空间。`svh_adapter` 的 9 通道映射本身是合理的 |
| "实验室中的为 **SVH 右手版**" | 第 11 页 | **单右手基线与硬件一致**，不是偷懒。`AGENTS.md` 第 2 条把双手列为后续任务是站得住的 |
| InterHand2.6M 选型理由是**双手交互 + 遮挡** | 第 6 页 | 该数据集的价值在于**检测鲁棒性**，不等于要做双手控制。`ai_roadmap.md`"先用于检测评测，不自动变成双手控制"的处理正确 |
| 轻量化模型架构未指定 | 全文 | LSTM 仅作为**他人工作**被引用（NASA、上海交大）。现有 residual GRU / TCN / Transformer 选型自由，**架构不是问题所在** |

## 3. 当前真实能力边界

| 能力 | 状态 | 证据 |
|---|---|---|
| 单右手 21 点检测（MediaPipe） | 可运行 | `src/perception/mediapipe_hand.py` |
| 统一 AI 入口，可换姿态模型 | 可运行 | `src/pipeline.py`；`src/perception/base.py` |
| 设备关键点直接入口 | 可运行 | `HandPipeline.process_detections` |
| 规则手势 open/fist/pinch/unknown | 可运行，**从未评测** | `src/gesture/rule_based_gesture.py` |
| 9 通道 `svh_preview` 映射 | 可运行 | `src/svh/svh_adapter.py` |
| 9 通道未来 50/100/150 ms 预测 | 可运行，**离线验收未通过、旁路运行** | `models/residual_motion4.json` `offline_gate_passed: false` |
| 姿态评测工具链（FreiHAND 口径） | 可运行，**无真实结果入库** | `experiments/freihand_eval/` |
| 21 点未来轨迹预测 | **未实现** | 见 4.A1 |
| MMPose / Cascade R-CNN / ResNet-101 / InterHand2.6M | **未实现** | 全仓库无相关代码 |
| ONNX / 量化 / 边缘部署 | **未实现** | 全仓库 grep 不到 onnx/quantize/tflite/tensorrt |
| 三条技术指标 | **0 条有实测证据** | 见 4.B1、4.B2 |

## 4. 不足之处

按"是否阻塞后续工作"分级。

---

### A 级：阻塞

#### A1. 流水线顺序与项目书相反，且当前任务已触到效果上限

> **修订**：初版称此为"任务定义不一致"。读到原文后，问题比这更具体——**是流水线顺序被倒置了**。

**项目书要求**（2.1）：`检测关键点 → 预测未来关键点 → 由预测关键点判手势 → 选映射 → SVH 9 通道`

**当前实现**：`检测关键点 → 判手势 → 映射成 SVH 9 通道 → 预测 9 通道`

预测被放在了映射**之后**，而项目书要求它在映射**之前**。两个直接后果：

1. 学习目标变成了映射产物而非关键点，效果触顶（见下）。
2. 手势分类吃的是**当前帧**特征，而项目书要求它吃**预测出来的**关节位置。也就是说"提前量"根本没有传到手势和映射环节。

**效果证据。** `reports/second_round/20260829T024400_143036Z_report.json`：

| 判据 | 要求 | 实测 | 结论 |
|---|---:|---:|---|
| `overall_rmse_improvement_percent` | ≥ 3.0 | **2.5296** | 不通过 |
| `q90_rmse_improvement_percent` | ≥ 5.0 | **3.7548** | 不通过 |
| `q90_p95_improvement_percent` | > 0 | 5.5600 | 通过 |
| `overall_mae_regression_percent` | ≤ 0.5 | −2.2018 | 通过 |
| `range_violation_rate` | = 0 | 0.0 | 通过 |
| `single_window_p95_ms` | ≤ 10.0 | 0.9533 | 通过 |

`acceptance.offline_gate_passed = false`。延迟注入回放同样 `retention gate: false`，决策字段为 `retain_hold_last_as_control_reference_and_keep_model_shadow_only`，聚合 RMSE 改善仅 1.95%（`reports/delay_injection/20260829T112008_630685Z_report.md`）。

**原因分析（推断，需接手方验证）。** 9 通道是几何特征经参考点归一、手势上下文分支、`clip(0,1)` 之后的产物，运动学结构被大幅压缩。旁证：`finger_spread` 通道 hold MAE 仅 0.0068，接近常数——对高度饱和的信号 persistence 天然极强。

**建议的验证方法**（一次性脚本，不需训练）：统计已预处理 NPZ 中 `controls_9ch` 逐通道贴边（≈0 或 ≈1）的帧占比与逐通道方差。若大比例贴边，饱和假设成立。

**注意**：9 通道表示本身**不需要废弃**——它对应 SVH 的 9 个真实驱动器（2.5 节）。要改的是**预测发生的位置**，不是映射本身。

---

#### A2. 预测没有进入控制链路，创新点 4.1 的核心主张目前无支撑

> **新增条目**（初版遗漏，因为未读到创新点原文）。

**项目书**（2.4）："通过在本地提前预测操作者的动作意图，**对灵巧手进行前馈控制**"。

**现状**：预测是 `PredictionShadow` 影子模式，默认关闭，`observe()` 明确"既不改写 `svh_preview`，也不参与 UDP 发送"（`src/prediction/shadow_predictor.py` 模块 docstring）。延迟注入实验的最终决策是保留 hold-last 作为控制参考。

**影响**：项目书把"前馈控制补偿网络抖动"列为第一创新点，当前**没有任何一条链路让预测值影响下发的控制量**。影子模式是正确的工程谨慎，但不能停在这里——否则创新点 4.1 只完成了"预测"，没完成"前馈"。

**注意**：不要为了凑创新点就贸然把未通过验收的预测接进控制路径。正确顺序是先解决 A1 让预测真正有增益，再设计带门控和回退的前馈接入，并明确安全边界。

---

#### A3. 训练标签的几何口径与运行时几何不一致，偏差量级大于待验收收益

**现象。** H2O pose-only 发布包中绝大多数 take 缺 `cam_intrinsics.txt`，预处理走了 `legacy_camera_xy_wrist_origin_palm_scale` 回退——直接取相机坐标 x/y，**没做透视除法**。运行时 MediaPipe 输出的是真正的透视归一化坐标。

**证据。** `reports/label_semantics/20260830_h2o_v2_label_semantics_audit.json`，`groups.all`：

- `projection_label_different_fraction` = **0.9509**（subject1 = 0.9988）
- `projection_mean_channel_abs_error_over_all_frames` = **0.0519**
- `projection_max_channel_abs_error` = 0.9624

对照量级：模型自身总 MAE 是 0.0217，追求的改善约 2%。**标签口径偏差比整个待验收收益大一个量级。**

代码位置：`h2o_adapter.py:32` 定义回退策略，`:293` 作为默认值。

**注意**：按 `AGENTS.md` 第 7 条，既有数据与 checkpoint 保留不覆盖；整改应产出**新**数据目录与新模型。

---

### B 级：技术指标零证据

> **修订**：初版把"90% 准确率"错误归到手势分类，据此把手势评测排为第二优先级。读到原文后更正——该指标的对象是**手部姿态估计模型**。手势评测仍有价值，但不是这条官方指标，优先级下调。

#### B1. 指标 1「姿态估计验证集准确率 90%」无任何实测

- **现状**：姿态评测工具链已就绪（`experiments/freihand_eval/`，可算 PCK@5/10/20/30px、MPJPE、关键点完整率），但**从未跑过真实数据入库**。
- **证据**：`experiments/freihand_eval/examples/sample_report.md` 开头自述"表格里的数值是展示用口径，不代表真实评估结果"。仓库内不存在其他姿态指标结果。
- **待决策**：项目书点名 InterHand2.6M，当前工具链面向 FreiHAND。用哪个数据集报这条指标，需要明确并记录理由（见 B4）。
- **还需明确**："准确率"的口径。PCK@某阈值是最自然的落点，但阈值必须先定死再报数，不能挑一个好看的报。

#### B2. 指标 2「手部检测 ≤50 ms」无任何实测

- **现状**：`run_current_pipeline_predictions.py` 已记录 `detector.detect + 手选择` 的单样本耗时，`freihand/report.py` 已能出 P95/P99——只差跑一次真实数据。
- **必须同时记录**：输入尺寸、硬件、批大小。在 PC 上测出的 20 ms 不能直接用来声称 HoloLens2 上满足 50 ms。

#### B3. 指标 3「总响应 ≤100 ms」——AI 侧应交付的分段耗时未成体系

- **现状**：`src/main.py:444` 的 `timing` 字段已按 unix ms 记录 `source_read` / `detection_end` / `baseline_end` / `preview_end` / `payload_ready` / `udp_send_attempt` 各时刻，数据结构是够的，但**没有据此产出过统计报告**。
- **边界**：AI 侧只交付采集、检测、预测/输出这几段；完整链路指标由各模块计时共同组成。**不要单方面声称满足 100 ms。**

#### B4. 数据集与网络选型偏离项目书，且偏离未被记录

> **新增条目**。

| 项目书点名 | 代码实际使用 | 状态 |
|---|---|---|
| InterHand2.6M（姿态/检测） | FreiHAND（姿态评测） | 替换，**理由未记录** |
| —（未提及） | H2O（时序预测） | 新增，**理由未记录** |
| Cascade R-CNN（手部包围框） | MediaPipe 内置检测 | 未实现 |
| ResNet-101（特征提取骨干） | 无 | 未实现 |
| MMPose 或 MediaPipe 二选一，后期择优 | 仅 MediaPipe | 择优实验未做 |

**影响**：这些替换可能都是合理的工程选择（FreiHAND 单手标注更适合当前单右手基线；H2O 提供 InterHand2.6M 没有的时序姿态），**但没有任何一处文档写明为什么这样换**。答辩时"为什么没用项目书里写的 InterHand2.6M"是必然被问的问题。

**建议**：这属于文档债，不是代码债。写一份选型说明记录进 `docs/`，成本很低，收益很高。

#### B5. 轻量化与边缘部署零实现

- **证据**：`grep -rniI "onnx|quantiz|int8|tflite|tensorrt|openvino"` 只在 `docs/ai_roadmap.md` 的计划文字里命中。
- **项目书要求**（2.4、1.1）：模型压缩与量化，部署在 VR 头显边缘端。
- **目标设备已明确**：HoloLens2（骁龙 850 / Adreno 630 / Windows 全息 OS，ARM64）。这决定了导出路线应走 **ONNX**（配合 Windows ML 或 Unity Barracuda），而不是 TensorRT / TFLite。
- **预期管理**：待导出模型只有 **172,699 参数**。量化收益很可能有限，但对照表本身就是交付物，**负结论也算完成**。

#### B6. 预测 horizon 在真实摄像头下覆盖不足

- **证据**：`reports/delay_injection/20260829T112008_630685Z_report.md` 真实摄像头回放段：`prediction 覆盖率 = 0.651739`，`超 150 ms horizon 比例 = 0.305544`。
- **影响**：约 30% 的帧落在最大预测距离之外。结合 2.2 的预算分解，整改时应重新论证 horizon 取值，不要照抄 50/100/150。

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

**注意 C3 的替换方式**：`ResidualGRUForecaster` 已有 `max_delta` 参数（配置为 0.35），改成 `tanh(·) × max_delta` 的有界残差是最自然的替换，能保留"初始状态严格等于 persistence"这个设计意图。不要简单删掉激活函数了事。

---

### D 级：轻微，不阻塞

| 编号 | 位置 | 说明 |
|---|---|---|
| D1 | `src/pipeline.py:237` | `_apply_extension_chain` 返回的 `diagnostics` 被直接丢弃，只有测试消费返回值。扩展失败只进日志、不进 payload/JSONL，离线回放分不出哪帧是降级帧。建议挂到 payload（需同步更新 `schemas/frame_payload.schema.json` 与 `frame_payload_contract.py` 的可选字段列表） |
| D2 | `src/pipeline.py:243` 与 `src/main.py:457` | `prepare_frame_payload` 调用两次。**不是缺陷**——主循环在 pipeline 返回后才补 `timing`，必须重新规范化。仅记录 |
| D3 | 协作方事项，非 AI 范围 | 项目书 3.2.1 指定 Socket（TCP/IP）通信；当前预览链路是 UDP（`configs/unity_udp_preview.yaml`）。请通信/Unity 负责人确认，AI 侧不动 |

## 5. 整改方案

优先级依据：先拿到官方技术指标的实测证据（最便宜、最刚性），再修流水线顺序（结构性问题），最后做交付物类工作。

> **修订**：初版顺序为「21 点预测 → 手势评测 → 轻量化 → MMPose」。本版据项目书技术指标重排——姿态评测提到第一位（它一次覆盖两条官方指标且工具链现成），手势评测降到最后（它不是官方指标）。

---

### P0 — 姿态估计准确率与检测耗时实测（技术指标 1、2）

**为什么排第一**：一次跑通同时拿到三条官方技术指标里的两条，工具链已经写好，只差真实数据。这是当前投入产出比最高的一件事。

**改动清单**

1. 准备数据集，并**先决定用 FreiHAND 还是 InterHand2.6M**（见 B4）。若时间紧张，建议：先用 FreiHAND 拿到数字，同时把选型理由写进文档；InterHand2.6M 留给遮挡场景专项。
2. 依次执行（命令见 `experiments/freihand_eval/README.md`）：
   `run_current_pipeline_predictions.py` → `evaluate_predictions.py` → `make_report_table.py`
3. **先定死"准确率"口径再报数**：建议以 PCK@某固定像素阈值为主指标，MPJPE 与关键点完整率为辅。阈值一旦选定写入配置，不允许事后调整。
4. 用实测结果替换 `examples/sample_report.md` 中的展示用数值，或另存真实报告并在该文件注明何处查看真值。
5. 记录硬件、输入分辨率、批大小、计时范围（当前口径是 `detector.detect + 手选择`，不含读图与后续映射）。

**验收判据**：一张带口径说明的 PCK/MPJPE/完整率表 + 一张带硬件说明的检测耗时 P50/P95 表。

**"失败也算完成"**：若 PCK 达不到 90%，如实记录并分析原因（分辨率、遮挡、手选择策略），这本身就是有效交付，也正是 P3 择优实验的动机。

**规模估计**：数据就位后半天到一天。

---

### P1 — 21 点关键点预测，恢复项目书的流水线顺序

**目标**：把学习目标从映射后的 9 通道换成关键点，让预测回到映射**之前**，使"预测关键点 → 判手势 → 选映射"这条链路成立。

**关键前提（省一大笔工作）**：21 点数据**已经在盘里且从未被使用**。`h2o_adapter.py:255-256` 每条序列都写入了 `landmarks_3d`（21×3 相机坐标）和 `landmarks_2d`（21×2）。除写入方外，全仓库无任何训练/评测代码读取它们（`sequence_data.py:188-189` 只在合成数据里写空数组占位）。**不需要重新预处理即可开工。**

**改动清单**

1. `sequence_data.py`：`_iter_sequence_windows` / `build_window_split` 增加 `target_key` 参数（`controls_9ch` | `landmarks_3d` | `landmarks_2d`），21×3 展平为 63 维；形状校验改为按实际通道数。**默认值保持 `controls_9ch`**，确保既有 9 通道实验可原样复现。
2. **先定坐标系（不要跳过）**：建议腕点为原点 + 掌长归一化，与 `h2o_adapter.canonicalize_h2o_camera_xy` 口径一致；全局位移单独作为一路目标。定完写进数据 manifest，不要只留在代码里。
3. `models.py`：`build_model` 参数化 `input_size` / `output_size`；按 C3 说明替换 sigmoid。
4. `metrics.py`：新增关键点口径——MPJPE、逐关节误差、按时间距分解。**不要沿用逐通道 MAE 汇报关键点结果。**`range_violation_rate` 对关键点无意义，应按目标类型分支。
5. `baselines.py`：解绑 9 通道；**务必纳入线性外推**——关键点任务上常速度外推是很强的基线，只比 hold-last 会高估模型收益。
6. 复用 `second_round.py` 已有协议（train → validation 选型冻结 → test 一次），**不要重写**。
7. **接上下游**：验证"由预测关键点判手势"这条链路——`infer_gesture_raw` 吃的是 `hand_features`，需要一条从预测关键点重算特征的路径。这是项目书 2.1 的核心要求，不能只做预测就收工。
8. 顺带处理 A3：用 `preprocess_h2o.py --pose-only-projection normalized_perspective_wrist_origin_palm_scale` 重跑一份新数据到**新的空目录**（该参数见 `preprocess_h2o.py:40-45`）。旧数据保留不动。

**验收判据**
- 合成 smoke 跑通：`python -X utf8 experiments/intent_prediction/scripts/run_second_round.py --synthetic-smoke --output-root outputs/keypoint_smoke`
- 真实数据上给出 21 点 MPJPE 对照表：hold-last / 线性外推 / 神经网络。
- 9 通道旧路径回归不变：既有 9 通道测试全部仍然通过。
- 能展示"预测关键点 → 手势"这一步跑通。

**"失败也算完成"**：若 21 点任务上神经网络同样打不过线性外推，**如实记录并说明原因**即为有效交付（`AGENTS.md` 第 6 条）。

**规模估计**：第 1、3 步是纯接线，一天内可跑通合成 smoke。第 2 步的坐标系设计需要认真想，别赶。

---

### P2 — 轻量化导出与边缘可行性（创新点 4.1 的后半）

**目标设备**：HoloLens2（骁龙 850 / Adreno 630 / ARM64 / Windows 全息 OS）。导出路线走 **ONNX**（Windows ML 或 Unity Barracuda），不要选 TensorRT / TFLite。

**改动清单**

1. `torch.onnx.export` 导出选中模型；FP32 vs 量化对照，报告：数值误差、模型文件大小、推理 P50/P95。
2. 如果拿不到 HoloLens2 实测环境，**明确标注是在 PC 上的代理测量**，不要写成边缘端实测。这一点在答辩时必须诚实。
3. 项目书第 6 节还提到剪枝、知识蒸馏。以 17 万参数的规模，**建议只做量化并说明为何不做剪枝**，而不是硬凑三种压缩手段。

**验收判据**：FP32/量化数值误差与模型大小对照表 + 明确标注测量平台的推理耗时表。

---

### P3 — MMPose vs MediaPipe 择优（项目书 3.1.2a 明确留待"后期择优"）

**目标**：完成项目书自己写明要做的选型决策，依据是"运算精度与速度"。

**改动清单**

1. 新增 `src/perception/mmpose_hand.py`，实现 `HandDetector` 协议（`detect` + `close`）。字段约定见 `docs/ai_interfaces.md`：`landmarks_2d` 为 21×2 图像归一化坐标，关节顺序 wrist → thumb → index → middle → ring → little；`handedness` 必须是已校正镜像的真实左右手标签。
2. 预测结果导出为 `experiments/freihand_eval/README.md` 定义的 `predictions.json` 格式，直接复用 P0 的评测脚本。
3. 对照必须统一：同数据、同输入分辨率、同点顺序、**同计时范围**。
4. **产出一个明确结论**："本项目选择 X，因为在同口径下精度 A、耗时 B"。项目书把这个决定留给了"后期"，现在就是后期。

**验收判据**：同口径对照表 + 一句可写进结题材料的选型结论。

---

### P4 — 手势分类评测

> **修订**：初版排第二并称其对应"90% 准确率"指标。更正后，此项**不是官方技术指标**，但仍有价值——项目书 2.1 里手势分类决定"选择相应的映射方法"，是控制链路的一环，目前完全无评测。

**改动清单**

1. 从 V1–V7 摄像头视频抽帧，标 open / fist / pinch / unknown 四类，每类约 500 帧。按视频/人切分，不要让同一视频相邻帧落进不同集合。
2. 输出逐类 precision / recall / F1 + 混淆矩阵。
3. `rule_based_gesture.py` 的八个阈值全部来自 cfg，可在开发集上扫，**测试集只评一次**。
4. H2O 的 `action_labels`（`h2o_adapter.py:216` 读入）是**动作类别**，不是 open/fist/pinch，不能直接当手势标签。

---

### 待决策：Cascade R-CNN + ResNet-101 + InterHand2.6M

项目书 3.1.2c 明确点名了这两个网络及其分工（包围框检测 + 特征提取骨干），比 `ai_roadmap.md` 转述的"候选路线"更具承诺性。

**建议**：先做 P0 和 P3 拿到 MediaPipe / MMPose 的实测数据，再决定是否投入自训练检测网络。如果 MediaPipe 已满足 90% 与 50 ms，就把这条写成"经实测比较后选择 X，未采用 Cascade R-CNN 路线"并给出数据依据——**这是合理的项目演进，不是没完成任务**，但必须有数据支撑，不能只是没做。

## 6. 技术指标追踪表

| 项目书指标 | 当前证据 | 由谁的工作补齐 |
|---|---|---|
| 姿态估计验证集准确率 ≥ 90% | **无** | P0 |
| 手部检测 ≤ 50 ms | **无**（现有数值是展示用示例） | P0 |
| 遥操作总响应 ≤ 100 ms | **无**；`timing` 字段已具备但无统计报告 | P0/B3 + 协作方 |
| 创新点 4.1 轻量化边缘部署 | **无** | P2 |
| 创新点 4.1 前馈控制补偿抖动 | **无**；预测当前为旁路 | P1 → A2 |

通用要求（`AGENTS.md` 第 5、6 条）：保留模型维度、通道、采样率、数据值域与时序连续性检查；数据切分、validation 选型、真实误差与失败案例如实记录。

## 7. 禁止事项

1. **不要在 9 通道任务上继续调超参或加模型。** 两轮已证明上限在 3% 以下（4.A1）。
2. **不要为了凑创新点 4.1 就把未通过验收的预测接进控制路径。** 先解决 A1，再谈前馈接入，且必须带门控与回退。
3. **不要重新引入 SHA / AST 指纹、Git 身份核验、冻结收据、一次性运行限制。** `AGENTS.md` 第 5 条明确禁止；这些在 `e92092a` 中被移除。
4. **不要覆盖既有数据、checkpoint 或机器报告。** `AGENTS.md` 第 7 条。新实验一律写新目录。
5. **不要把 9 通道预测称为"已实现项目书的关键点预测"。** 两者在项目书里处于流水线的不同位置。
6. **不要用 PC 上的耗时直接声称满足边缘端指标。** 目标设备是 HoloLens2，测量平台必须标注。
7. **不要单方面声称满足"总响应 ≤100 ms"。** AI 侧只交付自己那几段耗时。
8. **不要用合成 smoke 的指标作为算法效果结论。** manifest 里 `research_claims_allowed: false` 就是这个用途。
9. **不要把可选扩展变成 AI 启动前提**（`AGENTS.md` 第 9 条）。PyTorch 目前是延迟导入的可选依赖，`process_detections` 路径不导入 MediaPipe/PyTorch 也能产出规范输出——保持这个性质。
10. **不要动 `integrations/` 下的外部 Unity 工程和协作方接口语义。**

## 8. 建议同步修订的既有文档

本次核对原文后发现 `docs/ai_roadmap.md` 有两处可以更精确（**本次未改动，留给负责人决定**）：

1. 指标表"验证集准确率 90% | 先明确'准确率'的定义……手势任务报告逐类 precision/recall/F1"——原文已明确对象是**手部姿态估计模型**，这个悬置可以收敛掉。
2. 推进顺序第 2 条描述了关键点预测，但未点明项目书要求**手势分类消费预测后的关键点**。补上这句能避免后续实现只做预测就收工。

## 9. 关键路径速查

**代码**

| 用途 | 路径 |
|---|---|
| 单帧 AI 主入口 | `src/pipeline.py` → `HandPipeline` |
| 姿态模型替换接口 | `src/perception/base.py` → `HandDetector` / `HandDetection` |
| 输出协议 | `schemas/frame_payload.schema.json`、`src/output/frame_payload_contract.py` |
| 分段计时字段 | `src/main.py:444` `payload["timing"]` |
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
| 离线验收未通过 | `reports/second_round/20260829T024400_143036Z_report.json` |
| 延迟注入 retention gate 未通过 | `reports/delay_injection/20260829T112008_630685Z_report.md` |
| 投影口径不一致 | `reports/label_semantics/20260830_h2o_v2_label_semantics_audit.json` |

（以上相对 `experiments/intent_prediction/`）

## 10. 给接手 agent 的第一步建议

**如果只做一件事**：做 P0。它一次覆盖三条官方技术指标中的两条，工具链现成，风险最低，且结果直接决定 P3 和"是否需要 Cascade R-CNN"这两个后续决策。

**如果能做两件事**：P0 之后做 P1 的第 1、3 步（`sequence_data` 加 `target_key`、`models` 解绑 9 通道并替换 sigmoid），跑通合成 smoke。纯接线工作，是后面所有算法工作的前置。

动手前请确认三件事：

1. 已读 `AGENTS.md` 全部九条，以及本文第 2 节的项目书原文摘录。
2. 已亲自打开第 4 节引用的三份报告，确认结论没有过期。
3. 已确认新实验写入新目录，不覆盖 `experiments/intent_prediction/reports/` 与既有 checkpoint。
