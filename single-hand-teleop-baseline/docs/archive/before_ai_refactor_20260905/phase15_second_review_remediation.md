# Phase 1.5 二审修补与算法复核记录

日期：2026-08-30
范围：单右手摄像头/视频 → `control_representation` → `svh_preview` → 本机 Unity UDP 虚拟预览；预测仅为默认关闭的 shadow 诊断。

## 1. 结论先行

DS 二审指出的 CI 闭环、环境污染、真实域样本不足和完成说明口径问题成立。本轮没有为了“通过”而改模型门槛，也没有把预测接入 UDP；控制参考仍是 `hold-last`，现役 v2 retention gate 仍为 **4/6 未通过**。

独立代码审查另外发现并修补了三个问题：

1. latest-only worker 可能覆盖一帧 invalid，导致重新见到手后沿用失效前历史；
2. 旧 mapping SHA 只覆盖 YAML 参数，没有覆盖实际决定 9 通道标签的公式和去抖上下文；
3. 延迟报告把“已有包的接收 tick 上预测可用”写成笼统覆盖率，遗漏初始无包 tick。

H2O 算法审计还发现一个不能原地修改 v2 的数据域问题：pose-only 压缩包没有 `cam_intrinsics.txt`，v2 用相机坐标 `x/y` 代替图像透视投影。该路径可以复现实验，但只能称为代理标签。

## 2. DS 二审意见的处理

| 二审项 | 本轮判断与处理 |
|---|---|
| 新提交没有 GitHub Actions | 成立；本轮完成回归后为当前分支开 PR，使用干净 Windows/Ubuntu CI 验证。合并仍需用户决定。 |
| baseline 环境 `pip check` 污染 | 成立，但缺失项是该共享环境中的 Jupyter/音频包，不属于 baseline 依赖；不向共享环境盲装无关包，以仓库依赖和干净 CI 为准。 |
| 真实摄像头只一段日志 | 成立；仍需固定动作视频集，当前数据不能支持跨场景泛化或延迟补偿结论。 |
| 完成说明把历史 PR/CI 与当前阶段混在一起 | 成立；清单已拆分为历史 PR #14 和当前 Phase 1.5 PR/CI。 |
| 200 ms horizon | 目前不直接增加。旧日志帧龄 P95 超过 150 ms，但先解决低 FPS/丢检和数据域问题；只有冻结的目标延迟确实落在 100–200 ms 时再预注册 200 ms。 |

## 3. 代码修补

### 3.1 invalid 历史断点不再被 latest-only 队列吞掉

`PredictionShadowWorker` 仍保留容量 1、普通有效帧 latest-only 的实时策略，但为 invalid 输入增加单调 `reset_generation`：

- invalid 帧即使被下一有效帧覆盖，generation 仍随下一帧到达 worker；
- worker 在调用模型前先清空历史；
- 重新检测到手后必须重新 warm up；
- 不修改原始 `svh_preview`，不影响 UDP 发送顺序。

新增并发回归用例会故意阻塞推理、依次提交 valid → invalid → valid，确认 invalid 本身被覆盖后，最后一帧仍是 `warming_up` 而不是错误的 `predicted`。

### 3.2 mapping implementation 指纹

原 `mapping_contract_sha256` 保持不变，因而现有 v2 selection/checkpoint 仍可加载。另增加冻结的 implementation SHA：

`9373dc0857df09609a7a06179afa7e933f918d276384211646cf152c07d690ce`

它覆盖：

- 几何与 hand feature 公式；
- raw 手势和现役 `GestureStabilizer`；
- `control_representation` 公式；
- SVH 9 通道映射公式；
- H2O 右手解析、投影和逐帧标签转换；
- `stable_gesture_min_consecutive` 与 `stable_unknown_consecutive`；
- H2O `raw_gesture_as_stable_proxy` 与实时 consecutive stabilizer 的语义差别。

指纹基于去掉注释和文档字符串后的 AST，因此改说明文字不会让模型失效；改权重、公式或有效去抖参数会 fail closed。旧 v2 工件没有该字段时按冻结版本表兼容，新生成 manifest/selection/checkpoint 会显式携带它。

### 3.3 覆盖率分母拆分

报告现在同时输出：

- `conditional_prediction_available_fraction`：已有至少一个到达包的 tick 中，预测是否可用；
- `receiver_coverage_fraction`：全部接收 tick 中，是否已有包；
- `end_to_end_prediction_coverage_fraction`：全部接收 tick 中，是否实际有预测。

冻结 v1 retention gate 仍使用原先预注册的条件覆盖率，避免看完结果后换门槛；新增字段只用于解释和下一版协议设计。

同一冻结矩阵重放的结论：

| 域 | 条件预测覆盖 | 接收覆盖 | 端到端预测覆盖 |
|---|---:|---:|---:|
| H2O primary | 100.00% | 99.52% | 99.52% |
| 2026-08-28 摄像头 JSONL primary | 65.17% | 94.97% | **61.90%** |

retention gate 仍为 4/6，结论没有改变。

机器可读覆盖率审计报告：
[20260830T093122_803505Z_coverage_audit_report.json](../experiments/intent_prediction/reports/delay_injection/20260830T093122_803505Z_coverage_audit_report.json)，SHA-256 `ae97caeae3d8b5e46a1ac06b159a94fc4d133a63b1544ece693e584cf63e70ef`。

## 4. H2O 标签语义全量审计

可复现脚本：

```powershell
python -X utf8 experiments\intent_prediction\scripts\audit_h2o_label_semantics.py `
  --h2o-root D:\VR\HandAi\datasets\H2O\raw `
  --output experiments\intent_prediction\reports\label_semantics\20260830_h2o_v2_label_semantics_audit.json
```

机器可读报告：
[20260830_h2o_v2_label_semantics_audit.json](../experiments/intent_prediction/reports/label_semantics/20260830_h2o_v2_label_semantics_audit.json)，SHA-256 `c7a1a33eeaecc322c3e25a98f9496f7bc852e93487f1864429ab3332cdd78366`。

### 4.1 21 点顺序

H2O 官方可视化代码连接顺序为腕点 0，随后 1–4、5–8、9–12、13–16、17–20 五条手指链，与项目当前索引一致；右手从 128 数值记录的第 65 个数值开始读取也与官方“前 64 左手、后 64 右手”格式一致。因此没有发现关节重排错误。

### 4.2 stateless 代理手势与实时去抖

在 184 个 take、113,396 个有效右手帧上：

- raw/stable 手势不同：1,959 帧，`1.73%`；
- 实际改变 9 通道标签：1,020 帧，`0.90%`；
- 全帧全通道平均绝对差：`0.00240`；
- 最大单通道差：`0.92081`；
- subject4 标签变化比例：`1.15%`。

均值较小，但转场单帧可很大。v2 必须继续标为 `stateless_raw_gesture_as_stable_proxy`；若继续训练 v3，应按连续片段复现实时 stabilizer 后重标，不能覆盖 v2。

### 4.3 pose-only 无内参投影

H2O 官方播放器读取 `cam_intrinsics.txt` 并用 `cv2.projectPoints` 投影 3D 手点。本机 pose-only 数据根目录中该文件数量为 0；v2 因而使用 legacy 相机 `x/y` 代理。

将 v2 legacy 与不需要内参的归一化透视 `x/z, y/z` 对照：

- 9 通道标签有变化：107,832 / 113,396 帧，`95.09%`；
- 全帧全通道平均绝对差：`0.05191`；
- 最大单通道差：`0.96238`；
- subject4 平均绝对差：`0.05605`。

这比手势去抖差异更值得优先处理。本轮新增显式 `normalized_perspective_wrist_origin_palm_scale` 预处理选项，但默认仍是 legacy，以保证 v2 可复现。下一轮若重训，必须新建数据目录、冻结投影 policy、重跑 selection/test/retention，不能把新标签冒充旧 v2。

## 5. 独立逻辑检查

本轮逐项确认：

- 窗口只从同一连续 sequence 构造，train/validation/test 按 subject/sequence 隔离；
- validation 负责候选与 gate 拟合，selection 落盘后才加载 test；没有发现 test 反向参与选型；
- motion gate 推理只使用已观测历史，不读未来目标；
- 网络接收器按“已到达包中 source index 最新”选择，迟到旧包不会覆盖新包；
- 预测按实际帧龄在 0/50/100/150 ms 间插值，超过 150 ms 明确夹紧并计数；
- invalid、非单调 frame/timestamp、过大 gap、checkpoint/selection/mapping 不一致均 fail closed；
- H2O/摄像头指标仍是样本加权，重叠窗口不能当独立统计样本；
- v1 每个网络场景使用独立确定性随机流，适合复现但不是 common-random-number 配对设计。若做 v2 统计比较，应预注册配对随机流和 sequence/subject bootstrap。

## 6. 保留边界与下一步

当前允许表述：完成了单右手 Unity 虚拟预览、UDP 安全门控、默认关闭的预测 shadow，以及对 H2O 代理标签/网络扰动的可复现实验。

当前不允许表述：预测已补偿真实遥操作延迟、H2O 是 SVH 真值、已经完成 VR/5G/双手/真机链路。

优先顺序：

1. 当前 PR 双平台 CI 绿灯；
2. 用固定脚本录制多段个人右手动作视频，报告端到端 prediction coverage；
3. 若继续模型研究，新建 normalized-perspective + sequential-stabilizer 标签版本并重新预注册；
4. 只有真实域收益与覆盖先达标，再做多 seed/LOSO；
5. 实体 SVH 始终走独立协议、标定、限位、watchdog、急停和断线安全验收。

## 7. 验证与 Git 闭环

| 检查 | 结果 |
|---|---|
| `handai-intent-prediction` 全量 pytest | `168 passed` |
| `single-right-hand-baseline` 全量 pytest | `161 passed, 7 skipped` |
| Ruff / compileall | PASS |
| 真实 v2 checkpoint 无摄像头 shadow smoke | 36 帧，`preview_unchanged=true`，最终 `predicted` |
| 冻结 36 场景 + 摄像头 JSONL 覆盖率复跑 | 4/6 未通过；端到端覆盖 61.90% |
| PR | [#15](https://github.com/Save-xi/HandAi/pull/15)，保持 open，未自动合并 |
| GitHub Actions | Windows / Ubuntu baseline job 均 success |

共享 `single-right-hand-baseline` Conda 环境的 `pip check` 仍会报告历史 Jupyter/音频包缺少
`idna/cffi/certifi/jinja2` 等依赖；这些包不在 baseline requirements 中。本轮没有为消除共享环境
噪声而盲装无关依赖，PR 的干净 Python 3.10 Windows/Ubuntu job 均通过 `pip check`。
