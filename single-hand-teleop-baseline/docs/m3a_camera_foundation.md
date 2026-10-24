# M3-A：摄像头几何、输入与释放整改

2026-09-22，在负责人批准 M3-A 后实施。新入口为 `configs/ai_m3a.yaml`。完成输入尺度、坏帧处理、一个连续释放候选与 CPU 预测 CI；没有训练新的 H2O 模型，也未进入 M3-B/C。

## 实现与兼容

- `HandDetection` 显式携带原图宽高与 `mediapipe_image_xyz` 坐标声明。原始 XY/XYZ 保留给显示与记录；另以 `g=(x, y*H/W, z)` 计算比例和角度。几何任务为 `camera_mediapipe_geometry_xyz_v1`，单位是图像宽度，非公制三维。
- 流式入口按结构/有限值 → 几何 → 掌尺度/骨段 → 图像边界 → 特征/手势执行。掌长采用 XY 的 wrist→middle MCP；小于等于 1e-6 才尝试 index MCP→little MCP 掌宽，最终尺度至少 1e-4 图像宽。五指各三条相邻骨段的三维长度须大于 `max(1e-6, palm_size*0.001)`，二维重叠但深度不同不会直接被拒绝。
- 无效帧返回 `input_diagnostics.reason`，不输出可消费控制值。结构、数值、尺度错误清空坐标；单纯越界保留已核验的有限原始点。立即重置手势、释放状态；预测旁路因预览失效清空历史。下一有效帧重新确认手势。有效 `unknown` 可保留连续控制量。
- 新释放权重由张开比例、非拇指平均 curl 和捏合距离连续决定，使用 0.01 幅度滞回，不增加时间平均。curl 过渡半宽 0.02，捏合距离过渡宽度 0.10；端点直接到达 0/1。它取消释放函数对稳定 open 的硬门控，未重写全部 grasp/pinch 分支。
- `fork_mapping()` 复制手势候选、确认计数、释放权重、上次时间、图像尺寸与源节奏估计；复制不更新状态。新流遇倒序、真实断流或尺寸变化会重启状态。M3-A 交付时断流阈值固定为 100ms；2026-10-20 起随源周期放宽，见[实时慢帧断流修订](#实时慢帧断流修订2026-10-20)。M3-B 仍需实现预测网格与分数时刻查询。
- `ai.yaml`、`svh_9ch_preview.yaml` 保留历史几何/释放含义，M2 的独立标签计算未改。统一坏帧检查应用于两种流式模式。M3-A 交付版本为 `svh9-camera-v3-continuous-release`，断流修订后为 `svh9-camera-v4-cadence-gap`，参数明文保存；旧模型无法静默加载到新配置，当前新配置默认不启用预测。

## 固定事件实测

数字见 [机器报告](reports/m3a_development.json)，完整 trace 保存在本机 `outputs/m3a/20260922T105434_910453Z/`。以下为控制特征构造的工程样例，六类转换线性推进 15 个源采样间隔，FPS=30；不是人工标签或真实动作准确率。

转换时刻定义为首次进入并持续保持在最终通道响应的 90% 范围内，每路使用自己的起止值；滞后从事件起点计。新增滞后取新减旧，负值表示该定义下更早到达。每通道 P95/最大单帧变化、具体输入、微扰和丢手 trace 均在机器报告中；差分不跨失效帧。

| 事件 | 旧释放最大跳变 | 新释放最大跳变 | 新增滞后 ms |
|---|---:|---:|---:|
| curl 0.4499、0.4499、0.4501 | 0.470343 | 0.000106 | 微扰，不定义转换时刻 |
| open→unknown | 0.488113 | 0.406642 | 0 |
| unknown→open | 0.439361 | 0.419543 | -33.333 |
| open→pinch | 0.309327 | 0.190492 | 0 |
| pinch→open | 0.275642 | 0.245014 | 0 |
| 正常张开 | 0.135160 | 0.097070 | 0 |
| 正常闭合 | 0.100000 | 0.096292 | 0 |

微扰要求最大跳变 ≤0.05，实测通过；六类转换最大新增滞后 0ms，低于一个源周期 33.333ms。丢手当帧无效，恢复后的有效观测才重新产生输出。该结果仅证明固定样例的工程门槛；较大姿态变化仍可产生较大输出变化，不宣称所有转换已平滑。

## 两段开发视频检查

使用既有 V1/V5 各前 180 帧，共 360 帧，每帧只调用一次 MediaPipe，三路共用检测结果。所有帧使用容器 PTS，FPS 约为 29.927/29.996；没有 PTS 回退。这是历史开发片段，人员/会话未恢复，不是独立测试集。

| 视频 | 旧几何+旧释放 | 新几何+旧释放 | 新几何+新释放 | 有效帧（三路各自） |
|---|---:|---:|---:|---:|
| V1 最大通道单帧变化 | 0.700000 | 0.569221 | 0.226504 | 180/180 |
| V5 最大通道单帧变化 | 0.700000 | 0.673411 | 0.582843 | 180/180 |

新几何分别改变 V1 的 21 帧、V5 的 22 帧原始手势标签；没有人工标签，无法判断变化是否更准确。本轮保留旧手势阈值，未用标签变化率选阈值。V5 仍有 0.582843 的大跳变，既有 pinch 路由也仍按上下文切换。M3-B 应记录并定位这些事件；若基础几何/规则主导误差，按路线返回 M3-A，不能用预测掩盖。

视频均未在这 360 帧触发坏帧，坏帧与恢复证据来自合成回归，不从视频有效率推断鲁棒性。没有预测 RMSE、姿态真值或设备时间验收结论。

## 验证与复现

本机 `handai-intent-prediction` 环境：**251 passed，0 skipped**；Ruff、compileall 通过。较 M2 增加 53 项用例，覆盖 24 组编码宽高比/旋转/尺度、异常输入与恢复、输出契约、完整状态复制和释放事件。几何特征/通道等价容差为 1e-8；原始显示点保持原值。另用真实 CLI 完成 V1 的 30 帧与 JSONL 输出。

CI 新增 Linux CPU PyTorch 2.7.0 job，明确导入 torch、检查没有 CUDA，并执行 M1/M2 的 21 项可重建用例；JUnit 中任意 skip 会导致失败。用例实际训练、检查掩码梯度、保存与重载，无需 H2O 或 ignored checkpoint。现有 Windows/Ubuntu baseline 检查继续保留；远端执行结果见 [PR #21](https://github.com/Save-xi/HandAi/pull/21)。

摄像头运行：

```bat
conda activate handai-intent-prediction
cd /d D:\VR\HandAi\single-hand-teleop-baseline
python -X utf8 src\main.py --config configs\ai_m3a.yaml --camera-index 0 --save-jsonl
```

开发诊断（省略两个 `--video` 即只运行固定释放事件）：

```bat
conda activate handai-intent-prediction
cd /d D:\VR\HandAi\single-hand-teleop-baseline
python -X utf8 scripts\evaluate_m3a.py --video D:\HandAiVideos\camera_domain_dev_v1\v1.mp4 --video D:\HandAiVideos\camera_domain_dev_v1\v5.mp4 --max-frames 180
```

自检：

```bat
python -X utf8 -m pytest -q tests\test_m3a_geometry.py tests\test_m3a_release.py tests\test_mediapipe_detector.py
python -X utf8 -m pytest -q tests\test_keypoint_prediction.py tests\test_representation_comparison.py
python -X utf8 -m ruff check src tests scripts experiments examples/use_ai_api.py
```

本次交付时，下一阶段为 M3-B 的三个简单基线与完整计时；当时主循环时间戳仍在检测后生成，开发脚本则明确传媒体 PTS。该计时缺口现已由 [M3-B](m3b_camera_prediction.md) 在 2026-09-23 补充独立源时间和成本重放，仍不是设备端到端证明。M3-C 再补新人员/会话与人工标签，设备、驱动和通信保持协作方边界。

## 实时慢帧断流修订（2026-10-20）

**问题。** 原规则把相邻有效帧间隔超过固定 100ms 当作断流，立即清空手势去抖与释放状态。实时摄像头以 `source.read` 返回时刻为源时间，暗光或本机负载让帧率降到约 13fps 时，正常抖动就会越过 100ms：已确认的 open/fist/pinch 被打回 `unknown`，`preferred_mapping` 可能在 grasp/pinch 间切换。现场操作员容易把这种闪烁误判为相机或检测问题。

**规则。** 断流阈值 = min(`mapping_frame_gap_ceiling_ms`, max(`mapping_max_frame_gap_ms`, `mapping_frame_gap_period_factor` × 最近 `mapping_frame_gap_period_window` 个源间隔的中位数))；`ai_m3a.yaml` 为 min(1000ms, max(100ms, 2.5×中位周期))，窗口 9。

- 实现为 `capture.timeline.SourceCadence`，属于 `HandPipeline` 实例。每个到达的源帧（含失效帧）记录一次正间隔，当前帧只用它之前的间隔；触发重置的长间隔也进入窗口，单次卡顿不会抬高中位数。尚无间隔时用 100ms 下限，所以低于 10fps 的流在启动时仍重启一次（多一帧 unknown）。
- 坏帧（`input_valid=false`）、时间倒序/重复、图像尺寸变化与原来一样当帧立即重置。`reset()` 只清映射状态；源节奏属于该输入流，换流应新建实例。`fork_mapping()` 复制源节奏，分数时刻查询不推进它。
- `input_diagnostics` 新增 `reset_reason`（`invalid_input`、`frame_gap`、`timestamp_not_increasing`、`image_size_changed`，未重置为 null）和 `frame_gap_limit_ms`（本帧采用的阈值，旧几何为 null），schema 与运行期合同同步检查二者一致性。

**取舍。**

- 固定加大阈值不能同时满足“30fps 下 100ms 卡顿要断流”与“9.5fps 的正常帧不断流”，相对阈值可以同时满足。上限 1s 保证再慢的流超过 1s 也一定断流。
- 手势去抖按帧计数（确认 2 帧），本身随源周期伸缩；释放权重只有 0.01 幅度滞回、没有时间常数，重置后一帧内即回到目标 ±0.01 内。闪烁和映射切换来自手势去抖被清空，所以二者仍共用一个连续性判定，不为释放另设阈值。
- 倍数取 2.5：两段现场会话中超过 100ms 的间隔最多是同期 9 帧中位周期的 1.73 倍；同时近 9 个间隔中位数不超过 40ms（如 ≥25fps 恒定帧率）时 2.5 倍恒小于 100ms，下限主导，与固定阈值逐帧相同。

**版本与重放兼容。** 映射版本升级为 `svh9-camera-v4-cadence-gap`，契约参数新增上述三项（`ai_m3a.yaml` 显式写出）。`validate_sequence` 按同一规则复算缓存段号。M3-B 已缓存的 V1–V7 观测（两次运行目录，各 7×300 帧，均为容器 PTS）用改动前后代码重放：逐帧映射输出 2100 条、三个场景被调度帧的映射输出 5563 条、共同参考 2100 条、algorithm_only 五种候选 31500 行，除版本标签与实测耗时字段外全部一致。这些录像并非严格恒定帧率（间隔中位 32.0ms、P95 46.6ms、最大 96ms），v4 阈值在 100–116.6ms 间变化，但没有相邻有效帧间隔超过 100ms，所以 M3-B 结论不变。低于 25fps 或可变帧率录像的媒体 PTS 重放只会少掉仅由计时造成的重置，坏帧、倒序、换尺寸行为不变。`sample_pose` 的 100ms 插值上限未改，参考与速度窗口仍不跨越超过 100ms 的间隔，属于 M3-B 评分口径，不在本次修改范围。历史报告原样保留；需要固定 100ms 对照时，把 `mapping_frame_gap_ceiling_ms` 设为与下限相同（版本标签仍为 v4）。

**实测。** 以下回放按 1280×720 用 `process_detections` 重放 `ai_m3a.yaml`，“被改帧”和“最大通道偏差”相对“从不因计时重置”的参照计算：

| 回放 | 源帧/有效帧 | 间隔中位/P95/最大 ms | 规则 | 计时重置 | 手势被改帧 | 映射切换帧 | 最大通道偏差 |
|---|---|---|---|---:|---:|---:|---:|
| 08-28 21:01 会话（开 UDP→Unity） | 808/712 | 77.8/94.6/127.9 | 固定 100ms（v3） | 25（100.6–115.1ms） | 20（open 12、fist 5、pinch 3→unknown） | 3 | 0.519392 |
| | | | 随周期（v4） | 0 | 0 | 0 | 0 |
| 08-28 20:51 会话 | 931/668 | 44.8/78.6/642.1 | 固定 100ms（v3） | 3（107.5–108.7ms） | 3（fist→unknown） | 0 | 0 |
| | | | 随周期（v4） | 0 | 0 | 0 | 0 |

两段会话的失效帧（96/263 帧，均为 `no_right_hand`/`out_of_bounds`）在改动前后逐帧一致，并全部以 `invalid_input` 重置；21:01 会话中 v4 阈值为 100–319.8ms（中位 195.0ms）。合成 9.5fps 张手 30 帧：v3 全部 `unknown`（29 次重置），v4 仅启动时重置 1 次，其后 28 帧为 open。回放时间取 08-28 版本检测后生成的 `timestamp`（当时尚无 `source_timing`），关键点来自当时的 MediaPipe 输出；这只说明计时重置的影响，不是新相机实测或手势准确率。

**验证。** 新增 `tests/test_m3a_frame_gap.py` 29 项：30/13/9.5/7fps 慢而连续不重置、13fps 下 ≥200ms 真实断流与 1s 上限、坏帧立即重置且不清源节奏、倒序/换尺寸原因、副本与查询、诊断契约、配置校验、缓存段号及 M3-B 重放与提取一致；其中 28 项在改动前代码上失败。本机 `handai-intent-prediction` 环境全套 pytest **311 passed，0 skipped**，Ruff 通过。

复现（会话日志在原 checkout 的 ignored `outputs/`）：

```bat
conda activate handai-intent-prediction
cd /d D:\VR\HandAi\single-hand-teleop-baseline
python -X utf8 scripts\replay_frame_gap.py --session outputs\session_20260828_210132.jsonl --width 1280 --height 720
```

现场 A1/A3 若仍看到手势闪烁，先看逐帧 `input_diagnostics.reset_reason`：`frame_gap` 表示间隔确实超过了 `frame_gap_limit_ms`（相机或进程卡顿），`invalid_input` 表示无手、越界等输入失效（看 `reason`）；两类分别排查，不必先改阈值。
