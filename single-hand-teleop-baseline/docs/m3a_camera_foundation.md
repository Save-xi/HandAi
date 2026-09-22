# M3-A：摄像头几何、输入与释放整改

2026-09-22，在负责人批准 M3-A 后实施。新入口为 `configs/ai_m3a.yaml`。完成输入尺度、坏帧处理、一个连续释放候选与 CPU 预测 CI；没有训练新的 H2O 模型，也未进入 M3-B/C。

## 实现与兼容

- `HandDetection` 显式携带原图宽高与 `mediapipe_image_xyz` 坐标声明。原始 XY/XYZ 保留给显示与记录；另以 `g=(x, y*H/W, z)` 计算比例和角度。几何任务为 `camera_mediapipe_geometry_xyz_v1`，单位是图像宽度，非公制三维。
- 流式入口按结构/有限值 → 几何 → 掌尺度/骨段 → 图像边界 → 特征/手势执行。掌长采用 XY 的 wrist→middle MCP；小于等于 1e-6 才尝试 index MCP→little MCP 掌宽，最终尺度至少 1e-4 图像宽。五指各三条相邻骨段的三维长度须大于 `max(1e-6, palm_size*0.001)`，二维重叠但深度不同不会直接被拒绝。
- 无效帧返回 `input_diagnostics.reason`，不输出可消费控制值。结构、数值、尺度错误清空坐标；单纯越界保留已核验的有限原始点。立即重置手势、释放状态；预测旁路因预览失效清空历史。下一有效帧重新确认手势。有效 `unknown` 可保留连续控制量。
- 新释放权重由张开比例、非拇指平均 curl 和捏合距离连续决定，使用 0.01 幅度滞回，不增加时间平均。curl 过渡半宽 0.02，捏合距离过渡宽度 0.10；端点直接到达 0/1。它取消释放函数对稳定 open 的硬门控，未重写全部 grasp/pinch 分支。
- `fork_mapping()` 复制手势候选、确认计数、释放权重、上次时间与图像尺寸；复制不更新状态。新流遇倒序、超过 100ms 的时间间隔或尺寸变化会重启状态。M3-B 仍需实现预测网格与分数时刻查询。
- `ai.yaml`、`svh_9ch_preview.yaml` 保留历史几何/释放含义，M2 的独立标签计算未改。统一坏帧检查应用于两种流式模式。新版本为 `svh9-camera-v3-continuous-release`，参数明文保存；旧模型无法静默加载到新配置，当前新配置默认不启用预测。

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

下一阶段为 M3-B 的保持通道、保持姿态后映射、常速度姿态后映射三种基线，以及源时间、目标时间和全部后处理完成时间。当前主循环时间戳仍在检测后生成；本次开发脚本明确传媒体 PTS，两者不能混为端到端延迟证明。M3-C 再补新人员/会话与人工标签，设备、驱动和通信保持协作方边界。
