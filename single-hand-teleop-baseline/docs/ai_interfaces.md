# AI 对接入口

## 输入

`src/pipeline.py` 的 `HandPipeline` 负责一次单手输入流。每个流使用一个实例，内部保留手势去抖、释放权重和连续性状态。新摄像头开发使用 `configs/ai_m3a.yaml`。

- `process_frame(bgr_frame, frame_index=..., timestamp=..., fps=...)`：处理图像，构造实例时注入检测器。
- `process_detections(detections, frame_index=..., timestamp=..., fps=...)`：直接处理设备或其他模型给出的关键点，无需初始化 MediaPipe。

`src/perception/base.py` 定义 `HandDetection` 与 `HandDetector`。替换模型只需要实现 `detect(frame) -> list[HandDetection]` 和 `close()`；检测器资源由创建它的调用方释放。

| 字段 | 约定 |
|---|---|
| `landmarks_2d` | 21×2 图像归一化坐标，顺序为 wrist、thumb、index、middle、ring、little |
| `landmarks_xyz` | MediaPipe 原始 21×3 x/y/z；XY 分别按宽高归一化，z 是相对深度；保持原值输出 |
| `image_width` / `image_height` | 原图正整数宽高；M3-A 外部关键点入口必须提供，图像入口可从实际帧取得并核对 |
| `coordinate_space` | M3-A 必须明确为 `mediapipe_image_xyz`；缺失或其他坐标域产生失效帧 |
| `handedness` | 已校正镜像含义的真实 Right/Left 标签 |
| `confidence` | [0,1] 的置信度；不同模型置信度含义在适配器内说明 |
| `timestamp` | API 显式传入时为调用方定义的秒数；缺省取检测调用前的单调时钟。主 CLI 的顶层字段保留 Unix 兼容含义，源时间另见 `source_timing` |
| `frame_index` | 本次流中的递增帧号 |

设备成员负责 RGB-D 采集、内参和坐标转换；单位为米的 3D 点不能直接冒充 MediaPipe 的相对坐标。

[M3-A](m3a_camera_foundation.md) 已实现 `camera_mediapipe_geometry_xyz_v1`：`g=(x, y*H/W, z)`，以图像宽度为单位，x 向右、y 向下、z 越小表示越靠近相机。这只是单目近似相对几何，不能当作公制 3D 或 H2O 输入。原始点用于显示和边界，几何点用于比例和角度。处理顺序为结构/有限值、几何转换、掌部/骨段、图像边界、特征与手势。短输入、缺 XYZ、非有限值和退化姿态会返回原因，坏帧当帧不可控，后续有效帧可恢复。

例如：`HandDetection(xy, xyz, "Right", 0.99, image_width=1920, image_height=1080, coordinate_space="mediapipe_image_xyz")`。旧四参数调用仍适用于 `ai.yaml` 兼容入口；它不自动推断新的坐标域。纯 `extract_hand_features` 保留历史二维退化接口，不能替代严格的流式入口。

## 输出

返回普通 Python 字典，协议定义在 `schemas/frame_payload.schema.json`。主要结果是 `landmarks_2d`、`landmarks_3d`、`gesture_stable`、`finger_curl`、`control_representation` 和 `svh_preview`。

`svh_preview.target_positions` 为归一化 9 通道代理表示，保留它是为了兼容现有预测数据与下游接入。实体关节映射和驱动不属于这个返回值的语义。`target_ticks_preview` 仅是旧预览参考字段。

下游收到 `control_ready=false` 或 `svh_preview.valid=false` 时，该帧没有有效的可消费目标。字段会保留稳定形状；下游自行处理设备相关策略。

新增可选 `input_diagnostics` 包含 `task_id`、图像宽高、`input_valid`、`reason`、`geometry_landmarks`、`mapping_version`、`release_weight` 和 `state_reset`。原始 `landmarks_2d/landmarks_3d` 的含义不变。结构/数值/尺度失效时不输出坏坐标；仅越界时保留已核验的有限坐标。`release_weight` 是新连续候选的状态，旧映射为 null。输入有效不等于已分类；有效 `unknown` 可输出连续表示，失效输入则立即清空状态。

`reset()` 用于显式换流。新配置在时间倒序、间隔超过 100ms 或图像尺寸改变时重启映射状态。`fork_mapping()` 复制手势候选、计数、释放权重、上次时间和图像尺寸，副本不携带检测器；复制不推进状态。M3-B 的 `query_detections()` 在副本上查询，既不多确认一次手势，也不更新释放权重。现有预测旁路遇到无效预览会清空历史；外部消费者也必须处理每次断流信号。

主 CLI 新增 `source_timing`：视频使用 `media_pts_ms` 和记录来源的 PTS/FPS 回退；实时摄像头使用 `monotonic_ms`，起点为 `source.read` 返回，不能称为曝光时间。字段为 `source_time_ms`、`timebase`、`timestamp_source`、`read_return_monotonic_ms`、`nominal_fps`。这些时间送入流水线用于状态推进；主 CLI 的顶层 `timestamp` 在输出时保留读回时刻的 Unix 秒数，兼容旧影子日志，M3 不使用它作为媒体锚点。检测完成时刻单独记录，不再从 `timestamp` 推断。

最小调用见 [examples/use_ai_api.py](../examples/use_ai_api.py)。JSON/JSONL 输出由 `JsonExporter` 提供；可选 UDP 使用 `configs/unity_udp_preview.yaml`。协作方可直接调用 AI API，也可订阅文件/数据报，不需要把自己的 SDK 加进 AI 模块。

## 预测

`PredictionShadow.observe(payload)` 读取已生成的 9 通道序列，返回未来 50/100/150 ms 的 hold/raw/gated 结果。实时 CLI 通过容量为 1 的后台队列推理，结果写入独立 prediction JSONL。

模型配置指定历史长度、采样率和预测时间距。加载时比较可读参数、模型维度与通道顺序；无效输入、断帧和推理异常仍会产生明确诊断。

M3-A 新映射版本为 `svh9-camera-v3-continuous-release`，旧 `svh9-label-v2-open-release` 模型加载时会因版本不符拒绝接入新配置。当前主 CLI 默认关闭旧神经预测；M3-B 通过独立实验入口提供同域简单基线，不复用 H2O checkpoint。

## 摄像头简单预测（M3-B）

[camera_baselines.py](../src/prediction/camera_baselines.py) 的 `predict_camera(history, anchor, current, method=..., window=..., fps=30)` 只接收锚点及过去观测、完整映射状态和当前基础输出。方法为 `hold_channels`、`hold_pose`、`linear_pose`；返回 50/100/150ms 的 9 通道、手势、失效原因和本机拟合/全部映射后处理耗时。`reference_camera` 单独读取真实未来，不能传入预测器。

[run_camera_m3b.py](../experiments/intent_prediction/scripts/run_camera_m3b.py) 保存独立 `camera-simple-forecast-v1` JSONL。`source_time_ms`、`target_time_ms=source_time_ms+horizon_ms`、`ready_time_ms` 全在 `media_pts_schedule_ms` 上；后者是测量成本经两阶段有界队列调度得到的就绪时间。完整 `timing` 包含到达、检测开始/结束、基础映射完成、预测入队/开始/结束和全部未来映射完成时间，另存队列等待及注入的扰动。`algorithm_only` 的就绪时间是零成本虚拟源时刻。

`candidate_positions` 是实际被调度候选的数学输出，未调度时为 null；`prediction_valid` 只说明候选可用。`prediction_on_time` 才表示候选未过期且没有在完成前跨过历史重置。`output_positions` 为最终候选或已及时就绪的当前通道回退；`valid/on_time` 表示这次时距有可及时提供的输出。`used_fallback`、`fallback_reason`、`deadline_miss` 明确区分预热、坏帧、源丢弃、队列替换、历史重置和逾期。没有及时可用当前观测时不产生回退。

回退的当前通道在基础映射完成时即可提供；它不是在较晚的失败时刻重新发送旧值。后续源流失效仍须由消费者立即处理。该协议是逐源锚点的研究候选接口，未接入设备驱动或实时神经旁路；完整分母和范围见 [M3-B 文档](m3b_camera_prediction.md)。

9 通道新模型可通过 `scripts/export_prediction_model.py` 从对应第二轮报告导出配置；运行时使用 `--prediction-model`。9 通道训练选中神经网络后也会在结果目录生成 `model.json`。

## 原生 21 点预测（M1）

[keypoint_model_loader.py](../src/prediction/keypoint_model_loader.py) 提供 `load_keypoint_prediction_model(model_path, expected_task_id="h2o_camera3d_right21", device="auto")`。它消费 M1 生成的 `handai-keypoint-model-v1` 配置，checkpoint 相对该配置文件所在目录解析。加载只需模型配置和权重，不需要原始数据集或历史报告。

返回对象的 `predict(points, timestamps_s, input_task=task, frame_ids=..., mask=..., frame_valid=..., segment_ids=...)` 使用原始 `[T,21,3]` 相机米坐标。`input_task` 是调用方声明的输入定义（本任务采用 [M0 定义](../experiments/intent_prediction/configs/keypoint_m0.json)）；任务名、轴向、单位、点序、历史长度及时间语义须兼容。H2O 模型会拒绝声明为 `camera_mediapipe_image_xyz` 的任务，不直接消费现有 `HandPipeline.landmarks_xyz`。

历史必须能形成完整的 30Hz 网格；缺点、断段、尺度退化或历史不足时抛出带原因的 `ValueError`。本阶段不在此函数内隐藏回退或预测失败。未来输出字段：

| 字段 | 含义 |
|---|---|
| `task_id` | 输入输出所属的坐标任务 |
| `source_timestamp_s` | 最后观测所在的媒体时间 |
| `forecast_timestamps_s`、`landmarks_3d_m` | 未来 5 个规则时刻及 `[5,21,3]` 米坐标 |
| `evaluation_horizons_ms`、`evaluation_timestamps_s` | 50/100/150ms 及对应绝对媒体时间 |
| `evaluation_landmarks_3d_m` | `[3,21,3]` 查询姿态，按预测轨迹插值 |
| `anchor_scale_m` | 本次锚点用于归一化和反变换的尺度 |
| `valid` | 输入历史及模型输出可用；不是精度、按时送达或下游控制许可 |

独立 NPZ 推理入口为 [predict_keypoint_sequence.py](../experiments/intent_prediction/scripts/predict_keypoint_sequence.py)，已用真实 H2O 数据验证。环境、命令与实际产物见 [M1 文档](m1_keypoint_prediction.md)。

## 表示实验接口（M2）

[representation_data.py](../experiments/intent_prediction/intent_prediction/representation_data.py) 的 `RepresentationBatch.training_view("A"/"B"/"C")` 提供实验输入、监督掩码及残差参考，供已有 `predict_neural_checkpoint` 批量重载。A/B 输出共同七个时刻的 9 通道；C 输出五个规则姿态，再由 `map_predictions` 用当前腕点、尺度和手势状态副本转换成共同七个查询。该函数不读取未来真实姿态、标签掩码或真实手势，返回通道与映射有效位。

该入口属于 H2O 离线实验，没有替换 M1 的独立原始坐标 API，也不是摄像头实时输出协议。共同查询、映射失败回退和结果见 [M2 文档](m2_representation_comparison.md)。摄像头新几何和带时间候选输出由 M3-A/B 的独立入口提供，不能混用两种坐标任务。

## 兼容变化

- 运行会话信息使用 v2，保存实际配置、帧数和路径；新输出不计算日志哈希。
- 新预测诊断不再输出 selection/checkpoint SHA；解析器仍可读取携带这些旧字段的历史日志。
- `protocol_hint` 中的旧硬件地址变为可选兼容字段；新预览只描述通道和单位。
- 旧的 freeze、receipt 和 Unity 验收 CLI 已移除。现有外部 Unity 工程未改动。
