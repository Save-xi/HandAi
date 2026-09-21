# AI 对接入口

## 输入

`src/pipeline.py` 的 `HandPipeline` 负责一次单手输入流。每个流使用一个实例，内部保留手势去抖状态。

- `process_frame(bgr_frame, frame_index=..., timestamp=..., fps=...)`：处理图像，构造实例时注入检测器。
- `process_detections(detections, frame_index=..., timestamp=..., fps=...)`：直接处理设备或其他模型给出的关键点，无需初始化 MediaPipe。

`src/perception/base.py` 定义 `HandDetection` 与 `HandDetector`。替换模型只需要实现 `detect(frame) -> list[HandDetection]` 和 `close()`；检测器资源由创建它的调用方释放。

| 字段 | 约定 |
|---|---|
| `landmarks_2d` | 21×2 图像归一化坐标，顺序为 wrist、thumb、index、middle、ring、little |
| `landmarks_xyz` | 21×3 同尺度相对坐标；当前 MediaPipe 的 z 不是米/毫米真值 |
| `handedness` | 已校正镜像含义的真实 Right/Left 标签 |
| `confidence` | [0,1] 的置信度；不同模型置信度含义在适配器内说明 |
| `timestamp` | 秒；实时同机可用 Unix 时间，离线视频使用连续媒体时间 |
| `frame_index` | 本次流中的递增帧号 |

设备成员负责 RGB-D 采集、内参和坐标转换；单位为米的 3D 点须先转换成上述约定，不能直接冒充 MediaPipe 的相对坐标。

## 输出

返回普通 Python 字典，协议定义在 `schemas/frame_payload.schema.json`。主要结果是 `landmarks_2d`、`landmarks_3d`、`gesture_stable`、`finger_curl`、`control_representation` 和 `svh_preview`。

`svh_preview.target_positions` 为归一化 9 通道代理表示，保留它是为了兼容现有预测数据与下游接入。实体关节映射和驱动不属于这个返回值的语义。`target_ticks_preview` 仅是旧预览参考字段。

下游收到 `control_ready=false` 或 `svh_preview.valid=false` 时，该帧没有有效的可消费目标。字段会保留稳定形状；下游自行处理设备相关策略。

最小调用见 [examples/use_ai_api.py](../examples/use_ai_api.py)。JSON/JSONL 输出由 `JsonExporter` 提供；可选 UDP 使用 `configs/unity_udp_preview.yaml`。协作方可直接调用 AI API，也可订阅文件/数据报，不需要把自己的 SDK 加进 AI 模块。

## 预测

`PredictionShadow.observe(payload)` 读取已生成的 9 通道序列，返回未来 50/100/150 ms 的 hold/raw/gated 结果。实时 CLI 通过容量为 1 的后台队列推理，结果写入独立 prediction JSONL。

模型配置指定历史长度、采样率和预测时间距。加载时比较可读参数、模型维度与通道顺序；无效输入、断帧和推理异常仍会产生明确诊断。

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

该入口属于 H2O 离线实验，没有替换 M1 的独立原始坐标 API，也不是摄像头实时输出协议。共同查询、映射失败回退和结果见 [M2 文档](m2_representation_comparison.md)。摄像头坐标适配、计算就绪时间和带目标时间的实验流输出在 M3 完成。

## 兼容变化

- 运行会话信息使用 v2，保存实际配置、帧数和路径；新输出不计算日志哈希。
- 新预测诊断不再输出 selection/checkpoint SHA；解析器仍可读取携带这些旧字段的历史日志。
- `protocol_hint` 中的旧硬件地址变为可选兼容字段；新预览只描述通道和单位。
- 旧的 freeze、receipt 和 Unity 验收 CLI 已移除。现有外部 Unity 工程未改动。
