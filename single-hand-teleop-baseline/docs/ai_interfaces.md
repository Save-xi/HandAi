# AI 对接入口

## 输入

`src/pipeline.py` 的 `HandPipeline` 负责一次单手输入流。每个流使用一个实例，内部保留手势去抖、释放权重、连续性和可选逐指滤波状态。当前摄像头映射入口为 `configs/ai_camera_v10.yaml`，该入口关闭控制 flex 的时间滤波；`configs/ai_m3a.yaml` 保留 v4 兼容重放。

- `process_frame(bgr_frame, frame_index=..., timestamp=..., fps=...)`：处理图像，构造实例时注入检测器。
- `process_detections(detections, frame_index=..., timestamp=..., fps=...)`：直接处理设备或其他模型给出的关键点，无需初始化 MediaPipe。

`src/perception/base.py` 定义 `HandDetection` 与 `HandDetector`。替换模型只需要实现 `detect(frame) -> list[HandDetection]` 和 `close()`；检测器资源由创建它的调用方释放。

| 字段 | 约定 |
|---|---|
| `landmarks_2d` | 21×2 图像归一化坐标，顺序为 wrist、thumb、index、middle、ring、little |
| `landmarks_xyz` | MediaPipe 原始 21×3 x/y/z；XY 分别按宽高归一化，z 是相对深度；保持原值输出 |
| `image_width` / `image_height` | 原图正整数宽高（Python 或 numpy 整数，bool/浮点不接受；诊断中统一输出 Python int）；M3-A 外部关键点入口必须提供，图像入口可从实际帧取得并核对 |
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

新增可选 `input_diagnostics` 包含 `task_id`、图像宽高、`input_valid`、`reason`、`geometry_landmarks`、`mapping_version`、`release_weight`、`state_reset`、`reset_reason` 和 `frame_gap_limit_ms`。原始 `landmarks_2d/landmarks_3d` 的含义不变。结构/数值/尺度失效时不输出坏坐标；仅越界时保留已核验的有限坐标。`release_weight` 在 v2 为 null、v4/v5 为连续释放状态，关闭全局释放的 v6–v9 为 0。输入有效不等于已分类；有效 `unknown` 可输出连续表示，失效输入则立即清空状态。

`state_reset=true` 时 `reset_reason` 说明原因：`invalid_input`（当前输入失效，细节见 `reason`）、`frame_gap`（与上一有效帧的间隔超过 `frame_gap_limit_ms`）、`timestamp_not_increasing`（时间倒序或重复）、`image_size_changed`；未重置为 null。`frame_gap_limit_ms` 是本帧采用的断流阈值，旧几何为 null。

新配置在时间倒序、真实断流或图像尺寸改变时重启映射状态。断流阈值为 min(`mapping_frame_gap_ceiling_ms`, max(`mapping_max_frame_gap_ms`, `mapping_frame_gap_period_factor` × 最近 `mapping_frame_gap_period_window` 个源间隔的中位数))，`ai_m3a.yaml` 为 min(1000ms, max(100ms, 2.5×中位周期))、窗口 9：中位周期不超过 40ms（如 ≥25fps 恒定帧率）时即 100ms，约 13fps 的实时相机约 195ms。每个源帧（含失效帧）都计入间隔，当前帧只用之前的间隔；尚无间隔时用 100ms 下限，因此低于 10fps 的流启动时仍会重启一次。坏帧不受该阈值影响，当帧立即重置。理由与实测见 [M3-A 断流修订](m3a_camera_foundation.md#实时慢帧断流修订2026-10-20)。

`reset()` 用于明确中断，只清映射状态；源节奏估计属于该输入流，不随 `reset()` 清空，换流应新建 `HandPipeline` 实例。`fork_mapping()` 复制手势候选、计数、释放权重、可选逐指滤波状态、上次时间、图像尺寸和源节奏，副本不携带检测器；复制不推进状态。M3-B 的 `query_detections()` 在副本上查询，既不多确认一次手势，也不更新释放权重或源节奏。v9 查询可计算该时刻的假想滤波输出，不写回活跃流状态。现有预测旁路遇到无效预览会清空历史；外部消费者也必须处理每次断流信号。

主 CLI 新增 `source_timing`：视频使用 `media_pts_ms` 和记录来源的 PTS/FPS 回退；实时摄像头使用 `monotonic_ms`，起点为 `source.read` 返回，不能称为曝光时间。字段为 `source_time_ms`、`timebase`、`timestamp_source`、`read_return_monotonic_ms`、`nominal_fps`。这些时间送入流水线用于状态推进；主 CLI 的顶层 `timestamp` 在输出时保留读回时刻的 Unix 秒数，兼容旧影子日志，M3 不使用它作为媒体锚点。检测完成时刻单独记录，不再从 `timestamp` 推断。

最小调用见 [examples/use_ai_api.py](../examples/use_ai_api.py)。JSON/JSONL 输出由 `JsonExporter` 提供；可选 UDP 使用 `configs/unity_udp_preview.yaml`。协作方可直接调用 AI API，也可订阅文件/数据报，不需要把自己的 SDK 加进 AI 模块。

## 预测

`PredictionShadow.observe(payload)` 读取已生成的 9 通道序列，返回未来 50/100/150 ms 的 hold/raw/gated 结果。实时 CLI 通过容量为 1 的后台队列推理，结果写入独立 prediction JSONL。

模型配置指定历史长度、采样率和预测时间距。加载时比较可读参数、模型维度与通道顺序；无效输入、断帧和推理异常仍会产生明确诊断。

`ai_m3a.yaml` 的摄像头映射版本仍为 `svh9-camera-v4-cadence-gap`（M3-A 交付时为 `svh9-camera-v3-continuous-release`，v4 只改断流阈值）；旧 `svh9-label-v2-open-release` 模型加载时会因版本不符拒绝接入新配置。当前主 CLI 默认关闭旧神经预测；M3-B 通过独立实验入口提供同域简单基线，不复用 H2O checkpoint。

## 2026-10-24 复审后的当前映射 v10

当前入口是 [ai_camera_v10.yaml](../configs/ai_camera_v10.yaml)，版本 `svh9-camera-v10-low-lag-thumb`，完整结果见 [R1/R2、S1/S4 修订记录](mapping_review_revision_20261024.md)。该文件包含全部 112 个有效配置键，不继承下面的五个首轮中间配置。旧模型与缓存按新版本和明文参数拒绝混用。

v10 的 `control_flex_filter_mode=none`：控制层不再添加时间平滑，`finger_flex` 为当帧范围归一化后的值，拇指保留原始 curl；`release_weight=0`。无滤波的 30/13 fps 阶跃首帧响应均为 100%，两段录像九通道相对同配置不滤波均为 0 帧滞后。检测器、队列、网络与机械运动的总延迟没有因此归零。设备侧限速和限位仍由 DS 按设备能力实现。

R2 把拇指指尖参考改为开端 0.98、闭端 0.60，区间从 0.154 放宽到 0.38，指尖项权重从 0.70 降至 0.25。为了降低既有抓握引导的噪声并保留握拳行程，拇指抓握引导另用 `grasp_close` 的 0.10/0.65 作为开闭端点，屈曲混合权重为引导 0.95、自身 flex 0.05。它仅影响抓握端点的 ch0/ch1，不改四指或 ch8，不测直接对掌角，也不使用时序平均。

抓握拇指限制仍为 0.85/0.75；捏合端点及向捏合的连续混合不套抓握上限。`preferred_mapping` 仍只描述上下文，消费者直接读九通道 `target_positions`。不滤波的 ch0/ch1 抖动最高为旧 v4 的 1.212 倍，B 张手基准 ch1 为 0.026，两个主要录像的握拳拇指中位数约为 0.812/0.75 和 0.817/0.75。它们是代理指标，不是实体精度或人工 GT。

## 首轮 v5–v9 历史复现

完整实现、开发数据分母、逐项前后输出及检查见 [映射修复记录](mapping_repairs_20261024.md)。这些问题编号 M1/M2 等与路线图的科研里程碑不同。

| 配置 | 版本 | 对下游有影响的差异 |
|---|---|---|
| `ai_mapping_m1.yaml` | `svh9-camera-v5-per-finger-curl` | 原始 curl 改用逐指指尖-掌心参考；仍是代理量 |
| `ai_mapping_m2.yaml` | `svh9-camera-v6-local-finger-control` | 不再用一个释放权重缩放全部手指 |
| `ai_mapping_m7.yaml` | `svh9-camera-v7-pinch-depth-guard` | 捏合距离取同尺度 2D/3D 距离较大者，只作投影重叠排除 |
| `ai_mapping_m6.yaml` | `svh9-camera-v8-continuous-route` | 抓握/捏合公式按强度 0.20–0.60 连续混合；标签不直接切公式 |
| `ai_mapping_m8.yaml` | `svh9-camera-v9-long-finger-range` | 四指范围归一化，握紧度按四指计算，控制 flex 使用逐指 One Euro 自适应滤波 |

配置按上述顺序累积继承。语义参数（含滤波参数）全部进入映射契约，旧模型/缓存不兼容时重新提取，不覆盖旧工件。所有版本仍输出原顺序的九个 [0,1] 代理通道，未实现直接外展、直接对掌或 MCP/远端分别测量。

v8/v9 的 `preferred_mapping` 是描述性字段。实际通道按 `smoothstep((effective_pinch_strength−0.20)/0.40)` 混合两组端点，消费方应直接读 `target_positions`，不能按 `preferred_mapping` 再硬切一次。v9 抓握端点拇指屈曲/对掌分别限制到 0.85/0.75；捏合端点不受此限制，混合后的值可能随捏合权重超过抓握上限。这不是已确认的实体机械限位。

v9 的原始 `finger_curl` 不经时间滤波；`control_representation.finger_flex` 是范围重定后经过自适应滤波的控制量。滤波参数为最小截止频率 2 Hz、速度系数 0.1、导数截止频率 1 Hz。没有固定窗口平均，仍有跟随滞后：受控动作从首个改变输入帧到 90% 响应在 30 fps 时约 167–200 ms、13 fps 时约 231 ms。启动、无效帧及断流恢复时不使用旧值缓慢追赶，首个有效帧直接初始化滤波；该行为不代替设备侧限速。直接调用控制层时需要显式传入该流的 `AdaptiveFlexState`，常规调用应通过 `HandPipeline` 统一管理。

开发录像的两段主要计数合计由 10 降到 4，但 V5 单段由 2 增至 3，张手转换后也存在滤波残留；详见修复记录。没有人工 GT 或真实捏合召回结论，没有真机流畅度验收。

## 外部 Unity 接口差异（X1–X3，只记录）

2026-10-24 对 `D:\SVH\RoboticArm` 仅作只读核对，本轮没有修改 C#。

- **X1：增益与角度上限。** `RobotControlScript.cs` 的九通道增益为 `[1,1,1,1,1.12,1.12,1.18,1.20,1]`，乘完后再截到 1；虚拟手还按自身角度上限展开。例：拇指屈曲约 0.90 rad，食指/中指远端约 1.30 rad，食指近端约 0.75 rad；旧驱动对应量程约 0.97/1.33/0.80 rad。Python `target_positions` 不包含这些 Unity 增益。直接用于实体适配时会与 Unity 看到的姿态不同，增益归属与限位应由协作方统一确认。
- **X2：旧 payload 的 fallback。** Unity 对缺少 `svh_preview` 的 payload 使用另一套映射，张手时 ch8 约为 1，而当前 Python 抓握代理为 0.25。当前 Python 总会提供 `svh_preview`（失效时为空目标且 `valid=false`），正常接入应按有效性处理，不能通过删除此对象转到旧映射。
- **X3：ticks 仅为参考。** `target_ticks_preview` 的拇指对掌闭合参考为 −150000；旧驱动同处先赋“保守值”−105000，随后又以“全量程”−150000 覆盖。两个数都不能当作这台设备已标定的限位。接实体前按现场 SDK、方向、零位、机械范围重新确认；ch8 实际方向也由 DS 记录。

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
