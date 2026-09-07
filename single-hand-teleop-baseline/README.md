# HandAi：AI 开发入口

本项目负责单右手的姿态估计、手势理解、时序预测及训练评测。项目书作为后续 AI 研究路线，当前实现从普通摄像头或视频开始。

```text
图像 / 设备关键点 → HandPipeline → 21 点、手势、连续手部表示
                                  └→ 9 通道预览 → JSON / JSONL / 可选 UDP
                                        └→ 时序预测 → 独立预测结果
```

日常开发看这三个入口：

- [AI 路线图](docs/ai_roadmap.md)：项目书要求、当前实现、下一步 AI 工作。
- [对接入口](docs/ai_interfaces.md)：替换检测器、接设备关键点、消费 AI 结果。
- [训练与评测](experiments/intent_prediction/README.md)：数据、训练、模型导出和视频评测。

## 运行

使用已有 Conda 环境 `handai-intent-prediction`：

```bat
conda activate handai-intent-prediction
cd /d D:\VR\HandAi\single-hand-teleop-baseline
python -X utf8 src\main.py --config configs\ai.yaml --camera-index 0 --save-jsonl
```

启用现有预测模型：

```bat
conda activate handai-intent-prediction
cd /d D:\VR\HandAi\single-hand-teleop-baseline
python -X utf8 src\main.py --config configs\ai.yaml --camera-index 0 --prediction-shadow --save-jsonl
```

日志应显示 `label=residual_motion4` 和实际 `device`，本机可用 CUDA。`offline_gate_passed=False` 是既有模型的评测状态，不是加载错误。预测仍是研究旁路。

无摄像头自检：

```bat
conda activate handai-intent-prediction
cd /d D:\VR\HandAi\single-hand-teleop-baseline
python -X utf8 scripts\run_prediction_shadow_smoke.py --config configs\ai.yaml
```

实时 CLI 也可用 `--video-file 视频路径 --headless` 处理本地视频；算法效果评测使用下述媒体时间轴命令。

## 配置和代码

| 入口 | 用途 |
|---|---|
| [configs/default.yaml](configs/default.yaml) | 采集、感知、手势和映射的通用参数 |
| [configs/ai.yaml](configs/ai.yaml) | 日常 AI 配置，继承通用参数 |
| [models/residual_motion4.json](models/residual_motion4.json) | 权重路径、采样率、预测时间距、映射参数及门控 |
| [src/pipeline.py](src/pipeline.py) | 不依赖设备和网络的单帧 AI 处理 |
| [src/perception/base.py](src/perception/base.py) | 姿态模型/外部 21 点的公共数据结构 |
| [src/prediction](src/prediction) | 模型加载、连续历史、预测和后台 worker |
| [experiments/freihand_eval](experiments/freihand_eval) | 带真值的二维姿态评测 |
| [experiments/intent_prediction](experiments/intent_prediction) | H2O 数据、预测训练、回放与摄像头域评测 |

配置通过 `extends` 复用基础 YAML。替换预测模型使用 `--prediction-model 模型配置.json`；模型配置内的相对 checkpoint 路径按子项目根目录解析。

运行输出在 `outputs/`：最新 AI 帧、可选逐帧 JSONL、独立预测 JSONL，以及记录实际配置和帧数量的会话信息。两份 JSONL 使用同一个 `run_id`，按帧号和时间戳配对。

## 本次整理

移除了源码/配置/模型/视频 SHA 核验、AST 指纹、盲测冻结器和一次性 campaign 收据；移除了 Unity 源码快照、硬件协议草稿、mock 发送及相应测试。模型输入维度、通道顺序、采样率、有效值域和断帧处理仍保留。

Unity、5G、RS485、SVH/AUBO 控制、设备驱动由协作方实现。Python 保留 JSON/JSONL 和本机 UDP 入口，需要现有 Unity 预览时使用 `configs/unity_udp_preview.yaml`。

已有 checkpoint、数据集、机器报告和外部 Unity 工程原样保留。旧验收/冻结文档放在 [历史文档](docs/archive/README.md)，日常运行以本 README 为准。

## 检查

```bat
conda activate handai-intent-prediction
cd /d D:\VR\HandAi\single-hand-teleop-baseline
python -X utf8 -m pytest -q
python -X utf8 -m ruff check src tests scripts experiments
```

重构的实际验证见 [重构记录](docs/ai_refactor_20260905.md)。
