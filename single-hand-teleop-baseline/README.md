# HandAi：AI 开发入口

本项目负责单右手的姿态估计、手势理解、时序预测及训练评测。项目书作为后续 AI 研究路线，当前实现从普通摄像头或视频开始。

```text
图像 / 设备关键点 → HandPipeline → 21 点、手势、连续手部表示
                                  └→ 9 通道预览 → JSON / JSONL / 可选 UDP
                                        └→ 时序预测 → 独立预测结果
```

日常开发看这三个入口：

- [AI 路线图（M0/M1 与 M2 H2O 对照已完成）](docs/ai_roadmap.md)：实测结论、后续阶段与审核范围。
- [对接入口](docs/ai_interfaces.md)：替换检测器、接设备关键点、消费 AI 结果。
- [训练与评测](experiments/intent_prediction/README.md)：数据、训练、模型导出和视频评测。

阶段规划和成果复核可查 [AI 交接说明](docs/agent_handoff/README.md)，其中已合并项目书审核与本机二审意见，区分历史实测、当前缺口及后续计划。

已完成 [M0：数据定义与核对](docs/m0_data_definition.md)、[M1：原生 21 点预测](docs/m1_keypoint_prediction.md)及 [M2：三种子 A/B/C 对照](docs/m2_representation_comparison.md)。M2 覆盖 H2O 全部 217 段、每段最多抽样 64 个窗口；C 优于 A/B，但未稳定胜过验证集预选的简单基线，三条神经路线均未通过本轮研发继续条件。下一步审核摄像头域的简单基线与计算成本评测。现有摄像头与设备输出继续使用原路径。

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

若 pytest 只在 `%TEMP%\pytest-of-31948` 等历史临时目录或 `.pytest_cache` 出现 `WinError 5` / `PermissionError`，先保持上述 `handai-intent-prediction` 环境，在 PowerShell 中改用一次性专用目录并停用测试缓存：

```powershell
Set-Location -LiteralPath 'D:\VR\HandAi\single-hand-teleop-baseline'
conda activate handai-intent-prediction
$handaiTestTemp = Join-Path (Get-Location) ('work/pytest-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path (Split-Path -Parent $handaiTestTemp) -Force | Out-Null
python -X utf8 -m pytest -q -p no:cacheprovider --basetemp $handaiTestTemp
```

`--basetemp` 会清理指定目标，故只能指向这种新建的测试专用路径。若仍失败，依据新的报错区分依赖、输入和断言问题；不要把临时目录权限错误记作算法失败，也不要为此更改系统权限。
