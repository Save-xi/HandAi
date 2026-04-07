# 单右手遥操作 Baseline

`single-hand-teleop-baseline/` 是当前仓库里最成熟、最适合直接运行的部分。

如果你是第一次 clone 这个仓库，建议从这个子项目开始。当前支持的 baseline 是一条以单右手视觉链路为核心的运行路径，输入来自 webcam 或本地视频文件，输出是一份已经冻结的逐帧 payload contract。

## 当前阶段

- 当前主目标是单右手视觉 baseline。
- baseline 的标准输入入口是 webcam 或本地视频文件。
- 导出的逐帧 payload contract 已经固定，供下游稳定消费。
- `control_representation` 和 `svh_preview` 是建立在 baseline 之上的可选扩展。
- Unity 集成、真实硬件传输链路、真实 SVH 控制都不是安装或启动 baseline 的前置条件。

Baseline 主链路：

`输入 -> MediaPipe 手部检测 -> 右手筛选 -> 特征提取 -> 手势稳定 -> JSON / JSONL / OpenCV 可视化`

可选扩展链路：

`baseline payload -> control_representation -> svh_preview -> mock 传输`

## 范围说明

当前范围：

- 仅支持单右手主流程
- 运行路径以右手为中心
- 输入来自 webcam 或本地视频
- 基于 MediaPipe 的手部感知
- 基于规则的手势分类与稳定
- 导出带有冻结 contract 的 JSON / JSONL
- 可选的 `control_representation`
- 可选的 SVH preview 适配器与 mock transport

不属于当前 baseline 启动路径的内容：

- 双手 / 多手运行主流程
- Unity runtime 集成
- 真实 TCP / 串口 / RS485 传输
- 真实 SVH 硬件控制
- ROS / 数据库 / Web 前端

这些方向未来都可以继续扩展，但它们应该建立在 baseline 之上，而不是默认混进 baseline 的启动前提里。

## 关键文件

配置：

- [configs/default.yaml](configs/default.yaml)
- [configs/svh_9ch_preview.yaml](configs/svh_9ch_preview.yaml)

示例：

- [examples/sample_output.json](examples/sample_output.json)
- [examples/sample_output_svh_9ch.json](examples/sample_output_svh_9ch.json)
- [examples/sample_session.jsonl](examples/sample_session.jsonl)

Schema 与 payload contract：

- [schemas/frame_payload.schema.json](schemas/frame_payload.schema.json)
- [src/output/frame_payload_contract.py](src/output/frame_payload_contract.py)

文档与测试：

- [docs/downstream_preview_contract.md](docs/downstream_preview_contract.md)
- [docs/svh_real_hardware_calibration_table.md](docs/svh_real_hardware_calibration_table.md)
- [CONTRIBUTING.md](CONTRIBUTING.md)
- [tests/](tests/)
- [tests/test_cli_smoke.py](tests/test_cli_smoke.py)
- [tests/test_json_schema.py](tests/test_json_schema.py)

仓库级工作流：

- [../.github/workflows/single-hand-teleop-baseline-ci.yml](../.github/workflows/single-hand-teleop-baseline-ci.yml)

## 环境

推荐 Python 版本：

- Python `3.10`

你既可以在子项目目录里安装依赖，也可以在仓库根目录安装依赖。

### 方式 A：在 `single-hand-teleop-baseline/` 目录下运行

Conda：

```bash
cd single-hand-teleop-baseline
conda env create -f environment.yml
conda activate single-right-hand-baseline
```

Pip：

```bash
cd single-hand-teleop-baseline
python -m pip install -r requirements.txt
```

### 方式 B：在仓库根目录运行

Conda：

```bash
conda env create -f single-hand-teleop-baseline/environment.yml
conda activate single-right-hand-baseline
```

Pip：

```bash
python -m pip install -r single-hand-teleop-baseline/requirements.txt
```

## 最小命令

### 当前目录是 `single-hand-teleop-baseline/`

查看 CLI 帮助：

```bash
python src/main.py --help
```

运行默认 baseline：

```bash
python src/main.py --config configs/default.yaml
```

运行最小 smoke test：

```bash
pytest -q tests/test_cli_smoke.py
```

### 当前目录是仓库根目录

查看 CLI 帮助：

```bash
python single-hand-teleop-baseline/src/main.py --help
```

运行默认 baseline：

```bash
python single-hand-teleop-baseline/src/main.py --config single-hand-teleop-baseline/configs/default.yaml
```

运行最小 smoke test：

```bash
pytest -q single-hand-teleop-baseline/tests/test_cli_smoke.py
```

## 常用运行参数

- `--camera-index 1`
- `--video-file path/to/demo.mp4`
- `--input-mirrored`
- `--no-gui`
- `--headless`
- `--enable-control`
- `--preview-svh`
- `--print-json`
- `--save-jsonl`
- `--max-frames 300`

在 `single-hand-teleop-baseline/` 目录下的示例命令：

```bash
python src/main.py --config configs/default.yaml --print-json
python src/main.py --config configs/default.yaml --headless --video-file path/to/demo.mp4 --max-frames 300
python src/main.py --config configs/default.yaml --enable-control --print-json
python src/main.py --config configs/svh_9ch_preview.yaml --print-json
```

## 运行模式

| 模式 | 进入方式 | 启用内容 | 额外环境要求 |
|---|---|---|---|
| Baseline | `configs/default.yaml` | 检测、右手筛选、特征、手势、可视化、JSON / JSONL | 实时模式需要摄像头；也可以配合 `--video-file` 读取视频 |
| Baseline 无界面 | `--no-gui` 或 `--headless` | 同一条 baseline 链路，但不打开 OpenCV 窗口 | 不需要 GUI |
| control 扩展 | `enable_control_extension: true` 或 `--enable-control` | baseline + `control_representation` | 不需要硬件 |
| SVH 预览扩展 | `svh_enable_preview: true` 或 `--preview-svh` | baseline + `control_representation` + `svh_preview` + mock 传输 | 不需要真实 SVH |

默认行为：

- [configs/default.yaml](configs/default.yaml) 默认运行 baseline-only 模式。
- 默认启动目标是单右手 webcam / video baseline。
- Control 和 SVH preview 默认关闭。
- 扩展失败时会退化为无效占位对象，而不是拖垮 baseline 主循环。
- 不支持的非 mock SVH transport 会打 warning，并保持在 preview-only 模式。

## 帧载荷契约

导出的逐帧 payload 使用一份固定的 canonical schema。

Canonical 顶层字段：

- `timestamp`
- `frame_index`
- `detected`
- `handedness`
- `confidence`
- `control_ready`
- `gesture_raw`
- `gesture_stable`
- `pinch_distance_norm`
- `hand_open_ratio`
- `finger_curl`
- `landmarks_2d`
- `landmarks_3d`
- `control_representation`
- `svh_preview`
- `fps`
- `latency_ms`

已弃用别名：

- `gesture` -> `gesture_stable`
- `svh` -> `svh_preview`

参考文件：

- [schemas/frame_payload.schema.json](schemas/frame_payload.schema.json)
- [src/output/frame_payload_contract.py](src/output/frame_payload_contract.py)

示例 payload 文件：

- [examples/sample_output.json](examples/sample_output.json) 用于默认 baseline 模式
- [examples/sample_output_svh_9ch.json](examples/sample_output_svh_9ch.json) 用于 `svh_9ch` preview 模式

开启 `--save-jsonl` 后，运行期 JSONL 日志会写入 `outputs/`。

## 这条 Baseline 不是什么

- Unity runtime 不是启动、测试或验证 baseline 的必需条件。
- 真实 SVH 硬件控制不是 baseline 启动前提。
- 双手或更广义的多手行为不属于当前支持的 baseline 主路径。
- preview 类扩展不能被当成生产可用的机器人控制或游戏引擎集成证明。

## 扩展说明

`control_representation`：

- 可选扩展层
- 与硬件无关的中间表示
- 不要求 baseline 启动时必须启用

`svh_preview`：

- 可选的 preview-only 扩展
- 适合 JSON / JSONL 记录和后续集成规划
- 不是可直接用于真实硬件的安全命令链路

关于下游字段语义，请见 [docs/downstream_preview_contract.md](docs/downstream_preview_contract.md)。

## 未来扩展方向

可能的后续工作包括：

- 更广泛的双手或多手感知实验
- Unity 或其他下游 runtime 适配器
- 超出当前 mock preview 路径的真实传输和硬件控制层

这些方向与默认 baseline 有意保持解耦，这样单右手视觉链路仍然能保持易安装、易运行、易验证。

## 验证

在 `single-hand-teleop-baseline/` 目录下的最小验证：

```bash
python src/main.py --help
pytest -q tests/test_cli_smoke.py
```

这个子项目常用的更完整验证：

```bash
python -m compileall -q src
pytest -q
```

如果本地也安装了 CI 才需要的工具，还可以运行：

```bash
python -m ruff check src tests
```
