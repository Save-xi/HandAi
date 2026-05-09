# 单右手遥操作 Baseline

`single-hand-teleop-baseline/` 是当前仓库里最成熟、最适合直接运行的部分。  
如果你是第一次 clone 这个仓库，建议从这个子项目开始。当前 baseline 是一条以单右手视觉链路为核心的运行路径，输入来自 webcam 或本地视频文件，输出是一份已经冻结的逐帧 payload contract。

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
- 可选的 `svh_preview`

不属于当前 baseline 启动路径的内容：

- 双手 / 多手主流程
- Unity runtime 不是 baseline 启动前提
- 真实 TCP / 串口 / RS485 传输
- 真实 SVH 硬件控制
- ROS / 数据库 / Web 前端

这些方向后续都可以继续扩展，但应建立在 baseline 之上，而不是默认混进 baseline 的启动前提里。

## 关键文件

配置：

- [configs/default.yaml](configs/default.yaml)
- [configs/svh_9ch_preview.yaml](configs/svh_9ch_preview.yaml)
- [configs/unity_udp_preview.yaml](configs/unity_udp_preview.yaml)

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

## Unity 仿真联动

这一节只说明“本地 Unity 虚拟手联动预览”怎么开。  
它属于可选扩展，不是 baseline 启动前提。

### 预期链路

联动链路是：

`21 个视觉关键点 -> 连续手部特征 -> 9 个 SVH 预览通道 -> Unity 20 个关节展开`

注意：

- Unity 虚拟手的运动不等于 21 个关键点一一直接映射。
- 机械手本身存在自由度和关节耦合限制，所以不可能完全等同于真人手。
- 如果中指、无名指、小指的表现和真人有差异，优先先判断是否是机械结构限制，而不是先怀疑 Python 检测失效。

### 前置条件

- 已安装 Unity Editor，并能打开本地 Unity 工程 `RoboticArm`
- 已安装本项目 Python 依赖
- 本机可以同时运行 Unity 和 Python

### Unity 侧操作

1. 用 Unity Hub 打开 Unity 工程 `RoboticArm`
2. 打开场景：`Assets/Scenes/Hnad.unity`
3. 在层级面板中选中挂载了 `RobotControlScript` 的对象
4. 在 `Inspector` 里确认以下参数：

- `Enable Baseline Udp Preview = true`
- `Baseline Udp Listen Port = 18080`
- `Apply Baseline Preview To Hardware = false`
- `Log Baseline Preview Packets = false`
- `Enable Legacy Gesture Snapping = false`

说明：

- `Log Baseline Preview Packets` 默认建议关闭，否则 Unity Console 会每帧刷日志，影响实时性观察。
- `Enable Legacy Gesture Snapping` 默认必须关闭。它是旧的离散手势模板吸附逻辑，会把连续动作压回固定姿态，干扰当前 UDP 连续预览链。

5. 点击 Unity 的 `Play`

### Python 侧操作

在 `single-hand-teleop-baseline/` 目录下运行：

```bash
python src/main.py --config configs/unity_udp_preview.yaml
```

这份配置会默认启用：

- `unity_udp_enabled: true`
- `unity_udp_host: 127.0.0.1`
- `unity_udp_port: 18080`
- `enable_control_extension: true`
- `svh_enable_preview: true`
- `svh_preview_layout: svh_9ch`

### 联动成功时你会看到什么

- OpenCV 预览窗口正常运行
- Unity 场景里的虚拟手开始跟着当前右手动作更新
- `open / fist / pinch` 会比较明显
- 中间连续动作也会跟着变化，不再只剩几种离散姿态

### 如果没动，优先检查这些

1. Unity 场景是否真的是 `Assets/Scenes/Hnad.unity`
2. `RobotControlScript` 上的 `Enable Baseline Udp Preview` 是否开启
3. Unity 监听端口是否是 `18080`
4. Python 侧是否真的用了 `configs/unity_udp_preview.yaml`
5. `Enable Legacy Gesture Snapping` 是否被错误打开
6. `Log Baseline Preview Packets` 是否开着导致 Console 过度刷屏，影响观察

### 如果动作“不像真人手”

先区分两类原因：

1. 代码映射问题

- 完全不动
- 只有拇指和食指动
- 明显只会固定几种姿态
- 一开 `Enable Legacy Gesture Snapping` 就吸到模板手势

2. 机械手自由度限制

- 中指、无名指、小指跟随存在耦合
- 某些手指不可能像真人那样完全独立
- 半握、分指、捏合的视觉效果与真人不同，但整体趋势一致

如果已经确认是第二类，那更像是机械手设计约束，而不是 baseline 检测主链路故障。

## 运行模式

| 模式 | 进入方式 | 启用内容 | 额外环境要求 |
|---|---|---|---|
| Baseline | `configs/default.yaml` | 检测、右手筛选、特征、手势、可视化、JSON / JSONL | 实时模式需要摄像头；也可以配合 `--video-file` 读取视频 |
| Baseline 无界面 | `--no-gui` 或 `--headless` | 同一条 baseline 链路，但不打开 OpenCV 窗口 | 不需要 GUI |
| control 扩展 | `enable_control_extension: true` 或 `--enable-control` | baseline + `control_representation` | 不需要硬件 |
| SVH 预览扩展 | `svh_enable_preview: true` 或 `--preview-svh` | baseline + `control_representation` + `svh_preview` + mock 传输 | 不需要真实 SVH |
| Unity UDP 预览 | `configs/unity_udp_preview.yaml` | baseline + `control_representation` + `svh_preview` + Unity 本地 UDP 联动 | 需要本地 Unity 工程与场景 |

默认行为：

- [configs/default.yaml](configs/default.yaml) 默认运行 baseline-only 模式
- 默认启动目标是单右手 webcam / video baseline
- `control_representation` 和 `svh_preview` 默认关闭
- 扩展失败时会退化为无效占位对象，而不是拖垮 baseline 主循环
- 不支持的非 mock SVH transport 会打 warning，并保持在 preview-only 模式

## Payload 契约

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

- Unity runtime 不是启动、测试或验证 baseline 的必需条件
- 真实 SVH 硬件控制不是 baseline 启动前提
- 双手或更广义的多手行为不属于当前支持的 baseline 主路径
- preview 类扩展不能被当成生产可用的机器人控制或游戏引擎集成证明

## 扩展说明

`control_representation`：

- 可选扩展层
- 与硬件无关的中间表示
- 不要求 baseline 启动时必须启用

`svh_preview`：

- 可选的 preview-only 扩展
- 适合 JSON / JSONL 记录和后续集成联调
- 不等于真实硬件安全控制链路

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
