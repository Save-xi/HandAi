# 单右手视觉遥操作 Baseline

这个子项目的目标很简单：

**用普通摄像头看右手，把右手动作变成一份稳定的逐帧数据，供 Unity 预览或后续机械手控制链路使用。**

先不用急着理解 SVH、协议、ticks、Unity 关节这些词。你可以把当前项目先看成三层：

```text
摄像头/视频
  -> MediaPipe 检测右手
  -> 提取手部特征和手势
  -> 输出 JSON / JSONL / 可选 Unity UDP 预览
```

当前最成熟、最适合直接运行的代码就在这个目录里：`single-hand-teleop-baseline/`。

## 现在已经做到什么

- 可以从摄像头或本地视频读取画面。
- 可以用 MediaPipe 检测手部 21 个关键点。
- 会只选择“右手”作为当前控制对象。
- 会识别几个基础手势：`open`、`fist`、`pinch`、`unknown`。
- 会提取连续控制特征，比如捏合距离、手掌张开程度、每根手指弯曲程度。
- 可以导出冻结格式的逐帧 payload，方便后续程序稳定读取。
- 可以可选生成 `control_representation`，也就是与硬件无关的控制中间层。
- 可以可选生成 `svh_preview`，也就是给 SVH / Unity 用的预览控制量。
- 可以通过 UDP 把 9 通道预览数据发给本机 Unity 场景。
- 有测试覆盖当前主要行为。

还没做到的事情也很重要：

- 还没有接真实 SVH 硬件。
- 还没有完成真实 TCP / 串口 / RS485 传输。
- `svh_preview` 只是预览和联调用，不是可以直接下发给实体机械手的安全命令。
- 当前主线只做单右手，不做双手或多手控制。

## 推荐理解顺序

如果你是现在回来看这个项目，建议按这个顺序读：

1. 先看本 README，把整体链路和运行方式搞清楚。
2. 再跑默认 baseline，确认摄像头、MediaPipe、JSON 输出都正常。
3. 再开 `--enable-control`，看连续控制量。
4. 再开 `configs/svh_9ch_preview.yaml`，看 SVH 9 通道预览数据。
5. 最后才看 Unity UDP 或真机校准文档。

不用一上来读协议文档。那是后期接硬件时才需要啃的骨头。

## 环境安装

推荐 Python 版本是 `3.10`。

如果你用 Conda：

```bash
cd single-hand-teleop-baseline
conda env create -f environment.yml
conda activate single-right-hand-baseline
```

如果你直接用 pip：

```bash
cd single-hand-teleop-baseline
python -m pip install -r requirements.txt
```

主要依赖包括：

- `mediapipe`
- `opencv-contrib-python`
- `numpy`
- `Pillow`
- `PyYAML`
- `pytest`

## 最快跑起来

进入子项目目录：

```bash
cd single-hand-teleop-baseline
（我自己本地是cd /d D:\VR\HandAi\single-hand-teleop-baseline）
```

查看命令行帮助：

```bash
python src/main.py --help
```

运行默认摄像头 baseline：

```bash
python src/main.py --config configs/default.yaml
```

默认会打开 OpenCV 窗口。窗口中按 `q` 退出。

如果你不想开窗口，只想处理和输出数据：

```bash
python src/main.py --config configs/default.yaml --headless
```

如果要从视频文件读取：

```bash
python src/main.py --config configs/default.yaml --video-file path/to/demo.mp4 --headless
```

如果要把每隔几帧的 JSON 打印到控制台：

```bash
python src/main.py --config configs/default.yaml --print-json
```

如果要保存逐帧 JSONL 日志：

```bash
python src/main.py --config configs/default.yaml --save-jsonl
```

输出会写到：

- `examples/sample_output.json`：最近一帧的 JSON
- `outputs/session_*.jsonl`：开启 `--save-jsonl` 后的逐帧日志

## 常用模式

### 1. 默认 baseline

```bash
python src/main.py --config configs/default.yaml
```

适合检查：

- 摄像头能不能打开
- 是否检测到右手
- `open / fist / pinch` 是否大致正确
- JSON 输出是否正常

默认配置有意保持保守：

- 不要求 Unity
- 不要求真实 SVH
- 不启用控制扩展
- 不启用 SVH 预览扩展

### 2. 无界面处理

```bash
python src/main.py --config configs/default.yaml --headless
```

适合：

- 跑视频文件
- 只记录 JSON / JSONL
- 在没有图形界面的环境里做 smoke test

### 3. 打开控制表示

```bash
python src/main.py --config configs/default.yaml --enable-control --print-json
```

会额外输出 `control_representation`。

你可以重点看这些字段：

- `command_ready`：这一帧是否适合下游消费
- `preferred_mapping`：当前更像 `grasp` 还是 `pinch`
- `grasp_close`：抓握闭合程度，范围 `[0, 1]`
- `effective_pinch_strength`：有效捏合强度，范围 `[0, 1]`
- `finger_flex`：五根手指各自的弯曲程度

### 4. 打开 SVH 9 通道预览

```bash
python src/main.py --config configs/svh_9ch_preview.yaml --print-json
```

会额外输出 `svh_preview`，并使用更接近 SVH / Unity 参考实现的 9 通道顺序：

1. `thumb_flexion`
2. `thumb_opposition`
3. `index_finger_distal`
4. `index_finger_proximal`
5. `middle_finger_distal`
6. `middle_finger_proximal`
7. `ring_finger`
8. `pinky`
9. `finger_spread`

重点看：

- `svh_preview.valid`
- `svh_preview.target_channels`
- `svh_preview.target_positions`
- `svh_preview.target_ticks_preview`

注意：`target_ticks_preview` 只是 preview ticks，不是已经验证过的真机编码器命令。

### 5. Unity UDP 预览

如果你是隔一段时间回来看这个联动，或者准备在新对话里续接上下文，先看：

- [docs/unity_handoff.md](docs/unity_handoff.md)

```bash
python src/main.py --config configs/unity_udp_preview.yaml
```

这份配置会启用：

- `control_representation`
- `svh_preview`
- `svh_9ch` 布局
- UDP 发送到 `127.0.0.1:18080`

Unity 侧需要有对应脚本监听端口 `18080`。这条链路适合看虚拟手是否能跟随右手动作，不代表真实硬件已经可控。

## 输入镜像和左右手

MediaPipe 的左右手标签会受镜像视角影响。

默认情况下，项目按“普通摄像头画面”处理。如果你的输入已经是自拍镜像视角，运行时加：

```bash
python src/main.py --config configs/default.yaml --input-mirrored
```

这个选项会影响右手筛选。如果你发现自己伸右手却一直检测不到右手，优先试一下它。

## 输出数据长什么样

每一帧最终都会被整理成一份固定格式的 payload。顶层主要字段是：

- `timestamp`
- `frame_index`
- `detected`
- `handedness`
- `confidence`
- `gesture_raw`
- `gesture_stable`
- `pinch_distance_norm`
- `hand_open_ratio`
- `finger_curl`
- `control_ready`
- `control_representation`
- `svh_preview`
- `fps`
- `latency_ms`

最常看的几个字段：

- `detected`：这一帧有没有检测到目标右手
- `gesture_stable`：去抖后的稳定手势
- `control_ready`：下游是否应该消费这一帧
- `finger_curl`：五指弯曲程度
- `control_representation`：硬件无关的连续控制层
- `svh_preview`：给 SVH / Unity 看的预览层

完整 contract 在这里：

- [schemas/frame_payload.schema.json](schemas/frame_payload.schema.json)
- [src/output/frame_payload_contract.py](src/output/frame_payload_contract.py)

示例数据在这里：

- [examples/sample_output.json](examples/sample_output.json)
- [examples/sample_output_svh_9ch.json](examples/sample_output_svh_9ch.json)
- [examples/sample_session.jsonl](examples/sample_session.jsonl)

## 代码地图

核心入口：

- [src/main.py](src/main.py)：主循环、命令行参数、运行模式、扩展链路

输入：

- [src/capture/webcam.py](src/capture/webcam.py)：摄像头输入
- [src/capture/video_file.py](src/capture/video_file.py)：视频文件输入

感知：

- [src/perception/mediapipe_hand.py](src/perception/mediapipe_hand.py)：MediaPipe 手部检测
- [src/perception/hand_filter.py](src/perception/hand_filter.py)：选择右手
- [src/perception/landmark_quality.py](src/perception/landmark_quality.py)：判断当前关键点质量是否适合控制

特征与手势：

- [src/features/hand_features.py](src/features/hand_features.py)：提取 pinch、张开比例、五指 curl
- [src/features/geometry_utils.py](src/features/geometry_utils.py)：几何工具函数
- [src/gesture/rule_based_gesture.py](src/gesture/rule_based_gesture.py)：规则手势分类和去抖

控制和 SVH 预览：

- [src/control/control_representation.py](src/control/control_representation.py)：把视觉特征变成控制中间表示
- [src/svh/svh_adapter.py](src/svh/svh_adapter.py)：把控制中间表示变成 SVH preview
- [src/svh/svh_layout.py](src/svh/svh_layout.py)：SVH 9 通道顺序和 preview ticks 参考值
- [src/svh/svh_protocol.py](src/svh/svh_protocol.py)：协议 preview skeleton
- [src/svh/svh_transport_mock.py](src/svh/svh_transport_mock.py)：mock 传输层

输出和可视化：

- [src/output/json_exporter.py](src/output/json_exporter.py)：JSON、JSONL、UDP 输出
- [src/output/frame_payload_contract.py](src/output/frame_payload_contract.py)：payload 标准化和校验
- [src/visualize/status_panel.py](src/visualize/status_panel.py)：OpenCV 状态面板

配置：

- [configs/default.yaml](configs/default.yaml)：默认 baseline
- [configs/svh_9ch_preview.yaml](configs/svh_9ch_preview.yaml)：SVH 9 通道预览
- [configs/unity_udp_preview.yaml](configs/unity_udp_preview.yaml)：Unity UDP 联动预览

## 测试

最小检查：

```bash
python src/main.py --help
pytest -q tests/test_cli_smoke.py
```

完整测试：

```bash
python -m compileall -q src
pytest -q
```

如果本地装了 ruff：

```bash
python -m ruff check src tests
```

当前测试主要覆盖：

- CLI 能否正常启动
- 配置路径解析
- 手势规则
- 特征提取
- 控制表示
- payload contract
- JSON / JSONL / UDP 输出
- SVH preview 映射
- 扩展失败时是否安全降级

## 常见问题

### 摄像头打不开

试试换相机索引：

```bash
python src/main.py --config configs/default.yaml --camera-index 1
```

### 伸右手却识别不到右手

试试镜像输入参数：

```bash
python src/main.py --config configs/default.yaml --input-mirrored
```

### 没检测到手，但是程序还在跑

这是正常的。payload 会输出：

- `detected=false`
- `gesture_stable="unknown"`
- `control_ready=false`
- `svh_preview.valid=false`

这比直接崩掉更适合实时系统。

### Unity 没反应

先确认：

- Python 用的是 `configs/unity_udp_preview.yaml`
- Unity 监听端口是 `18080`
- Unity 侧开启了 UDP preview
- 本机防火墙没有拦截本地 UDP
- Python 侧确实检测到了右手，并且 `svh_preview.valid=true`

### 动作不像真人手

先别急着怀疑检测坏了。SVH / Unity 虚拟手不是 21 个关键点一一映射到真人手，它更像：

```text
21 个视觉关键点 -> 连续手部特征 -> 9 个 SVH 预览通道 -> Unity 关节展开
```

中指、无名指、小指的联动和真人不同，可能是机械结构和自由度限制，也可能是映射参数还需要调。

## 后续真正要做的事

如果继续推进，这几件事最关键：

- 清理运行时调试输出。
- 用真实视频和摄像头多录几段 JSONL，观察 `gesture_stable` 和连续控制量是否稳定。
- 调 Unity 侧 9 通道到 20 关节的展开效果。
- 逐项确认 SVH 真机协议、通道方向、零位、限位、homing 和安全策略。
- 在真机接入前实现真正的 transport、ACK、timeout、watchdog、急停和速率限制。

更多下游和真机相关细节放在：

- [docs/README.md](docs/README.md)
- [docs/ai_scope_and_proposal_alignment.md](docs/ai_scope_and_proposal_alignment.md)
- [docs/unity_handoff.md](docs/unity_handoff.md)
- [docs/downstream_preview_contract.md](docs/downstream_preview_contract.md)
- [docs/svh_real_hardware_calibration_table.md](docs/svh_real_hardware_calibration_table.md)

## 一句话总结

当前项目已经完成了“单右手视觉理解 -> 连续控制表示 -> SVH / Unity 预览输出”的 baseline。

它适合继续做仿真联动和后续硬件接入准备，但还不能被称为真实 SVH 硬件控制系统。
