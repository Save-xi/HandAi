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
- 可以默认关闭地加载与当前映射契约一致的 v2 residual GRU；UDP 后只做非阻塞后台诊断，结果写入独立 prediction 日志。
- 有测试覆盖当前主要行为。

还没做到的事情也很重要：

- 还没有接真实 SVH 硬件。
- 还没有完成真实 TCP / 串口 / RS485 传输。
- `svh_preview` 只是预览和联调用，不是可以直接下发给实体机械手的安全命令。
- 意图预测目前只处于 shadow mode；v2 离线 gate 未全部通过，没有证明延迟补偿有效，也不驱动 Unity。
- 冻结的 36 场景延迟/抖动/丢包回放已完成：H2O primary RMSE 只改善 1.95%，retention gate 4/6，控制参考继续使用 hold-last。
- 二审已补上 invalid 历史断点和映射公式指纹；真实日志端到端 prediction 覆盖率为 61.90%，H2O v2 仍明确属于 pose-only 代理标签。
- 当前主线只做单右手，不做双手或多手控制。

网络扰动的完整协议、指标和决策见
[延迟、抖动、丢包冻结回放报告](docs/intent_prediction_delay_injection.md)。
二审代码修补、覆盖率口径和 H2O 投影/去抖审计见
[Phase 1.5 二审修补与算法复核记录](docs/phase15_second_review_remediation.md)。
真实摄像头视频域的媒体时间轴、开发集/盲测集隔离和三分支决策见
[真实摄像头域开发评测协议](docs/camera_domain_protocol_v1.md)。

## 推荐理解顺序

如果你是现在回来看这个项目，建议按这个顺序读：

1. 先看本 README，把整体链路和运行方式搞清楚。
2. 再跑默认 baseline，确认摄像头、MediaPipe、JSON 输出都正常。
3. 再开 `--enable-control`，看连续控制量。
4. 再开 `configs/svh_9ch_preview.yaml`，看 SVH 9 通道预览数据。
5. 没有摄像头时可跑无摄像头的预测影子 smoke，并先读 Phase 1.5 风险加固记录和延迟注入报告。
6. 最后才看 Unity UDP 或真机校准文档。

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

如果一个长期使用的 Conda 环境在 `pip check` 中出现本项目没有声明的 Jupyter、音频或
网页包缺失，不要为了让那份污染清单归零而给本项目继续堆无关依赖。更稳妥的做法是从
`environment.yml` 新建干净环境；需要同时运行 PyTorch 影子预测时，使用
`experiments/intent_prediction/environment.yml` 对应的 `handai-intent-prediction` 环境。

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

运行产物会写到（`examples/` 只保留冻结示例，不再被实时运行覆盖）：

- `outputs/latest_frame.json`：最近一帧的 JSON
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
- `finger_flex`：五根手指用于下游控制的弯曲程度；预览配置可对稳定张手做连续释放校正

`finger_curl` 始终保留视觉侧原始几何测量。`configs/unity_udp_preview.yaml`
和 `configs/svh_9ch_preview.yaml` 默认启用 `control_open_release_*`，用于处理
单目摄像头下拇指、小指已经伸直但 curl 仍偏大的常见视角误差；默认 baseline
配置保持关闭，不改变原始测量语义。

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

### 6. 默认关闭的预测影子模式

没有摄像头时，先在独立 PyTorch 环境运行冻结 checkpoint smoke：

```powershell
conda activate handai-intent-prediction
python -X utf8 scripts\run_prediction_shadow_smoke.py `
  --device auto `
  --output outputs\prediction_shadow_smoke\latest.json
```

将来有本地右手视频时，可在 9 通道配置上显式开启：

```powershell
python -X utf8 src\main.py `
  --config configs\svh_9ch_preview.yaml `
  --video-file D:\path\to\right_hand_video.mp4 `
  --prediction-shadow --headless --save-jsonl
```

预测使用按 timestamp 重采样到 30 Hz 的 30 帧历史，输出未来 `50/100/150 ms` 的
hold-last、raw residual 和 gated residual。当前帧 UDP 发送后只做非阻塞提交；后台结果
进入独立的 `latest_prediction_shadow.json` / `prediction_session_*.jsonl`，不改
`svh_preview`，也不把预测发给 Unity。当前 v2 离线 gate 未全部通过，只允许研究诊断。详见
[docs/intent_prediction_shadow_mode.md](docs/intent_prediction_shadow_mode.md)。

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
- `prediction_diagnostics`（只在独立 prediction 结果文件中出现）

最常看的几个字段：

- `detected`：这一帧有没有检测到目标右手
- `gesture_stable`：去抖后的稳定手势
- `control_ready`：下游是否应该消费这一帧
- `finger_curl`：五指弯曲程度
- `control_representation`：硬件无关的连续控制层
- `svh_preview`：给 SVH / Unity 看的预览层
- `prediction_diagnostics`：后台生成的 hold/raw/gated 研究诊断，不是控制命令

完整 contract 在这里：

- [schemas/frame_payload.schema.json](schemas/frame_payload.schema.json)
- [src/output/frame_payload_contract.py](src/output/frame_payload_contract.py)

示例数据在这里：

- [examples/sample_output.json](examples/sample_output.json)
- [examples/sample_output_svh_9ch.json](examples/sample_output_svh_9ch.json)
- [examples/sample_prediction_diagnostics.json](examples/sample_prediction_diagnostics.json)
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
- [src/svh/mapping_contract.py](src/svh/mapping_contract.py)：H2O 标签、checkpoint 与实时映射的语义指纹

意图预测影子层：

- [src/prediction/shadow_predictor.py](src/prediction/shadow_predictor.py)：映射/SHA 闸门、30 Hz 重采样、门控诊断和安全回退
- [src/prediction/shadow_worker.py](src/prediction/shadow_worker.py)：容量 1 的 latest-only 非阻塞后台推理
- [scripts/run_prediction_shadow_smoke.py](scripts/run_prediction_shadow_smoke.py)：无需摄像头的真实 checkpoint smoke

输出和可视化：

- [src/output/json_exporter.py](src/output/json_exporter.py)：JSON、JSONL、UDP 输出
- [src/output/frame_payload_contract.py](src/output/frame_payload_contract.py)：payload 标准化和校验
- [src/visualize/status_panel.py](src/visualize/status_panel.py)：OpenCV 状态面板

配置：

- [configs/default.yaml](configs/default.yaml)：默认 baseline
- [configs/svh_9ch_preview.yaml](configs/svh_9ch_preview.yaml)：SVH 9 通道预览
- [configs/unity_udp_preview.yaml](configs/unity_udp_preview.yaml)：Unity UDP 联动预览

Unity 最小可恢复快照：

- [integrations/unity_phase15_snapshot/README.md](integrations/unity_phase15_snapshot/README.md)：
  已验收 UDP 接收器、batch 安全脚本、包清单和编辑器版本的 Git 快照
- [integrations/unity_phase15_snapshot/snapshot_manifest.json](integrations/unity_phase15_snapshot/snapshot_manifest.json)：
  7 个快照文件的路径、大小与 SHA-256

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
- 影子历史/断帧/推理失败回退，以及 UDP 前后 payload 隔离
- Unity UDP loopback、invalid/乱序/过期/watchdog 安全策略的源码与 batch 行为回归
- Unity 最小源码快照自身哈希及本机外部工程漂移检查

当前二轮风险清理、v2 重训结果和诚实边界见：

- [docs/phase15_risk_hardening.md](docs/phase15_risk_hardening.md)

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
- 按摄像头域协议录制 V1–V7 开发视频，用媒体 PTS 确定性重放；不要把离线处理 FPS 当作源帧率。
- 用确定性回放注入延迟、抖动和丢包，比较 hold-last/raw/gated，证明或否定延迟补偿价值。
- 调 Unity 侧 9 通道到 20 关节的展开效果。
- 逐项确认 SVH 真机协议、通道方向、零位、限位、homing 和安全策略。
- 在真机接入前实现真正的 transport、ACK、timeout、watchdog、急停和速率限制。

更多下游和真机相关细节放在：

- [docs/README.md](docs/README.md)
- [docs/ai_scope_and_proposal_alignment.md](docs/ai_scope_and_proposal_alignment.md)
- [docs/intent_prediction_shadow_mode.md](docs/intent_prediction_shadow_mode.md)
- [docs/unity_handoff.md](docs/unity_handoff.md)
- [docs/downstream_preview_contract.md](docs/downstream_preview_contract.md)
- [docs/svh_real_hardware_calibration_table.md](docs/svh_real_hardware_calibration_table.md)

## 一句话总结

当前项目已经完成了“单右手视觉理解 -> 连续控制表示 -> SVH / Unity 预览输出”的 baseline。

它适合继续做仿真联动和后续硬件接入准备，但还不能被称为真实 SVH 硬件控制系统。
