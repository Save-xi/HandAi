# Single Right Hand Teleop Baseline

## 1. 项目目标

这个目录实现的是一个单右手视觉控制 baseline。当前主流程是：

`摄像头输入 -> MediaPipe 单手检测 -> Right hand 过滤 -> 特征提取 -> 手势稳定 -> control_representation -> svh preview -> JSON / JSONL / OpenCV 可视化`

当前阶段的目标不是直接控制真实机械手，而是先把“右手视觉理解”整理成一套稳定、可测、可扩展的中间表示，方便以后继续接更真实的 SVH 接口。

## 2. 当前范围

当前已经支持：

- 摄像头输入
- MediaPipe 单右手检测
- OpenCV 关键点、骨架和状态面板
- 稳定特征提取
  - `pinch_distance_norm`
  - `hand_open_ratio`
  - `finger_curl`
- `quality gate`
- `gesture_raw / gesture`
- `control_representation`
- `svh adapter / command preview`
- `svh protocol skeleton`
- `svh mock transport`
- 统一 JSON 输出
- 可选 JSONL 日志
- 一圈基础测试

当前明确不做：

- 双手
- Unity 集成
- 真实 SVH 联调
- 真实 TCP/IP 发送到串口服务器
- 真实 RS485 桥接
- ROS
- 数据库
- Web 前端
- 时序模型训练
- 大规模工程重构

## 3. 环境安装

推荐使用 `environment.yml`：

```bash
cd single-hand-teleop-baseline
conda env create -f environment.yml
conda activate single-right-hand-baseline
```

也可以使用 `requirements.txt`：

```bash
cd single-hand-teleop-baseline
conda create -n single-right-hand-baseline python=3.10 -y
conda activate single-right-hand-baseline
pip install -r requirements.txt
```

## 4. 运行方式

基础运行：

```bash
cd single-hand-teleop-baseline
python src/main.py --config configs/default.yaml
```

常用参数：

- `--camera-index 1`
  说明：覆盖配置文件中的摄像头索引
- `--print-json`
  说明：按配置节流打印每帧 JSON console preview
- `--save-jsonl`
  说明：为当前 session 额外写入逐帧 `.jsonl` 日志

常见组合：

```bash
python src/main.py --config configs/default.yaml --print-json
python src/main.py --config configs/default.yaml --print-json --save-jsonl
python src/main.py --config configs/svh_9ch_preview.yaml --print-json
```

## 5. 目录结构

- `src/capture/`
  摄像头输入
- `src/perception/`
  MediaPipe 检测、右手过滤、关键点质量判断
- `src/features/`
  手部几何工具和特征提取
- `src/gesture/`
  规则手势识别和去抖稳定器
- `src/control/`
  从原始特征整理连续控制表示
- `src/svh/`
  SVH preview adapter、command object、layout、protocol skeleton、mock transport
- `src/visualize/`
  OpenCV 叠加绘制和状态面板
- `src/output/`
  JSON、JSONL 和 console preview
- `src/utils/`
  配置、日志、计时等通用工具
- `configs/`
  默认配置和 9 通道 preview 配置
- `examples/`
  样例输出
- `tests/`
  单元测试

## 6. 每帧输出字段

当前顶层 payload 字段包括：

- `timestamp`
- `detected`
- `handedness`
- `confidence`
- `gesture_raw`
- `gesture`
- `pinch_distance_norm`
- `hand_open_ratio`
- `finger_curl`
- `landmarks_2d`
- `control_representation`
- `svh`
- `fps`
- `latency_ms`

### JSON 示例

```json
{
  "timestamp": 1774965000.123,
  "detected": true,
  "handedness": "Right",
  "confidence": 0.972,
  "gesture_raw": "pinch",
  "gesture": "pinch",
  "pinch_distance_norm": 0.145,
  "hand_open_ratio": 0.793,
  "finger_curl": {
    "thumb": 0.036,
    "index": 0.316,
    "middle": 0.017,
    "ring": 0.013,
    "little": 0.018
  },
  "landmarks_2d": [
    [0.185, 0.968],
    [0.223, 0.919],
    [0.246, 0.858]
  ],
  "control_representation": {
    "valid": true,
    "features_valid": true,
    "command_ready": true,
    "source": "features",
    "gesture_context": "pinch",
    "preferred_mapping": "pinch",
    "grasp_close": 0.17,
    "thumb_index_proximity": 0.843,
    "effective_pinch_strength": 0.843,
    "pinch_strength": 0.843,
    "support_flex": 0.016,
    "finger_flex": {
      "thumb": 0.036,
      "index": 0.316,
      "middle": 0.017,
      "ring": 0.013,
      "little": 0.018
    }
  },
  "svh": {
    "enabled": true,
    "mode": "preview",
    "valid": true,
    "command_source": "control_representation",
    "target_channels": [0, 1, 2, 3, 4],
    "target_positions": [0.722, 0.711, 0.169, 0.169, 0.169],
    "target_ticks_preview": [],
    "protocol_hint": {
      "set_control_state_addr": "0x09",
      "set_all_channels_addr": "0x03",
      "transport": "mock",
      "channel_layout": "compact5",
      "channel_order": "thumb,index,middle,ring,little",
      "position_units": "normalized_preview",
      "target_tick_units": "none"
    }
  },
  "fps": 27.77,
  "latency_ms": 20.04
}
```

完整样例见：

- [sample_output.json](/D:/VR/HandAi/single-hand-teleop-baseline/examples/sample_output.json)
- [sample_output_svh_9ch.json](/D:/VR/HandAi/single-hand-teleop-baseline/examples/sample_output_svh_9ch.json)

## 7. `control_representation` 说明

`control_representation` 是一层硬件无关的中间表示。它不直接声称自己就是最终机械手命令，而是把当前视觉帧整理成更适合控制映射的连续量。

主要字段：

- `features_valid`
  当前帧连续特征是否可算
- `command_ready`
  当前帧是否已经有足够稳定的上下文来生成动作命令
- `grasp_close`
  整体抓握闭合程度
- `thumb_index_proximity`
  拇指和食指的接近程度
- `effective_pinch_strength`
  经过手势上下文约束后，真正用于 pinch 控制的强度

设计上刻意区分：

- “特征可算”
- “命令可发”

这样在过渡帧、低质量帧或语义尚未稳定时，不会过早发出 preview 命令。

## 8. `svh` Preview Adapter 说明

当前 `svh` 字段只是 preview / skeleton：

- 它把 `control_representation` 映射成一个可打印、可测试、可写入 JSON 的命令预览对象
- 它不直接连接真实 SVH
- 它不包含真实 TCP/IP + RS485 + SVH 联调

当前 `target_channels / target_positions / target_ticks_preview` 都只是 preview abstraction，不是已经校准好的真实电机命令。

### 当前已经做的事情

- 根据 `open / fist / pinch` 选择不同映射分支
- 输出 preview 级别的目标通道和目标值
- 支持 `compact5` 与 `svh_9ch` 两种 preview 布局
- 在 `svh_9ch` 模式下额外输出 `target_ticks_preview`
- 提供 `mock transport`
- 提供 `svh_protocol.py` 中的 skeleton packet builder

### 当前明确没做的事情

- 没有实现真实 TCP 客户端
- 没有实现串口服务器联调
- 没有实现真实 RS485 通信
- 没有声称已经完成官方协议全部细节

## 9. SVH Preview Layout

### `compact5`

默认配置使用 `compact5`。它是一个更轻量的 preview 向量：

- `thumb`
- `index`
- `middle`
- `ring`
- `little`

优点是简单、直观，适合前期调试。

### `svh_9ch`

如果想让 preview 更贴近学长 Unity/C# 工程里的 SVH 驱动顺序，可以使用：

```bash
python src/main.py --config configs/svh_9ch_preview.yaml --print-json
```

此时 preview 通道顺序为：

`thumb_flexion, thumb_opposition, index_finger_distal, index_finger_proximal, middle_finger_distal, middle_finger_proximal, ring_finger, pinky, finger_spread`

这个模式仍然是 preview，不代表已经完成真实硬件标定，但会更方便以后继续对齐：

- 通道顺序
- ticks 量纲
- 全通道 packet 组织方式

相关文件：

- [svh_9ch_preview.yaml](/D:/VR/HandAi/single-hand-teleop-baseline/configs/svh_9ch_preview.yaml)
- [sample_output_svh_9ch.json](/D:/VR/HandAi/single-hand-teleop-baseline/examples/sample_output_svh_9ch.json)

## 10. SVH Protocol Skeleton

`src/svh/svh_protocol.py` 当前是一个更贴近论文和 Unity/C# 驱动信息的协议骨架，但仍然只是 skeleton。

当前默认参考值：

- `SYNC1 = 0x4C`
- `SYNC2 = 0xAA`
- `SetControlState = 0x09`
- `SetControlCommand AllChannels = 0x03`
- command payload `40` bytes
- command frame `48` bytes
- response payload `64` bytes
- response frame `72` bytes

当前 builder 还会给出：

- `CHECK1(sum)` preview
- `CHECK2(xor)` preview

这些只是为了让骨架更像真实协议，不等于已经完成真实设备可用的协议实现。

未来仍需继续校准：

- 真实通道与地址字段的编码关系
- 长度字段
- 零填充细节
- `CHECK1 / CHECK2`
- 小端字节序下的真实打包
- TCP/IP -> 串口服务器 -> RS485 的实际链路

## 11. Gesture Fallback

当前 `svh adapter` 支持可配置的 gesture fallback：

- 配置项：`svh_enable_gesture_fallback`
- 默认值：`false`

默认关闭的原因是：

- 对 demo 来说，fallback 可以让低质量帧看起来更“连续”
- 但对更接近真实硬件的阶段，这样做不够保守

所以当前默认策略是：

- continuous features 无效时
- 不自动合成 fallback 命令

只有显式打开 `svh_enable_gesture_fallback: true` 时，才允许在连续特征失效但 gesture 标签仍可用时生成 `gesture_fallback` preview。

## 12. 配置项

与 SVH preview 相关的关键配置包括：

- `svh_enable_preview`
- `svh_enable_gesture_fallback`
- `svh_preview_layout`
- `svh_preview_channel_count`
- `svh_preview_mode`
- `svh_transport`
- `svh_protocol_sync_bytes`
- `svh_protocol_use_little_endian`
- `svh_position_open_value`
- `svh_position_closed_value`
- `svh_thumb_grasp_scale`
- `svh_thumb_opposition_scale`
- `svh_pinch_support_scale`
- `svh_open_spread_scale`
- `svh_grasp_spread_scale`
- `svh_pinch_spread_scale`
- `svh_9ch_open_ticks`
- `svh_9ch_closed_ticks`

默认配置见：

- [default.yaml](/D:/VR/HandAi/single-hand-teleop-baseline/configs/default.yaml)
- [svh_9ch_preview.yaml](/D:/VR/HandAi/single-hand-teleop-baseline/configs/svh_9ch_preview.yaml)

## 13. 手动验证

运行：

```bash
python src/main.py --config configs/default.yaml --print-json
```

检查：

1. 张开手
- `gesture = open`
- `grasp_close` 接近低值
- `svh.valid = true`

2. 握拳
- `gesture = fist`
- `grasp_close` 明显升高
- `effective_pinch_strength` 应接近 `0`

3. 捏合
- `gesture = pinch`
- `effective_pinch_strength` 升高
- `svh.target_positions` 前两路高于 support fingers

4. 无手 / 低质量帧
- `features_valid = false`
- `command_ready = false`
- `svh.valid = false`
- `target_positions = []`

如果切到 `svh_9ch_preview.yaml`，还可以再看：

- `target_ticks_preview`
- `protocol_hint.channel_order`

## 14. 测试

运行：

```bash
cd single-hand-teleop-baseline
pytest -q
```

当前测试覆盖包括：

- 配置解析
- JSON schema
- 特征与控制表示
- gesture-conditioned SVH preview mapping
- fallback 开关行为
- 9 通道布局
- protocol skeleton 常量和 sync bytes
- preview checksum / payload 提示
- mock transport

## 15. 后续方向

如果后面继续接真实 SVH，建议按这个顺序推进：

1. 校准真实 9 通道映射
2. 校准 preview ticks 与真实 encoder ticks 的关系
3. 校准真实 packet packing、长度和校验
4. 新增 `svh_transport_tcp.py`
5. 再去接串口服务器和 RS485

当前这个目录的目标仍然只是：

**先把单右手视觉 baseline 和 SVH preview 接口骨架做稳。**
