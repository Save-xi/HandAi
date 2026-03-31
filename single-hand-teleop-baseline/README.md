# Single Right Hand Teleop Baseline

## 1. 项目目标

这个目录实现的是一个**单右手视觉控制 baseline**。当前主流程是：

`摄像头输入 -> MediaPipe 单手检测 -> Right hand 过滤 -> 特征提取 -> 手势稳定 -> control_representation -> svh preview -> JSON / JSONL / OpenCV 可视化`

当前代码目标是把“右手视觉理解”整理成一个稳定、可调、可测试的控制前端表示，方便后续再继续接更真实的机械手接口。

## 2. 当前范围

当前已经支持：

- 摄像头输入
- MediaPipe 单手检测
- 只保留 `Right hand`
- OpenCV 关键点、骨架和状态面板
- 稳定特征提取：
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

当前明确**不做**：

- 双手
- Unity
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
  SVH preview adapter、command object、protocol skeleton、mock transport
- `src/visualize/`
  OpenCV 叠加绘制和状态面板
- `src/output/`
  JSON、JSONL 和 console preview
- `src/utils/`
  配置、日志、计时等通用工具
- `configs/`
  默认配置
- `examples/`
  样例输出
- `tests/`
  单元测试

## 6. 主要输出字段

每帧最终 payload 的顶层字段当前是：

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

下面是一个**字段名与当前真实输出一致**的示例：

```json
{
  "timestamp": 1774965000.123,
  "detected": true,
  "handedness": "Right",
  "confidence": 0.97,
  "gesture_raw": "pinch",
  "gesture": "pinch",
  "pinch_distance_norm": 0.16,
  "hand_open_ratio": 0.82,
  "finger_curl": {
    "thumb": 0.04,
    "index": 0.28,
    "middle": 0.02,
    "ring": 0.02,
    "little": 0.02
  },
  "landmarks_2d": [
    [0.19, 0.95],
    [0.23, 0.90],
    [0.26, 0.84]
  ],
  "control_representation": {
    "valid": true,
    "features_valid": true,
    "command_ready": true,
    "source": "features",
    "gesture_context": "pinch",
    "preferred_mapping": "pinch",
    "grasp_close": 0.15,
    "thumb_index_proximity": 0.79,
    "effective_pinch_strength": 0.79,
    "pinch_strength": 0.79,
    "support_flex": 0.02,
    "finger_flex": {
      "thumb": 0.04,
      "index": 0.28,
      "middle": 0.02,
      "ring": 0.02,
      "little": 0.02
    }
  },
  "svh": {
    "enabled": true,
    "mode": "preview",
    "valid": true,
    "command_source": "control_representation",
    "target_channels": [0, 1, 2, 3, 4],
    "target_positions": [0.68, 0.66, 0.16, 0.16, 0.16],
    "protocol_hint": {
      "set_control_state_addr": "0x09",
      "set_all_channels_addr": "0x03",
      "transport": "mock"
    }
  },
  "fps": 29.4,
  "latency_ms": 20.1
}
```

完整样例请参考：

- [examples/sample_output.json](/D:/VR/HandAi/single-hand-teleop-baseline/examples/sample_output.json)

## 7. `control_representation` 说明

`control_representation` 是一层**硬件无关**的中间表示。它不直接声称自己就是最终机械手命令，而是把当前视觉帧整理成更适合控制映射的连续量。

当前主要字段：

- `features_valid`
  说明：当前帧连续特征是否可算
- `command_ready`
  说明：当前帧是否已经有足够稳定的上下文来生成动作命令
- `grasp_close`
  说明：整体抓握闭合程度
- `thumb_index_proximity`
  说明：拇指和食指的接近程度
- `effective_pinch_strength`
  说明：经过手势上下文约束后，真正用于 pinch 控制的强度

设计上刻意区分了：

- “特征可算”
- “命令可发”

这样在过渡帧、低质量帧或语义尚未稳定时，不会过早发出 preview 命令。

## 8. `svh` Preview Adapter 说明

当前 `svh` 字段只是一个**preview / skeleton**：

- 它把 `control_representation` 映射成一个可打印、可测试、可写入 JSON 的命令预览对象
- 它不直接连接真实 SVH
- 它不包含真实 TCP/IP + RS485 + SVH 联调

当前 `target_channels / target_positions` 只是**preview abstraction**，不是已经校准好的真实电机通道映射。

### 当前已经做的事情

- 根据 `open / fist / pinch` 选择不同映射分支
- 输出 preview 级别的 `target_channels / target_positions`
- 提供 `mock transport`
- 提供 `svh_protocol.py` 中的 skeleton packet builder

### 当前明确没做的事情

- 没有实现真实 TCP 客户端
- 没有实现串口服务器联调
- 没有实现真实 RS485 通信
- 没有声明已经完成官方协议全部细节

## 9. SVH Protocol Skeleton 说明

`src/svh/svh_protocol.py` 当前是一个**更贴近论文信息的协议骨架**，但仍然只是 skeleton。

当前默认参考值：

- `SYNC1 = 0x4C`
- `SYNC2 = 0xAA`
- `SetControlState = 0x09`
- `SetControlCommand AllChannels = 0x03`

这些默认值用于让 packet builder 更接近论文描述，但这**不等于**已经完成真实设备可用的协议实现。

仍需未来继续校准的内容包括：

- 真实通道与地址字段的编码关系
- 长度字段
- 零填充细节
- `CHECK1 / CHECK2`
- 小端字节序下的真实打包方式
- TCP/IP 到串口服务器，再到 RS485 的真实链路行为

## 10. Gesture Fallback 策略

当前 `svh adapter` 支持一个可配置的 gesture fallback：

- 配置项：`svh_enable_gesture_fallback`
- 默认值：`false`

默认关闭的原因是：

- 对 demo 来说，fallback 可以让低质量帧看起来更“连续”
- 但对更接近真实硬件的阶段，这样做不够保守
- 所以当前默认策略是：当 continuous features 无效时，不自动合成 fallback 命令

如果显式打开 `svh_enable_gesture_fallback: true`，才允许在连续特征失效但手势标签仍可用时，生成 `gesture_fallback` preview。

## 11. 配置项

当前和 SVH preview 相关的关键配置包括：

- `svh_enable_preview`
- `svh_enable_gesture_fallback`
- `svh_preview_channel_count`
- `svh_preview_mode`
- `svh_transport`
- `svh_protocol_sync_bytes`
- `svh_protocol_use_little_endian`
- `svh_position_open_value`
- `svh_position_closed_value`
- `svh_thumb_grasp_scale`
- `svh_pinch_support_scale`

完整默认配置请看：

- [configs/default.yaml](/D:/VR/HandAi/single-hand-teleop-baseline/configs/default.yaml)

## 12. 如何验证 demo 正常

1. 运行：

```bash
python src/main.py --config configs/default.yaml --print-json
```

2. 张开手：

- `gesture` 稳定为 `open`
- `grasp_close` 接近低值
- `svh.valid = true`

3. 握拳：

- `gesture` 稳定为 `fist`
- `grasp_close` 明显升高
- `effective_pinch_strength` 仍应接近 `0`

4. 捏合：

- `gesture` 稳定为 `pinch`
- `effective_pinch_strength` 升高
- `svh.target_positions` 前两个通道高于后面几个

5. 无手 / 低质量帧：

- `features_valid = false`
- `command_ready = false`
- `svh.valid = false`
- `target_positions = []`

## 13. 测试

运行：

```bash
cd single-hand-teleop-baseline
pytest -q
```

当前测试覆盖包括：

- JSON schema
- 特征与控制表示
- gesture-conditioned SVH preview mapping
- fallback 开关行为
- protocol skeleton 常量和 sync bytes
- mock transport

## 14. 后续方向

后续如果继续接真实 SVH，建议按下面的顺序推进：

1. 校准真实通道映射
2. 校准真实 packet packing、长度和校验
3. 新增 `svh_transport_tcp.py`
4. 再去连串口服务器和 RS485

当前这个目录的目标仍然只是：

**先把单右手视觉 baseline 和 SVH preview 接口骨架做稳。**
