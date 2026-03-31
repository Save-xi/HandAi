# Single Right Hand Teleop Baseline

## 1) 项目目标
本项目当前阶段目标：构建“**单右手遥操作 AI baseline**”。

数据流：`摄像头输入 -> 单右手关键点检测 -> 特征提取 -> 基础手势识别 -> 轻量可视化 -> 统一 JSON 输出`。

## 2) 当前阶段范围
本阶段仅做：
- 单右手（Right hand）关键点检测与过滤
- 几何规则特征与规则手势（open/fist/pinch/unknown）
- OpenCV 实时可视化
- 每帧统一 JSON 输出

本阶段不做：
- 双手
- Unity
- 实体机械手控制
- 5G
- 时序模型训练
- 深度相机复杂适配
- 复杂前端
- 数据库
- 云部署
- 多进程分布式

> **明确声明：当前仅支持 single right hand baseline，不包含 Unity、双手、实体机械手、5G 和时序模型训练。**

## 3) 环境安装（推荐与备选）
### 方案 1（推荐）：使用 `environment.yml`
```bash
cd single-hand-teleop-baseline
conda env create -f environment.yml
conda activate single-right-hand-baseline
```

### 方案 2（备选）：使用 `requirements.txt`
```bash
cd single-hand-teleop-baseline
conda create -n single-right-hand-baseline python=3.10 -y
conda activate single-right-hand-baseline
pip install -r requirements.txt
```

## 4) 运行方法
```bash
cd single-hand-teleop-baseline
python src/main.py --config configs/default.yaml
```

可选参数：
- `--camera-index 1`：切换摄像头索引
- `--print-json`：每帧打印 JSON 到控制台

## 5) 模块说明
- `src/capture/`：输入源抽象与 webcam 实现（预留 video file 接口）
- `src/perception/`：MediaPipe 单手检测、Right hand 过滤
- `src/features/`：几何工具与手部特征提取
- `src/gesture/`：规则手势识别
- `src/visualize/`：关键点覆盖与状态面板
- `src/output/`：JSON 输出（打印、保存最后一帧、预留 send）
- `src/utils/`：配置、日志、计时

## 6) JSON 输出格式说明
每帧统一对象（字段稳定）：

```json
{
  "timestamp": 1710000000.123,
  "detected": true,
  "handedness": "Right",
  "detection_confidence": 0.95,
  "gesture": "pinch",
  "pinch_distance": 0.08,
  "hand_open_ratio": 0.31,
  "finger_curl": {
    "thumb": 0.42,
    "index": 0.77,
    "middle": 0.81,
    "ring": 0.79,
    "little": 0.75
  },
  "landmarks_2d": [[0.50, 0.82], [0.47, 0.74], [0.45, 0.67], "... 共 21 个 [x, y] 点 ..."],
  "fps": 29.5,
  "latency_ms": 12.4
}
```

> `landmarks_2d` 在真实输出中固定为 21 个二维点。完整示例请参考 `examples/sample_output.json`。

## 7) 如何验证 demo 成功
1. 启动 `python src/main.py --config configs/default.yaml` 后，能打开摄像头窗口。
2. 检测到右手时，画面能绘制关键点与骨架。
3. 状态面板会显示并刷新：`gesture / fps / latency_ms`（以及 detected/handedness 等字段）。
4. 在 `configs/default.yaml` 的 `output_json_path` 指定路径，能看到 JSON 输出被刷新。
5. 按 `q` 安全退出，摄像头和窗口资源被释放。

## 8) 测试
```bash
cd single-hand-teleop-baseline
pytest -q
```

> 已通过 `pytest.ini` 配置测试路径，不需要手动设置 `PYTHONPATH`。

## 9) 后续扩展方向（当前不实现）
- 时序预测模型（如 TCN/Transformer/LSTM）
- Unity 接口对接
- SVH 右手控制接口
- Azure Kinect DK 输入适配

## 10) SVH Adapter Preview
当前 baseline 已新增一个 **SVH adapter / command preview skeleton**，目标是把现有单右手 AI 输出整理成未来可接灵巧手控制的接口骨架。

当前已经做到：
- 基于 `gesture / hand_open_ratio / pinch_distance_norm / finger_curl` 构造 `payload["svh"]`
- 输出 preview-oriented 的 `target_channels / target_positions`
- 提供 `mock transport`，用于让 adapter 层和未来真实发送层解耦
- 提供 `svh_protocol.py` 中的协议提示常量与 packet builder skeleton

当前明确没有做到：
- 没有连接真实 SVH
- 没有实现真实 TCP/IP + RS485 联调
- 没有声称已完成真实设备协议的全部长度/校验/字段验证

`svh` 字段示意：
```json
{
  "enabled": true,
  "mode": "preview",
  "valid": true,
  "command_source": "control_representation",
  "target_channels": [0, 1, 2, 3, 4],
  "target_positions": [0.12, 0.15, 0.18, 0.18, 0.18],
  "protocol_hint": {
    "set_control_state_addr": "0x09",
    "set_all_channels_addr": "0x03",
    "transport": "mock"
  }
}
```

说明：
- 这里的通道是 **preview-level abstraction**，不是最终真实 SVH 电机编号
- `pinch / open / fist` 会走不同的简化映射分支
- `invalid`、无手、质量门失败时不会残留旧命令，`svh.valid = false`
- 未来若要接真实硬件，应新增例如 `svh_transport_tcp.py`，并结合真实协议继续校准小端字节序、长度字段、校验字段与通道映射
