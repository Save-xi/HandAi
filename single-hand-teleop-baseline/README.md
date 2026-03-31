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
- 双手、Unity、实体机械手控制、5G、深度相机复杂适配、时序模型训练、复杂前端、数据库、云部署、多进程分布式。

> **明确声明：当前仅支持 single right hand baseline，不包含 Unity 和实体机械手。**

## 3) 环境安装方法（Python 3.10+ / conda）
```bash
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
  "landmarks_2d": [[0.11, 0.22], [0.12, 0.25]],
  "fps": 29.5,
  "latency_ms": 12.4
}
```

## 7) 如何验证 demo 成功
1. 启动 `python src/main.py --config configs/default.yaml` 后，能打开摄像头窗口。
2. 右手进入画面，左侧显示 21 关键点和骨架。
3. 右侧状态区会更新：`detected / handedness / gesture / pinch_distance / hand_open_ratio / finger_curl / FPS / latency_ms`。
4. 在 `configs/default.yaml` 的 `output_json_path` 指定路径，看到最后一帧 JSON 被刷新。
5. 按 `q` 安全退出，摄像头和窗口资源被释放。

## 8) 测试
```bash
cd single-hand-teleop-baseline
pytest -q
```

## 9) 后续扩展方向（当前不实现）
- 时序预测模型（如 TCN/Transformer/LSTM）
- Unity 接口对接
- SVH 右手控制接口
- Azure Kinect DK 输入适配
