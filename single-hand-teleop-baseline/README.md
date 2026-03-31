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

本阶段明确不做：
- 双手
- Unity
- 实体机械手控制
- 5G 通信
- 复杂网页前端
- 训练脚本/时序模型训练
- 数据库与云部署

> **当前仅支持 single right hand baseline，不包含 Unity，不包含实体机械手，不包含双手。**

## 3) 环境安装（推荐 conda）
### 方案 A（推荐）：使用 environment.yml
```bash
conda env create -f environment.yml
conda activate single-right-hand-baseline
```

### 方案 B：已有环境时使用 requirements.txt
```bash
conda create -n single-right-hand-baseline python=3.10 -y
conda activate single-right-hand-baseline
pip install -r requirements.txt
```

## 4) 运行方法
在 `single-hand-teleop-baseline` 根目录执行：
```bash
python src/main.py --config configs/default.yaml
```

可选参数：
- `--camera-index 1`：切换摄像头索引
- `--print-json`：每帧打印 JSON 到控制台

## 5) 测试方法
在 `single-hand-teleop-baseline` 根目录直接运行：
```bash
pytest -q
```

无需手动设置 `PYTHONPATH=src`。

## 6) 模块说明
- `src/capture/`：输入源抽象与 webcam 实现（预留 video file 接口）
- `src/perception/`：MediaPipe 单手检测、Right hand 过滤
- `src/features/`：几何工具与手部特征提取
- `src/gesture/`：规则手势识别
- `src/visualize/`：关键点覆盖与状态面板
- `src/output/`：JSON 输出（打印、保存最后一帧、预留 send）
- `src/utils/`：配置、日志、计时

## 7) JSON 输出格式（稳定字段）
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
  "landmarks_2d": [[0.512, 0.802], [0.538, 0.748], [0.561, 0.697], [0.576, 0.650], [0.583, 0.609], [0.497, 0.651], [0.499, 0.581], [0.500, 0.521], [0.501, 0.472], [0.451, 0.643], [0.444, 0.564], [0.438, 0.500], [0.435, 0.449], [0.410, 0.658], [0.392, 0.594], [0.378, 0.542], [0.369, 0.499], [0.378, 0.692], [0.349, 0.648], [0.331, 0.610], [0.319, 0.579]],
  "fps": 29.5,
  "latency_ms": 12.4
}
```

## 8) 如何判断 demo 成功运行
1. 能成功打开摄像头窗口。
2. 当右手进入画面时，左侧显示 21 个关键点和骨架连线。
3. 右侧状态面板持续更新 `gesture / fps / latency_ms` 等信息。
4. `configs/default.yaml` 指定的 `output_json_path` 能看到最近一帧 JSON 输出。
5. 按 `q` 后程序正常退出，并释放窗口与摄像头资源。

## 9) 后续扩展方向（当前未实现）
- 时序预测模型接入
- Unity 接口接入
- SVH 控制接口接入
- Azure Kinect DK 输入接入
