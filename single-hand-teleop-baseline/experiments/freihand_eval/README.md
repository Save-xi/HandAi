# FreiHAND 视觉感知评估模块

这个目录是一个独立的离线评估工具包，用于在初期答辩里说明“视觉感知部分可以怎样量化评估”。

它不训练模型，不接 Unity，也不接实体灵巧手。当前阶段只做四件事：

1. 读取 FreiHAND annotation JSON。
2. 把 FreiHAND 的 3D 手部关键点按相机内参投影成 2D 像素点。
3. 将预测结果与 FreiHAND 2D 标注对齐，计算 PCK、MPJPE、关键点完整率和单帧耗时。
4. 输出 Markdown / JSON 报告，方便整理进 PPT。

## 数据集定位

FreiHAND 是单手关键点姿态估计数据集，适合评估“手部关键点定位是否准确”。

它不适合作为无手场景误检率数据集使用，因为 FreiHAND 样本本身主要围绕有手图像构建。后续如果要评估“没有手时会不会误检”，需要额外准备无手场景数据，例如桌面、背景、人脸、普通物体等负样本。

## 当前主要指标

- `keypoint_complete_rate`：预测是否完整输出 21 个 2D 关键点。
- `mpjpe_2d_px`：2D 平均关键点误差，单位 pixel。
- `PCK@5/10/20/30px`：2D 误差小于阈值的关键点比例。
- `per_joint_error`：每个关节的 2D 平均误差。
- `latency_ms`：预测端记录的单帧处理时间，报告平均值和 P95。

本模块不再计算 3D MPJPE / 3D PCK。当前项目的 MediaPipe `z` 不是 FreiHAND 相机坐标系下的真实 3D 坐标，强行对齐会让指标失真；答辩阶段只保留可信的 2D 视觉感知指标。

## 目录结构

```text
experiments/freihand_eval/
├── README.md
├── configs/
│   └── freihand_eval.yaml
├── scripts/
│   ├── inspect_annotations.py
│   ├── project_xyz_to_uv.py
│   ├── run_current_pipeline_predictions.py
│   ├── evaluate_predictions.py
│   ├── make_report_table.py
│   └── make_report_figures.py
├── freihand/
│   ├── __init__.py
│   ├── io.py
│   ├── projection.py
│   ├── metrics.py
│   └── report.py
└── examples/
    ├── sample_prediction.json
    └── sample_report.md
```

## 数据文件

把 FreiHAND annotation 放在项目根目录的 `dataset/` 下，例如：

```text
single-hand-teleop-baseline/dataset/
├── training_K.json
├── training_xyz.json
├── training_scale.json
├── evaluation_K.json
├── evaluation_xyz.json
└── evaluation_scale.json
```

`dataset/` 已经写入 `.gitignore`，不要提交到 GitHub。

## 配置

所有输入输出路径都从 `configs/freihand_eval.yaml` 读取。路径相对于 yaml 文件所在目录解析。

默认评估 `evaluation` split。如果需要切到 training，可以在命令行加：

```bash
python scripts/project_xyz_to_uv.py --config ../configs/freihand_eval.yaml --split training
```

## 使用流程

进入模块目录：

```bash
cd /d D:\VR\HandAi\single-hand-teleop-baseline\experiments\freihand_eval
```

检查 annotation：

```bash
python scripts/inspect_annotations.py --config configs/freihand_eval.yaml
```

输出：

```text
reports/annotation_inspection.md
```

把 3D annotation 投影成 2D keypoints：

```bash
python scripts/project_xyz_to_uv.py --config configs/freihand_eval.yaml
```

输出：

```text
outputs/freihand_2d_keypoints.json
```

运行当前项目里的 MediaPipe 单右手 CV 管线，生成 FreiHAND predictions：

```bash
python scripts/run_current_pipeline_predictions.py --config configs/freihand_eval.yaml
```

默认读取：

```text
dataset/evaluation/rgb/*.jpg
```

输出：

```text
outputs/current_pipeline_predictions.json
```

评估预测结果：

```bash
python scripts/evaluate_predictions.py --config configs/freihand_eval.yaml --predictions outputs/current_pipeline_predictions.json
```

默认读取：

```text
examples/sample_prediction.json
```

这个 sample 文件只用于确认 JSON 格式和报告生成流程。它只包含 2 个样本，并不是完整模型预测结果；直接拿它和完整 FreiHAND evaluation split 对齐时，完整率和 PCK 会很低，这是正常的。真正答辩或实验时，请把 MediaPipe / MMPose / 自己模型的完整预测导出为同样格式，再改配置里的 `paths.prediction_json` 或使用 `--predictions` 指向新文件。

输出：

```text
reports/eval_metrics.json
reports/eval_report.md
```

生成 PPT 可用表格：

```bash
python scripts/make_report_table.py --config configs/freihand_eval.yaml
```

输出：

```text
reports/ppt_report_table.md
```

生成 PPT 可用 SVG 图表：

```bash
python scripts/make_report_figures.py --config configs/freihand_eval.yaml
```

输出：

```text
reports/figures/detection_matrix.svg
reports/figures/pck_curve.svg
reports/figures/latency_card.svg
reports/figures/pipeline_eval_dashboard.svg
```

## predictions.json 格式

```json
{
  "00000000": {
    "keypoints_2d": [[x, y], "... 21 points"],
    "latency_ms": 12.3
  }
}
```

样本 id 使用 8 位字符串，例如 `00000000`、`00000001`。这和 FreiHAND annotation 的 list 顺序对齐。

## 后续接入方向

后续可以把 MediaPipe、MMPose 或其他手部关键点模型的预测结果转换成上述 `predictions.json` 格式，然后直接复用这里的评估脚本。

当前项目里的实时 baseline 主要输出归一化 2D landmark。后续如果要重新加入严格 3D 指标，需要先接入和 FreiHAND 相机坐标系一致的 3D 预测来源。
