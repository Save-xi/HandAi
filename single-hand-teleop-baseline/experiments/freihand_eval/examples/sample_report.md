# FreiHAND Sample Report

这是一个格式示例，用于确认报告长什么样。表格里的数值是展示用口径，不代表 `sample_prediction.json` 的真实评估结果。真实评估结果请运行：

```bash
python scripts/run_current_pipeline_predictions.py --config configs/freihand_eval.yaml
python scripts/evaluate_predictions.py --config configs/freihand_eval.yaml --predictions outputs/current_pipeline_predictions.json
python scripts/make_report_table.py --config configs/freihand_eval.yaml
```

## PPT 表格示例

| 指标 | 数值 | PPT 口径 |
| --- | ---: | --- |
| 关键点完整率 | 85.7% | 预测是否完整输出 21 个 2D 关键点 |
| PCK@20px | 80.8% | 全数据口径，未检测帧计为失败 |
| 2D MPJPE | 8.102 px | 2D 平均关键点误差 |
| 平均 latency_ms | 19.134 ms | 单帧平均处理时间 |
| P95 latency_ms | 21.137 ms | 95% 帧不超过该耗时 |

## 解释口径

- PCK 越高越好，表示落在 2D 像素误差阈值内的关键点比例更高。
- MPJPE 越低越好，表示 2D 平均关键点误差更小。
- 关键点完整率用于说明模型是否稳定输出 21 个 2D 点。
- latency_ms 用于说明单帧处理速度，适合和实时遥操作需求放在同一页讲。
